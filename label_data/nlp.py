import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig  # Important for HF model saving
)
import torch
from torch.utils.data import Dataset
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import json
import traceback
from collections import Counter
from tqdm import tqdm

from utils.vhdl_segmenter import VHDLSegmenter
from utils.model_logger import ModelLogger

class SegmentDataset(Dataset):
    """Dataset class for VHDL code segments during training.
    
    This class handles the batching and indexing of our tokenized code segments
    and their corresponding labels."""
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Tokenized input texts
        self.labels = labels        # Corresponding labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class HDLSegmentTrainer:
    """Trains models for analyzing VHDL code segments and patterns.
    
    This trainer handles both segment-level features and full-code design patterns,
    managing the entire pipeline from data loading through training and model saving."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """Initialize trainer with model and connections."""


        # Initialize connections and base components
        load_dotenv()
        self.mongo_client = MongoClient(os.getenv('DB_URI'))
        self.db = self.mongo_client['hdl_database']
        self.collection = self.db['hdl_codes']
        
        # Verify HuggingFace token exists
        if not os.getenv('HF_TOKEN'):
            raise ValueError("HF_TOKEN environment variable required for model saving")
        
        # Set up model components
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = ModelLogger()
        self.segmenter = VHDLSegmenter()
        
        # Classification mappings
        self.feature_mapping = {}  # For segment features
        self.pattern_mapping = {}  # For design patterns
        self.component_mapping = {}  # For component types


    def load_pattern_data(self) -> Tuple[List[str], List[int]]:
        """Load full-code design pattern data from MongoDB."""
        texts, labels = [], []
        
        # Get all analyzed documents
        cursor = self.collection.find({"analysis": {"$exists": True}})
        unique_patterns = set()
        
        # First pass: collect unique patterns
        for doc in tqdm(cursor, desc="Collecting patterns"):
            pattern = doc.get('analysis', {}).get('design_pattern')
            if pattern:
                unique_patterns.add(str(pattern))
        
        # Create pattern mapping
        self.pattern_mapping = {
            pattern: idx for idx, pattern in enumerate(sorted(unique_patterns))
        }
        
        # Second pass: collect full code samples and their patterns
        cursor.rewind()
        for doc in tqdm(cursor, desc="Processing documents"):
            content = doc.get('content')
            pattern = doc.get('analysis', {}).get('design_pattern')
            if content and pattern and pattern in self.pattern_mapping:
                texts.append(content)
                labels.append(self.pattern_mapping[pattern])
        
        print(f"\nCollected {len(texts)} samples with {len(unique_patterns)} unique patterns")
        return texts, labels

    def load_segment_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load pre-labeled segment data using VHDLSegmenter."""
        texts = []
        labels = []
        segment_types = []
        feature_counts = Counter()

        # Get all analyzed documents
        cursor = self.collection.find({"analysis": {"$exists": True}})
        
        print("Analyzing feature distribution...")
        # First pass: count feature occurrences
        for doc in cursor:
            features = doc.get('analysis', {}).get('key_features', [])
            for feature in features:
                if isinstance(feature, dict) and feature.get('key_feature'):
                    feature_counts[feature['key_feature']] += 1
        
        # Filter for features that appear at least 10 times
        common_features = {feature: count for feature, count in feature_counts.items() 
                        if count >= 10}
        
        print(f"\nFound {len(common_features)} features with 10+ occurrences")
        
        # Create mapping of common features to indices
        self.feature_mapping = {feature: idx for idx, feature 
                            in enumerate(sorted(common_features.keys()))}
        
        # Reset cursor for main data collection
        cursor.rewind()
        print("\nProcessing documents...")
        for doc in tqdm(cursor):
            content = doc.get('content')
            if not content:
                continue

            # Get segments for this document
            segments = self.segmenter.segment_code(content)
            segment_by_type = {seg.segment_type: seg for seg in segments if seg.segment_type}

            # Match features to segments
            features = doc.get('analysis', {}).get('key_features', [])
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                    
                key_feature = feature.get('key_feature')
                seg_type = feature.get('segment_type')
                
                # Only use features that appear frequently
                if key_feature in self.feature_mapping and seg_type in segment_by_type:
                    segment = segment_by_type[seg_type]
                    texts.append(segment.content)
                    labels.append(self.feature_mapping[key_feature])
                    segment_types.append(seg_type)

        print(f"\nCollected {len(texts)} segments with {len(common_features)} common features")
        
        # Print distribution of top 10 features
        print("\nTop 10 most common features:")
        for feature, count in sorted(common_features.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{feature}: {count} samples")

        return texts, labels, segment_types

    def train_model(self, model_type: str, output_dir: str = "hdl_models"):
        """Train model on common HDL code features."""
        try:
            # Load labeled segments
            texts, labels, segment_types = self.load_segment_data()
            num_labels = len(self.feature_mapping)
            print(f"\nStarting training for {num_labels} common features")
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels,
                test_size=0.2,
                random_state=42
            )
            print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")
            
            # Prepare data
            train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
            val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)
            
            train_dataset = SegmentDataset(train_encodings, train_labels)
            val_dataset = SegmentDataset(val_encodings, val_labels)
            
            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
            
            # Setup training
            training_args = TrainingArguments(
                output_dir=f"{output_dir}/{model_type}",
                num_train_epochs=27,
                per_device_train_batch_size=16,
                learning_rate=2e-5,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                push_to_hub=True
            )
            
            # Custom metrics
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return {
                    'accuracy': accuracy_score(labels, predictions)
                }
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )
            
            print("\nTraining started...")
            trainer.train()
            
            # Save final model
            final_path = f"{output_dir}/{model_type}_final"
            trainer.save_model(final_path)
            self.tokenizer.save_pretrained(final_path)
            
            # Save feature mapping for inference
            mapping_path = os.path.join(final_path, "feature_mapping.json")
            with open(mapping_path, 'w') as f:
                json.dump(self.feature_mapping, f, indent=2)
            
            print(f"\nTraining complete. Model and mappings saved to {final_path}")
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()
def main():
    """Main execution function."""
    try:
        print("\n=== Starting HDL Model Training ===")
        
        trainer = HDLSegmentTrainer()
        
        # Train both classifiers
        trainer.train_model('segment')
        #trainer.train_model('pattern')
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        if 'trainer' in locals():
            trainer.close()
        print("\n=== HDL Model Training Complete ===")

if __name__ == "__main__":
    main()