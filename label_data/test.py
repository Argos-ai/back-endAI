from pymongo import MongoClient
import os
from dotenv import load_dotenv
from typing import List, Dict
import json
from datetime import datetime
from utils.vhdl_segmenter import VHDLSegmenter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

class VHDLTester:
    def __init__(self, sample_size: int = 5):
        # Initialize connections
        load_dotenv()
        self.mongo = MongoClient(os.getenv('DB_URI')).hdl_database.hdl_codes
        self.segmenter = VHDLSegmenter()
        self.sample_size = sample_size

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.segment_model = AutoModelForSequenceClassification.from_pretrained("hdl_models/segment_final")
        
        # Load feature mapping
        with open("hdl_models/segment_final/feature_mapping.json") as f:
            self.feature_mapping = json.load(f)
        print(f"Loaded {len(self.feature_mapping)} features")

    def predict_segment(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.segment_model(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
        
        # Convert prediction index to feature name
        for feature, idx in self.feature_mapping.items():
            if idx == pred_idx:
                return feature
        return f"Unknown-{pred_idx}"

    def analyze_code(self, code: str) -> Dict:
        # Segment and analyze code
        segments = self.segmenter.segment_code(code)
        results = []
        
        for seg in segments:
            if not seg.content.strip():
                continue
                
            prediction = self.predict_segment(seg.content)
            results.append({
                'type': seg.segment_type,
                'lines': (seg.start_line, seg.end_line),
                'prediction': prediction,
                'content': seg.content[:100] + '...' if len(seg.content) > 100 else seg.content
            })
            
        return results

    def run_tests(self):
        print(f"\nTesting model on {self.sample_size} samples...")
        
        # Get samples with analysis
        samples = list(self.mongo.aggregate([
            {"$match": {"analysis": {"$exists": True}}},
            {"$sample": {"size": self.sample_size}}
        ]))
        
        results = []
        for sample in tqdm(samples, desc="Analyzing samples"):
            code = sample.get('content', '')
            if not code:
                continue
                
            # Get predictions
            predictions = self.analyze_code(code)
            
            # Get actual features
            actual = []
            if 'analysis' in sample and 'key_features' in sample['analysis']:
                actual = [f['key_feature'] for f in sample['analysis']['key_features']]
            
            # Store results
            results.append({
                'id': str(sample['_id']),
                'predictions': [p['prediction'] for p in predictions],
                'actual_features': actual,
                'details': predictions
            })
            
            # Print results for this sample
            print(f"\nSample {sample['_id']}:")
            print("Predictions:")
            for p in predictions:
                print(f"- {p['type']}: {p['prediction']}")
            print("Actual features:", actual)
        
        # Save results
        output_file = f"test_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to {output_file}")

def main():
    print("\n=== Starting HDL Model Testing ===")
    tester = VHDLTester(sample_size=5)
    tester.run_tests()
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main()