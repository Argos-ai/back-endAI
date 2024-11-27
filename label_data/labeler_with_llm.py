import os
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from utils.vhdl_segmenter import VHDLSegmenter, CodeSegment
import time

class HDLAnalyzer:
    """Simple HDL code analyzer using OpenAI API and code segmentation."""
    
    def __init__(self):
        # Initialize connections
        load_dotenv()
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.mongo = MongoClient(os.getenv('DB_URI')).hdl_database.hdl_codes
        self.segmenter = VHDLSegmenter()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup basic logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'hdl_analysis_{datetime.now():%Y%m%d_%H%M%S}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def analyze_code(self, code: str) -> Dict:
        """Analyze HDL code by combining overall analysis with segment-specific analysis."""
        try:
            # Get overall design pattern analysis
            design_analysis = self._get_design_pattern(code)
            if not design_analysis:
                return {'error': 'Failed to analyze design pattern'}

            # Get segment-specific features
            segments = self.segmenter.segment_code(code)
            segment_features = self._analyze_segments(segments)

            # Combine analyses
            return {
                'design_pattern': design_analysis.get('design_pattern'),
                'component_type': design_analysis.get('component_type'),
                'complexity_level': design_analysis.get('complexity_level'),
                'key_features': segment_features,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {'error': str(e)}

    def _get_design_pattern(self, code: str) -> Optional[Dict]:
        """Get high-level design pattern analysis."""
        prompt = """Analyze this VHDL code and provide a response in strict JSON format with these fields:
            {
                "design_pattern": "The main architectural pattern used",
                "component_type": "One of: [StateMachine/Counter/ALU/Package/Interface/Memory/Controller/Decoder/TestBench/Other]",
                "complexity_level": "Integer 1-5"
            }
            
            Ensure your response is a valid JSON object with exactly these fields."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Code to analyze:\n{code}"}
                ],
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Design pattern analysis failed: {str(e)}")
            return None

    def _analyze_segments(self, segments: List[CodeSegment]) -> List[Dict]:
        """Analyze each code segment for its key features and semantic tags."""
        features = []

        for segment in segments:
            # Use semantic tags in analysis
            tags = ", ".join(segment.semantic_tags or [])
            prompt = f"""You are analyzing a {segment.segment_type} code segment with the following semantic tags: {tags}. 
            Respond with a **strict JSON object** containing exactly the following fields:
            {{
                "key_feature": "The single most important feature/characteristic of the segment.",
                "purpose": "A brief description of the segment's role or purpose."
            }}
            Your response must be valid JSON with no extra fields or text outside the JSON object."""

            try:
                # Call OpenAI's Chat API
                response = self.openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Code segment:\n{segment.content}"}
                    ],
                    temperature=0.3
                )

                # Extract and validate the response
                response_content = response.choices[0].message.content.strip()
                analysis = json.loads(response_content)

                # Add the analysis to the results
                features.append({
                    "segment_type": segment.segment_type,
                    "semantic_tags": segment.semantic_tags,
                    "key_feature": analysis.get("key_feature"),
                    "purpose": analysis.get("purpose"),
                    "lines": (segment.start_line, segment.end_line),
                    "parent": segment.parent,
                })

            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON response for segment: {segment.segment_type}")
            except Exception as e:
                self.logger.warning(f"Segment analysis failed: {str(e)}")
                continue

        return features

    def process_batch(self, batch_size: int = 20) -> None:
        """Process a batch of unanalyzed documents."""
        try:
            # Get unanalyzed documents
            docs = list(self.mongo.find(
                {"analysis": {"$exists": False}},
                {"_id": 1, "content": 1}
            ).limit(batch_size))

            for doc in docs:
                try:
                    self.logger.info(f"Processing document {doc['_id']}")
                    
                    # Analyze the code
                    analysis = self.analyze_code(doc['content'])
                    
                    # Update database
                    self.mongo.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"analysis": analysis}}
                    )
                    
                    self.logger.info(f"Successfully analyzed document {doc['_id']}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process document {doc['_id']}: {str(e)}")
                    self.mongo.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"processing_failed": True}}
                    )

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")

def main():
    """Run the analyzer to process 500 documents in total."""
    analyzer = HDLAnalyzer()
    total_to_process = 500  # Total number of documents to process
    batch_size = 20  # Process 20 documents per batch
    processed = 0

    while processed < total_to_process:
        analyzer.logger.info(f"Processing batch {processed + 1} to {processed + batch_size}...")
        
        # Process a batch
        analyzer.process_batch(batch_size=batch_size)
        
        # Update the number of processed documents
        processed += batch_size
        
        # Log progress
        analyzer.logger.info(f"Processed {processed}/{total_to_process} documents.")
        
        # Optional: Add a delay if needed (e.g., to prevent overloading the database or API)
        time.sleep(1)

    analyzer.logger.info(f"Completed processing {total_to_process} documents.")
if __name__ == "__main__":
    main()