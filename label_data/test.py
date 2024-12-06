from pymongo import MongoClient
import os
from dotenv import load_dotenv
from typing import List, Dict
import json
from datetime import datetime
from utils.vhdl_segmenter import VHDLSegmenter 
class VHDLTester:
    def __init__(self, sample_size: int = 5):
        """Initialize the tester with database connection and segmenter."""
        load_dotenv()
        self.mongo = MongoClient(os.getenv('DB_URI')).hdl_database.hdl_codes  # Ensure DB_URI is in .env
        self.segmenter = VHDLSegmenter()
        self.sample_size = sample_size

    def fetch_samples(self) -> List[Dict]:
        """Fetch a sample of VHDL files from the entire database index."""
        return list(self.mongo.aggregate([
            {"$sample": {"size": self.sample_size}}
        ]))

    def analyze_sample(self, sample: Dict, idx: int) -> Dict:
        """Analyze a single VHDL file."""
        code = sample.get('content', '')
        if not code:
            return {'error': f"No content in sample {idx}"}
        
        try:
            segments = self.segmenter.segment_code(code)
            return {
                'document_id': str(sample['_id']),
                'segments_found': len(segments),
                'segment_details': [
                    {
                        'type': seg.segment_type,
                        'name': seg.name,
                        'lines': (seg.start_line, seg.end_line),
                        'content_preview': seg.content[:100] + '...' if len(seg.content) > 100 else seg.content
                    }
                    for seg in segments
                ]
            }
        except Exception as e:
            return {'error': f"Error analyzing sample {idx}: {str(e)}"}

    def run_tests(self) -> None:
        """Run tests and generate a detailed summary."""
        samples = self.fetch_samples()
        if not samples:
            print("No samples found.")
            return

        results = []
        segment_types = {}
        total_segments = 0

        print(f"\nRunning VHDL segmentation tests on {len(samples)} samples...\n")
        
        for idx, sample in enumerate(samples, 1):
            print(f"Processing sample {idx}...")
            result = self.analyze_sample(sample, idx)
            if 'error' in result:
                print(f"Error: {result['error']}")
                continue

            # Aggregate results
            total_segments += result['segments_found']
            for detail in result['segment_details']:
                segment_types[detail['type']] = segment_types.get(detail['type'], 0) + 1

            results.append(result)

            # Print details for current sample
            print(f"Found {result['segments_found']} segments:")
            for detail in result['segment_details']:
                print(f"- {detail['type']}: {detail['name'] if detail['name'] else 'unnamed'} "
                      f"(lines {detail['lines'][0]}-{detail['lines'][1]})")

        # Generate summary
        print("\n=== Test Summary ===")
        print(f"Total files processed: {len(samples)}")
        print(f"Total segments found: {total_segments}")
        print(f"Average segments per file: {total_segments / len(samples):.1f}")

        print("\nSegment type distribution:")
        for seg_type, count in sorted(segment_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_segments) * 100 if total_segments > 0 else 0
            print(f"- {seg_type}: {count} ({percentage:.1f}%)")

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vhdl_test_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(samples),
                'total_segments': total_segments,
                'segment_types': segment_types,
                'results': results
            }, f, indent=2)

        print(f"\nDetailed results saved to {filename}")

def main():
    """Entry point for the tester."""
    print("\n=== Starting VHDL Tester ===\n")
    tester = VHDLTester(sample_size=50)  # Adjust sample size as needed
    tester.run_tests()
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main()
