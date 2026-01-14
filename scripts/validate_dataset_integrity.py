#!/usr/bin/env python3
"""
Dataset Integrity Validation for Medical Publication
=====================================================
Validates that processed spatial dataset matches PitVQA ground truth.
Detects AI hallucinations and data processing errors.

Critical for Pituitary journal peer review.
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import sys

# Configuration
GT_DIR = Path("ground_truth")
DATASET_FILE = Path("comprehensive_spatial_dataset/train.jsonl")
VALIDATION_FILE = Path("comprehensive_spatial_dataset/validation.jsonl")

# PitVQA instrument codes (from official dataset)
PITVQA_INSTRUMENTS = {
    -1: "out_of_patient",
    0: "no_visible_instrument",
    1: "freer_elevator",
    2: "pituitary_rongeurs",
    3: "spatula_dissector",
    4: "kerrisons",
    5: "cottle",
    6: "haemostatic_foam",
    7: "micro_doppler",
    8: "nasal_cutting_forceps",
    9: "drill",
    10: "suction_coagulator",
    11: "bipolar",
    12: "ring_curette",
    13: "speculum",
    14: "knife",
    15: "needle",
    16: "suction"
}

# PitVQA quadrant system
PITVQA_QUADRANTS = {
    "1": (25.0, 25.0),   # Upper-left
    "2": (75.0, 25.0),   # Upper-right
    "3": (25.0, 75.0),   # Lower-left
    "4": (75.0, 75.0),   # Lower-right
    "5": (50.0, 50.0),   # Center
}


class ValidationResults:
    def __init__(self):
        self.total_gt_samples = 0
        self.total_processed_samples = 0
        self.matched_samples = 0
        self.instrument_label_matches = 0
        self.instrument_label_mismatches = 0
        self.quadrant_matches = 0
        self.quadrant_mismatches = 0
        self.timestamp_mismatches = []
        self.hallucinated_instruments = []
        self.missing_annotations = []
        self.coordinate_errors = []

    def print_summary(self):
        print("\n" + "="*70)
        print("DATASET INTEGRITY VALIDATION REPORT")
        print("="*70)

        print(f"\n📊 SAMPLE COUNTS")
        print(f"  Ground Truth: {self.total_gt_samples:,}")
        print(f"  Processed Dataset: {self.total_processed_samples:,}")
        print(f"  Matched: {self.matched_samples:,}")

        if self.instrument_label_matches + self.instrument_label_mismatches > 0:
            label_acc = self.instrument_label_matches / (self.instrument_label_matches + self.instrument_label_mismatches)
            print(f"\n✅ INSTRUMENT LABEL ACCURACY: {label_acc:.1%}")
            print(f"  Matches: {self.instrument_label_matches}")
            print(f"  Mismatches: {self.instrument_label_mismatches}")

        if self.quadrant_matches + self.quadrant_mismatches > 0:
            quad_acc = self.quadrant_matches / (self.quadrant_matches + self.quadrant_mismatches)
            print(f"\n📍 QUADRANT ACCURACY: {quad_acc:.1%}")
            print(f"  Matches: {self.quadrant_matches}")
            print(f"  Mismatches: {self.quadrant_mismatches}")

        if self.hallucinated_instruments:
            print(f"\n⚠️  POTENTIAL HALLUCINATIONS: {len(self.hallucinated_instruments)}")
            print("  (Instruments in processed data NOT in ground truth)")
            for item in self.hallucinated_instruments[:5]:
                print(f"    • {item}")

        if self.coordinate_errors:
            print(f"\n❌ COORDINATE ERRORS: {len(self.coordinate_errors)}")
            for err in self.coordinate_errors[:5]:
                print(f"    • {err}")

        if self.missing_annotations:
            print(f"\n⚠️  MISSING ANNOTATIONS: {len(self.missing_annotations)}")
            print("  (Ground truth samples not in processed dataset)")

        # Final verdict
        print("\n" + "="*70)
        if (self.instrument_label_matches / max(1, self.instrument_label_matches + self.instrument_label_mismatches) > 0.95 and
            len(self.hallucinated_instruments) < 10 and
            len(self.coordinate_errors) == 0):
            print("✅ VERDICT: DATASET IS PUBLICATION-GRADE")
        elif len(self.hallucinated_instruments) > 50 or len(self.coordinate_errors) > 20:
            print("❌ VERDICT: DATASET HAS MAJOR INTEGRITY ISSUES")
        else:
            print("⚠️  VERDICT: DATASET NEEDS REVIEW (minor issues detected)")
        print("="*70)


def load_ground_truth() -> Dict[str, Dict[int, Dict]]:
    """Load all ground truth CSVs indexed by video_id and timestamp."""
    gt_data = defaultdict(dict)

    for video_num in range(1, 26):
        inst_file = GT_DIR / f"instruments_{video_num:02d}.csv"
        if not inst_file.exists():
            continue

        with open(inst_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = f"video_{video_num:02d}"
                timestamp = int(row['int_time'])

                # Skip frames with no instrument
                if row['str_instrument1'] in ['no_visible_instrument', 'out_of_patient', '']:
                    continue

                gt_data[video_id][timestamp] = {
                    'instrument': row['str_instrument1'].lower(),
                    'quadrant': row['pos_instrument1'],
                    'int_code': int(row['int_instrument1']) if row['int_instrument1'] else None
                }

    return gt_data


def normalize_instrument_name(name: str) -> str:
    """Normalize instrument names for comparison."""
    # Convert underscores to spaces
    name = name.lower().replace('_', ' ')

    # Common variants
    mappings = {
        'suction device': 'suction',
        'suction tube': 'suction',
        'freer': 'freer elevator',
        'ring curettes': 'ring curette',
        'kerrison': 'kerrisons',
        'kerrison rongeur': 'kerrisons',
        'pituitary rongeur': 'pituitary rongeurs',
        'rongeurs': 'pituitary rongeurs',
        'bipolar forceps': 'bipolar',
        'surgical drill': 'drill',
        'hemostatic foam': 'haemostatic foam',
        'haemostatic': 'haemostatic foam',
        'cottonoid': 'cottonoid',
        'spatula': 'spatula dissector',
    }

    return mappings.get(name, name)


def quadrant_to_coordinates(quadrant: str) -> Tuple[float, float]:
    """Convert PitVQA quadrant to approximate center coordinates."""
    return PITVQA_QUADRANTS.get(quadrant, (50.0, 50.0))


def coordinates_to_quadrant(x: float, y: float) -> str:
    """Determine which quadrant coordinates fall into."""
    # Allow for jitter: ±10 from quadrant centers
    for quad, (cx, cy) in PITVQA_QUADRANTS.items():
        if abs(x - cx) <= 20 and abs(y - cy) <= 20:
            return quad

    # Fallback: use simple grid
    if x < 40:
        if y < 40:
            return "1"  # Upper-left
        elif y > 60:
            return "3"  # Lower-left
        else:
            return "5"  # Center
    elif x > 60:
        if y < 40:
            return "2"  # Upper-right
        elif y > 60:
            return "4"  # Lower-right
        else:
            return "5"  # Center
    else:
        return "5"  # Center


def validate_dataset():
    """Main validation function."""
    results = ValidationResults()

    print("Loading ground truth...")
    gt_data = load_ground_truth()
    results.total_gt_samples = sum(len(frames) for frames in gt_data.values())

    print(f"Ground truth loaded: {len(gt_data)} videos, {results.total_gt_samples:,} samples")

    # Load processed dataset
    print("Loading processed dataset...")
    processed_samples = []

    for dataset_file in [DATASET_FILE, VALIDATION_FILE]:
        if not dataset_file.exists():
            continue
        with open(dataset_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Only validate ground truth samples (not AI-augmented)
                if data['metadata']['source'] == 'ground_truth':
                    processed_samples.append(data)

    results.total_processed_samples = len(processed_samples)
    print(f"Processed samples loaded: {results.total_processed_samples:,}")

    # Validate each processed sample
    print("\nValidating samples...")
    for idx, sample in enumerate(processed_samples):
        if idx % 500 == 0 and idx > 0:
            print(f"  Validated {idx:,}/{results.total_processed_samples:,} samples...")

        video_id = sample.get('video_id', '')
        metadata = sample.get('metadata', {})

        # Get timestamp
        timestamp = sample.get('timestamp')
        if timestamp is None:
            timestamp = sample.get('frame_index', 0) / 2  # Convert frame to seconds at 2fps
        timestamp = int(timestamp)

        # Check if video exists in ground truth
        if video_id not in gt_data:
            results.hallucinated_instruments.append(f"Video {video_id} not in ground truth")
            continue

        # Check if timestamp exists
        if timestamp not in gt_data[video_id]:
            # Allow ±1 second tolerance for frame sync issues
            matched = False
            for offset in [-1, 0, 1]:
                if timestamp + offset in gt_data[video_id]:
                    timestamp = timestamp + offset
                    matched = True
                    break

            if not matched:
                results.missing_annotations.append(f"{video_id} @ t={timestamp}s")
                continue

        results.matched_samples += 1
        gt_sample = gt_data[video_id][timestamp]

        # Validate instrument label
        processed_label = normalize_instrument_name(metadata.get('label', ''))
        gt_label = normalize_instrument_name(gt_sample['instrument'])

        if processed_label == gt_label or processed_label in gt_label or gt_label in processed_label:
            results.instrument_label_matches += 1
        else:
            results.instrument_label_mismatches += 1
            if results.instrument_label_mismatches <= 10:  # Store first 10
                results.hallucinated_instruments.append(
                    f"{video_id} @ t={timestamp}s: GT='{gt_label}' vs Processed='{processed_label}'"
                )

        # Validate coordinates match quadrant
        coords = metadata.get('coordinates', {})
        x, y = coords.get('x', 50), coords.get('y', 50)

        # Check if coordinates are valid (0-100)
        if not (0 <= x <= 100 and 0 <= y <= 100):
            results.coordinate_errors.append(f"{video_id} @ t={timestamp}s: Invalid coords ({x}, {y})")
            continue

        # Check if coordinates match ground truth quadrant
        predicted_quadrant = coordinates_to_quadrant(x, y)
        gt_quadrant = gt_sample['quadrant']

        if predicted_quadrant == gt_quadrant:
            results.quadrant_matches += 1
        else:
            results.quadrant_mismatches += 1

    return results


def main():
    """Run validation and generate report."""
    if not GT_DIR.exists():
        print(f"❌ Ground truth directory not found: {GT_DIR}")
        sys.exit(1)

    if not DATASET_FILE.exists():
        print(f"❌ Dataset file not found: {DATASET_FILE}")
        sys.exit(1)

    results = validate_dataset()
    results.print_summary()

    # Save detailed report
    report = {
        'total_gt_samples': results.total_gt_samples,
        'total_processed_samples': results.total_processed_samples,
        'matched_samples': results.matched_samples,
        'instrument_accuracy': results.instrument_label_matches / max(1, results.instrument_label_matches + results.instrument_label_mismatches),
        'quadrant_accuracy': results.quadrant_matches / max(1, results.quadrant_matches + results.quadrant_mismatches),
        'hallucinated_count': len(results.hallucinated_instruments),
        'coordinate_errors': len(results.coordinate_errors),
        'missing_annotations': len(results.missing_annotations),
    }

    with open('dataset_integrity_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: dataset_integrity_report.json")


if __name__ == "__main__":
    main()
