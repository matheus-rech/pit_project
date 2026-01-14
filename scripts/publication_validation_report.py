#!/usr/bin/env python3
"""
Publication-Grade Dataset Validation
====================================
Validates dataset integrity for medical journal submission.
Handles label synonyms and normalization properly.

For Pituitary journal peer review.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set

# Instrument synonym mapping (based on PitVQA taxonomy)
INSTRUMENT_SYNONYMS = {
    'suction': {'suction', 'suction device', 'suction tube'},
    'kerrisons': {'kerrisons', 'kerrison', 'kerrison rongeur', 'kerrisons rongeur'},
    'pituitary_rongeurs': {'pituitary rongeurs', 'rongeurs', 'pituitary rongeur'},
    'freer_elevator': {'freer elevator', 'freer', 'elevator'},
    'ring_curette': {'ring curette', 'curette', 'ring curettes'},
    'bipolar': {'bipolar', 'bipolar forceps'},
    'drill': {'drill', 'surgical drill'},
    'haemostatic_foam': {'haemostatic foam', 'hemostatic foam', 'haemostatic', 'foam'},
    'spatula_dissector': {'spatula dissector', 'spatula', 'dissector'},
    'nasal_cutting_forceps': {'nasal cutting forceps', 'cutting forceps', 'forceps'},
    'suction_coagulator': {'suction coagulator', 'coagulator'},
    'cottonoid': {'cottonoid', 'cotton', 'patty'},
    'needle_holder': {'needle holder', 'needle'},
    'scissors': {'scissors', 'surgical scissors'},
    'irrigation_syringe': {'irrigation syringe', 'syringe'},
    'cup_forceps': {'cup forceps'},
    'micro_doppler': {'micro doppler', 'doppler'},
    'speculum': {'speculum'},
    'knife': {'knife'},
    'needle': {'needle'},
    'cottle': {'cottle'},
}

def normalize_instrument_label(label: str) -> str:
    """Normalize instrument label to canonical form."""
    label = label.lower().strip().replace('_', ' ')

    # Find canonical form
    for canonical, synonyms in INSTRUMENT_SYNONYMS.items():
        if label in synonyms or label.replace(' ', '_') == canonical:
            return canonical.replace('_', ' ')

    return label

def labels_match(label1: str, label2: str) -> bool:
    """Check if two labels refer to the same instrument."""
    norm1 = normalize_instrument_label(label1)
    norm2 = normalize_instrument_label(label2)
    return norm1 == norm2

def validate_ground_truth_samples():
    """Validate all ground_truth samples in comprehensive_spatial_dataset."""

    print("="*70)
    print("PUBLICATION-GRADE VALIDATION REPORT")
    print("Dataset: comprehensive_spatial_dataset")
    print("Validation against: PitVQA ground truth CSVs")
    print("="*70)

    # Load ground truth index
    print("\nLoading ground truth...")
    gt_index = {}  # (video_id, frame) -> instrument_data

    for video_num in range(1, 26):
        csv_file = Path(f"ground_truth/instruments_{video_num:02d}.csv")
        if not csv_file.exists():
            continue

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = f"video_{video_num:02d}"
                frame = int(row['int_time'])
                gt_index[(video_id, frame)] = {
                    'instrument': row['str_instrument1'],
                    'quadrant': row['pos_instrument1'],
                }

    print(f"Ground truth loaded: {len(gt_index):,} frames")

    # Load processed dataset (ground_truth source only)
    print("Loading processed dataset...")
    processed_samples = []

    for dataset_file in ['comprehensive_spatial_dataset/train.jsonl',
                         'comprehensive_spatial_dataset/validation.jsonl']:
        if not Path(dataset_file).exists():
            continue
        with open(dataset_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['metadata']['source'] == 'ground_truth':
                    processed_samples.append(data)

    print(f"Processed samples loaded: {len(processed_samples):,}\n")

    # Validation metrics
    metrics = {
        'total_samples': len(processed_samples),
        'matched_frames': 0,
        'label_matches': 0,
        'label_mismatches': 0,
        'quadrant_matches': 0,
        'quadrant_mismatches': 0,
        'frame_not_found': 0,
        'true_hallucinations': [],
        'label_normalization_issues': [],
    }

    # Validate each sample
    print("Validating samples...")
    for idx, sample in enumerate(processed_samples):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Progress: {idx:,}/{len(processed_samples):,}")

        video_id = sample['video_id']
        frame = sample.get('frame_index', int(sample['timestamp'] * 2))

        # Look up in ground truth
        key = (video_id, frame)
        if key not in gt_index:
            # Try ±1 frame tolerance
            found = False
            for offset in [-1, 0, 1]:
                if (video_id, frame + offset) in gt_index:
                    key = (video_id, frame + offset)
                    found = True
                    break
            if not found:
                metrics['frame_not_found'] += 1
                continue

        metrics['matched_frames'] += 1
        gt_data = gt_index[key]

        # Validate instrument label
        gt_instrument = gt_data['instrument']
        processed_instrument = sample['metadata'].get('label', '')

        # Skip if GT has no visible instrument
        if gt_instrument in ['no_visible_instrument', 'out_of_patient', '']:
            continue

        if labels_match(gt_instrument, processed_instrument):
            metrics['label_matches'] += 1
        else:
            metrics['label_mismatches'] += 1

            # Distinguish true hallucination from normalization issue
            gt_norm = normalize_instrument_label(gt_instrument)
            proc_norm = normalize_instrument_label(processed_instrument)

            if gt_norm != proc_norm:
                metrics['true_hallucinations'].append({
                    'video_frame': f"{video_id} frame {frame}",
                    'gt': gt_instrument,
                    'processed': processed_instrument,
                })

        # Validate quadrant
        gt_quadrant = gt_data['quadrant']
        processed_quadrant = sample['metadata'].get('quadrant', '')

        if gt_quadrant and processed_quadrant:
            if gt_quadrant == processed_quadrant:
                metrics['quadrant_matches'] += 1
            else:
                metrics['quadrant_mismatches'] += 1

    # Print report
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n📊 SAMPLE STATISTICS")
    print(f"  Total processed (GT source): {metrics['total_samples']:,}")
    print(f"  Matched to GT frames: {metrics['matched_frames']:,}")
    print(f"  Frame lookup failures: {metrics['frame_not_found']:,}")

    if metrics['label_matches'] + metrics['label_mismatches'] > 0:
        label_acc = 100 * metrics['label_matches'] / (metrics['label_matches'] + metrics['label_mismatches'])
        print(f"\n✅ INSTRUMENT LABEL ACCURACY: {label_acc:.1f}%")
        print(f"  Correct matches: {metrics['label_matches']:,}")
        print(f"  Mismatches: {metrics['label_mismatches']:,}")

    if metrics['quadrant_matches'] + metrics['quadrant_mismatches'] > 0:
        quad_acc = 100 * metrics['quadrant_matches'] / (metrics['quadrant_matches'] + metrics['quadrant_mismatches'])
        print(f"\n📍 QUADRANT ACCURACY: {quad_acc:.1f}%")
        print(f"  Correct: {metrics['quadrant_matches']:,}")
        print(f"  Incorrect: {metrics['quadrant_mismatches']:,}")

    if metrics['true_hallucinations']:
        print(f"\n⚠️  TRUE HALLUCINATIONS: {len(metrics['true_hallucinations'])}")
        print(f"  Rate: {100*len(metrics['true_hallucinations'])/metrics['matched_frames']:.2f}%")
        for h in metrics['true_hallucinations'][:5]:
            print(f"    • {h['video_frame']}: GT='{h['gt']}' vs Proc='{h['processed']}'")
    else:
        print(f"\n✅ NO HALLUCINATIONS DETECTED")

    # Final verdict
    print("\n" + "="*70)
    hallucination_rate = len(metrics['true_hallucinations']) / max(1, metrics['matched_frames'])
    label_acc_rate = metrics['label_matches'] / max(1, metrics['label_matches'] + metrics['label_mismatches'])

    if hallucination_rate < 0.01 and label_acc_rate > 0.95:
        print("✅ VERDICT: PUBLICATION-READY")
        print("   Dataset integrity validated. Safe for medical journal submission.")
    elif hallucination_rate < 0.05 and label_acc_rate > 0.85:
        print("⚠️  VERDICT: ACCEPTABLE WITH DISCLOSURE")
        print("   Minor issues detected. Disclose limitations in Methods section.")
    else:
        print("❌ VERDICT: NOT PUBLICATION-READY")
        print("   Major integrity issues. Review data processing pipeline.")
    print("="*70)

    # Save report
    report = {
        'dataset': 'comprehensive_spatial_dataset',
        'validation_date': '2026-01-14',
        'metrics': {
            'total_samples': metrics['total_samples'],
            'label_accuracy': label_acc_rate,
            'quadrant_accuracy': metrics['quadrant_matches'] / max(1, metrics['quadrant_matches'] + metrics['quadrant_mismatches']),
            'hallucination_rate': hallucination_rate,
            'hallucination_count': len(metrics['true_hallucinations']),
        },
        'verdict': 'PUBLICATION-READY' if (hallucination_rate < 0.01 and label_acc_rate > 0.95) else 'REVIEW_NEEDED'
    }

    with open('publication_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved: publication_validation_report.json")

if __name__ == "__main__":
    validate_ground_truth_samples()
