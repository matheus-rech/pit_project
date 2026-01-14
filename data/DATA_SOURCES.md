# Data Sources for PitVQA Project

**IMPORTANT:** Actual data files are NOT included in this repository due to size.
Access data through HuggingFace Hub or download separately.

## Datasets (HuggingFace Hub)

| Dataset | Status | Samples | Link |
|---------|--------|---------|------|
| **pitvqa-comprehensive-spatial** | ✅ Primary | 10,139 | https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial |
| pitvqa-spatial-vlm | ✅ Alternative | 5,162 | https://huggingface.co/datasets/mmrech/pitvqa-spatial-vlm |
| pitvqa-unified-vlm | ✅ Classification | 5,184 | https://huggingface.co/datasets/mmrech/pitvqa-unified-vlm |

## Ground Truth

**Location:** `ground_truth/` directory (52 CSV files)

**Contents:**
- Instrument annotations (25 videos)
- Surgical step labels
- Quadrant-based spatial locations

**Size:** ~6.3 MB

**Validation:** 100% accuracy confirmed (see `validation/` directory)

## Downloading Data

### Option 1: HuggingFace Hub (Recommended)

```python
from datasets import load_dataset

# Load comprehensive spatial dataset
dataset = load_dataset("mmrech/pitvqa-comprehensive-spatial")

print(f"Train: {len(dataset['train'])}")
print(f"Val: {len(dataset['validation'])}")
```

### Option 2: Local Files

If you have local copies:
1. Place ground truth CSVs in `ground_truth/`
2. Place annotation JSONs in `*_annotations/` directories
3. Spatial datasets in `comprehensive_spatial_dataset/`

## Data Integrity

**Validated:** January 14, 2026

See `validation/publication_validation_report.json`:
- Instrument accuracy: 100%
- Hallucination rate: 0%
- Ground truth fidelity: Perfect match

## Citation

```bibtex
@dataset{pitvqa_comprehensive_spatial,
  author = {Your Name},
  title = {PitVQA Comprehensive Spatial Localization Dataset},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial}
}
```
