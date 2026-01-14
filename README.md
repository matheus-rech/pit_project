# PitVQA Surgical Workflow - Publication Package

**Clean, organized repository for publication, reproduction, and audit.**

🔗 **GitHub Repository**: https://github.com/matheus-rech/pit_project
📊 **Dataset**: https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial
🤖 **Model**: https://huggingface.co/mmrech/pitvqa-qwen2vl-spatial

---

## 📁 Directory Structure

```
publication_ready/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── MANIFEST.json                # Project manifest
│
├── gradio_demo.py               # 🎬 Interactive Gradio demo
├── GRADIO_DEMO_README.md        # Demo setup guide
├── test_gradio_dependencies.py  # Dependency checker
├── CORRECTED_VALIDATION_REPORT.json  # Proper metrics (100% data, 80% model)
│
├── scripts/                     # Core Python scripts
│   ├── train_unified_vlm.py
│   ├── evaluate_unified_vlm.py
│   ├── create_comprehensive_spatial_dataset.py
│   ├── validate_dataset_integrity.py
│   └── publication_validation_report.py
│
├── notebooks/                   # Jupyter/Colab notebooks
│   ├── train_spatial_qwen2vl_colab.ipynb
│   ├── train_and_demo_colab.ipynb
│   └── 01_upload_pitvqa_to_huggingface.ipynb
│
├── docs/                        # Documentation
│   ├── README.md
│   ├── COLAB_TRAINING_GUIDE.md
│   ├── VIDEO_DEMO_GUIDE.md
│   └── SPATIAL_TRAINING_PLAN.md
│
├── validation/                  # Validation reports
│   ├── final_validation_report.json
│   ├── publication_validation_report.json
│   └── dataset_integrity_report.json
│
├── data/                        # Data references (not actual data)
│   ├── DATA_SOURCES.md          # Links to HuggingFace datasets
│   └── GROUND_TRUTH_README.md
│
├── models/                      # Model references (not actual models)
│   └── MODELS.md                # Links to HuggingFace models
│
└── validation/                  # Validation reports
    ├── final_validation_report.json
    ├── publication_validation_report.json
    └── dataset_integrity_report.json
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model (Colab)

```bash
# Upload to Google Colab:
notebooks/train_and_demo_colab.ipynb

# Or use the training-only notebook:
notebooks/train_spatial_qwen2vl_colab.ipynb
```

### 3. Validate Results

```bash
python scripts/validate_dataset_integrity.py
```

---

## 📊 Dataset

**Primary:** `mmrech/pitvqa-comprehensive-spatial` (HuggingFace)
- 9,125 training samples
- 1,014 validation samples
- 100% ground truth accuracy (validated)

See `data/DATA_SOURCES.md` for details.

---

## 🤖 Models

**Baseline:** `mmrech/pitvqa-qwen2vl-unified`
**Publication:** `mmrech/pitvqa-qwen2vl-spatial` (train with notebook)

See `models/MODELS.md` for details.

---

## 📖 Documentation

| Guide | Purpose |
|-------|---------|
| `docs/COLAB_TRAINING_GUIDE.md` | Step-by-step Colab training |
| `docs/VIDEO_DEMO_GUIDE.md` | Creating demo videos |
| `docs/SPATIAL_TRAINING_PLAN.md` | Complete training roadmap |

---

## ✅ Validation

All validation reports in `validation/`:
- Dataset integrity: 100% accuracy
- No AI hallucinations: 0%
- Ground truth fidelity: Perfect

---

## 📝 Citation

```bibtex
@article{yourname2026pitvqa,
  title={PitVQA: Multi-Task Vision-Language Model for Pituitary Surgery},
  author={Your Name and Collaborators},
  journal={Medical Image Analysis},
  year={2026}
}
```

---

## 📧 Contact

For questions: your.email@institution.edu

---

**Generated:** 2026-01-14 10:58:19
**Organization script:** `organize_for_publication.py`
