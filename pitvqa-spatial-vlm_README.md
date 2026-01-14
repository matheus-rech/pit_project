---
license: cc-by-nc-nd-4.0
task_categories:
  - visual-question-answering
  - object-detection
  - image-to-text
language:
  - en
tags:
  - medical
  - surgery
  - pituitary
  - spatial-reasoning
  - instrument-detection
  - surgical-workflow
  - vision-language
  - qwen2-vl
  - lora
  - coordinates
  - prototype
size_categories:
  - 1K<n<10K
pretty_name: PitVQA Spatial VLM Dataset (Early Version)
---

# PitVQA Spatial VLM Dataset (Early Version)

Early prototype spatial localization dataset for pituitary surgery. **Note**: For production use, please use [mmrech/pitvqa-comprehensive-spatial](https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial) which has 10,139 validated samples.

🔗 **GitHub**: https://github.com/matheus-rech/pit_project
🚀 **Updated Version**: [mmrech/pitvqa-comprehensive-spatial](https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial) (recommended)
📄 **Original Dataset**: [UCL Research Data Repository](https://doi.org/10.5522/04/27004666)

## ⚠️ Important Notice

This is an **early prototype version** of the spatial localization dataset. For current research and production use, we recommend:

**👉 Use [mmrech/pitvqa-comprehensive-spatial](https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial) instead**

### Why Use the Comprehensive Version?

| Feature | This Dataset (Early) | Comprehensive (Current) |
|---------|---------------------|------------------------|
| Samples | ~3,000-5,000 | 10,139 |
| Validation | Partial | 100% verified |
| Coverage | Limited | Complete workflow |
| Documentation | Basic | Comprehensive |
| Model Performance | Baseline | State-of-the-art |
| Recommended | ❌ No | ✅ Yes |

## Dataset Description

This early-stage dataset contains spatial annotations for surgical instrument localization in pituitary surgery. It served as a proof-of-concept for the spatial localization task.

### Key Features

- 🎯 **Spatial Coordinates**: Normalized (x, y) coordinates in 0-100 scale
- 🔧 **Surgical Instruments**: Basic instrument categories
- 🧪 **Prototype Phase**: Early development version
- 📊 **Limited Coverage**: Subset of complete surgical workflow

### Historical Context

This dataset was created during the **initial development phase** of the PitVQA spatial localization project. It helped establish:

1. Feasibility of spatial localization with VLMs
2. Coordinate format (normalized 0-100 scale)
3. Question-answering structure for spatial queries
4. Baseline performance metrics

### Evolution Path

```
pitvqa-unified-vlm (Classification)
         ↓
pitvqa-spatial-vlm (Early Spatial) ← You are here
         ↓
pitvqa-comprehensive-spatial (Production) ← Recommended
```

## Data Format

### Sample Structure

```python
{
    "image": PIL.Image,           # Surgical frame
    "question": str,               # Spatial query
    "answer": str,                 # Format: "<point x='45.2' y='68.3'>object</point>"
    "video_id": str,               # Source video
    "frame_number": int            # Frame index
}
```

### Coordinate Format

```xml
<point x='45.2' y='68.3'>suction device</point>
```

## Migration Guide

### Upgrading to Comprehensive Version

If you're currently using this dataset, migration is straightforward:

```python
# Old (Early Version)
from datasets import load_dataset
dataset_old = load_dataset("mmrech/pitvqa-spatial-vlm")

# New (Comprehensive Version) - Recommended
dataset_new = load_dataset("mmrech/pitvqa-comprehensive-spatial")

# Same format, just more data and better validation!
```

### Training Configuration

For LoRA training, use the same configuration as the comprehensive version:

```python
from trl import SFTTrainer
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**However**, we recommend training on the comprehensive version for better performance.

## Performance Comparison

### Early Version (This Dataset)

| Metric | Value |
|--------|-------|
| Quadrant Accuracy | ~35-40% |
| Coordinate MAE | ~18-20% |
| Status | Baseline |

### Comprehensive Version (Recommended)

| Metric | Value | Improvement |
|--------|-------|-------------|
| Quadrant Accuracy | 80.3% | +124% |
| Coordinate MAE | 12.1% | -40% |
| Status | State-of-the-art | ✅ |

**Performance increase**: Models trained on the comprehensive version achieve **124% improvement** in quadrant accuracy.

## Use Cases

### Appropriate Use Cases

1. **Historical Research**: Understanding evolution of spatial VLMs
2. **Ablation Studies**: Comparing data quantity effects
3. **Baseline Comparisons**: Establishing improvement metrics
4. **Educational Demos**: Simple proof-of-concept examples

### Not Recommended For

- ❌ Production models (use comprehensive version)
- ❌ MICCAI/journal publications (use comprehensive version)
- ❌ Clinical research (use comprehensive version)
- ❌ Benchmark evaluations (use comprehensive version)

## Training Usage

### Recommended Approach

**Don't train on this dataset**. Instead:

```python
# Use the comprehensive version
from datasets import load_dataset

dataset = load_dataset("mmrech/pitvqa-comprehensive-spatial")

# Follow training guide:
# https://github.com/matheus-rech/pit_project/blob/main/notebooks/train_spatial_qwen2vl_colab.ipynb
```

### If You Must Use This Dataset

```python
from datasets import load_dataset

# Load early version (not recommended)
dataset = load_dataset("mmrech/pitvqa-spatial-vlm")

# Same training procedure as comprehensive version
# But expect lower performance (35-40% vs 80.3%)
```

## Limitations

### Dataset Limitations

- **Limited Samples**: Smaller dataset than comprehensive version
- **Incomplete Coverage**: Not all surgical phases covered
- **Partial Validation**: Not fully validated for ground truth fidelity
- **Lower Performance**: Models trained on this achieve 35-40% accuracy vs 80.3%

### Technical Limitations

- **Data Quality**: Less rigorous validation than comprehensive version
- **Documentation**: Limited compared to production dataset
- **Support**: Community support focused on comprehensive version

### Superseded Status

⚠️ **This dataset has been superseded** by [mmrech/pitvqa-comprehensive-spatial](https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial)

## Ethical Considerations

Same ethical considerations as comprehensive version:

- ✅ De-identified patient data
- ✅ Institutional ethics approval
- ❌ Not for clinical use

## License

**CC-BY-NC-ND-4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International)

Same license as comprehensive version.

## Citation

If citing this early work, please also cite the comprehensive version:

```bibtex
@misc{rech2026pitvqa_spatial_early,
  author = {Rech, Matheus},
  title = {PitVQA Spatial VLM Dataset (Early Version)},
  year = {2026},
  publisher = {HuggingFace},
  note = {Early prototype. See pitvqa-comprehensive-spatial for production use.},
  howpublished = {\url{https://huggingface.co/datasets/mmrech/pitvqa-spatial-vlm}}
}

@misc{rech2026pitvqa_spatial_dataset,
  author = {Rech, Matheus},
  title = {PitVQA Comprehensive Spatial Dataset},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial}},
  note = {Recommended version with 10,139 validated samples}
}
```

## Recommended Resources

### Instead of This Dataset, Use:

1. **Dataset**: [mmrech/pitvqa-comprehensive-spatial](https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial)
2. **Model**: [mmrech/pitvqa-qwen2vl-spatial](https://huggingface.co/mmrech/pitvqa-qwen2vl-spatial)
3. **GitHub**: https://github.com/matheus-rech/pit_project
4. **Training Guide**: [Colab Notebook](https://github.com/matheus-rech/pit_project/blob/main/notebooks/train_spatial_qwen2vl_colab.ipynb)

## Dataset Card Authors

Matheus Rech

## Contact

- **GitHub**: https://github.com/matheus-rech/pit_project
- **HuggingFace**: https://huggingface.co/mmrech
- **Questions**: Please open an issue on GitHub

## Changelog

### Version 1.0.0 (Early 2026)
- Initial early prototype release
- Basic spatial localization annotations
- Proof-of-concept for spatial VLM task

### Status: Superseded (Current)
- **Superseded by**: [mmrech/pitvqa-comprehensive-spatial](https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial)
- **Recommendation**: Use comprehensive version for all new projects

---

**⚠️ Deprecation Notice**: This early version is provided for historical reference and reproducibility of early experiments. For current research, please use [mmrech/pitvqa-comprehensive-spatial](https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial) which provides 10,139 validated samples and achieves 80.3% quadrant accuracy vs 35-40% with this early version.
