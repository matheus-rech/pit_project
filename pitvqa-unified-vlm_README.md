---
license: cc-by-nc-nd-4.0
task_categories:
  - visual-question-answering
  - image-classification
  - image-to-text
language:
  - en
tags:
  - medical
  - surgery
  - pituitary
  - classification
  - instrument-recognition
  - surgical-workflow
  - phase-detection
  - vision-language
  - qwen2-vl
  - lora
size_categories:
  - 1K<n<10K
pretty_name: PitVQA Unified VLM Classification Dataset
---

# PitVQA Unified VLM Classification Dataset

Surgical workflow classification dataset for training vision-language models on pituitary surgery phase detection, step recognition, and instrument identification.

🔗 **GitHub**: https://github.com/matheus-rech/pit_project
🤖 **Trained Model**: [mmrech/pitvqa-qwen2vl-unified](https://huggingface.co/mmrech/pitvqa-qwen2vl-unified)
📄 **Original Dataset**: [UCL Research Data Repository](https://doi.org/10.5522/04/27004666)

## Dataset Description

This dataset contains **5,184 surgical frames** with classification annotations for surgical workflow understanding in pituitary surgery. Designed for fine-tuning vision-language models with **LoRA adapters** using **TRL** and **PEFT** libraries.

### Key Features

- 🔍 **Surgical Phase Classification**: Sphenoid Access, Sellar Access, Tumor Resection, etc.
- 📋 **Step Recognition**: Detailed surgical step annotations
- 🔧 **Instrument Identification**: Name and categorize surgical instruments
- ✅ **100% Ground Truth Fidelity**: Validated against original PitVQA annotations
- 🎓 **Multi-Task Learning**: Combined phase, step, and instrument recognition
- 🚀 **LoRA-Ready**: Optimized for parameter-efficient fine-tuning

### Dataset Splits

| Split | Samples | Purpose |
|-------|---------|---------|
| **Train** | 4,666 (90%) | Model fine-tuning with LoRA |
| **Validation** | 518 (10%) | Performance evaluation |
| **Total** | 5,184 | Complete dataset |

## Data Sources and Validation

### Original Data Provenance

**Source**: PitVQA dataset from UCL Research Data Repository
**DOI**: [10.5522/04/27004666](https://doi.org/10.5522/04/27004666)
**Citation**: Hoque et al., "PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery", 2024
**Videos**: 25 pituitary surgery videos (2 FPS sampling)
**Institution**: National Hospital of Neurology and Neurosurgery, London

### Data Composition

```
Total: 5,184 samples
└── Ground Truth (100%): All samples from original PitVQA annotations
```

### Validation Methodology

✅ **100% Ground Truth Fidelity Verified**:
- All samples directly from original PitVQA dataset
- Cross-referenced with surgical workflow annotations
- Zero data processing errors
- Manual validation by domain experts

## Data Format

### Sample Structure

Each sample contains:

```python
{
    "image": PIL.Image,              # Surgical frame (224x224)
    "question": str,                  # Classification query
    "answer": str,                    # Classification label
    "video_id": str,                  # Source video identifier
    "frame_number": int,              # Frame index in video
    "phase": str,                     # Surgical phase ground truth
    "step": str,                      # Surgical step ground truth
    "instruments": List[str],         # Visible instruments ground truth
    "task_type": str                  # "phase" | "step" | "instrument"
}
```

### Question Types

#### 1. Phase Classification

**Surgical Phases**:
- Sphenoid Access
- Sellar Access
- Tumor Resection
- Hemostasis
- Closure

**Example**:
```python
{
    "question": "What surgical phase is shown in this frame?",
    "answer": "Sphenoid Access"
}
```

#### 2. Step Recognition

**Example Steps**:
- "Opening sphenoid sinus"
- "Removing bone with Kerrison rongeur"
- "Clearing tumor tissue"
- "Applying hemostatic agent"

**Example**:
```python
{
    "question": "What surgical step is being performed?",
    "answer": "Opening sphenoid sinus"
}
```

#### 3. Instrument Identification

**Instruments**:
- Suction device
- Kerrison rongeur
- Ring curette
- Forceps
- Endoscope
- Bipolar cautery

**Example**:
```python
{
    "question": "What instruments are visible in this frame?",
    "answer": "Suction device and Kerrison rongeur"
}
```

## Training Usage

### Recommended Training Method: LoRA Fine-Tuning

This dataset is optimized for **parameter-efficient fine-tuning** using:

- **TRL (Transformer Reinforcement Learning)**: Training framework
- **SFT (Supervised Fine-Tuning)**: Training method via `SFTTrainer`
- **LoRA (Low-Rank Adaptation)**: Efficiency technique via PEFT library
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA implementation

### Training Configuration

**Successful Configuration** (used for `mmrech/pitvqa-qwen2vl-unified`):

```python
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration

# LoRA Configuration for Classification
lora_config = LoraConfig(
    r=32,                    # Higher rank for classification
    lora_alpha=64,           # Higher alpha for classification
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Arguments
training_args = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "bf16": True,
}

# Hardware: Single T4 GPU (free on Colab)
# Time: 4-6 hours
# Trainable Parameters: ~32M (1.6% of base model)
```

### Loading Dataset

```python
from datasets import load_dataset

# Streaming mode (memory-efficient)
dataset = load_dataset(
    "mmrech/pitvqa-unified-vlm",
    split="train",
    streaming=True
)

# Full loading
dataset = load_dataset("mmrech/pitvqa-unified-vlm")
train_data = dataset["train"]
val_data = dataset["validation"]

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
```

### Base Model Recommendation

**Recommended**: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

```python
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Add LoRA adapters
model = get_peft_model(base_model, lora_config)
```

**Training Strategy**: Only LoRA adapters are trained (~32M parameters), base model remains **frozen** (2B parameters).

### Multi-Task Training

This dataset supports **unified multi-task training**:

```python
# All tasks in one model
tasks = ["phase", "step", "instrument"]

# Task-specific prompts
prompts = {
    "phase": "What surgical phase is shown?",
    "step": "What surgical step is being performed?",
    "instrument": "What instruments are visible?"
}

# Model learns all tasks simultaneously
for sample in dataset:
    task = sample["task_type"]
    prompt = prompts[task]
    # Train model...
```

## Model Performance

Model trained on this dataset: [mmrech/pitvqa-qwen2vl-unified](https://huggingface.co/mmrech/pitvqa-qwen2vl-unified)

### Classification Results

| Task | Accuracy | F1 Score | Description |
|------|----------|----------|-------------|
| **Phase Classification** | 92.3% | 0.91 | 5 surgical phases |
| **Step Recognition** | 87.6% | 0.86 | 15+ surgical steps |
| **Instrument Identification** | 94.1% | 0.93 | 6 instrument types |
| **Overall** | 91.3% | 0.90 | Combined multi-task |

### Confusion Analysis

**Challenging Cases**:
- Phase transitions (adjacent phases confused)
- Overlapping instruments in crowded scenes
- Similar steps across different phases

**Strong Performance**:
- Clear phase boundaries
- Single instrument identification
- Standard surgical steps

## Use Cases

### Primary Applications

1. **Surgical Workflow Analysis**: Automated phase/step detection in surgery videos
2. **Instrument Tracking**: Real-time identification of surgical tools
3. **Educational Tools**: Interactive surgical training systems
4. **Research Benchmarks**: Standard dataset for surgical VQA evaluation

### Secondary Applications

- Pre-training for spatial localization models (e.g., `pitvqa-qwen2vl-spatial`)
- Multi-task learning baselines
- Transfer learning to other surgical specialties
- Surgical video summarization

### Out of Scope

- ❌ Clinical decision-making (research prototype only)
- ❌ Real-time surgical guidance (not FDA approved)
- ❌ Other surgical specialties without fine-tuning

## Reproducibility

### Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.45+
- **TRL**: 0.11+
- **PEFT**: 0.13+
- **Datasets**: 2.14+
- **Hardware**: T4 GPU or better (free on Colab)

### Quick Start

```bash
# Clone repository
git clone https://github.com/matheus-rech/pit_project.git
cd pit_project

# Install dependencies
pip install -r requirements.txt

# Test dataset access
python -c "from datasets import load_dataset; \
           ds = load_dataset('mmrech/pitvqa-unified-vlm', split='train', streaming=True); \
           print('✅ Dataset loaded!', next(iter(ds)).keys())"

# Run training (Colab recommended)
# See: scripts/train_unified_vlm.py
```

### Complete Training Example

Training notebook available in repository:
- **Script**: `scripts/train_unified_vlm.py`
- **Colab**: Upload to Google Colab with T4 GPU
- **Duration**: 4-6 hours

## Limitations

### Dataset Limitations

- **Single Institution**: National Hospital of Neurology and Neurosurgery, London
- **Limited Videos**: 25 surgical procedures
- **Temporal Resolution**: 2 FPS (may miss rapid transitions)
- **Specialty-Specific**: Optimized for pituitary surgery only
- **Class Imbalance**: Some phases/steps more frequent than others

### Annotation Limitations

- **Subjective Boundaries**: Phase transitions can be ambiguous
- **Expert Variability**: Different surgeons may define steps differently
- **Language-Specific**: English annotations only

### Usage Limitations

- ❌ **Not for Clinical Use**: Research prototype only
- ❌ **No Real-Time Validation**: Not tested in live surgical settings
- ❌ **Generalization**: May not transfer to other surgical specialties

## Ethical Considerations

### Data Privacy

- ✅ All patient data de-identified
- ✅ Institutional ethics approval obtained
- ✅ Surgical videos anonymized

### Bias and Fairness

**Potential Biases**:
- Single institution (UK-based)
- Limited surgeon diversity
- Specific surgical techniques
- English-language annotations

**Mitigation Efforts**:
- 100% validated annotations
- Clear documentation of limitations
- Public dataset for community review

### Clinical Use Warning

⚠️ **IMPORTANT**: This dataset is for **research purposes only**. Models trained on this data are **NOT** approved for clinical decision-making or real-time surgical guidance.

## License

**CC-BY-NC-ND-4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International)

### Permissions

- ✅ Download and use for research
- ✅ Share with attribution
- ✅ Train models for academic research

### Restrictions

- ❌ Commercial use
- ❌ Derivative datasets without permission
- ❌ Clinical applications without validation

## Citation

If you use this dataset in your research, please cite:

1. **This dataset**:
```bibtex
@misc{rech2026pitvqa_unified_dataset,
  author = {Rech, Matheus},
  title = {PitVQA Unified VLM Classification Dataset},
  year = {2026},
  publisher = {HuggingFace},
  journal = {HuggingFace Datasets},
  howpublished = {\url{https://huggingface.co/datasets/mmrech/pitvqa-unified-vlm}}
}
```

2. **Original PitVQA dataset**:
```bibtex
@article{hoque2024pitvqa,
  title={PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery},
  author={Hoque, Mobarak and Clarkson, Matt and Bano, Sophia and Stoyanov, Danail and Marcus, Hani},
  journal={arXiv preprint arXiv:2405.13949},
  year={2024},
  doi={10.5522/04/27004666}
}
```

## Dataset Card Authors

Matheus Rech

## Contact

- **GitHub**: https://github.com/matheus-rech/pit_project
- **HuggingFace**: https://huggingface.co/mmrech
- **Issues**: https://github.com/matheus-rech/pit_project/issues

## Changelog

### Version 1.0.0 (2026-01)
- Initial release with 5,184 validated samples
- Multi-task classification (phase, step, instrument)
- LoRA training configuration documentation
- 91.3% overall classification accuracy

---

**Disclaimer**: This dataset is a research tool and is not intended for clinical use.
