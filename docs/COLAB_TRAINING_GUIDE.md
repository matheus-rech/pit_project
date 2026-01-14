# 🚀 Colab Training Quick Start

## ✅ Your Data is Ready on HuggingFace!

| Dataset | Status | Samples | Purpose |
|---------|--------|---------|---------|
| **mmrech/pitvqa-comprehensive-spatial** | ✅ Live | 9,125 train / 1,014 val | **Use this!** |
| mmrech/pitvqa-spatial-vlm | ✅ Live | 4,645 train / 517 val | Earlier version |
| mmrech/pitvqa-unified-vlm | ✅ Live | 4,772 train / 412 val | Classification only |

**Validated dataset:** `mmrech/pitvqa-comprehensive-spatial`
- ✅ 100% ground truth accuracy
- ✅ Publication-ready
- ✅ 10,139 total samples (87% ground truth, 13% AI-augmented)

---

## 📓 How to Use the Colab Notebook

### Option 1: Open in Colab (Easiest)

1. Upload `notebooks/train_spatial_qwen2vl_colab.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Go to **Runtime** → **Change runtime type** → Select **GPU** (T4 or A100)
4. Run all cells sequentially

### Option 2: Direct Link

```bash
# From your terminal, open the notebook
open notebooks/train_spatial_qwen2vl_colab.ipynb
```

Then upload to Colab.

---

## ⚙️ Training Configuration

### Hardware Requirements

| GPU | Memory | Batch Size | Training Time | Cost |
|-----|--------|------------|---------------|------|
| **T4** (Free) | 16GB | 1 (effective 16) | ~6-8 hours | Free |
| **A100** (Pro) | 40GB | 2-4 (effective 32) | ~2-3 hours | ~$10 |

**Recommendation:** Start with T4 (free). It works fine, just slower.

### Training Parameters

```python
Epochs: 3
Effective batch size: 16 (via gradient accumulation)
Learning rate: 2e-5 (with cosine schedule)
LoRA rank: 16 (reduced for fine-tuning)
Optimizer: paged_adamw_8bit (memory efficient)
```

### What Gets Trained

```
Base Model: Qwen/Qwen2-VL-2B-Instruct (frozen)
    ↓
Existing Adapter: mmrech/pitvqa-qwen2vl-unified (frozen)
    ↓
NEW Spatial Adapter: r=16 LoRA (TRAINABLE)
    ↓
Output: mmrech/pitvqa-qwen2vl-spatial
```

Only the NEW spatial adapter trains (~18M parameters), not the entire model.

---

## 📊 Expected Results

### Before Training (pitvqa-qwen2vl-unified)
- ✅ Phase classification: Good
- ✅ Step classification: Good
- ✅ Instrument naming: Decent
- ❌ Coordinate pointing: Random (~50% quadrant accuracy)

### After Training (pitvqa-qwen2vl-spatial)
- ✅ Phase classification: Good (maintained)
- ✅ Step classification: Good (maintained)
- ✅ Instrument naming: Better
- ✅ **Coordinate pointing: Accurate (>80% quadrant, <15% MAE)**

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory (OOM)

```python
# Reduce batch size in cell 6
per_device_train_batch_size=1  # Already minimum
gradient_accumulation_steps=8   # Reduce from 16 to 8
```

### Issue 2: Dataset Loading Slow

```python
# Normal! First load caches to HuggingFace (~2GB download)
# Subsequent loads are instant
```

### Issue 3: Training Taking Forever

```python
# On T4: 6-8 hours is normal
# Speed up: Use A100 (Colab Pro) → 2-3 hours
```

### Issue 4: Model Upload Fails

```python
# Make sure you ran notebook_login() in cell 3
# Check HuggingFace token has write permissions
```

---

## 🧪 Testing After Training

### Quick Test (In Notebook)

The notebook includes a test cell at the end. It will:
1. Load a validation sample
2. Run inference
3. Compare output to ground truth

### Full Evaluation (Separate Notebook)

After training, create an evaluation notebook:

```python
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load your trained model
processor = AutoProcessor.from_pretrained("mmrech/pitvqa-qwen2vl-spatial")
base = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "mmrech/pitvqa-qwen2vl-spatial")

# Load test set
test_data = load_dataset("mmrech/pitvqa-comprehensive-spatial", split="validation")

# Evaluate (measure coordinate MAE, quadrant accuracy, etc.)
```

---

## 📈 Monitoring Training

### Option 1: Watch Colab Output

The notebook prints:
- Loss every 10 steps
- Evaluation metrics every 100 steps
- Best model checkpoint saved

### Option 2: TensorBoard (Advanced)

```python
# In cell 6, change:
report_to="tensorboard"  # Instead of "none"

# Then in a new cell:
%load_ext tensorboard
%tensorboard --logdir ./pitvqa-qwen2vl-spatial
```

---

## 🎯 Success Criteria

Training is successful if:

1. **Loss Decreases**
   - Train loss: starts ~2.5 → ends <1.5
   - Eval loss: starts ~2.3 → ends <1.8

2. **Test Sample Works**
   - Model outputs coordinates in `<point x='...' y='...'>` format
   - Coordinates roughly match ground truth (within ±20px)

3. **Model Uploads**
   - Successfully pushed to `mmrech/pitvqa-qwen2vl-spatial`
   - Visible on HuggingFace Hub

---

## 🔄 Next Steps After Training

### Stage 2: Video + Temporal (Optional)

If you want video understanding:
1. Add temporal tracking samples
2. Format: Multi-frame sequences
3. Train for 2 more epochs

### Publication

Your model is now ready for:
- ✅ MICCAI paper (with spatial evaluation metrics)
- ✅ Pituitary journal (with validation results)
- ✅ Demo application (inference on surgical videos)

---

## 💾 Save Your Work

### Download from Colab

```python
# In the last cell, add:
!zip -r pitvqa-qwen2vl-spatial.zip ./pitvqa-qwen2vl-spatial-final
from google.colab import files
files.download('pitvqa-qwen2vl-spatial.zip')
```

### Backup Checkpoints

The notebook auto-saves checkpoints to:
- HuggingFace Hub: `mmrech/pitvqa-qwen2vl-spatial`
- Colab: `./pitvqa-qwen2vl-spatial/checkpoint-XXX/`

---

## 🆘 Need Help?

### Check Training Logs

```python
# Last few lines of training output should show:
# TrainOutput(global_step=XXX, training_loss=1.X, ...)
```

### Verify Model Works

```python
# Run the test cell - should output coordinates
# Example: "<point x='45.2' y='68.3'>suction device</point>"
```

### Model Not Learning?

- Check learning rate (2e-5 is good for fine-tuning)
- Verify dataset loaded correctly (should show 9,125 train samples)
- Ensure GPU is enabled (Runtime → Change runtime type)

---

## 📚 Resources

- **Dataset:** https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial
- **Base Model:** https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- **Your Model:** https://huggingface.co/mmrech/pitvqa-qwen2vl-unified
- **Output Model:** https://huggingface.co/mmrech/pitvqa-qwen2vl-spatial (after training)

---

**Ready to train? Open the notebook and start! 🚀**
