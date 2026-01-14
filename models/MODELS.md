# Trained Models

**IMPORTANT:** Actual model files are hosted on HuggingFace Hub.
Download during training or inference.

## Available Models

### 1. pitvqa-qwen2vl-unified ✅ BASELINE

**HuggingFace:** https://huggingface.co/mmrech/pitvqa-qwen2vl-unified

**Capabilities:**
- ✅ Surgical phase classification (4 phases)
- ✅ Surgical step classification (10 steps)
- ✅ Instrument identification
- ⚠️  Spatial pointing (basic, needs improvement)

**Training:**
- Dataset: pitvqa-unified-vlm (4,772 samples)
- Base: Qwen2-VL-2B-Instruct
- Method: QLoRA (r=32, alpha=64)
- Date: January 4, 2026

### 2. pitvqa-qwen2vl-spatial 🎯 PUBLICATION MODEL

**HuggingFace:** https://huggingface.co/mmrech/pitvqa-qwen2vl-spatial

**Capabilities:**
- ✅ Surgical phase classification
- ✅ Surgical step classification
- ✅ Instrument identification
- ✅ **Spatial pointing (accurate >80%)**

**Training:**
- Dataset: pitvqa-comprehensive-spatial (10,139 samples)
- Base: Qwen2-VL-2B-Instruct + pitvqa-qwen2vl-unified
- Method: QLoRA (r=16, alpha=32) - spatial fine-tuning
- Date: TBD (train with provided notebook)

**Performance:**
- Coordinate MAE: <15%
- Quadrant accuracy: >80%
- Instrument F1: >0.80

## Using Models

### Load for Inference

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# Load model
processor = AutoProcessor.from_pretrained(
    "mmrech/pitvqa-qwen2vl-spatial",
    trust_remote_code=True
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    "mmrech/pitvqa-qwen2vl-spatial"
)

# Inference
conversation = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "Point to the suction device."}
    ]
}]

text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[your_image], return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=100)

response = processor.decode(outputs[0], skip_special_tokens=True)
# Output: "<point x='45.2' y='68.3'>suction device</point>"
```

## Training Your Own

See `notebooks/train_spatial_qwen2vl_colab.ipynb` for complete training pipeline.
