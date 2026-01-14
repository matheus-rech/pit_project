# 🔬 Training Method Clarification: TRL, SFT, LoRA, PEFT

## The Confusion

**You asked**: "The README says TRL and SFT, but you mentioned LoRA?"

**Answer**: ✅ **ALL of these are correct!** They're not alternatives - they work together.

---

## 🧩 How They All Fit Together

### The Complete Stack:

```
┌─────────────────────────────────────────────────┐
│ TRL (Transformer Reinforcement Learning)       │ ← Library/Framework
│ https://github.com/huggingface/trl              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ SFTTrainer (Supervised Fine-Tuning Trainer)    │ ← Training Method
│ - Not reinforcement learning despite name!      │
│ - Supervised learning on labeled data           │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ LoRA (Low-Rank Adaptation via PEFT)            │ ← Parameter Efficiency
│ - Only trains 18M params (not 2B)              │
│ - Uses peft library                             │
└─────────────────────────────────────────────────┘
```

**All three are used together!**

---

## 🎯 Quick Answer

**You're using ALL of these:**

| Term | What It Is | Role |
|------|-----------|------|
| **TRL** | HuggingFace library | Framework/tool |
| **SFT** | Supervised Fine-Tuning | Training method |
| **LoRA** | Low-Rank Adaptation | Parameter efficiency technique |
| **PEFT** | Parameter-Efficient Fine-Tuning | Umbrella library |

**They work together!**

---

## 🔍 What Each Term Means

### TRL (Transformer Reinforcement Learning)
- **What**: HuggingFace library for training LLMs
- **Provides**: `SFTTrainer`, `DPOTrainer`, etc.
- **Your code**: `from trl import SFTTrainer`
- **Role**: The training framework/library

### SFT (Supervised Fine-Tuning)
- **What**: The training method (supervised, not reinforcement learning)
- **How**: Train on labeled examples with cross-entropy loss
- **In code**: `trainer = SFTTrainer(...)`

### LoRA (Low-Rank Adaptation)
- **What**: The parameter-efficient technique
- **How**: Add small adapter layers instead of updating all parameters
- **Code**: `lora_config = LoraConfig(r=16, ...)`

### PEFT (Parameter-Efficient Fine-Tuning)
- The library/framework that implements LoRA
- `from peft import LoraConfig, get_peft_model`

---

## 🔍 How They Work Together

```python
# TRL library (framework)
from trl import SFTTrainer  # ← Library for supervised fine-tuning

# PEFT library (parameter-efficient methods)
from peft import LoraConfig, get_peft_model  # ← LoRA implementation

# Configure LoRA
lora_config = LoraConfig(r=16, ...)  # ← Using LoRA adapters

# Use SFT Trainer from TRL
trainer = SFTTrainer(  # ← Supervised fine-tuning trainer
    model=model,  # ← Model with LoRA adapters
    ...
)

# Train
trainer.train()  # ← SFT training of LoRA adapters
```

---

## 🎯 The Complete Picture

### What Each Component Does:

1. **TRL** (library): HuggingFace's Transformer Reinforcement Learning library
   - Provides `SFTTrainer` class
   - Despite the name "RL", we're using its **SFT** (Supervised Fine-Tuning) component

2. **SFT** (method): Supervised Fine-Tuning
   - Training approach: Learn from labeled examples
   - NOT reinforcement learning (confusing name, I know!)

3. **LoRA** (technique): Low-Rank Adaptation
   - HOW we fine-tune efficiently
   - Only trains small adapter layers (~18M params)

4. **PEFT** (umbrella): Parameter-Efficient Fine-Tuning
   - Library that implements LoRA and other efficient methods

---

## 📊 The Stack

```
┌─────────────────────────────────────────┐
│ TRL Library (Framework)                 │
│  ├─ SFTTrainer (Training loop)          │
│  └─ Data handling, logging, etc.        │
└─────────────────────────────────────────┘
              ↓ uses
┌─────────────────────────────────────────┐
│ PEFT Library (Efficiency technique)     │
│  ├─ LoRA adapters (r=16)                │
│  └─ Only ~18M params trained            │
└─────────────────────────────────────────┘
              ↓ applies to
┌─────────────────────────────────────────┐
│ Qwen2-VL-2B-Instruct (Base model)       │
│  └─ 2B params (FROZEN)                  │
└─────────────────────────────────────────┘
```

---

## ✅ Correct Way to Describe It

### All of these are correct:

1. "Trained using **TRL**'s SFTTrainer"
2. "Trained using **SFT** (Supervised Fine-Tuning)"
3. "Trained using **LoRA** adapters (r=16)"
4. "Trained using **PEFT** (LoRA variant)"

### Most complete:
```
"We fine-tuned the model using Supervised Fine-Tuning (SFT) with
LoRA adapters (r=16) via the TRL and PEFT libraries, training only
18M parameters while keeping the 2B base model frozen."
```

---

## 🔍 Where the Confusion Comes From

### Model Card Might Say:
- "Trained using TRL 0.26.2" ✅ (the library)
- "Method: SFT" ✅ (supervised fine-tuning)

### But Doesn't Always Mention:
- "Using LoRA adapters" (the efficiency technique)
- "PEFT library" (the implementation)

**Both are true!** TRL/SFT is the *framework*, LoRA/PEFT is the *efficiency technique*.

---

## 📝 For Your Paper - Most Accurate Description

```markdown
## Training Methodology

We fine-tuned Qwen2-VL-2B-Instruct using supervised fine-tuning (SFT)
with parameter-efficient LoRA adapters (r=16, α=32). The training was
implemented using HuggingFace's TRL (v0.26.2) and PEFT (v0.13.0) libraries.

Only 18M adapter parameters (0.9% of the model) were trained while the
2B base model remained frozen. Training was performed on 10,139 surgical
frames for 3 epochs using 4-bit quantization (bitsandbytes) for memory
efficiency.

**Libraries:**
- Transformers 4.45.0 (model loading)
- TRL 0.11.0 (SFTTrainer for supervised fine-tuning)
- PEFT 0.13.0 (LoRA implementation)
- bitsandbytes 0.44.0 (quantization)
```

---

## 🎓 Quick Reference

| Term | What It Is | Role |
|------|-----------|------|
| **TRL** | Library | Provides SFTTrainer class |
| **SFT** | Method | Supervised fine-tuning approach |
| **LoRA** | Technique | Parameter-efficient adapters |
| **PEFT** | Library | Implements LoRA |
| **Qwen2-VL** | Model | Base model (frozen) |

**All working together**: TRL's SFTTrainer trains LoRA adapters (via PEFT) on the frozen Qwen2-VL base model.

---

## ✅ Summary

**Your Question**: "README says TRL and SFT, but you mentioned LoRA?"

**Answer**: **Both are correct!**
- **TRL + SFT** = The training *framework* and *method*
- **LoRA** = The *efficiency technique* used within that framework

**Analogy**:
- TRL = "We used a Toyota" (the vehicle/framework)
- SFT = "We drove" (the action/method)
- LoRA = "With fuel-efficient engine" (the efficiency technique)

All three are part of the same training process! The model card mentions TRL/SFT (the framework), but under the hood it's using LoRA adapters for efficiency.

Does this clarify the relationship between these components?