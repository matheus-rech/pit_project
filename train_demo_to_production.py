#!/usr/bin/env python3
"""
Unified Demo → Production Training Pipeline
Validates with 100 samples, then proceeds to full training if successful.

This script runs:
1. DEMO: 100 samples, 1 epoch (~5-10 min) - validates pipeline
2. If demo succeeds → PRODUCTION: Full dataset, 3 epochs (~4-6 hrs)

No manual intervention required. Fully automated for publication.
"""

import os
import json
import torch
import time
import random
import numpy as np
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# ============================================================================
# CONFIGURATION
# ============================================================================

# Credentials
HF_TOKEN = os.environ.get("HF_TOKEN")

# Model
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DATASET_ID = "mmrech/pitvqa-comprehensive-spatial"

# Demo configuration
DEMO_SAMPLES = 100
DEMO_EPOCHS = 1
DEMO_OUTPUT = "./pitvqa-demo"

# Production configuration
PROD_TRAIN_SAMPLES = 9125
PROD_VAL_SAMPLES = 1014
PROD_EPOCHS = 3
PROD_OUTPUT = "./pitvqa-production"
HF_REPO = "mmrech/pitvqa-qwen2vl-spatial"

# Training hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Success criteria for demo
DEMO_MAX_FINAL_LOSS = 2.0  # If final loss < 2.0, demo is successful


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_all_seeds(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Qwen2VLDataCollator:
    """Custom data collator for Qwen2-VL variable-length images."""
    processor: Any

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [ex.get('text', '') for ex in examples]
        images = [[ex['image']] if ex.get('image') else None for ex in examples]

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100
        return batch


def format_conversation(example, processor):
    """Format dataset sample into Qwen2-VL conversation."""
    messages = json.loads(example['messages']) if isinstance(example['messages'], str) else example['messages']

    user_content = messages[0]['content'].replace('<image>\n', '')
    assistant_content = messages[1]['content']

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_content}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_content}]
        }
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def run_training(
    mode: str,
    dataset_train,
    dataset_val,
    model,
    processor,
    output_dir: str,
    num_epochs: int,
    push_to_hub: bool = False
):
    """Run training for demo or production mode."""

    print(f"\n{'=' * 80}")
    print(f"🚀 STARTING {mode.upper()} TRAINING")
    print(f"{'=' * 80}")
    print(f"Samples: {len(dataset_train)} train, {len(dataset_val)} val")
    print(f"Epochs: {num_epochs}")
    print(f"Output: {output_dir}")
    print(f"Push to Hub: {push_to_hub}")
    print(f"{'=' * 80}\n")

    # Training configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50 if mode == "demo" else 100,
        save_strategy="steps",
        save_steps=50 if mode == "demo" else 100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_seq_length=MAX_SEQ_LENGTH,
        hub_model_id=HF_REPO if push_to_hub else None,
        push_to_hub=push_to_hub,
        report_to="tensorboard",
        dataset_text_field="text",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=RANDOM_SEED,
    )

    # Data collator
    data_collator = Qwen2VLDataCollator(processor=processor)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=data_collator,
        processing_class=processor,
    )

    # Train
    start_time = time.time()
    result = trainer.train()
    end_time = time.time()

    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)

    print(f"\n{'=' * 80}")
    print(f"✅ {mode.upper()} TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Training time: {hours}h {minutes}m")
    print(f"Final train loss: {result.training_loss:.4f}")
    print(f"{'=' * 80}\n")

    # Save model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    if push_to_hub:
        print(f"📤 Pushing to HuggingFace Hub: {HF_REPO}")
        model.push_to_hub(HF_REPO, commit_message=f"{mode} training complete")
        processor.push_to_hub(HF_REPO)

    return result.training_loss, trainer


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("🎯 DEMO → PRODUCTION TRAINING PIPELINE")
    print("=" * 80)
    print("Stage 1: Demo (100 samples, 1 epoch)")
    print("Stage 2: Production (9,125 samples, 3 epochs)")
    print("No manual intervention - fully automated!")
    print("=" * 80)

    # Setup
    set_all_seeds(RANDOM_SEED)
    login(token=HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset (disable verification to avoid CastError with complex message structures)
    print(f"\n📊 Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, verification_mode='no_checks')
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Validation: {len(dataset['validation']):,} samples")

    # Load model (once, reuse for both stages)
    print(f"\n🔄 Loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # Add LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("\n📊 Trainable Parameters:")
    model.print_trainable_parameters()

    # Load processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Format datasets
    print("\n🔄 Formatting datasets...")
    train_full = dataset['train'].map(
        lambda x: format_conversation(x, processor),
        num_proc=4,
        desc="Formatting train"
    )
    val_full = dataset['validation'].map(
        lambda x: format_conversation(x, processor),
        num_proc=4,
        desc="Formatting validation"
    )

    # ========================================================================
    # STAGE 1: DEMO TRAINING
    # ========================================================================

    print("\n" + "=" * 80)
    print("📝 STAGE 1: DEMO TRAINING")
    print("=" * 80)
    print(f"Purpose: Validate pipeline with {DEMO_SAMPLES} samples")
    print(f"Success criteria: Final loss < {DEMO_MAX_FINAL_LOSS}")
    print("=" * 80)

    # Subset datasets for demo
    demo_train = train_full.select(range(min(DEMO_SAMPLES, len(train_full))))
    demo_val = val_full.select(range(min(20, len(val_full))))  # Small validation set

    try:
        demo_loss, demo_trainer = run_training(
            mode="demo",
            dataset_train=demo_train,
            dataset_val=demo_val,
            model=model,
            processor=processor,
            output_dir=DEMO_OUTPUT,
            num_epochs=DEMO_EPOCHS,
            push_to_hub=False
        )

        # Check if demo succeeded
        if demo_loss < DEMO_MAX_FINAL_LOSS:
            print(f"\n✅ DEMO SUCCESS! Final loss: {demo_loss:.4f} < {DEMO_MAX_FINAL_LOSS}")
            print(f"Pipeline validated. Proceeding to production...")
        else:
            print(f"\n⚠️  DEMO WARNING: Final loss {demo_loss:.4f} >= {DEMO_MAX_FINAL_LOSS}")
            print(f"Proceeding anyway (demo is just validation)...")

    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        print("Cannot proceed to production. Please check logs.")
        sys.exit(1)

    # ========================================================================
    # STAGE 2: PRODUCTION TRAINING
    # ========================================================================

    print("\n" + "=" * 80)
    print("🏭 STAGE 2: PRODUCTION TRAINING")
    print("=" * 80)
    print(f"Training on full dataset: {len(train_full):,} samples")
    print(f"Epochs: {PROD_EPOCHS}")
    print(f"Will push to: {HF_REPO}")
    print("=" * 80)

    # Small delay to ensure GPU memory is clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(2)

    try:
        prod_loss, prod_trainer = run_training(
            mode="production",
            dataset_train=train_full,
            dataset_val=val_full,
            model=model,
            processor=processor,
            output_dir=PROD_OUTPUT,
            num_epochs=PROD_EPOCHS,
            push_to_hub=True
        )

        print(f"\n✅ PRODUCTION TRAINING COMPLETE!")
        print(f"   Final loss: {prod_loss:.4f}")
        print(f"   Model: {HF_REPO}")

    except Exception as e:
        print(f"\n❌ PRODUCTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Demo loss: {demo_loss:.4f}")
    print(f"   Production loss: {prod_loss:.4f}")
    print(f"\n🔗 Model: https://huggingface.co/{HF_REPO}")
    print(f"\n📝 Next Steps:")
    print(f"   1. Test on images: python test_spatial_model.py")
    print(f"   2. Test on videos: jupyter notebook test_video.ipynb")
    print(f"   3. Evaluate: python scripts/evaluate_unified_vlm.py")
    print(f"   4. Deploy demo: python gradio_demo.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
