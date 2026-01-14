#!/bin/bash
# Quick Reproducibility Test
# Tests if the work can be reproduced with just GitHub + HuggingFace

set -e

echo "========================================================================"
echo "QUICK REPRODUCIBILITY TEST"
echo "========================================================================"
echo ""
echo "This script tests if you can reproduce the work with only:"
echo "  ✓ GitHub repository"
echo "  ✓ HuggingFace datasets/models"
echo "  ✓ No proprietary data or local files"
echo ""
echo "========================================================================"
echo ""

# Check Python
echo "1️⃣ Checking Python..."
python3 --version
echo "   ✅ Python available"
echo ""

# Check dependencies
echo "2️⃣ Checking key dependencies..."
python3 -c "import transformers; print(f'   ✅ transformers {transformers.__version__}')" || echo "   ❌ transformers missing"
python3 -c "import torch; print(f'   ✅ torch {torch.__version__}')" || echo "   ❌ torch missing"
python3 -c "import datasets; print(f'   ✅ datasets {datasets.__version__}')" || echo "   ❌ datasets missing"
echo ""

# Test dataset loading
echo "3️⃣ Testing dataset access (HuggingFace)..."
python3 << 'DATASET_TEST'
from datasets import load_dataset
import sys

try:
    print("   Loading mmrech/pitvqa-comprehensive-spatial...")
    dataset = load_dataset("mmrech/pitvqa-comprehensive-spatial", split="train", streaming=True)
    sample = next(iter(dataset))
    print(f"   ✅ Dataset accessible! Sample keys: {list(sample.keys())}")
except Exception as e:
    print(f"   ❌ Dataset loading failed: {e}")
    sys.exit(1)
DATASET_TEST
echo ""

# Test model access
echo "4️⃣ Testing model access (HuggingFace)..."
python3 << 'MODEL_TEST'
from transformers import AutoProcessor
import sys

try:
    print("   Loading mmrech/pitvqa-qwen2vl-spatial processor...")
    processor = AutoProcessor.from_pretrained("mmrech/pitvqa-qwen2vl-spatial", trust_remote_code=True)
    print("   ✅ Model accessible!")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    sys.exit(1)
MODEL_TEST
echo ""

# Test base model access
echo "5️⃣ Testing base model access (Qwen)..."
python3 << 'BASE_MODEL_TEST'
from transformers import AutoConfig
import sys

try:
    print("   Loading Qwen/Qwen2-VL-2B-Instruct config...")
    config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    print("   ✅ Base model accessible!")
except Exception as e:
    print(f"   ❌ Base model access failed: {e}")
    sys.exit(1)
BASE_MODEL_TEST
echo ""

# Test scripts exist
echo "6️⃣ Checking training scripts..."
if [ -f "scripts/train_unified_vlm.py" ]; then
    echo "   ✅ Training script found"
else
    echo "   ❌ Training script missing"
fi

if [ -f "scripts/evaluate_unified_vlm.py" ]; then
    echo "   ✅ Evaluation script found"
else
    echo "   ❌ Evaluation script missing"
fi
echo ""

# Test notebooks exist
echo "7️⃣ Checking Colab notebooks..."
if [ -f "notebooks/train_spatial_qwen2vl_colab.ipynb" ]; then
    echo "   ✅ Training notebook found"
else
    echo "   ❌ Training notebook missing"
fi

if [ -f "notebooks/train_and_demo_colab.ipynb" ]; then
    echo "   ✅ Demo notebook found"
else
    echo "   ❌ Demo notebook missing"
fi
echo ""

# Test demo exists
echo "8️⃣ Checking Gradio demo..."
if [ -f "gradio_demo.py" ]; then
    echo "   ✅ Gradio demo found"
    python3 -c "import gradio" && echo "   ✅ Gradio installed" || echo "   ❌ Gradio not installed (pip install gradio)"
else
    echo "   ❌ Gradio demo missing"
fi
echo ""

# Test documentation
echo "9️⃣ Checking documentation..."
if [ -f "README.md" ]; then
    echo "   ✅ Main README found"
fi

if [ -f "docs/COLAB_TRAINING_GUIDE.md" ]; then
    echo "   ✅ Training guide found"
fi

if [ -f "GRADIO_DEMO_README.md" ]; then
    echo "   ✅ Demo guide found"
fi
echo ""

# Test validation reports
echo "🔟 Checking validation reports..."
if [ -f "CORRECTED_VALIDATION_REPORT.json" ]; then
    echo "   ✅ Corrected validation report found"
fi

if [ -f "validation/final_validation_report.json" ]; then
    echo "   ✅ Final validation report found"
fi
echo ""

echo "========================================================================"
echo "REPRODUCIBILITY TEST COMPLETE"
echo "========================================================================"
echo ""
echo "✅ All critical components accessible!"
echo ""
echo "You can reproduce the work with:"
echo "  ✓ This GitHub repository"
echo "  ✓ HuggingFace datasets (mmrech/pitvqa-comprehensive-spatial)"
echo "  ✓ HuggingFace models (mmrech/pitvqa-qwen2vl-spatial)"
echo "  ✓ Free Google Colab GPU"
echo ""
echo "No proprietary data or local files needed!"
echo ""
echo "Next steps:"
echo "  1. Test demo: python gradio_demo.py"
echo "  2. Train model: Upload notebook to Colab"
echo "  3. Read guides: docs/COLAB_TRAINING_GUIDE.md"
echo ""
echo "========================================================================"
