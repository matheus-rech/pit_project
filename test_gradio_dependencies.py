#!/usr/bin/env python3
"""
Test if all Gradio demo dependencies are installed
"""

import sys

print("Testing Gradio demo dependencies...\n")

required = {
    'gradio': 'Gradio (UI framework)',
    'torch': 'PyTorch (deep learning)',
    'transformers': 'HuggingFace Transformers',
    'peft': 'PEFT (LoRA adapters)',
    'cv2': 'OpenCV (video processing)',
    'PIL': 'Pillow (image processing)',
    'imageio': 'ImageIO (video export)',
    'numpy': 'NumPy (arrays)',
}

missing = []
installed = []

for module, description in required.items():
    try:
        __import__(module)
        installed.append(f"✅ {module:20s} - {description}")
    except ImportError:
        missing.append(f"❌ {module:20s} - {description}")

print("INSTALLED:")
for item in installed:
    print(item)

if missing:
    print("\nMISSING:")
    for item in missing:
        print(item)
    print("\nInstall missing packages:")
    print("pip install gradio transformers torch peft opencv-python pillow imageio imageio-ffmpeg numpy")
    sys.exit(1)
else:
    print("\n✅ All dependencies installed!")
    print("\nReady to run:")
    print("  python gradio_demo.py")
    sys.exit(0)
