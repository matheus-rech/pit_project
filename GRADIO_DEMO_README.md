# 🎬 Gradio Interactive Demo Setup

## Quick Start

```bash
# 1. Install dependencies
pip install gradio transformers accelerate peft torch qwen-vl-utils opencv-python imageio imageio-ffmpeg pillow

# 2. Run demo
python gradio_demo.py
```

The demo will launch at `http://localhost:7860` with a **public shareable link**!

---

## Features

### 📷 Single Frame Mode
- Upload a surgical frame (JPG/PNG)
- Ask natural language questions
- Get instant bounding boxes and labels
- Choose between bbox or point annotations

### 🎬 Video Processing Mode
- Upload surgical video (MP4/AVI)
- Process up to 100 frames
- Get annotated video output
- Statistical summary of detections

### ℹ️ About Tab
- Model information
- Performance metrics (80% quadrant accuracy)
- Citations and links

---

## Demo Questions

Pre-loaded questions you can use:
- "Point to all surgical instruments visible in this frame."
- "Identify and locate all anatomical structures."
- "Where is the suction device?"
- "Point to the Kerrison rongeur."
- "Locate all instruments and anatomy visible."
- "Where is the sphenoid sinus?"
- "Identify the surgical tools being used."

Or type your own custom questions!

---

## Example Usage

### Single Frame
1. Upload a surgical endoscopy frame
2. Select "Point to all surgical instruments visible in this frame."
3. Choose "bbox" annotation style
4. Click "🔍 Analyze Frame"
5. See bounding boxes with labels!

### Video
1. Upload a surgical video clip
2. Set max frames (30 recommended for testing)
3. Click "🎥 Process Video"
4. Wait for processing (1-2 minutes for 30 frames)
5. Download annotated video!

---

## Requirements

- **GPU**: Recommended (CUDA-enabled)
- **RAM**: 8GB minimum
- **Disk**: 5GB for model weights
- **Internet**: Required for first run (downloads model from HuggingFace)

---

## Model Loading

The demo automatically downloads:
- Base model: `Qwen/Qwen2-VL-2B-Instruct` (~4.8GB)
- Fine-tuned adapter: `mmrech/pitvqa-qwen2vl-spatial` (~30MB)

First launch takes 5-10 minutes to download models.

---

## Annotation Colors

- **Green boxes**: Surgical instruments
- **Orange boxes**: Anatomical structures
- **White text**: Instrument/anatomy labels

---

## Performance

| Hardware | Speed (single frame) | Speed (30-frame video) |
|----------|---------------------|------------------------|
| **CPU only** | ~10s per frame | ~5 minutes |
| **GPU (T4)** | ~2s per frame | ~1 minute |
| **GPU (A100)** | ~0.5s per frame | ~15 seconds |

---

## Troubleshooting

### Model loading fails
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface
```

### Out of memory
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

### Video processing hangs
- Try reducing max frames (10-20 instead of 100)
- Ensure video format is supported (MP4, AVI)
- Check video isn't corrupted

### Dependencies missing
```bash
# Install all at once
pip install -r requirements.txt

# Or install manually
pip install gradio==4.10.0
pip install transformers==4.36.0
pip install torch torchvision
pip install peft accelerate
pip install opencv-python pillow
pip install imageio imageio-ffmpeg
```

---

## Sharing the Demo

The demo creates a **public shareable link** automatically:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
```

Share the public URL with:
- ✅ Reviewers (for paper demonstration)
- ✅ Collaborators (for testing)
- ✅ Conference attendees (for live demos)

**Note**: Public links expire after 72 hours. Restart demo to get new link.

---

## For HuggingFace Spaces

To deploy permanently on HuggingFace Spaces:

1. **Create Space**: https://huggingface.co/spaces
2. **Upload files**:
   - `gradio_demo.py` (rename to `app.py`)
   - `requirements.txt`
3. **Space will auto-launch**!

Example Space URL: `https://huggingface.co/spaces/mmrech/pitvqa-demo`

---

## Example Video Clips

For testing, extract surgical clips from your dataset:

```python
# Extract 10-second clips
import cv2

cap = cv2.VideoCapture("full_surgical_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []

for i in range(int(fps * 10)):  # 10 seconds
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

# Save clip
out = cv2.VideoWriter('clip.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                      (frame.shape[1], frame.shape[0]))
for frame in frames:
    out.write(frame)
out.release()
```

---

## Metrics Display

The demo shows:
- **Dataset Fidelity**: 100% (validated against ground truth)
- **Quadrant Accuracy**: 80.3% (model predictions)
- **Coordinate MAE**: 12.1% (mean absolute error)
- **Confidence**: 95%+ on validated samples

These are the **corrected metrics** that distinguish:
- Data quality (100%) ✅
- Model performance (80%) ✅

---

## Citation

If using this demo for presentations or papers:

```bibtex
@misc{rech2026pitvqa_demo,
  title={PitVQA: Interactive Demo for Surgical Instrument Localization},
  author={Rech, Matheus},
  year={2026},
  howpublished={\url{https://huggingface.co/mmrech/pitvqa-qwen2vl-spatial}}
}
```

---

## Security Note

This is a **research prototype**, not for clinical use:
- ⚠️ Not FDA approved
- ⚠️ Not validated for patient care
- ⚠️ For research and education only

---

## Next Steps

1. **Test locally**: Run `python gradio_demo.py`
2. **Share link**: Send to reviewers/collaborators
3. **Deploy to Spaces**: For permanent public demo
4. **Add to paper**: Include demo link in supplementary materials

---

**Questions?** Check the "About" tab in the demo for full documentation!
