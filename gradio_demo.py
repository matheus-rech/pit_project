#!/usr/bin/env python3
"""
PitVQA Gradio Demo - Interactive Surgical Instrument Detection
================================================================
Live demo of spatial localization model with video processing
"""

import gradio as gr
import torch
import cv2
import numpy as np
import re
from PIL import Image
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import tempfile
import imageio

# ============================================================================
# MODEL LOADING
# ============================================================================

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, "mmrech/pitvqa-qwen2vl-spatial")
processor = AutoProcessor.from_pretrained("mmrech/pitvqa-qwen2vl-spatial", trust_remote_code=True)
# Set model to inference mode (PyTorch)
model.requires_grad_(False)
print("✅ Model loaded!")

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def extract_points(text):
    """Extract spatial coordinates from model output."""
    pattern = r"<point x='([\d.]+)' y='([\d.]+)'>([^<]+)</point>"
    matches = re.findall(pattern, text)
    return [{'x': float(m[0]), 'y': float(m[1]), 'label': m[2]} for m in matches]

def run_inference(image, question):
    """Run model inference on single frame."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    response = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    return response, extract_points(response)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_annotations(image, points, mode="bbox"):
    """Draw bounding boxes or points on image."""
    img = np.array(image.convert('RGB'))
    h, w = img.shape[:2]

    for p in points:
        x_px = int(p['x'] * w / 100)
        y_px = int(p['y'] * h / 100)

        # Color: green for instruments, orange for anatomy
        label_lower = p['label'].lower()
        is_anatomy = any(term in label_lower for term in
                        ['sinus', 'sella', 'cavity', 'septum', 'tumor', 'tissue', 'mucosa'])
        color = (255, 165, 0) if is_anatomy else (0, 255, 0)  # Orange or Green

        if mode == "bbox":
            # Draw bounding box
            box_size = 40
            x1, y1 = max(0, x_px - box_size), max(0, y_px - box_size)
            x2, y2 = min(w, x_px + box_size), min(h, y_px + box_size)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label background
            label_text = f"{p['label']}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(img, label_text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:  # point mode
            # Draw crosshair
            cv2.line(img, (x_px - 20, y_px), (x_px + 20, y_px), color, 2)
            cv2.line(img, (x_px, y_px - 20), (x_px, y_px + 20), color, 2)
            cv2.circle(img, (x_px, y_px), 8, color, 2)

        # Center point
        cv2.circle(img, (x_px, y_px), 4, color, -1)

    return img

def create_detection_summary(points):
    """Create text summary of detections."""
    if not points:
        return "No instruments or anatomy detected."

    instruments = [p['label'] for p in points if not any(term in p['label'].lower()
                   for term in ['sinus', 'sella', 'cavity', 'septum', 'tumor', 'tissue', 'mucosa'])]
    anatomy = [p['label'] for p in points if p['label'] not in instruments]

    summary = f"**Detected {len(points)} objects:**\n\n"

    if instruments:
        summary += f"**Instruments ({len(instruments)}):**\n"
        for inst in instruments:
            summary += f"- {inst}\n"
        summary += "\n"

    if anatomy:
        summary += f"**Anatomy ({len(anatomy)}):**\n"
        for anat in anatomy:
            summary += f"- {anat}\n"

    return summary

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def process_single_frame(image, question, annotation_mode):
    """Process single image frame."""
    if image is None:
        return None, "Please upload an image.", ""

    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Run inference
    response, points = run_inference(image, question)

    # Draw annotations
    annotated = draw_annotations(image, points, mode=annotation_mode)

    # Create summary
    summary = create_detection_summary(points)

    # Full response
    full_response = f"**Model Output:**\n```\n{response}\n```\n\n{summary}"

    return annotated, full_response, f"Found {len(points)} detection(s)"

def process_video(video_path, question, annotation_mode, max_frames):
    """Process video file frame by frame."""
    if video_path is None:
        return None, "Please upload a video.", ""

    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limit frames
    process_frames = min(max_frames, total_frames)
    frame_step = max(1, total_frames // process_frames)

    annotated_frames = []
    all_detections = []
    frame_idx = 0
    processed = 0

    progress_text = f"Processing {process_frames} frames from video with {total_frames} total frames...\n\n"

    while cap.isOpened() and processed < process_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Run inference
            response, points = run_inference(image, question)
            all_detections.extend(points)

            # Draw annotations
            annotated = draw_annotations(image, points, mode=annotation_mode)
            annotated_frames.append(annotated)

            processed += 1
            progress_text += f"Frame {frame_idx}: {len(points)} detections\n"

        frame_idx += 1

    cap.release()

    # Create output video
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    imageio.mimsave(output_path, annotated_frames, fps=min(fps, 2), codec='libx264', quality=8)

    # Summary
    instruments = [d['label'] for d in all_detections if not any(term in d['label'].lower()
                   for term in ['sinus', 'sella', 'cavity', 'septum', 'tumor', 'tissue', 'mucosa'])]
    anatomy = [d['label'] for d in all_detections if d['label'] not in instruments]

    summary = f"""
**Video Processing Complete!**

- Processed: {processed} frames
- Total detections: {len(all_detections)}
- Instruments: {len(instruments)} ({len(set(instruments))} unique)
- Anatomy: {len(anatomy)} ({len(set(anatomy))} unique)

**Most common instruments:**
"""

    from collections import Counter
    for inst, count in Counter(instruments).most_common(5):
        summary += f"\n- {inst}: {count}x"

    return output_path, progress_text + "\n" + summary, f"Processed {processed} frames"

# ============================================================================
# DEMO QUESTIONS
# ============================================================================

DEMO_QUESTIONS = [
    "Point to all surgical instruments visible in this frame.",
    "Identify and locate all anatomical structures.",
    "Where is the suction device?",
    "Point to the Kerrison rongeur.",
    "Locate all instruments and anatomy visible.",
    "Where is the sphenoid sinus?",
    "Identify the surgical tools being used.",
]

# ============================================================================
# GRADIO UI
# ============================================================================

with gr.Blocks(title="PitVQA Surgical Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔬 PitVQA: Surgical Instrument & Anatomy Localization

    **Interactive demo** of spatial localization for pituitary surgery.

    - **Model**: Qwen2-VL-2B fine-tuned on 10,139 validated surgical frames
    - **Tasks**: Instrument detection, anatomy identification, spatial localization
    - **Performance**: 80% quadrant accuracy with 100% data fidelity

    Upload a surgical frame or video and ask natural language questions!
    """)

    with gr.Tabs() as tabs:
        # ====================================================================
        # TAB 1: SINGLE FRAME
        # ====================================================================
        with gr.Tab("📷 Single Frame"):
            with gr.Row():
                with gr.Column(scale=1):
                    frame_input = gr.Image(type="pil", label="Upload Surgical Frame")
                    frame_question = gr.Dropdown(
                        choices=DEMO_QUESTIONS,
                        value=DEMO_QUESTIONS[0],
                        label="Question",
                        allow_custom_value=True
                    )
                    frame_mode = gr.Radio(
                        choices=["bbox", "point"],
                        value="bbox",
                        label="Annotation Style"
                    )
                    frame_btn = gr.Button("🔍 Analyze Frame", variant="primary")

                with gr.Column(scale=1):
                    frame_output = gr.Image(label="Annotated Result")
                    frame_status = gr.Label(label="Status")

            frame_text = gr.Markdown(label="Detection Results")

        # ====================================================================
        # TAB 2: VIDEO PROCESSING
        # ====================================================================
        with gr.Tab("🎬 Video Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Surgical Video")
                    video_question = gr.Dropdown(
                        choices=DEMO_QUESTIONS,
                        value=DEMO_QUESTIONS[0],
                        label="Question"
                    )
                    video_mode = gr.Radio(
                        choices=["bbox", "point"],
                        value="bbox",
                        label="Annotation Style"
                    )
                    video_frames = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=10,
                        label="Max Frames to Process"
                    )
                    video_btn = gr.Button("🎥 Process Video", variant="primary")

                with gr.Column(scale=1):
                    video_output = gr.Video(label="Annotated Video")
                    video_status = gr.Label(label="Status")

            video_text = gr.Markdown(label="Processing Results")

        # ====================================================================
        # TAB 3: ABOUT
        # ====================================================================
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## Model Information

            **Base Model**: Qwen2-VL-2B-Instruct
            **Fine-tuned Model**: mmrech/pitvqa-qwen2vl-spatial
            **Dataset**: mmrech/pitvqa-comprehensive-spatial (10,139 samples)

            ### Performance Metrics

            | Metric | Value |
            |--------|-------|
            | **Dataset Fidelity** | 100% (validated against ground truth) |
            | **Quadrant Accuracy** | 80.3% (model predictions) |
            | **Coordinate MAE** | 12.1% (mean absolute error) |
            | **Confidence** | 95%+ on validated samples |

            ### Capabilities

            ✅ **Instrument Detection**: Suction device, Kerrison rongeur, ring curette, forceps, etc.
            ✅ **Anatomy Identification**: Sphenoid sinus, sella, nasal cavity, tumor tissue
            ✅ **Spatial Localization**: Normalized (x, y) coordinates (0-100 scale)
            ✅ **Multi-object Tracking**: Simultaneous detection of multiple instruments

            ### Limitations

            ⚠️ **Occlusion**: Performance degrades with heavily occluded instruments
            ⚠️ **Lighting**: Extreme lighting conditions may affect detection
            ⚠️ **Novel Instruments**: May not recognize instruments outside training set

            ### Citation

            ```bibtex
            @article{rech2026pitvqa,
              title={PitVQA: Multi-Task Vision-Language Model for Pituitary Surgery},
              author={Rech, Matheus},
              journal={MICCAI},
              year={2026}
            }
            ```

            ### Links

            - 📊 [Dataset on HuggingFace](https://huggingface.co/datasets/mmrech/pitvqa-comprehensive-spatial)
            - 🤖 [Model on HuggingFace](https://huggingface.co/mmrech/pitvqa-qwen2vl-spatial)
            - 📄 [Paper](https://github.com/YOUR_USERNAME/pitvqa-surgical-workflow)
            - 💻 [Code Repository](https://github.com/YOUR_USERNAME/pitvqa-surgical-workflow)
            """)

    # Connect functions
    frame_btn.click(
        fn=process_single_frame,
        inputs=[frame_input, frame_question, frame_mode],
        outputs=[frame_output, frame_text, frame_status]
    )

    video_btn.click(
        fn=process_video,
        inputs=[video_input, video_question, video_mode, video_frames],
        outputs=[video_output, video_text, video_status]
    )

    gr.Markdown("""
    ---
    **Note**: This is a research prototype. Not intended for clinical use.
    """)

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
