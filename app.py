import os
import cv2
import time
import torch
import tempfile
import numpy as np
import streamlit as st
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# =========================================
# SETTINGS
# =========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "best_ckpt"
IMG_SIZE = 384
THRESHOLD = 0.5
TARGET_CLASS = 1

# =========================================
# TEST SET METRICS
# =========================================
TEST_MAE = 0.056
TEST_FMEASURE = 0.764
TEST_SMEASURE = 0.791

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Camouflaged Object Detection Demo", layout="wide")
st.title("Camouflaged Object Detection Demo")
st.write("Upload a video and run SegFormer segmentation on each frame.")

st.markdown("### Model Evaluation on Test Set")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{TEST_MAE:.3f}")
m2.metric("F-measure", f"{TEST_FMEASURE:.3f}")
m3.metric("S-measure", f"{TEST_SMEASURE:.3f}")

st.info(
    "These metrics are computed on the test set using ground truth masks. "
    "They are shown here to clarify the model performance in addition to the demo video."
)

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("Model Settings")
st.sidebar.write("Model: SegFormer")
st.sidebar.write(f"Device: {DEVICE}")
st.sidebar.write(f"Image Size: {IMG_SIZE}")
st.sidebar.write(f"Threshold: {THRESHOLD}")
st.sidebar.write(f"Target Class: {TARGET_CLASS}")

st.sidebar.header("Test Set Metrics")
st.sidebar.write(f"MAE: {TEST_MAE:.3f}")
st.sidebar.write(f"F-measure: {TEST_FMEASURE:.3f}")
st.sidebar.write(f"S-measure: {TEST_SMEASURE:.3f}")

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    return processor, model

# =========================================
# PREDICTION
# =========================================
def predict_mask(frame_bgr, processor, model):
    rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_original.shape[:2]

    img_resized = cv2.resize(rgb_original, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    enc = processor(images=img_resized, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(DEVICE)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        up = torch.nn.functional.interpolate(
            logits,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )
        prob = torch.softmax(up, dim=1)[0, TARGET_CLASS].detach().cpu().numpy()

    prob_map = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    pred_mask = (prob_map > THRESHOLD).astype(np.uint8) * 255

    return rgb_original, pred_mask

# =========================================
# OVERLAY MASK
# =========================================
def overlay_mask(rgb_frame, binary_mask):
    overlay = rgb_frame.copy()
    mask_bool = binary_mask > 0

    red_layer = np.zeros_like(rgb_frame, dtype=np.uint8)
    red_layer[:, :, 0] = 255

    alpha = 0.45
    overlay[mask_bool] = (
        alpha * red_layer[mask_bool] + (1 - alpha) * overlay[mask_bool]
    ).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)

    return overlay_bgr

# =========================================
# DRAW METRICS ON VIDEO
# =========================================
def draw_metrics(frame, mae, fmeasure, smeasure):
    """
    Draw global test-set metrics on the output video frame.
    """
    overlay = frame.copy()

    # Transparent dark box behind text
    cv2.rectangle(overlay, (10, 10), (360, 125), (0, 0, 0), -1)
    alpha = 0.45
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)

    cv2.putText(frame, f"MAE: {mae:.3f}", (20, 40), font, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"F-measure: {fmeasure:.3f}", (20, 75), font, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"S-measure: {smeasure:.3f}", (20, 110), font, 0.8, color, 2, cv2.LINE_AA)

    return frame

# =========================================
# PROCESS VIDEO
# =========================================
def process_video(input_path, output_path, processor, model, frame_skip=1):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_fps = fps / frame_skip if frame_skip > 1 else fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    progress_bar = st.progress(0)
    status_text = st.empty()

    processed = 0
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_skip > 1 and frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        rgb_frame, pred_mask = predict_mask(frame, processor, model)
        result_frame = overlay_mask(rgb_frame, pred_mask)

        # Add metrics text on output video
        result_frame = draw_metrics(
            result_frame,
            TEST_MAE,
            TEST_FMEASURE,
            TEST_SMEASURE
        )

        writer.write(result_frame)

        processed += 1
        frame_idx += 1

        if total_frames > 0:
            progress = min(frame_idx / total_frames, 1.0)
            progress_bar.progress(progress)

        elapsed = time.time() - start_time
        status_text.text(
            f"Processed frames: {processed} | Original frame index: {frame_idx}/{total_frames} | Time: {elapsed:.1f}s"
        )

    cap.release()
    writer.release()
    progress_bar.empty()
    status_text.empty()

# =========================================
# UI
# =========================================
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
frame_skip = st.selectbox("Frame Skip", [1, 2, 3], index=0)

if uploaded_video is not None:
    st.subheader("Original Video")
    st.video(uploaded_video)

    if st.button("Run Segmentation Demo"):
        with st.spinner("Loading model and processing video..."):
            processor, model = load_model()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
                temp_input.write(uploaded_video.read())
                input_video_path = temp_input.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
                output_video_path = temp_output.name

            process_video(
                input_path=input_video_path,
                output_path=output_video_path,
                processor=processor,
                model=model,
                frame_skip=frame_skip
            )

        st.success("Done.")

        st.subheader("Segmented Output Video")
        with open(output_video_path, "rb") as f:
            output_bytes = f.read()
            st.video(output_bytes)

        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Output Video",
                data=f,
                file_name="segformer_demo_output.mp4",
                mime="video/mp4"
            )