"""Streamlit research demo for the Deepfake Detection System."""

import os
import io
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from inference.demo_predictor import DemoPredictor
from inference.gradcam import gradcam_plus_plus, gradcam_video


ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Baseline Model": {
        "path": ROOT / "checkpoints" / "baselines" / "efficientnet_b0_enhanced_best.pth",
        "type": "baseline",
        "description": "EfficientNet-B0 baseline classifier",
    },
    "EfficientNet - B0 + CBAM": {
        "path": ROOT / "checkpoints" / "experiments" / "cbam_20260630_235749" / "efficientnet_cbam_stage2_epoch_15.pth",
        "type": "cbam",
        "description": "EfficientNet-B0 with Channel and Spatial Attention",
    },
}
CONFIG_PATH = ROOT / "configs" / "cbam_config.json"

st.set_page_config(page_title="Deepfake Detection System", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#17212b; --muted:#64717d; --line:#dce4e8; --blue:#176b87; --blue-soft:#e8f3f6; --red:#b94a4a; --red-soft:#f9eceb; --green:#17745b; --green-soft:#e8f5ef; }
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color:var(--ink); }
.block-container { max-width: 1440px; padding-top: 2.5rem; }
.eyebrow { color:var(--blue); font-family:'DM Mono',monospace; font-size:.75rem; letter-spacing:.13em; text-transform:uppercase; }
.hero h1 { font-size:clamp(2rem,4vw,3.8rem); letter-spacing:-.055em; margin:.2rem 0 .3rem; }
.hero p { color:var(--muted); font-size:1.08rem; margin-bottom:2rem; }
.card { background:#171821; color:#fff; border:1px solid #A9A3CA; border-radius:16px; padding:1.25rem; box-shadow:0 8px 28px rgba(23,33,43,.12); }
.media-card { background:#11121A; border:2px solid #A9A3CA; border-radius:18px; min-height:420px; padding:1.1rem; display:flex; flex-direction:column; justify-content:center; }
.media-card-wrapper { background:#11121A; border:2px solid #A9A3CA; border-radius:18px; min-height:420px; padding:1rem; }
.media-card h3 { color:#fff; margin:.8rem 0 .3rem; }
.media-help { color:#c8c4dc; text-align:center; margin-bottom:1rem; }
.result { border-radius:20px; padding:1.6rem; color:white; background:linear-gradient(135deg,#173d4b,#176b87); min-height:210px; }
.result.fake { background:linear-gradient(135deg,#6f3038,#b94a4a); }
.result .label { opacity:.75; font-family:'DM Mono',monospace; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; }
.result .prediction { font-size:3.3rem; font-weight:700; letter-spacing:-.06em; margin:.3rem 0; }
.metric-label { color:#c8c4dc; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-family:'DM Mono',monospace; font-size:1.4rem; font-weight:500; }
.mono { font-family:'DM Mono',monospace; }
.section-title { color:#fff; font-size:1.45rem; font-weight:600; margin:2.2rem 0 .8rem; letter-spacing:-.03em; }
.pipeline { display:flex; gap:.35rem; align-items:stretch; overflow-x:auto; padding-bottom:.3rem; }
.pipe-step { min-width:110px; flex:1; background:var(--blue-soft); border:1px solid #cbe3e8; border-radius:10px; padding:.75rem .55rem; text-align:center; font-size:.78rem; }
.pipe-step strong { display:block; font-size:1.1rem; color:var(--blue); margin-bottom:.2rem; }
div[data-testid="stFileUploader"] section { border-color:#A9A3CA; background:#242331; }
div[data-testid="stFileUploader"] section small, div[data-testid="stFileUploader"] section span { color:#fff; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#11121A; border:2px solid #A9A3CA; border-radius:18px; min-height:420px; padding:1rem; }
.dark-copy { color:#c8c4dc; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_predictor(checkpoint: str, model_type: str):
    config = str(CONFIG_PATH) if model_type == "cbam" else None
    return DemoPredictor(checkpoint, config, model_type=model_type, max_frames=32)


def card_title(title: str, detail: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>' + (f'<div style="color:#64717d;margin-top:-.6rem;margin-bottom:1rem">{detail}</div>' if detail else ""), unsafe_allow_html=True)


def boxed_image(image, caption):
    st.image(image, caption=caption, use_container_width=True)


st.markdown('<div class="hero"><div class="eyebrow">Research prototype · visual inference</div><h1>Deepfake Detection System</h1><p>Facial Image Analysis using EfficientNet-B0 + CBAM</p></div>', unsafe_allow_html=True)

model_name = st.selectbox("Select detection model", list(MODEL_OPTIONS), key="model_name")
model = MODEL_OPTIONS[model_name]
st.caption(model["description"] + " · " + str(model["path"]))
if st.session_state.get("active_model") != model_name:
    st.session_state.active_model = model_name
    st.session_state.pop("analysis", None)
    st.session_state.pop("cam", None)
    st.session_state.pop("cam_video", None)
    st.session_state.pop("cam_image_model", None)
    st.session_state.pop("cam_video_model", None)
    st.session_state.pop("cam_video_key", None)
    st.session_state.pop("cam_image_key", None)

with st.container(border=True):
    upload = st.file_uploader("Upload video or image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"], label_visibility="visible")
    if upload:
        if upload.type.startswith("video"):
            st.video(upload.getvalue())
        else:
            st.image(upload.getvalue(), use_container_width=True)
        size_mb = upload.size / (1024 * 1024)
        st.caption(f"`{upload.name}`  ·  {upload.type or 'unknown type'}  ·  {size_mb:.2f} MB")
    else:
        st.markdown("<h3 style='text-align:center;color:#fff'>Upload a video or image</h3><div class='media-help'>Drag and drop a file here, or use Browse files below.<br>MP4, MOV, AVI, JPG, JPEG, PNG</div>", unsafe_allow_html=True)

controls = st.columns([1, 1, 3])
with controls[0]:
    analyze = st.button("Analyze Deepfake", type="primary", use_container_width=True, disabled=upload is None)
with controls[1]:
    clear = st.button("Clear", use_container_width=True)
if clear:
    st.session_state.pop("analysis", None)
    st.rerun()

if analyze and upload:
    if not model["path"].exists():
        st.error(f"{model_name} checkpoint not found at `{model['path']}`.")
    else:
        suffix = Path(upload.name).suffix.lower()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(upload.getvalue())
                temp_path = temp.name
            progress_bar = st.progress(0)
            status = st.empty()
            predictor = load_predictor(str(model["path"]), model["type"])
            result = predictor.run(temp_path, is_video=upload.type.startswith("video"), progress=lambda text, value: (status.info(text), progress_bar.progress(value)))
            result["model_name"] = model_name
            result["model_type"] = model["type"]
            st.session_state.analysis = result
            status.success("Analysis complete")
        except Exception as error:
            st.error(str(error))
        finally:
            if temp_path:
                os.unlink(temp_path)

result = st.session_state.get("analysis")
if result:
    card_title("Final Prediction", f"{result['model_name']} · mean frame probability determines the video-level classification.")
    final_col, stats_col = st.columns([1, 2], gap="large")
    with final_col:
        tone = "fake" if result["prediction"] == "FAKE" else ""
        st.markdown(f'<div class="result {tone}"><div class="label">Prediction</div><div class="prediction">{result["prediction"]}</div><div>Fake probability <b>{result["fake_probability"]:.1%}</b></div><div>Real probability <b>{result["real_probability"]:.1%}</b></div></div>', unsafe_allow_html=True)
    with stats_col:
        metrics = st.columns(4)
        for column, label, value in zip(metrics, ["Frames analyzed", "Faces detected", "Fake frames", "Real frames"], [result["total_frames"], result["detected_faces"], result["fake_frames"], result["real_frames"]]):
            column.markdown(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card" style="margin-top:1rem"><span class="mono">mean(fake probability)</span> = <b>{result["fake_probability"]:.3f}</b> &nbsp; · &nbsp; threshold = <b>{result["threshold"]:.2f}</b> &nbsp; · &nbsp; final = <b>{result["prediction"]}</b></div>', unsafe_allow_html=True)

    valid = [frame for frame in result["frames"] if frame["fake_probability"] is not None]
    st.markdown("### Export Model Attention", unsafe_allow_html=True)
    st.caption("0.4× speed · cropped facial regions only · MP4 download" if upload and upload.type.startswith("video") else "Grad-CAM++ explanation image download")
    export_key = f"{upload.name if upload else 'input'}:{result['model_name']}:{result['frames'][0]['frame_index']}"
    if upload and upload.type.startswith("video") and st.session_state.get("cam_video_key") != export_key:
        try:
            predictor = load_predictor(str(model["path"]), model["type"])
            st.session_state.cam_video = gradcam_video(
                predictor.model,
                result["frames"],
                predictor.transform,
                predictor.device,
                result["metadata"].get("fps") or 5,
                result["metadata"].get("duration") or 0,
                speed=0.4,
            )
            st.session_state.cam_video_key = export_key
        except Exception as error:
            st.warning(f"Grad-CAM++ export could not be generated: {error}")
    elif upload and not upload.type.startswith("video") and st.session_state.get("cam_image_key") != export_key:
        try:
            predictor = load_predictor(str(model["path"]), model["type"])
            image_result = gradcam_plus_plus(predictor.model, valid[0]["face"], predictor.transform, predictor.device)
            image_buffer = io.BytesIO()
            Image.fromarray(image_result["overlay"]).save(image_buffer, format="PNG")
            st.session_state.cam_image = image_buffer.getvalue()
            st.session_state.cam_image_key = export_key
        except Exception as error:
            st.warning(f"Grad-CAM++ export could not be generated: {error}")

    if upload and upload.type.startswith("video") and st.session_state.get("cam_video_key") == export_key:
        st.download_button("Export Grad-CAM++", st.session_state.cam_video, file_name="gradcam_plus_plus_faces_0.4x.mp4", mime="video/mp4", key="export_cam_video", use_container_width=True)
    elif upload and not upload.type.startswith("video") and st.session_state.get("cam_image_key") == export_key:
        st.download_button("Export Grad-CAM++", st.session_state.cam_image, file_name="gradcam_plus_plus_face.png", mime="image/png", key="export_cam_image", use_container_width=True)

    card_title("Frame Probability Over Time", "Fake probability across analyzed frames; the classification threshold is 0.50.")
    chart = pd.DataFrame({"Frame": [frame["frame_index"] + 1 for frame in valid], "Fake probability": [frame["fake_probability"] for frame in valid], "Threshold": [result["threshold"]] * len(valid)}).set_index("Frame")
    st.line_chart(chart, y=["Fake probability", "Threshold"], color=["#b94a4a", "#64717d"], height=280)

    card_title("Frame-Level Analysis", "Representative frames from the model input.")
    frame_count = st.slider("Frames to show", min_value=min(6, len(valid)), max_value=len(valid), value=min(6, len(valid)))
    grid = valid[:frame_count]
    columns = st.columns(3)
    for index, frame in enumerate(grid):
        with columns[index % 3]:
            st.image(frame["face"], caption=f"Frame {frame['frame_index'] + 1} · {frame['prediction']} · fake {frame['fake_probability']:.1%}", use_container_width=True)

    card_title("Selected Frame Analysis", "RetinaFace bounding box, detected face, and normalized model crop.")
    selected_index = st.selectbox("Select analyzed frame", range(len(valid)), format_func=lambda index: f"Frame {valid[index]['frame_index'] + 1} · {valid[index]['timestamp']:.2f}s")
    selected = valid[selected_index]
    original = selected["image"].copy()
    box = selected["bbox"]
    cv2.rectangle(original, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (23, 107, 135), 4)
    image_cols = st.columns(3)
    with image_cols[0]: boxed_image(original, "Original frame · RetinaFace box")
    with image_cols[1]: boxed_image(selected["face"], f"Detected face · confidence {selected['face_confidence']:.1%}")
    with image_cols[2]: boxed_image(selected["face"], "224 × 224 model input")
    detail_cols = st.columns(4)
    for column, label, value in zip(detail_cols, ["Frame index", "Timestamp", "Fake probability", "Prediction"], [selected["frame_index"] + 1, f"{selected['timestamp']:.2f}s", f"{selected['fake_probability']:.1%}", selected["prediction"]]):
        column.metric(label, value)

    card_title("Model Information")
    info = st.columns(5)
    model_description = "EfficientNet-B0 baseline" if result["model_type"] == "baseline" else "EfficientNet-B0 + CBAM"
    for column, label, value in zip(info, ["Model", "Input size", "Face detector", "Classification", "Aggregation"], [model_description, "224 × 224", "RetinaFace", "Binary · Real / Fake", "Mean frame probability"]):
        column.markdown(f'<div class="card"><div class="metric-label">{label}</div><b>{value}</b></div>', unsafe_allow_html=True)
