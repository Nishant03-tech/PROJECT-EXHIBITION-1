"""Streamlit Web Application for Tuberculosis Detection & Clinical Lung-Masked Grad-CAM.

Official deployment script for Streamlit Community Cloud (share.streamlit.io).
"""

from __future__ import annotations
import os
from pathlib import Path
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

from model import IMG_SIZE, BinaryFocalLoss
from gradcam import extract_lung_mask, compute_gradcam_heatmap, render_clinical_gradcam

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TB Chest X-Ray AI Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-tb {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_classification_model(model_path: str = "best_tb_densenet_model.keras"):
    """Loads and caches the trained Keras model."""
    path = Path(model_path)
    if not path.exists():
        return None
    return tf.keras.models.load_model(
        str(path),
        custom_objects={"BinaryFocalLoss": BinaryFocalLoss},
        compile=False,
    )


def process_image(model, raw_image_rgb: np.ndarray):
    """Executes prediction and generates clinical Grad-CAM."""
    resized_rgb = cv2.resize(raw_image_rgb, IMG_SIZE)
    
    # Preprocess tensor for model input
    x_pre = tf.keras.applications.densenet.preprocess_input(resized_rgb.copy().astype(np.float32))
    input_tensor = tf.expand_dims(x_pre, axis=0)

    # Compute prediction & Grad-CAM
    cam, score = compute_gradcam_heatmap(model, input_tensor)
    lung_mask = extract_lung_mask(resized_rgb)
    overlay = render_clinical_gradcam(resized_rgb, cam, lung_mask)

    pred_label = "Tuberculosis Detected" if score >= 0.5 else "Normal (Clear Lungs)"
    confidence = score if score >= 0.5 else (1.0 - score)

    return pred_label, float(score), float(confidence), overlay


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=70)
    st.title("System Specs")
    st.markdown(
        """
        - **Architecture:** DenseNet121
        - **Attention:** CBAM (Channel + Spatial)
        - **Loss:** Binary Focal Loss
        - **Test Accuracy:** **99.07%**
        - **ROC-AUC:** **0.9971**
        - **Explainability:** Lung-Masked Grad-CAM
        """
    )
    st.divider()
    st.info("Academic research prototype evaluated on the Exhibition TB dataset.")


# --- MAIN INTERFACE ---
st.markdown('<div class="main-title">🫁 Tuberculosis AI Diagnostic Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Chest Radiograph Screening with Clinical Lung-Masked Grad-CAM</div>', unsafe_allow_html=True)

model = load_classification_model()

if model is None:
    st.error("⚠️ Model file `best_tb_densenet_model.keras` not found in current directory.")
    st.stop()

# Input Options
tab1, tab2 = st.tabs(["📤 Upload Chest X-Ray", "🖼️ Quick Test Samples"])

uploaded_file = None
sample_chosen = None

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a posteroanterior (PA) chest X-ray image (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
    )

with tab2:
    st.write("Or pick a sample from the test set:")
    samples_dir = Path("new_test_images")
    if samples_dir.exists():
        sample_files = list(samples_dir.glob("*.*"))[:4]
        if sample_files:
            cols = st.columns(len(sample_files))
            for i, sf in enumerate(sample_files):
                with cols[i]:
                    img = Image.open(sf)
                    st.image(img, caption=sf.name, use_container_width=True)
                    if st.button(f"Analyze {sf.name}", key=f"btn_{i}"):
                        sample_chosen = sf
        else:
            st.write("No images found in `new_test_images/`.")
    else:
        st.write("Directory `new_test_images/` not found.")

selected_path = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        selected_path = Path(tmp.name)
elif sample_chosen is not None:
    selected_path = sample_chosen

if selected_path is not None:
    st.divider()

    # Load raw image
    raw_bgr = cv2.imread(str(selected_path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # Validation check
    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    contrast = float(gray.std())
    r, g, b = raw_rgb[:, :, 0], raw_rgb[:, :, 1], raw_rgb[:, :, 2]
    color_diff = float(np.mean(np.abs(r.astype(float) - g.astype(float))) +
                        np.mean(np.abs(g.astype(float) - b.astype(float))))

    if color_diff > 28.0 or contrast < 18.0:
        st.error("❌ Input Rejected: This image does not meet thoracic radiograph contrast or spectrum criteria.")
        st.stop()

    with st.spinner("Analyzing radiograph with DenseNet121 + CBAM..."):
        label, score, conf, overlay = process_image(model, raw_rgb)

    # Results Header
    st.subheader("Diagnostic Results")
    col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

    with col_res1:
        if "Tuberculosis" in label:
            st.markdown(f'<div class="badge-tb">⚠️ {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-normal">✅ {label}</div>', unsafe_allow_html=True)

    with col_res2:
        st.metric(label="Diagnostic Confidence", value=f"{conf:.2%}")

    with col_res3:
        st.metric(label="TB Probability Score", value=f"{score:.4f}")

    st.write("")

    # Clinical Side-by-Side Comparison
    st.subheader("Explainable AI: Original vs Lung-Masked Grad-CAM")
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.image(cv2.resize(raw_rgb, IMG_SIZE), caption="Original Chest X-Ray", use_container_width=True)

    with col_img2:
        st.image(overlay, caption="Lung-Masked Grad-CAM (Targeted Pulmonary Heatmap)", use_container_width=True)

    st.caption("Warm regions (Red/Yellow) highlight focal activations in the lung parenchyma. Background purple wash eliminates non-lung artifacts.")
