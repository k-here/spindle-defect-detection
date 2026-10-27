import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spindle Defect Inspection System | AI Quality Control",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #4B5563;
        margin-bottom: 18px;
    }
    .status-badge-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        display: inline-block;
        border: 1px solid #31C48D;
    }
    .status-badge-defect {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        display: inline-block;
        border: 1px solid #F98080;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING WITH CACHE ---
@st.cache_resource
def load_yolo_model():
    # Look for best.pt first, then fallback to yolov8n.pt if not uploaded yet
    candidate_paths = ["best.pt", "weights/best.pt", "yolov8n.pt"]
    for p in candidate_paths:
        if Path(p).exists():
            return YOLO(p), p
    # Final fallback: load lightweight pre-trained weights
    return YOLO("yolov8n.pt"), "yolov8n.pt (Pre-trained Fallback)"

model, model_source = load_yolo_model()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Inspection Controls")
st.sidebar.markdown(f"**Loaded Model:** `{model_source}`")
st.sidebar.markdown("---")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.00, 0.25, 0.05)
iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Academic System Info")
st.sidebar.info("""
- **Course:** ANNA (2307411T)
- **Domain:** Industrial AI (Quality Control)
- **Defect Class:** Silver Lining Defect
- **Deployment:** Streamlit Cloud Free Tier
""")

# --- MAIN UI HEADER ---
st.markdown('<div class="main-header">🧵 Automated Silver Lining Defect Inspection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning-Powered Real-Time Defect Detection & Localization for Precision Thread Spindles</div>', unsafe_allow_html=True)

# --- INPUT SELECTION TABS ---
tab1, tab2 = st.tabs(["📁 Pre-Loaded Test Samples", "📤 Upload New Spindle Image"])

input_image = None
sample_name = "None"

with tab1:
    demo_dir = Path("demo_test_images")
    if demo_dir.exists():
        sample_files = sorted([f for f in demo_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if sample_files:
            selected_file = st.selectbox(
                "Select a sample from the factory inspection queue:",
                sample_files,
                format_func=lambda x: f"📦 {x.name}"
            )
            if selected_file:
                input_image = Image.open(selected_file).convert("RGB")
                sample_name = selected_file.name
        else:
            st.warning("The folder 'demo_test_images/' exists but contains no images.")
    else:
        st.info("💡 Add images to a `demo_test_images/` folder in your repo to select them directly.")

with tab2:
    uploaded_file = st.file_uploader("Upload an image of a spindle (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        sample_name = uploaded_file.name

# --- INFERENCE & RESULTS SECTION ---
if input_image is not None and model is not None:
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📷 Original Spindle View")
        st.image(input_image, use_container_width=True, caption=f"Active Sample: {sample_name}")

    with col_right:
        st.subheader("🔍 Automated Defect Localization")
        start_time = time.time()
        
        # Run inference
        results = model.predict(
            source=np.array(input_image),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        latency_ms = (time.time() - start_time) * 1000

        res = results[0]
        # Overlay bounding boxes
        annotated_bgr = res.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True, caption="YOLOv8 Detection Overlay")

    # --- TELEMETRY & DECISION METRICS ---
    st.markdown("---")
    st.subheader("📊 Quality Inspection Telemetry")

    num_defects = len(res.boxes)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown("**Spindle Decision**")
        if num_defects == 0:
            st.markdown('<span class="status-badge-ok">✅ ACCEPT (PASS)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-defect">❌ REJECT (FAIL)</span>', unsafe_allow_html=True)

    with m_col2:
        st.metric(label="Defects Count", value=num_defects)

    with m_col3:
        st.metric(label="Inference Latency", value=f"{latency_ms:.1f} ms")

    with m_col4:
        if num_defects > 0:
            top_conf = float(res.boxes.conf.max()) * 100
            st.metric(label="Peak Confidence", value=f"{top_conf:.1f}%")
        else:
            st.metric(label="Peak Confidence", value="N/A")

    # --- DETAILED DEFECT BOUNDING BOX TABLE ---
    if num_defects > 0:
        st.markdown("#### 📐 Defect Geometry & Bounding Coordinates")
        table_rows = []
        for idx, box in enumerate(res.boxes):
            coords = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            width_px = coords[2] - coords[0]
            height_px = coords[3] - coords[1]
            area_px = width_px * height_px
            severity = "Critical (>12,000 px²)" if area_px > 12000 else "Moderate"

            table_rows.append({
                "Defect #": idx + 1,
                "Class": "Silver Lining",
                "Confidence": f"{conf:.3f}",
                "X_min": int(coords[0]),
                "Y_min": int(coords[1]),
                "X_max": int(coords[2]),
                "Y_max": int(coords[3]),
                "Area (px²)": int(area_px),
                "Severity": severity
            })

        df_telemetry = pd.DataFrame(table_rows)
        st.dataframe(df_telemetry, use_container_width=True)
else:
    st.info("👈 Please select a pre-loaded sample or upload an image to begin automated defect inspection.")
