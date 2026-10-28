# ANNA silver-lining detector — Streamlit app (single file)
# requirements.txt: streamlit ultralytics opencv-python-headless pillow numpy
# packages.txt (Streamlit Cloud): libgl1
import os
import numpy as np, cv2, streamlit as st
from PIL import Image
from ultralytics import YOLO

# ================= CONFIG =================
WEIGHTS      = "spindle_silver_best.pt"
IMGSZ        = 640
DEFAULT_CONF = 0.25
GREEN_RGB    = (51, 153, 51)
GREEN_TOL    = 40
BAR_ROW_FRAC = 0.6
# ==========================================


@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS):
        st.error(f"Missing weights file: {WEIGHTS}. Commit it next to app.py.")
        st.stop()
    return YOLO(WEIGHTS)


def detect_bars(img):
    h = img.shape[0]
    g = np.array(GREEN_RGB[::-1], dtype=np.int16)
    m = (np.abs(img.astype(np.int16) - g).max(axis=2) < GREEN_TOL).mean(axis=1) > BAR_ROW_FRAC
    top = 0
    while top < h and m[top]: top += 1
    bot = 0
    while bot < h - top and m[h - 1 - bot]: bot += 1
    return top, bot


def strip_ui(img):
    """Same preprocessing as training. No-op on raw camera frames."""
    h = img.shape[0]
    top, bot = detect_bars(img)
    img = img[top:h - bot]
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    mask = ((r < 100) & (g > 180) & (b > 180)).astype(np.uint8) * 255
    if mask.any():
        img = cv2.inpaint(img, cv2.dilate(mask, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)
    return np.ascontiguousarray(img)


def run(model, pil, conf):
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = strip_ui(bgr)
    r = model.predict(bgr, conf=conf, imgsz=IMGSZ, verbose=False)[0]
    return cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), r.boxes


st.set_page_config(page_title="Silver Lining Detector", layout="wide")
st.title("Spindle Silver-Lining Detector")
conf = st.sidebar.slider("Confidence", 0.05, 0.95, DEFAULT_CONF, 0.05)
src = st.sidebar.radio("Input", ["Upload", "Camera"])
model = load_model()

files = []
if src == "Upload":
    files = st.file_uploader("Spindle images", type=["jpg", "jpeg", "png"], accept_multiple_files=True) or []
else:
    cam = st.camera_input("Capture")
    files = [cam] if cam else []

for f in files:
    out, boxes = run(model, Image.open(f), conf)
    c1, c2 = st.columns([3, 1])
    c1.image(out, use_container_width=True)
    if len(boxes):
        c2.error(f"DEFECT — {len(boxes)} silver lining(s)")
        for i, b in enumerate(boxes):
            c2.write(f"#{i+1}: conf {float(b.conf):.2f}")
    else:
        c2.success("CLEAN — no silver lining")
