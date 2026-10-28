import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Spindle Defect Detection", page_icon="🧵", layout="wide")

st.title("Edge-Deployed Quality Control: Spindle Defect Detection")
st.markdown("Automated inspection of silver lining defects in gold thread spindles using YOLO.")

# Sidebar Configuration
st.sidebar.header("Operator Controls")
st.sidebar.markdown("Adjust inference parameters in real-time.")
# Confidence slider as requested in the project proposal
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.00, value=0.15, step=0.01)
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Details**")
st.sidebar.markdown("Course: ANNA (2307411T)")
st.sidebar.markdown("Developer: Kanak Panchal")

@st.cache_resource
def load_model():
    # Ensure your best.pt file is in the same directory as this app.py file
    return YOLO('best.pt')

model = load_model()

# File Uploader
uploaded_file = st.file_uploader("Upload Spindle Image for Inspection", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Handle PNGs with alpha channels
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Camera Feed")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Inspection Overlay")
        with st.spinner("Running inference..."):
            # Run YOLO prediction
            results = model.predict(source=img_array, conf=conf_threshold)
            
            # Plot the results
            res_plotted = results[0].plot(line_width=3)
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_rgb, use_column_width=True)
            
    # Defect Telemetry & Quality Logging Summary
    st.markdown("### Inspection Summary")
    num_defects = len(results[0].boxes)
    
    if num_defects > 0:
        st.error(f"⚠️ QUALITY ALERT: {num_defects} silver lining defect(s) detected.")
        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf)
            st.warning(f"Defect {i+1}: Confidence {conf:.2%}")
    else:
        st.success("✅ PASS: No defects detected at current confidence threshold.")
