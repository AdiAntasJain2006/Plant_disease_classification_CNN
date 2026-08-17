"""
app.py
Interactive Streamlit dashboard for real-time plant disease diagnosis.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from src.utils import get_gradcam_heatmap, overlay_heatmap

st.set_page_config(page_title="Plant Disease Diagnostic AI", layout="wide")

st.title("🌿 AI Plant Disease Classifier & Explainability Dashboard")
st.write("Upload a leaf photo to diagnose potential crop diseases and inspect model focus using Grad-CAM heatmaps.")

# 1. Load Model & Class Names
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("models/plant_model.keras")

model = load_trained_model()

# Class names list (38 categories matching dataset folder order)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# 2. File Upload Interface
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp image file
    temp_path = "temp_leaf.jpg"
    image = Image.open(uploaded_file)
    image.save(temp_path)

    # Preprocess image tensor
    img = cv2.imread(temp_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)

    # Run Model Inference
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    predicted_class = CLASS_NAMES[pred_idx]

    # Generate Grad-CAM Overlay
    heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=pred_idx)
    gradcam_img = overlay_heatmap(temp_path, heatmap)

    # Render Side-by-Side Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Uploaded Image")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Model Attention Map")
        st.image(gradcam_img, use_container_width=True)

    st.success(f"**Diagnosis:** {predicted_class.replace('___', ' - ')}")
    st.info(f"**Confidence Score:** {confidence:.2f}%")