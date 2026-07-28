
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="🌿 Crop Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# Class names
CLASS_NAMES = sorted([
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew",
    "Cherry___healthy", "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_Black_Measles",
    "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
])

@st.cache_resource
def load_my_model():
    model = load_model("crop_disease_resnet50.h5")
    return model

def predict(image, model):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    top3_idx = predictions.argsort()[-3:][::-1]
    results = []
    for idx in top3_idx:
        name = CLASS_NAMES[idx].replace("___", " — ").replace("_", " ")
        confidence = predictions[idx] * 100
        results.append((name, confidence))
    return results

# UI
st.title("🌿 Crop Disease Detector")
st.markdown("**Upload a leaf image to detect plant disease using AI**")
st.markdown("---")

model = load_my_model()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            results = predict(image, model)
        
        st.markdown("### 🔍 Results")
        for i, (name, conf) in enumerate(results):
            if i == 0:
                st.success(f"**{name}**")
                st.metric("Confidence", f"{conf:.1f}%")
            else:
                st.info(f"{name} — {conf:.1f}%")

st.markdown("---")
st.markdown("*ML Term Project — Muhammad Anus Imran (231370129)*")
