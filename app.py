import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌱 Plant Disease Detection System")
st.write("Upload a leaf image to detect the disease.")

# Updated File Name
MODEL_PATH = 'crop_disease_resnet50.keras'

# 1. Load Model
@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ '{MODEL_PATH}' file nahi mili! Make sure ye GitHub repo me push hui ho.")
        st.stop()
        
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

try:
    model = load_keras_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Class Names (15 Classes)
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Yellow_Leaf_Curl_Virus',
    'Tomato__mosaic_virus',
    'Tomato_healthy'
]

# 3. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image read karo aur RGB ensure karo
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing Image..."):
            # Image Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize (1./255 scaling)
            img_array = img_array / 255.0
            
            # Add batch dimension -> (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100

            # Output Clean Formatting
            readable_class = predicted_class.replace('___', ' - ').replace('_', ' ')
            
            st.success(f"**Prediction:** {readable_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")
