import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌱 Plant Disease Detection System")
st.write("Upload a leaf image to detect the disease.")

# 1. Load Model (Cached taake app baar baar re-load na ho)
@st.cache_resource
def load_keras_model():
    # Apne saved model ka sahi path yahan do (.h5 ya .keras)
    model = tf.keras.models.load_model('plant_disease_resnet50.h5')
    return model

try:
    model = load_keras_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Class Names (Apni notebook ki exact 15 classes ki list yahan rakho)
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
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing Image..."):
            # Image Preprocessing (Notebook ke Mutabiq)
            img = image.resize((224, 224)) # ResNet50 input size
            img_array = np.array(img)
            
            # Agar image RGBA hai to RGB banao
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
            # Normalize (1./255 scaling)
            img_array = img_array / 255.0
            
            # Add batch dimension -> (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100

            # Result Display
            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")
