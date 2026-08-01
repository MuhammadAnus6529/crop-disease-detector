import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgriGuard | AI Crop Health Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 2. Custom CSS (Full Website Styling)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8faf9;
        font-family: 'Inter', sans-serif;
    }

    /* Header / Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(27, 67, 50, 0.15);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.9;
        max-width: 700px;
        margin: 0 auto;
    }

    /* Cards */
    .feature-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
        transition: transform 0.2s ease;
    }

    /* Result Box */
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 6px solid #2d6a4f;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    .result-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #52b788;
        font-weight: 700;
    }
    .result-disease {
        font-size: 1.6rem;
        font-weight: 800;
        color: #081c15;
        margin: 0.3rem 0;
    }

    /* Spec / info tables on the About tab */
    .spec-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }
    .spec-table td {
        padding: 0.45rem 0.6rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .spec-table td:first-child {
        color: #52796f;
        font-weight: 600;
        width: 45%;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        background-color: #2d6a4f;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        border: none;
        box-shadow: 0 4px 10px rgba(45, 106, 79, 0.2);
    }
    div.stButton > button:hover {
        background-color: #1b4332;
        color: white;
    }

    /* Hide default Streamlit padding at top */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Model & Data Initialization
# ---------------------------------------------------------
MODEL_PATH = 'crop_disease_resnet50.keras'

@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_keras_model()

# 15 classes — matches the PlantVillage subset (Pepper, Potato, Tomato) the
# ResNet50 model was actually trained on (Report, Section 3 & Table 1).
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

# ---------------------------------------------------------
# 4. Helper Functions (Leaf Image Validation)
# ---------------------------------------------------------
def is_likely_leaf_image(pil_img):
    """
    Basic heuristic check: Validates if the image contains enough
    green/brown vegetation pixels to filter out random non-crop images.
    """
    img_hsv = pil_img.convert('HSV')
    np_img = np.array(img_hsv)

    # Hue range for green vegetation (approx 35 to 85 out of 255)
    # & brown/yellowish diseased tones (approx 10 to 35)
    h_channel = np_img[:, :, 0]
    s_channel = np_img[:, :, 1]

    # Check green/brown leaf pixel ratio
    leaf_pixels = np.sum(((h_channel >= 20) & (h_channel <= 95)) & (s_channel > 30))
    total_pixels = h_channel.size
    leaf_ratio = leaf_pixels / total_pixels

    return leaf_ratio > 0.12  # Requires at least 12% plant/leaf coloration

# ---------------------------------------------------------
# 5. Website Header & Navigation Bar
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌱 AgriGuard AI</div>
        <div class="hero-subtitle">Next-Generation Precision Agriculture Diagnostic Engine</div>
    </div>
""", unsafe_allow_html=True)

# Top Navigation Tabs (Acts like a website menu)
tab_home, tab_detector, tab_crops, tab_about, tab_team = st.tabs([
    "🏠 Home",
    "🔬 Disease Detector",
    "🌾 Supported Crops",
    "📊 Project Details",
    "👥 About & Team"
])

# ---------------------------------------------------------
# TAB 1: HOME PAGE
# ---------------------------------------------------------
with tab_home:
    st.write("### Welcome to AgriGuard")
    st.write("AgriGuard leverages deep transfer learning (ResNet50) to instantly detect early-stage crop diseases from leaf imagery.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
            <div class="feature-card">
                <h4>⚡ Instant Diagnosis</h4>
                <p>Upload a photograph of any leaf sample and receive diagnostic classifications within seconds.</p>
            </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
            <div class="feature-card">
                <h4>🎯 High Precision</h4>
                <p>Trained on thousands of curated PlantVillage dataset samples across 15 healthy and diseased conditions.</p>
            </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
            <div class="feature-card">
                <h4>🛡️ Crop Protection</h4>
                <p>Early identification helps farmers prevent field-wide pathogen outbreaks and protect yields.</p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: DISEASE DETECTOR DASHBOARD
# ---------------------------------------------------------
with tab_detector:
    st.write("### 🔬 Automated Leaf Scan Diagnostic")
    st.caption("Upload a clear leaf photo of Pepper, Potato, or Tomato crops.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_file = st.file_uploader(
            "Select Leaf Image",
            type=["jpg", "jpeg", "png"],
            help="Supported Formats: JPG, JPEG, PNG"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Sample Preview", use_container_width=True)

    with col2:
        if uploaded_file is None:
            st.info("👈 Upload an image on the left panel to run AI analysis.")
        else:
            if st.button("Run Diagnostic Check"):
                if model is None:
                    st.error(f"❌ Model file `{MODEL_PATH}` not found in repository root.")
                else:
                    with st.spinner("Analyzing structural and chromatic leaf patterns..."):
                        # Step A: Domain Image Validation
                        if not is_likely_leaf_image(image):
                            st.error("⚠️ **Invalid Image Detected**")
                            st.warning("The uploaded photo does not appear to contain a valid crop leaf sample. Please upload a clear leaf photo.")
                        else:
                            # Step B: Model Inference
                            img_resized = image.resize((224, 224))
                            img_array = np.array(img_resized, dtype=np.float32) / 255.0
                            img_array = np.expand_dims(img_array, axis=0)

                            predictions = model.predict(img_array)
                            confidence = float(np.max(predictions[0]))
                            predicted_idx = int(np.argmax(predictions[0]))

                            # Step C: Confidence Thresholding
                            if confidence < 0.60:
                                st.error("⚠️ **Uncertain Diagnosis / Low Confidence**")
                                st.warning("The AI model cannot confidently match this image to a known disease category. Ensure the image is well-lit and focused on the leaf surface.")
                                st.caption(f"Top Match Score: {confidence * 100:.2f}% (Required: >60%)")
                            else:
                                raw_label = CLASS_NAMES[predicted_idx]
                                formatted_label = raw_label.replace('___', ' — ').replace('_', ' ')

                                st.markdown(f"""
                                    <div class="result-card">
                                        <div class="result-header">Diagnostic Result</div>
                                        <div class="result-disease">{formatted_label}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                st.write("")
                                st.write("**Model Certainty Score:**")
                                st.progress(confidence)
                                st.caption(f"Confidence: **{confidence * 100:.2f}%**")

                                # Recommendation Notice
                                if "healthy" in raw_label.lower():
                                    st.success("✅ **Status:** Plant foliage appears healthy. Maintain standard watering and nutrient schedules.")
                                else:
                                    st.warning("⚠️ **Status:** Pathogen infection identified. Isolate infected plants and consider recommended fungicides/treatments.")

# ---------------------------------------------------------
# TAB 3: SUPPORTED CROPS & CLASSES
# ---------------------------------------------------------
with tab_crops:
    st.write("### 🌾 Supported Crop Categories")
    st.write("The current ResNet50 model instance is calibrated to classify the following 15 conditions:")

    crop_col1, crop_col2, crop_col3 = st.columns(3)
    with crop_col1:
        st.write("#### 🫑 Pepper (Bell)")
        st.markdown("- Bacterial Spot\n- Healthy")
    with crop_col2:
        st.write("#### 🥔 Potato")
        st.markdown("- Early Blight\n- Late Blight\n- Healthy")
    with crop_col3:
        st.write("#### 🍅 Tomato")
        st.markdown("- Bacterial Spot\n- Early Blight\n- Late Blight\n- Leaf Mold\n- Septoria Leaf Spot\n- Spider Mites\n- Target Spot\n- Yellow Leaf Curl Virus\n- Mosaic Virus\n- Healthy")

# ---------------------------------------------------------
# TAB 4: PROJECT DETAILS (pulled from the project report)
# ---------------------------------------------------------
with tab_about:
    st.write("### 📊 Project Details")
    st.caption("Summary of the dataset, model and training pipeline described in the project report.")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("#### 🗂️ Dataset")
        st.markdown("""
            <table class="spec-table">
                <tr><td>Source</td><td>Kaggle — emmarex/plantdisease (PlantVillage)</td></tr>
                <tr><td>Classes</td><td>15 (Pepper, Potato, Tomato)</td></tr>
                <tr><td>Training images</td><td>16,516</td></tr>
                <tr><td>Validation images</td><td>4,122</td></tr>
                <tr><td>Train / validation split</td><td>80% / 20%</td></tr>
                <tr><td>Image size</td><td>224 × 224 px, rescaled 1/255</td></tr>
                <tr><td>Augmentation</td><td>Rotation 20°, H/V flip, zoom 0.2, shear 0.1</td></tr>
            </table>
        """, unsafe_allow_html=True)

        st.markdown("#### 🧠 Model Architecture")
        st.markdown("""
            <table class="spec-table">
                <tr><td>Backbone</td><td>ResNet50 (ImageNet weights, frozen)</td></tr>
                <tr><td>Head</td><td>GlobalAveragePooling → Dense(256, ReLU)</td></tr>
                <tr><td>Regularization</td><td>Dropout (rate = 0.4)</td></tr>
                <tr><td>Output</td><td>Dense(15, Softmax)</td></tr>
                <tr><td>Total parameters</td><td>24,116,111 (≈ 92.0 MB)</td></tr>
                <tr><td>Trainable parameters</td><td>528,399 (≈ 2.0 MB)</td></tr>
            </table>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown("#### ⚙️ Training Configuration")
        st.markdown("""
            <table class="spec-table">
                <tr><td>Framework</td><td>TensorFlow / Keras</td></tr>
                <tr><td>Optimizer</td><td>Adam</td></tr>
                <tr><td>Loss function</td><td>Categorical Crossentropy</td></tr>
                <tr><td>Epochs (configured)</td><td>15</td></tr>
                <tr><td>EarlyStopping</td><td>val_accuracy, patience 5</td></tr>
                <tr><td>ReduceLROnPlateau</td><td>val_loss, factor 0.3, patience 3</td></tr>
            </table>
        """, unsafe_allow_html=True)

        st.markdown("#### 📈 Reported Results")
        st.markdown("""
            <table class="spec-table">
                <tr><td>ResNet50 validation accuracy</td><td>41.99%</td></tr>
                <tr><td>Validation loss</td><td>1.7784</td></tr>
                <tr><td>Random Forest (CNN features)</td><td>49.53% acc / 49.52% F1</td></tr>
                <tr><td>SVM (CNN features)</td><td>49.95% acc / 49.95% F1</td></tr>
                <tr><td>Hybrid XGBoost</td><td>49.94% acc / 49.93% F1</td></tr>
            </table>
        """, unsafe_allow_html=True)
        st.caption(
            "Note: classical ML models were trained on 256-D CNN-extracted features "
            "for comparison purposes only — this app runs the end-to-end ResNet50 classifier."
        )

    st.markdown("---")
    st.markdown("#### 🔍 Explainability")
    st.write(
        "Grad-CAM (Gradient-weighted Class Activation Mapping) was applied to the final "
        "convolutional block (`conv5_block3_out`) during development to visualise which "
        "leaf regions drove each prediction, helping verify the model focuses on diseased "
        "tissue rather than background."
    )

# ---------------------------------------------------------
# TAB 5: ABOUT & TEAM
# ---------------------------------------------------------
with tab_team:
    st.write("### 👥 Project Information & Authors")
    st.info("""
        **Crop Disease Identification System**
        Developed as a Computer Vision & Deep Learning project utilizing Keras, ResNet50 Architecture, and Streamlit Web Engine.
    """)
    st.write("#### Development Team:")
    st.markdown("""
    * **Muhammad Anus Imran** — 231370129
    * **Ali Saqlain** — 231370119
    * **Hashir Ehsan** — 231370149
    * **Muhammad Khizar Hafeez** — 231370131
    * **Asad Ullah** — 231370151
    """)
    st.caption("GIFT University, Gujranwala — Department of Computer Science · August 2026")
