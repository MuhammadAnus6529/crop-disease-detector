import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===========================================================
# 1. Page Configuration
# ===========================================================
st.set_page_config(
    page_title="AgriGuard | AI Crop Health Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================================================
# 2. Global CSS — full website styling
# ===========================================================
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background-color: #f6f9f7;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* ---------- Sticky Nav ---------- */
    .nav-bar {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(6px);
        border-radius: 14px;
        padding: 0.9rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    }
    .nav-brand { font-weight: 800; font-size: 1.2rem; color: #1b4332; }
    .nav-links a {
        color: #2d6a4f;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.92rem;
        margin-left: 1.4rem;
    }
    .nav-links a:hover { color: #081c15; }

    /* ---------- Hero ---------- */
    .hero-container {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 55%, #40916c 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(27, 67, 50, 0.2);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.92;
        max-width: 680px;
        margin: 0 auto 1.8rem auto;
        line-height: 1.6;
    }

    /* ---------- Section headers ---------- */
    .section-tag {
        color: #40916c;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .section-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #081c15;
        margin: 0.2rem 0 0.5rem 0;
    }
    .section-sub {
        color: #52796f;
        font-size: 1rem;
        max-width: 700px;
        margin-bottom: 1.6rem;
    }

    /* ---------- Cards ---------- */
    .feature-card, .crop-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 1.6rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        height: 100%;
    }
    .feature-card h4, .crop-card h4 { margin-top: 0; color: #1b4332; }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px; height: 34px;
        border-radius: 50%;
        background: #2d6a4f;
        color: white;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    /* ---------- Verdict banners ---------- */
    .verdict-healthy {
        background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
        border-left: 8px solid #2d6a4f;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
    }
    .verdict-disease {
        background: linear-gradient(135deg, #ffe5e5, #ffcccc);
        border-left: 8px solid #c1121f;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
    }
    .verdict-unknown {
        background: linear-gradient(135deg, #fff3cd, #ffe8a1);
        border-left: 8px solid #b08900;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
    }
    .verdict-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        opacity: 0.75;
    }
    .verdict-name {
        font-size: 1.7rem;
        font-weight: 800;
        color: #081c15;
        margin: 0.2rem 0 0.3rem 0;
    }

    /* ---------- Score rows ---------- */
    .score-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        margin: 0.55rem 0 0.15rem 0;
        color: #333;
    }
    .score-row b { color: #081c15; }

    /* Spec table (project details) */
    .spec-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    .spec-table td { padding: 0.45rem 0.6rem; border-bottom: 1px solid #e2e8f0; }
    .spec-table td:first-child { color: #52796f; font-weight: 600; width: 45%; }

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
    div.stButton > button:hover { background-color: #1b4332; color: white; }

    /* Footer */
    .site-footer {
        text-align: center;
        color: #6c8f80;
        font-size: 0.85rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================================================
# 3. Model & Class Setup
# ===========================================================
MODEL_PATH = 'crop_disease_resnet50.keras'

@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_keras_model()

# 15 classes — matches the PlantVillage subset (Pepper, Potato, Tomato) the
# ResNet50 model was actually trained on.
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

def format_label(raw_label):
    return raw_label.replace('___', ' — ').replace('_', ' ').strip()

# ===========================================================
# 4. Helper: Leaf validity check
# ===========================================================
def is_likely_leaf_image(pil_img):
    """
    Heuristic check: validates if the image contains enough green/brown
    vegetation pixel coverage to plausibly be a crop leaf photo, so the
    app can honestly say 'not recognized' instead of forcing a guess.
    """
    img_hsv = pil_img.convert('HSV')
    np_img = np.array(img_hsv)

    h_channel = np_img[:, :, 0]
    s_channel = np_img[:, :, 1]

    leaf_pixels = np.sum(((h_channel >= 20) & (h_channel <= 95)) & (s_channel > 30))
    total_pixels = h_channel.size
    leaf_ratio = leaf_pixels / total_pixels

    return leaf_ratio > 0.12

# ===========================================================
# 5. Sticky Navigation
# ===========================================================
st.markdown("""
    <div class="nav-bar">
        <div class="nav-brand">🌱 AgriGuard AI</div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#detector">Detector</a>
            <a href="#crops">Crops</a>
            <a href="#details">Project Details</a>
            <a href="#team">Team</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# ===========================================================
# SECTION: HERO
# ===========================================================
st.markdown('<div id="home"></div>', unsafe_allow_html=True)
st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">Precision Agriculture · Deep Learning</div>
        <div class="hero-title">Diagnose Crop Diseases Instantly with AI</div>
        <div class="hero-subtitle">
            AgriGuard uses a ResNet50 transfer-learning model trained on the PlantVillage
            dataset to identify healthy and diseased Pepper, Potato and Tomato leaves —
            in seconds, from a single photo.
        </div>
    </div>
""", unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
for col, (num, label) in zip(
    [s1, s2, s3, s4],
    [("15", "Disease Classes"), ("16,516", "Training Images"),
     ("4,122", "Validation Images"), ("3", "Crops Supported")]
):
    with col:
        st.markdown(f"""
            <div class="feature-card" style="text-align:center;">
                <div style="font-size:1.7rem;font-weight:800;color:#1b4332;">{num}</div>
                <div style="color:#52796f;font-size:0.85rem;">{label}</div>
            </div>
        """, unsafe_allow_html=True)

st.write("")
st.write("")

# ===========================================================
# SECTION: HOW IT WORKS
# ===========================================================
st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Process</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How AgriGuard Works</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Three simple steps take you from a leaf photo to a clear, actionable diagnosis.</div>', unsafe_allow_html=True)

h1, h2, h3 = st.columns(3)
with h1:
    st.markdown("""
        <div class="feature-card">
            <div class="step-num">1</div>
            <h4>Upload a Leaf Photo</h4>
            <p>Take or select a clear, well-lit photo of a single leaf from your Pepper, Potato, or Tomato crop.</p>
        </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown("""
        <div class="feature-card">
            <div class="step-num">2</div>
            <h4>AI Analyzes the Image</h4>
            <p>A ResNet50 deep learning model examines colour, texture and lesion patterns across the leaf surface.</p>
        </div>
    """, unsafe_allow_html=True)
with h3:
    st.markdown("""
        <div class="feature-card">
            <div class="step-num">3</div>
            <h4>Get a Full Diagnosis</h4>
            <p>Receive a healthy/diseased verdict, the specific condition, and a full confidence breakdown across all classes.</p>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ===========================================================
# SECTION: DETECTOR
# ===========================================================
st.markdown('<div id="detector"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Live Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔬 Run a Diagnostic Scan</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Upload a leaf image below. AgriGuard will tell you whether the crop is healthy, and if not, exactly what it\'s showing signs of — with confidence scores for every class.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    uploaded_file = st.file_uploader(
        "Select Leaf Image",
        type=["jpg", "jpeg", "png"],
        help="Supported Formats: JPG, JPEG, PNG"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Sample Preview", use_container_width=True)
        run_scan = st.button("Run Diagnostic Check")
    else:
        image = None
        run_scan = False
        st.info("👆 Upload an image to enable the scan.")

with col2:
    if uploaded_file is None:
        st.markdown("""
            <div class="feature-card" style="text-align:center; color:#52796f;">
                Your diagnosis — healthy/diseased verdict, confidence score,
                and full class breakdown — will appear here.
            </div>
        """, unsafe_allow_html=True)
    elif run_scan:
        if model is None:
            st.error(f"❌ Model file `{MODEL_PATH}` not found. Place it next to this app to enable predictions.")
        else:
            with st.spinner("Analyzing structural and chromatic leaf patterns..."):

                # --- Step A: Is this even a plausible leaf photo? ---
                if not is_likely_leaf_image(image):
                    st.markdown("""
                        <div class="verdict-unknown">
                            <div class="verdict-label">Not Recognized</div>
                            <div class="verdict-name">Image Not In Category</div>
                            <p style="margin:0;color:#5c4b00;">
                                This photo doesn't show enough leaf-like colouring to be analyzed.
                                AgriGuard only works on Pepper, Potato, and Tomato leaf close-ups —
                                please upload a clear, well-lit leaf photo.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # --- Step B: Run the model ---
                    img_resized = image.resize((224, 224))
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array)[0]
                    top_idx = int(np.argmax(predictions))
                    top_confidence = float(predictions[top_idx])

                    # --- Step C: Confidence gate — still refuse to guess if unsure ---
                    if top_confidence < 0.60:
                        st.markdown("""
                            <div class="verdict-unknown">
                                <div class="verdict-label">Uncertain</div>
                                <div class="verdict-name">Not In Category / Low Confidence</div>
                                <p style="margin:0;color:#5c4b00;">
                                    The model can't confidently match this leaf to any of its
                                    15 known classes. It will not guess a crop or disease name.
                                    Try a closer, sharper, better-lit photo of a single leaf.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"Top match score was only {top_confidence*100:.2f}% (need ≥ 60%).")
                    else:
                        raw_label = CLASS_NAMES[top_idx]
                        is_healthy = "healthy" in raw_label.lower()

                        # Overall healthy-vs-diseased probability, summed across
                        # ALL healthy classes vs ALL disease classes
                        healthy_mask = np.array(["healthy" in c.lower() for c in CLASS_NAMES])
                        healthy_score = float(predictions[healthy_mask].sum())
                        disease_score = 1.0 - healthy_score

                        # --- Main verdict banner ---
                        if is_healthy:
                            st.markdown(f"""
                                <div class="verdict-healthy">
                                    <div class="verdict-label">Diagnosis</div>
                                    <div class="verdict-name">✅ Healthy Leaf</div>
                                    <p style="margin:0;color:#1b4332;">{format_label(raw_label)} — no disease symptoms detected.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="verdict-disease">
                                    <div class="verdict-label">Diagnosis</div>
                                    <div class="verdict-name">⚠️ Disease Detected</div>
                                    <p style="margin:0;color:#7a0c14;">{format_label(raw_label)}</p>
                                </div>
                            """, unsafe_allow_html=True)

                        st.write("")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Top Match", format_label(raw_label).split(' — ')[-1], f"{top_confidence*100:.1f}% confidence")
                        m2.metric("Healthy Probability", f"{healthy_score*100:.1f}%")
                        m3.metric("Disease Probability", f"{disease_score*100:.1f}%")

                        if not is_healthy:
                            st.warning("⚠️ **Recommendation:** Isolate affected plants and consider appropriate fungicide/treatment for this condition.")
                        else:
                            st.success("✅ **Recommendation:** Maintain standard watering and nutrient schedule.")

                        st.write("")
                        st.markdown("**Full Confidence Breakdown (all 15 classes):**")
                        ranked = sorted(zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True)
                        for cls, prob in ranked:
                            st.markdown(f"""
                                <div class="score-row"><span>{format_label(cls)}</span><b>{prob*100:.2f}%</b></div>
                            """, unsafe_allow_html=True)
                            st.progress(min(float(prob), 1.0))

st.write("")
st.write("")

# ===========================================================
# SECTION: SUPPORTED CROPS
# ===========================================================
st.markdown('<div id="crops"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Coverage</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌾 Supported Crop Categories</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">The model is calibrated to classify the following 15 conditions across three crops.</div>', unsafe_allow_html=True)

crop_col1, crop_col2, crop_col3 = st.columns(3)
with crop_col1:
    st.markdown("""
        <div class="crop-card">
            <h4>🫑 Pepper (Bell)</h4>
            <p>Bacterial Spot<br>Healthy</p>
        </div>
    """, unsafe_allow_html=True)
with crop_col2:
    st.markdown("""
        <div class="crop-card">
            <h4>🥔 Potato</h4>
            <p>Early Blight<br>Late Blight<br>Healthy</p>
        </div>
    """, unsafe_allow_html=True)
with crop_col3:
    st.markdown("""
        <div class="crop-card">
            <h4>🍅 Tomato</h4>
            <p>Bacterial Spot · Early Blight · Late Blight<br>
            Leaf Mold · Septoria Leaf Spot · Spider Mites<br>
            Target Spot · Yellow Leaf Curl Virus · Mosaic Virus<br>
            Healthy</p>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ===========================================================
# SECTION: PROJECT DETAILS
# ===========================================================
st.markdown('<div id="details"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Under The Hood</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Project Details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Dataset, model and training pipeline summary from the project report.</div>', unsafe_allow_html=True)

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
            <tr><td>Random Forest (CNN features)</td><td>49.53% acc</td></tr>
            <tr><td>SVM (CNN features)</td><td>49.95% acc</td></tr>
            <tr><td>Hybrid XGBoost</td><td>49.94% acc</td></tr>
        </table>
    """, unsafe_allow_html=True)
    st.caption("This app runs the end-to-end ResNet50 classifier; the classical ML models were a comparison experiment only.")

st.write("")
st.write("")

# ===========================================================
# SECTION: TEAM
# ===========================================================
st.markdown('<div id="team"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Credits</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">👥 Development Team</div>', unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.columns(5)
team = [
    ("Muhammad Anus Imran", "231370129"),
    ("Ali Saqlain", "231370119"),
    ("Hashir Ehsan", "231370149"),
    ("Muhammad Khizar Hafeez", "231370131"),
    ("Asad Ullah", "231370151"),
]
for col, (name, reg) in zip([t1, t2, t3, t4, t5], team):
    with col:
        st.markdown(f"""
            <div class="feature-card" style="text-align:center;">
                <div style="font-weight:700;color:#1b4332;">{name}</div>
                <div style="color:#52796f;font-size:0.85rem;">{reg}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <div class="site-footer">
        🌱 AgriGuard AI — Crop Disease Identification System<br>
        GIFT University, Gujranwala · Department of Computer Science · August 2026
    </div>
""", unsafe_allow_html=True)
