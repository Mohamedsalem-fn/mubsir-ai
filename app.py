# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Author: Mohamed Salem (Expert AI Engineer & Automation Architect)
# Focus: AI Engineering | Automation | Agentic Systems
# Copyright (c) 2026. All Rights Reserved.
# -------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import io

# -------------------------------------------------------------------------------
# App Configuration
# -------------------------------------------------------------------------------
st.set_page_config(
    page_title="مُبصر AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #7f8c8d;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------------
# Model Loading & Processing
# -------------------------------------------------------------------------------
MODEL_PATH = "model.tflite"
CLASSES = ['Drawings', 'Hentai', 'Neutral', 'Porn', 'Sexy']
NSFW_CLASSES = ['Hentai', 'Porn', 'Sexy']

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    
    # If image has alpha channel, convert to RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize if required (assuming 0-255 or 0-1, standard for these models is often /255.0)
    img_array /= 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    return predictions

def apply_blur(image, blur_radius):
    return image.filter(ImageFilter.GaussianBlur(blur_radius))

# -------------------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------------------
st.title("🛡️ مُبصر AI")
st.markdown("<p style='text-align: center; color: #34495e;'>The Intelligent Shield for Digital Family Protection.<br>A Comprehensive Multimodal AI Ecosystem for Proactive Threat Detection.</p>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings")
blur_radius = st.sidebar.slider("Blur Intensity", min_value=10, max_value=100, value=50, step=5)
threshold = st.sidebar.slider("Sensitivity Threshold (%)", min_value=1, max_value=100, value=70, step=1) / 100.0

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        with st.spinner("Analyzing content..."):
            interpreter = load_model()
            preds = predict(image, interpreter)
            
            # Anti-SFW Bias Correction
            safe_classes = ['Drawings', 'Neutral']
            for c in safe_classes:
                idx = CLASSES.index(c)
                preds[idx] *= 0.6
            preds = preds / np.sum(preds)
            
            # Find the max probability and its class
            max_idx = np.argmax(preds)
            max_class = CLASSES[max_idx]
            max_prob = preds[max_idx]
            
            # Check if it's NSFW and above threshold
            is_nsfw = False
            nsfw_score = sum([preds[CLASSES.index(c)] for c in NSFW_CLASSES])
            
            if nsfw_score >= threshold:
                is_nsfw = True

            with col2:
                st.subheader("Processed Result")
                if is_nsfw:
                    blurred_img = apply_blur(image, blur_radius)
                    st.image(blurred_img, use_container_width=True)
                    st.error(f"⚠️ **Inappropriate Content Detected!** (Confidence: {nsfw_score:.1%})")
                    st.warning("The image has been automatically blurred to protect viewers.")
                else:
                    st.image(image, use_container_width=True)
                    st.success(f"✅ **Content is Safe.** (Safe Score: {1-nsfw_score:.1%})")
            
            # Detailed Analysis Expander
            with st.expander("📊 Detailed AI Analysis"):
                st.markdown("### Class Probabilities")
                for i, c in enumerate(CLASSES):
                    st.progress(float(preds[i]), text=f"{c}: {preds[i]:.2%}")
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.markdown("""
    <div class="footer">
        Engineered with ❤️ by <b>Mohamed Salem</b> <br>
        <i>Expert AI Engineer & Automation Architect</i>
    </div>
""", unsafe_allow_html=True)
