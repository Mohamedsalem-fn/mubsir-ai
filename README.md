# مُبصر AI — Intelligent NSFW Content Detection

<div align="center">

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />

**نظام الكشف الذكي عن المحتوى غير الآمن باستخدام الذكاء الاصطناعي**

*Intelligent AI-powered NSFW content detection with face analysis and auto-blur*

</div>

---

## 📋 نظرة عامة | Overview

**مُبصر AI** is a production-ready Streamlit application that combines three AI models to accurately detect unsafe/NSFW content in images and automatically apply blurring:

| Model | Purpose | Format |
|-------|---------|--------|
| `nsfw.tflite` | 5-class NSFW image classifier | TFLite (22 MB) |
| `blazeface.tflite` | Real-time face detector | TFLite (224 KB) |
| `faceres.json/.bin` | Gender & age predictor (MobileNet) | TF.js Graph Model (6.8 MB) |

### Detection Pipeline

```
Image Upload
    │
    ▼
┌─────────────────────┐
│  NSFW TFLite Model  │ → 5-class probabilities (Drawing, Hentai, Neutral, Porn, Sexy)
│  224×224 input      │ + Anti-SFW Bias Correction (SFW × 0.6)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  BlazeFace TFLite   │ → Face bounding boxes + confidence scores
│  128×128 input      │ (OpenCV fallback if model incompatible)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  FaceRes TF.js      │ → Gender (female/male) + estimated age per face
│  224×224 crops      │ (MobileNet-based, vladmandic format)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Decision Engine    │ Rule 1: Female face + NSFW > 2% → UNSAFE
│                     │ Rule 2: NSFW total > SFW total → UNSAFE
│                     │ Rule 3: Strict mode + top NSFW > 15% → UNSAFE
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Auto Blur          │ Full image blur / Face-only blur / No blur
│  (configurable)     │ Gaussian kernel 61×61, σ=20
└─────────────────────┘
```

---

## 🚀 التثبيت والتشغيل | Installation & Running

### Prerequisites

- Python 3.10+
- pip / venv

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/MohamedSalem/mubsir-ai.git
cd mubsir-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### Model Files

Place the following model files in the **same directory** as `app.py`:

```
v1/
├── app.py
├── requirements.txt
├── nsfw.tflite        ← 5-class NSFW classifier
├── blazeface.tflite   ← Face detector
├── faceres.json       ← TF.js model topology
└── faceres.bin        ← TF.js model weights
```

---

## ☁️ النشر على Streamlit Cloud | Deploying to Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: initial Mubsir AI deployment"
   git remote add origin https://github.com/<your-username>/mubsir-ai.git
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"** → Select your repository

4. **Configure**:
   - Repository: `<your-username>/mubsir-ai`
   - Branch: `main`
   - Main file: `app.py` (or `v1/app.py` if in subfolder)

5. **Deploy!** Streamlit Cloud will automatically install `requirements.txt`

> **Note**: The model files (`.tflite`, `.json`, `.bin`) must be committed to the repository or loaded from an external URL (e.g., Hugging Face Hub) for cloud deployment.

---

## 🎛️ الميزات | Features

- 🧠 **5-Class NSFW Classification** — Drawing, Hentai, Neutral, Porn, Sexy
- 👤 **Real-time Face Detection** — BlazeFace + OpenCV fallback
- 🧬 **Gender & Age Analysis** — Per-face predictions via FaceRes
- 🔵 **Anti-SFW Bias Correction** — Mirrors the Mubsir Chrome extension logic
- 🎚️ **Adjustable Sensitivity** — From lenient to ultra-strict
- 🌫️ **Smart Blur Modes** — Full image blur / Face-only blur / No blur
- 📊 **Visual Probability Bars** — Animated class breakdown
- ⬇️ **Download Processed Image** — Export blurred result
- 🌙 **Dark Mode UI** — Premium glassmorphism design
- 🌐 **Arabic/English Bilingual** — RTL-aware interface

---

## 🏗️ المعمارية | Architecture

```
v1/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Theme & server configuration
├── nsfw.tflite             ← NSFW TFLite model
├── blazeface.tflite        ← BlazeFace TFLite model
├── faceres.json            ← FaceRes topology (TF.js graph model)
└── faceres.bin             ← FaceRes weights binary shard
```

### Key Design Decisions

- **`@st.cache_resource`** — Models are loaded once and cached across sessions
- **Graceful fallback** — BlazeFace → OpenCV Haar cascade if TFLite output format differs
- **FaceRes loading** — Uses `tensorflowjs` Python library to parse the vladmandic graph model format
- **Anti-SFW bias** — SFW class probabilities are multiplied by 0.60 before normalization (matches the Chrome extension behavior)
- **Decision rules** — Three-tier logic: female+NSFW signal → NSFW dominant → strict mode threshold

---

## 📦 المتطلبات | Dependencies

```
streamlit>=1.35.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python-headless>=4.8.0
tensorflow-cpu>=2.15.0
tensorflowjs>=4.18.0
```

---

## ⚠️ ملاحظات النشر | Deployment Notes

- **Model size**: The `.tflite` and `.bin` files total ~31 MB — within GitHub's 100 MB file limit
- **Memory**: Streamlit Cloud free tier has 1 GB RAM — sufficient for CPU inference
- **Cold start**: First inference may take 10–30 seconds while models warm up
- **TF.js model**: If `tensorflowjs` fails to load `faceres`, the app gracefully falls back to face detection without gender/age prediction

---

## 👤 المؤلف | Author

**Mohamed Salem**
Expert AI Engineer & Automation Architect
Focus: AI Engineering | Automation | LLMs | Agentic Systems | MLOps
Experience: 4+ years development · 1+ year AI specialization

---

## 📄 الترخيص | License

Copyright (c) 2026 Mohamed Salem. All Rights Reserved.
