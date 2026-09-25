# مُبصر AI — Intelligent Multimodal Parental Control System

<div align="center">

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/LLM-Integrated-8A2BE2?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />

**نظام الكشف الذكي متعدد الوسائط لحماية الأسرة الرقمية**

*Intelligent Multimodal AI (Computer Vision + LLMs) for comprehensive child protection against NSFW content and predatory behavior.*

</div>

---

## 📋 نظرة عامة | Overview

**مُبصر AI (Mubsir AI)** is not just an image classifier; it is a comprehensive, production-ready AI parental control ecosystem. It utilizes a **Multimodal AI Architecture** to protect children from both visual and textual digital threats across all applications.

The system integrates an ensemble of Computer Vision models for real-time NSFW image detection and leverages Large Language Models (LLMs) to analyze textual context, effectively blocking harmful content and detecting predatory behavior (grooming) before it escalates.

### Core Technologies

| Domain | Technology / Model | Purpose | Format |
|--------|-------------------|---------|--------|
| **Vision** | `nsfw.tflite` | 5-class NSFW image classifier | TFLite (22 MB) |
| **Vision** | `blazeface.tflite` | Real-time face detector | TFLite (224 KB) |
| **Vision** | `faceres.json/.bin` | Gender & age predictor (MobileNet) | TF.js Graph Model (6.8 MB) |
| **NLP** | Large Language Models (LLMs) | Contextual text & behavior analysis | API / Edge Integration |

---

## 🧠 معمارية الذكاء الاصطناعي | Multimodal AI Pipeline

The system operates on a dual-engine architecture to ensure comprehensive protection:

### 1. Computer Vision Pipeline (Image Analysis)

```
Image Upload / Screen Capture
    │
    ▼
┌─────────────────────┐
│  NSFW TFLite Model  │ → 5-class probabilities (Drawing, Hentai, Neutral, Porn, Sexy)
│  224×224 input      │ + Anti-SFW Bias Correction (SFW × 0.6)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  BlazeFace TFLite   │ → Face bounding boxes + confidence scores
│  128×128 input      │ (OpenCV fallback if model incompatible)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  FaceRes TF.js      │ → Gender (female/male) + estimated age per face
│  224×224 crops      │ (MobileNet-based, vladmandic format)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Decision Engine    │ Rule 1: Female face + NSFW > 2% → UNSAFE
│  (Rules Fusion)     │ Rule 2: NSFW total > SFW total → UNSAFE
│                     │ Rule 3: Strict mode + top NSFW > 1% → UNSAFE
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Auto Blur          │ Full image blur / Face-only blur / No blur
│  (configurable)     │ Gaussian kernel 61×61, σ=20
└─────────────────────┘
```

### 2. LLM NLP Engine (Text & Behavior Analysis)

To prevent threats that bypass visual filters, Mubsir AI integrates LLMs to analyze textual context across chats and browsers:

*   **Contextual Understanding:** Moves beyond static keyword blacklists to understand the intent behind messages.
*   **Grooming & Cyberbullying Detection:** Identifies predatory behavioral patterns, manipulation tactics, and cyberbullying.
*   **Cross-App Monitoring:** Provides continuous behavioral analysis across different digital environments.

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
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### Model Files Structure
Place the following model files in the **same directory** as `app.py`:

```
v1/
├── app.py
├── requirements.txt
├── nsfw.tflite        ← 5-class NSFW classifier (Custom trained on 130k images)
├── blazeface.tflite   ← Face detector
├── faceres.json       ← TF.js model topology
└── faceres.bin        ← TF.js model weights
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
   - Main file: `app.py`
5. **Deploy!** Streamlit Cloud handles the environment setup.

> **Note**: Model files must be committed to the repository or fetched dynamically.

---

## 🎛️ الميزات | Features

- 🧠 **5-Class Visual Classifier** — Distinguishes between Drawing, Hentai, Neutral, Porn, and Sexy.
- 🗣️ **LLM Behavioral Analysis** — Detects grooming, cyberbullying, and inappropriate text context.
- 👤 **Real-time Face & Demographic Analysis** — Predicts gender and age to inform the Decision Engine.
- 🔵 **Anti-SFW Bias Correction** — Algorithmic penalty (SFW × 0.6) to prioritize child safety.
- 🎚️ **Adjustable Sensitivity** — Customizable parental controls (Lenient to Ultra-strict).
- 🌫️ **Smart Blur Technology** — Context-aware blurring (Full image or Face-only).
- 📊 **Visual Probability Bars** — Real-time animated classification breakdown.
- 🌙 **Dark Mode & Bilingual UI** — Premium Arabic/English interface.

---

## 🏗️ التصميم المعماري | Key Design Decisions

- **Multimodal Approach:** Combines CV for immediate visual threats and LLMs for insidious textual threats (grooming).
- **Ensemble Vision Processing:** Relying on three distinct vision models reduces false positives significantly compared to a single monolithic model.
- **Edge-Ready (Privacy First):** The use of lightweight `.tflite` models ensures that visual processing can be handled locally on the device, maximizing user privacy and minimizing cloud infrastructure costs.
- **Deterministic Decision Engine:** Translates complex AI probabilities into actionable, interpretable rules (e.g., Strict Mode: NSFW > 1% = UNSAFE).

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

## 👤 المؤلف | Author

**Mohamed Salem**
Expert AI Engineer & Automation Architect
Focus: AI Engineering | Automation | LLMs | Agentic Systems | MLOps
Experience: 2+ years development · 1+ year AI specialization

---

## 📄 الترخيص | License

Copyright (c) 2026 Mohamed Salem. All Rights Reserved.