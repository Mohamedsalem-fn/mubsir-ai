# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Author: Mohamed Salem (Expert AI Engineer & Automation Architect)
# Focus: AI Engineering | Automation | Agentic Systems
# Copyright (c) 2026. All Rights Reserved.
# -------------------------------------------------------------------------------
"""
مُبصر AI — نظام الكشف الذكي عن المحتوى غير الآمن
Mubsir AI — Intelligent NSFW Content Detection System

Models are downloaded at runtime from:
  https://frank0mm0m.serv00.net/models/

  - nsfw.tflite      : 5-class NSFW image classifier   (224×224)
  - blazeface.tflite : MediaPipe BlazeFace face detector (128×128)
  - faceres.json     : TF.js MobileNet graph model topology
  - faceres.bin      : TF.js MobileNet model weights shard
"""

import io
import os
import time
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ── Silence TF noise ─────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf  # noqa: E402

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="مُبصر AI — كاشف المحتوى غير الآمن",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/MohamedSalem",
        "About": "مُبصر AI v2.0 · Built by Mohamed Salem · Expert AI Engineer",
    },
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_BASE_URL = "https://frank0mm0m.serv00.net/models"

# Cache directory — /tmp is writable on Streamlit Cloud
CACHE_DIR = Path("/tmp/mubsir_models")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "nsfw.tflite":      {"url": f"{MODELS_BASE_URL}/nsfw.tflite",      "size_mb": 18.0},
    "blazeface.tflite": {"url": f"{MODELS_BASE_URL}/blazeface.tflite", "size_mb": 0.22},
    "faceres.json":     {"url": f"{MODELS_BASE_URL}/faceres.json",     "size_mb": 0.07},
    "faceres.bin":      {"url": f"{MODELS_BASE_URL}/faceres.bin",      "size_mb": 6.7},
}

NSFW_MODEL_PATH   = CACHE_DIR / "nsfw.tflite"
BLAZE_MODEL_PATH  = CACHE_DIR / "blazeface.tflite"
FACERES_JSON_PATH = CACHE_DIR / "faceres.json"
FACERES_BIN_PATH  = CACHE_DIR / "faceres.bin"

NSFW_CLASSES = {
    0: {"name": "Drawing", "ar": "رسم",   "nsfw": False, "color": "#4CAF50"},
    1: {"name": "Hentai",  "ar": "هنتاي", "nsfw": True,  "color": "#FF5252"},
    2: {"name": "Neutral", "ar": "محايد", "nsfw": False, "color": "#4CAF50"},
    3: {"name": "Porn",    "ar": "إباحي", "nsfw": True,  "color": "#D50000"},
    4: {"name": "Sexy",    "ar": "مثير",  "nsfw": True,  "color": "#FF6D00"},
}

SFW_DISCOUNT  = 0.60
BLUR_KERNEL   = 61
BLUR_SIGMA    = 20
FACE_NSFW_THR = 0.02
FACERES_INPUT = 224

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary:    #6C63FF;
    --success:    #00C896;
    --danger:     #FF3860;
    --bg-dark:    #0D0E1A;
    --bg-card:    #161728;
    --border:     rgba(108,99,255,0.25);
    --text-main:  #E8E9FF;
    --text-muted: #8B8FA8;
    --glow:       0 0 30px rgba(108,99,255,0.3);
}
* { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
    background: var(--bg-dark);
    color: var(--text-main);
    font-family: 'Inter', 'Cairo', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0D0E1A 0%,#13142A 100%);
    border-right: 1px solid var(--border);
}

/* ── Hero ── */
.mubsir-hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg,#0D0E1A 0%,#1A1B35 50%,#0D0E1A 100%);
    border-radius: 20px;
    border: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.mubsir-hero::before {
    content:'';
    position:absolute; top:-50%; left:-50%;
    width:200%; height:200%;
    background: radial-gradient(ellipse at center,rgba(108,99,255,0.08) 0%,transparent 70%);
    pointer-events:none;
}
.mubsir-logo  { font-size:4.5rem; margin-bottom:.25rem; }
.mubsir-title {
    font-family:'Cairo',sans-serif; font-size:2.8rem; font-weight:900;
    background:linear-gradient(135deg,#6C63FF 0%,#A8A4FF 50%,#FF6B6B 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin:.25rem 0;
}
.mubsir-subtitle { font-family:'Cairo',sans-serif; color:var(--text-muted); font-size:1.05rem; margin:0; }
.stat-row { display:flex; gap:1rem; justify-content:center; margin-top:1.5rem; flex-wrap:wrap; }
.stat-badge {
    background:rgba(108,99,255,0.12); border:1px solid var(--border);
    border-radius:50px; padding:.4rem 1.2rem; font-size:.8rem; color:#A8A4FF;
    font-weight:500; letter-spacing:.04em;
}

/* ── Download progress ── */
.dl-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 2rem auto;
    max-width: 640px;
    text-align: center;
}
.dl-title { font-family:'Cairo',sans-serif; font-size:1.2rem; font-weight:700; color:#A8A4FF; margin-bottom:1rem; }
.dl-file  { font-size:.85rem; color:var(--text-muted); margin:.2rem 0; }

/* ── Result cards ── */
.result-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:16px; padding:1.5rem; margin-bottom:1rem; box-shadow:var(--glow);
}
.result-safe   { border-color:var(--success); box-shadow:0 0 20px rgba(0,200,150,.15); }
.result-unsafe { border-color:var(--danger);  box-shadow:0 0 20px rgba(255,56,96,.2);  }

/* ── NSFW bars ── */
.class-bar-wrapper { margin-bottom:.75rem; }
.class-bar-label { display:flex; justify-content:space-between; align-items:center; margin-bottom:.3rem; font-size:.9rem; }
.class-bar-track { height:8px; background:rgba(255,255,255,.08); border-radius:99px; overflow:hidden; }
.class-bar-fill  { height:100%; border-radius:99px; transition:width .6s ease; }

/* ── Verdict ── */
.verdict-banner {
    border-radius:12px; padding:1rem 1.5rem; text-align:center;
    font-family:'Cairo',sans-serif; font-size:1.3rem; font-weight:700; margin:1rem 0;
}
.verdict-safe   { background:rgba(0,200,150,.15); color:#00C896; border:1px solid rgba(0,200,150,.3); }
.verdict-unsafe { background:rgba(255,56,96,.15);  color:#FF3860; border:1px solid rgba(255,56,96,.3); }

/* ── Face chips ── */
.face-chip {
    display:inline-flex; align-items:center; gap:.4rem;
    background:rgba(108,99,255,.15); border:1px solid var(--border);
    border-radius:50px; padding:.3rem .9rem; margin:.25rem; font-size:.85rem;
}
.section-title {
    font-family:'Cairo',sans-serif; font-size:1.15rem; font-weight:700;
    color:var(--primary); margin-bottom:1rem; display:flex; align-items:center; gap:.5rem;
}
.sidebar-section {
    background:rgba(108,99,255,.07); border:1px solid var(--border);
    border-radius:12px; padding:1rem; margin-bottom:1rem;
}

/* ── Pulse animation ── */
@keyframes pulse-red {
    0%,100%{ box-shadow:0 0 0 0 rgba(255,56,96,.4); }
    50%    { box-shadow:0 0 0 10px rgba(255,56,96,0); }
}
.pulse-unsafe { animation:pulse-red 2s infinite; }

/* ── Misc ── */
#MainMenu,footer,header { visibility:hidden; }
[data-testid="metric-container"] {
    background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:1rem;
}
.stButton>button {
    background:linear-gradient(135deg,#6C63FF,#9B59B6); border:none; border-radius:10px;
    color:#fff; font-weight:600; padding:.6rem 1.5rem; transition:transform .2s,box-shadow .2s;
}
.stButton>button:hover { transform:translateY(-2px); box-shadow:0 8px 25px rgba(108,99,255,.4); }
.stFileUploader>div { background:var(--bg-card); border:2px dashed var(--border); border-radius:16px; }
</style>
"""

# ── Runtime model downloader ──────────────────────────────────────────────────

class _ProgressBar:
    """urllib progress hook that writes into a Streamlit progress widget."""
    def __init__(self, progress_widget, text_widget, filename: str, total: float):
        self._bar    = progress_widget
        self._text   = text_widget
        self._fname  = filename
        self._total  = total  # MB
        self._got    = 0.0

    def __call__(self, block_count: int, block_size: int, total_size: int):
        self._got = block_count * block_size / 1_048_576  # MB
        total     = max(self._total, self._got)
        pct       = min(self._got / total, 1.0)
        self._bar.progress(pct)
        self._text.markdown(
            f'<div class="dl-file">⬇️ {self._fname} — '
            f'{self._got:.1f} / {total:.1f} MB ({pct*100:.0f}%)</div>',
            unsafe_allow_html=True,
        )


def _download_model(name: str, dest: Path, progress_w, text_w) -> bool:
    """Download a single model file with live progress feedback."""
    url      = MODELS[name]["url"]
    size_mb  = MODELS[name]["size_mb"]
    hook     = _ProgressBar(progress_w, text_w, name, size_mb)
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=hook)
        progress_w.progress(1.0)
        return True
    except Exception as exc:
        text_w.error(f"❌ فشل تحميل {name}: {exc}")
        return False


@st.cache_resource(show_spinner=False)
def ensure_models() -> bool:
    """
    Download all required models to CACHE_DIR if not already present.
    Shows a polished download UI while doing so.
    Returns True when all models are ready.
    """
    needed = {
        name: info
        for name, info in MODELS.items()
        if not (CACHE_DIR / name).exists()
    }
    if not needed:
        return True  # everything already cached

    # --- Build download UI ---
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="dl-card">'
        '<div class="dl-title">🔄 جاري تحميل نماذج الذكاء الاصطناعي…</div>'
        '<p style="color:var(--text-muted);font-size:.85rem">يتم تحميل النماذج مرة واحدة فقط ثم تُخزَّن مؤقتاً.</p>',
        unsafe_allow_html=True,
    )

    all_ok = True
    for i, (name, _) in enumerate(needed.items()):
        dest     = CACHE_DIR / name
        prog_w   = st.progress(0.0, text="")
        text_w   = st.empty()
        ok       = _download_model(name, dest, prog_w, text_w)
        if ok:
            text_w.markdown(
                f'<div class="dl-file">✅ {name} — اكتمل التحميل</div>',
                unsafe_allow_html=True,
            )
        else:
            all_ok = False
            if dest.exists():
                dest.unlink()

    st.markdown("</div>", unsafe_allow_html=True)

    if all_ok:
        st.success("✅ تم تحميل جميع النماذج — جاري الإطلاق…")
        time.sleep(0.8)
        st.rerun()
    else:
        st.error("❌ فشل تحميل بعض النماذج. حاول إعادة تحميل الصفحة.")
        st.stop()

    return all_ok


# ── Model loaders (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nsfw_model():
    interp = tf.lite.Interpreter(model_path=str(NSFW_MODEL_PATH))
    interp.allocate_tensors()
    return interp


@st.cache_resource(show_spinner=False)
def load_blazeface_model():
    interp = tf.lite.Interpreter(model_path=str(BLAZE_MODEL_PATH))
    interp.allocate_tensors()
    return interp


@st.cache_resource(show_spinner=False)
def load_faceres_model():
    """
    Load FaceRes TF.js graph model (vladmandic format: graph-model).
    Uses tensorflowjs Python library. Gracefully returns None on failure.
    """
    try:
        import tensorflowjs as tfjs
        model = tfjs.converters.load_keras_model(str(FACERES_JSON_PATH))
        return model
    except Exception:
        return None


# ── Preprocessing helpers ─────────────────────────────────────────────────────
def preprocess_tflite(image: Image.Image, size: int, dtype) -> np.ndarray:
    img = image.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if dtype in (np.int8, np.uint8):
        return arr.astype(dtype)
    return arr / 255.0


def dequantize(output: np.ndarray, details: dict) -> np.ndarray:
    if details["dtype"] in (np.int8, np.uint8):
        scale, zp = details["quantization"]
        if scale > 0:
            output = (output.astype(np.float32) - zp) * scale
    return output.astype(np.float32)


# ── NSFW Detection ────────────────────────────────────────────────────────────
def run_nsfw(image: Image.Image) -> list[dict]:
    interp      = load_nsfw_model()
    in_det      = interp.get_input_details()
    out_det     = interp.get_output_details()
    size        = in_det[0]["shape"][1]
    dtype       = in_det[0]["dtype"]

    tensor = preprocess_tflite(image, size, dtype)
    interp.set_tensor(in_det[0]["index"], tensor)
    interp.invoke()
    out = interp.get_tensor(out_det[0]["index"])[0]
    out = dequantize(out, out_det[0])

    # Anti-SFW bias (mirrors Chrome extension logic)
    for i in range(len(out)):
        if not NSFW_CLASSES[i]["nsfw"]:
            out[i] *= SFW_DISCOUNT
    total = out.sum()
    if total > 0:
        out /= total

    results = []
    for i, prob in enumerate(out):
        results.append({
            "id": i, "name": NSFW_CLASSES[i]["name"],
            "ar": NSFW_CLASSES[i]["ar"], "nsfw": NSFW_CLASSES[i]["nsfw"],
            "color": NSFW_CLASSES[i]["color"],
            "prob": float(prob), "prob_pct": float(prob) * 100,
        })
    return sorted(results, key=lambda x: x["prob"], reverse=True)


# ── BlazeFace Detection ───────────────────────────────────────────────────────
def run_blazeface(image: Image.Image) -> list[dict]:
    interp  = load_blazeface_model()
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()
    size    = in_det[0]["shape"][1]
    dtype   = in_det[0]["dtype"]

    tensor = preprocess_tflite(image, size, dtype)
    interp.set_tensor(in_det[0]["index"], tensor)
    interp.invoke()

    try:
        boxes_idx = scores_idx = None
        for det in out_det:
            sh = det["shape"]
            if len(sh) >= 2 and sh[-1] == 4:
                boxes_idx = det["index"]
            elif len(sh) >= 2 and sh[-1] == 1:
                scores_idx = det["index"]

        if boxes_idx is None:
            return _cv_faces(image)

        raw_boxes  = interp.get_tensor(boxes_idx)
        raw_boxes  = raw_boxes[0] if raw_boxes.ndim == 3 else raw_boxes
        if scores_idx:
            raw_sc = interp.get_tensor(scores_idx)
            raw_sc = raw_sc[0] if raw_sc.ndim == 3 else raw_sc
            scores = raw_sc[:, 0] if raw_sc.ndim == 2 else raw_sc
            scores = 1.0 / (1.0 + np.exp(-scores))
        else:
            scores = np.ones(len(raw_boxes))

        faces = []
        for box, score in zip(raw_boxes, scores):
            if float(score) > 0.5:
                b = box.copy()
                if b.max() > 1.5:
                    b /= float(size)
                faces.append({"box": b.tolist(), "score": float(score), "gender": None, "age": None})
        return faces if faces else _cv_faces(image)

    except Exception:
        return _cv_faces(image)


def _cv_faces(image: Image.Image) -> list[dict]:
    """OpenCV Haar-cascade fallback face detector."""
    bgr     = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rects   = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    h, w    = bgr.shape[:2]
    return [
        {"box": [y/h, x/w, (y+fh)/h, (x+fw)/w], "score": 0.9, "gender": None, "age": None}
        for (x, y, fw, fh) in rects
    ]


# ── FaceRes Gender/Age ────────────────────────────────────────────────────────
def run_faceres(image: Image.Image, faces: list[dict]) -> list[dict]:
    model = load_faceres_model()
    if model is None or not faces:
        return faces

    h, w    = image.size[1], image.size[0]
    img_arr = np.array(image.convert("RGB"))
    enriched = []

    for face in faces:
        try:
            ymin, xmin, ymax, xmax = face["box"]
            y1, x1 = int(ymin * h), int(xmin * w)
            y2, x2 = int(ymax * h), int(xmax * w)
            margin = int(max(y2 - y1, x2 - x1) * 0.2)
            y1 = max(0, y1 - margin); x1 = max(0, x1 - margin)
            y2 = min(h, y2 + margin); x2 = min(w, x2 + margin)
            crop = img_arr[y1:y2, x1:x2]
            if crop.size == 0:
                enriched.append(face)
                continue

            face_t = np.expand_dims(
                np.array(Image.fromarray(crop).resize((FACERES_INPUT, FACERES_INPUT), Image.LANCZOS),
                         dtype=np.float32) / 255.0, axis=0)
            outputs = model(tf.constant(face_t), training=False)

            # Parse outputs by shape (gender=1, embed=1024, age=100)
            gender_sigmoid, age_softmax = 0.5, None
            outs = outputs if isinstance(outputs, (list, tuple)) else outputs.values()
            for o in outs:
                arr = np.array(o)
                if arr.shape[-1] == 1:
                    gender_sigmoid = float(arr.flat[0])
                elif arr.shape[-1] == 100:
                    age_softmax = arr[0]

            gender = "female" if gender_sigmoid > 0.5 else "male"
            g_score = gender_sigmoid if gender == "female" else (1 - gender_sigmoid)
            est_age = int(np.dot(np.arange(100), age_softmax)) if age_softmax is not None else None

            enriched.append({**face, "gender": gender, "gender_score": float(g_score), "age": est_age})
        except Exception:
            enriched.append(face)

    return enriched


# ── Blur Helpers ──────────────────────────────────────────────────────────────
def _cv_blur(image: Image.Image, k: int, sigma: float) -> Image.Image:
    kk  = k if k % 2 == 1 else k + 1
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return Image.fromarray(cv2.cvtColor(cv2.GaussianBlur(bgr, (kk, kk), sigma), cv2.COLOR_BGR2RGB))


def apply_full_blur(image: Image.Image) -> Image.Image:
    return _cv_blur(image, BLUR_KERNEL, BLUR_SIGMA)


def apply_face_blur(image: Image.Image, faces: list[dict]) -> Image.Image:
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    kk = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
    for face in faces:
        ymin, xmin, ymax, xmax = face["box"]
        y1, x1 = int(ymin * h), int(xmin * w)
        y2, x2 = int(ymax * h), int(xmax * w)
        mg = int(max(y2 - y1, x2 - x1) * 0.15)
        y1, x1 = max(0, y1 - mg), max(0, x1 - mg)
        y2, x2 = min(h, y2 + mg), min(w, x2 + mg)
        if y2 > y1 and x2 > x1:
            bgr[y1:y2, x1:x2] = cv2.GaussianBlur(bgr[y1:y2, x1:x2], (kk, kk), BLUR_SIGMA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def annotate_faces(image: Image.Image, faces: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    h, w = image.size[1], image.size[0]
    for face in faces:
        ymin, xmin, ymax, xmax = face["box"]
        y1, x1 = int(ymin * h), int(xmin * w)
        y2, x2 = int(ymax * h), int(xmax * w)
        color  = (255, 100, 100) if face.get("gender") == "female" else (100, 180, 255)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        if face.get("gender"):
            g_ar  = "أنثى" if face["gender"] == "female" else "ذكر"
            label = f"{g_ar} ({int(face.get('gender_score',0)*100)}%)"
            if face.get("age"):
                label += f" عمر≈{face['age']}"
            draw.text((x1 + 4, y1 + 4), label, fill=color)
    return image


# ── Decision Engine ───────────────────────────────────────────────────────────
def make_decision(nsfw: list[dict], faces: list[dict], sensitivity: float) -> dict:
    nsfw_total = sum(r["prob"] for r in nsfw if r["nsfw"])
    sfw_total  = sum(r["prob"] for r in nsfw if not r["nsfw"])
    top        = nsfw[0] if nsfw else {}
    has_female = any(f.get("gender") == "female" and f.get("gender_score", 0) > 0.25 for f in faces)

    reasons, is_unsafe = [], False

    female_thr = FACE_NSFW_THR * (1 - sensitivity * 0.5)
    if has_female and nsfw_total > female_thr:
        is_unsafe = True
        reasons.append(f"👩 وجود أنثى مع نسبة NSFW ({nsfw_total*100:.1f}%) > {female_thr*100:.1f}%")

    if not is_unsafe and nsfw_total > sfw_total:
        is_unsafe = True
        reasons.append(f"⚠️ نسبة NSFW ({nsfw_total*100:.1f}%) أعلى من SFW ({sfw_total*100:.1f}%)")

    if not is_unsafe and sensitivity > 0.7 and top.get("nsfw") and top.get("prob", 0) > 0.15:
        is_unsafe = True
        reasons.append(f"🔴 الوضع الصارم: '{top['ar']}' بنسبة {top['prob_pct']:.1f}%")

    return {
        "is_unsafe": is_unsafe, "nsfw_total": nsfw_total, "sfw_total": sfw_total,
        "has_female": has_female, "face_count": len(faces), "reasons": reasons, "top": top,
    }


# ── UI Helpers ────────────────────────────────────────────────────────────────
def render_nsfw_bars(results: list[dict]):
    for r in sorted(results, key=lambda x: x["prob"], reverse=True):
        pct   = r["prob_pct"]
        color = r["color"]
        label = f"{'🔴' if r['nsfw'] else '🟢'} {r['name']} ({r['ar']})"
        st.markdown(
            f'<div class="class-bar-wrapper">'
            f'<div class="class-bar-label"><span>{label}</span>'
            f'<span style="color:{color};font-weight:700">{pct:.2f}%</span></div>'
            f'<div class="class-bar-track">'
            f'<div class="class-bar-fill" style="width:{pct:.2f}%;background:{color}"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def render_face_chips(faces: list[dict]):
    if not faces:
        st.info("لم يتم اكتشاف وجوه في الصورة.")
        return
    html = ""
    for i, face in enumerate(faces, 1):
        gender = face.get("gender", "")
        g_ar   = "أنثى" if gender == "female" else ("ذكر" if gender == "male" else "غير معروف")
        g_icon = "👩" if gender == "female" else ("👨" if gender == "male" else "❓")
        g_conf = f" ({int(face.get('gender_score',0)*100)}%)" if face.get("gender_score") else ""
        age    = f" · عمر≈{face['age']}" if face.get("age") else ""
        conf   = int(face.get("score", 1) * 100)
        html  += f'<span class="face-chip">{g_icon} وجه {i}: {g_ar}{g_conf}{age} · ثقة {conf}%</span>'
    st.markdown(html, unsafe_allow_html=True)


def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1rem 0 .5rem">'
            '<span style="font-size:2.5rem">🛡️</span><br>'
            '<b style="font-size:1.2rem;color:#A8A4FF">مُبصر AI</b><br>'
            '<span style="font-size:.75rem;color:#8B8FA8">v2.0 · NTI Edition</span>'
            '</div>', unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**⚙️ إعدادات الكشف**")
        sensitivity = st.slider("مستوى الحساسية", 0.0, 1.0, 0.5, 0.05,
                                help="0 = متساهل · 1 = صارم جداً", key="sens")
        blur_mode = st.selectbox("نمط التمويه",
                                 ["تمويه كامل للصورة", "تمويه الوجوه فقط", "لا تمويه"], key="bmode")
        detect_faces = st.checkbox("تفعيل كشف الوجوه", value=True, key="dfaces")
        st.markdown("</div>", unsafe_allow_html=True)

        # Model status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**🤖 حالة النماذج**")
        statuses = {
            "nsfw.tflite":      NSFW_MODEL_PATH.exists(),
            "blazeface.tflite": BLAZE_MODEL_PATH.exists(),
            "faceres.json":     FACERES_JSON_PATH.exists(),
            "faceres.bin":      FACERES_BIN_PATH.exists(),
        }
        for fname, ok in statuses.items():
            icon = "✅" if ok else "⏳"
            sz   = f"{(CACHE_DIR / fname).stat().st_size/1_048_576:.1f} MB" if ok else "—"
            st.markdown(f"{icon} `{fname}` · {sz}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**ℹ️ حول النظام**")
        st.markdown(
            '<p style="font-size:.8rem;color:#8B8FA8;line-height:1.7">'
            "نظام مُبصر AI يستخدم ثلاثة نماذج للكشف الدقيق:<br>"
            "• <b>NSFW Classifier</b> — 5 فئات<br>"
            "• <b>BlazeFace</b> — كشف الوجوه<br>"
            "• <b>FaceRes</b> — الجنس + العمر<br>"
            "النماذج تُحمَّل من الخادم تلقائياً.</p>",
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<p style="text-align:center;font-size:.72rem;color:#555;margin-top:1rem">'
            "Built by <b style='color:#A8A4FF'>Mohamed Salem</b><br>"
            "Expert AI Engineer · © 2026</p>", unsafe_allow_html=True)

    return sensitivity, blur_mode, detect_faces


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Step 1: Ensure models are downloaded ─────────────────────────────────
    ensure_models()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    sensitivity, blur_mode, detect_faces = render_sidebar()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="mubsir-hero">
        <div class="mubsir-logo">🛡️</div>
        <div class="mubsir-title">مُبصر AI</div>
        <div class="mubsir-subtitle">
            نظام الكشف الذكي عن المحتوى غير الآمن · Intelligent NSFW Content Detection
        </div>
        <div class="stat-row">
            <span class="stat-badge">🧠 NSFW TFLite · 5 Classes</span>
            <span class="stat-badge">👁️ BlazeFace Detector</span>
            <span class="stat-badge">🧬 FaceRes Gender + Age</span>
            <span class="stat-badge">🔵 Anti-SFW Bias Correction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    col_up, _ = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "ارفع صورة للتحليل (JPG · PNG · WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploader",
            label_visibility="collapsed",
        )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#555">
            <div style="font-size:4rem;margin-bottom:1rem">📂</div>
            <div style="font-size:1.1rem">ارفع صورة لبدء التحليل</div>
            <div style="font-size:.85rem;margin-top:.5rem;color:#444">يدعم: JPG · PNG · WEBP</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Analyze ───────────────────────────────────────────────────────────────
    image  = Image.open(uploaded).convert("RGB")
    t0     = time.perf_counter()

    with st.spinner("🔍 جاري التحليل العميق…"):
        nsfw_results = run_nsfw(image)
        faces        = []
        if detect_faces:
            faces = run_blazeface(image)
            if faces:
                faces = run_faceres(image, faces)
        decision = make_decision(nsfw_results, faces, sensitivity)

    elapsed = (time.perf_counter() - t0) * 1000

    # ── Display ───────────────────────────────────────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-title">🖼️ الصورة المُحللة</div>', unsafe_allow_html=True)

        if decision["is_unsafe"]:
            if blur_mode == "تمويه كامل للصورة":
                disp = apply_full_blur(image)
                cap  = "🔴 صورة مُموَّهة (محتوى غير آمن)"
            elif blur_mode == "تمويه الوجوه فقط" and faces:
                disp = apply_face_blur(image, faces)
                cap  = "🔴 الوجوه مُموَّهة"
            else:
                disp = image.copy()
                cap  = "⚠️ صورة غير آمنة"
            card_cls = "result-card result-unsafe pulse-unsafe"
        else:
            disp     = annotate_faces(image.copy(), faces) if faces else image.copy()
            cap      = "✅ صورة آمنة"
            card_cls = "result-card result-safe"

        st.image(disp, use_container_width=True, caption=cap)
        st.download_button(
            "⬇️ تحميل الصورة المعالجة",
            data=img_to_bytes(disp),
            file_name="mubsir_result.png",
            mime="image/png",
            use_container_width=True,
            key="dl_btn",
        )
        st.caption(f"⏱️ وقت التحليل: {elapsed:.0f} مللي ثانية")

    with col_res:
        # Verdict
        if decision["is_unsafe"]:
            st.markdown('<div class="verdict-banner verdict-unsafe">🚨 محتوى غير آمن — تم التنبيه</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-banner verdict-safe">✅ المحتوى آمن</div>',
                        unsafe_allow_html=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("⚠️ NSFW", f"{decision['nsfw_total']*100:.1f}%")
        m2.metric("✅ SFW",  f"{decision['sfw_total']*100:.1f}%")
        m3.metric("👤 وجوه", str(decision["face_count"]))
        st.divider()

        # NSFW bars
        st.markdown('<div class="section-title">📊 تصنيف المحتوى</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        render_nsfw_bars(nsfw_results)
        st.markdown("</div>", unsafe_allow_html=True)

        # Faces
        if detect_faces:
            st.markdown('<div class="section-title">👤 تحليل الوجوه</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            render_face_chips(faces)
            st.markdown("</div>", unsafe_allow_html=True)

        # Reasons
        if decision["reasons"]:
            st.markdown('<div class="section-title">📋 أسباب القرار</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-card result-unsafe">', unsafe_allow_html=True)
            for r in decision["reasons"]:
                st.markdown(f"- {r}")
            st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;padding:1.5rem;border-top:1px solid rgba(108,99,255,.2);margin-top:2rem">'
        '<span style="color:#555;font-size:.8rem">'
        '🛡️ <b style="color:#A8A4FF">مُبصر AI</b> · Built by '
        '<b style="color:#6C63FF">Mohamed Salem</b> — Expert AI Engineer & Automation Architect · '
        '© 2026 All Rights Reserved'
        '</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
