"""VehicleVision - single-page Streamlit app for custom-CNN vehicle classification."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from html import escape

import streamlit as st

from utils.helpers import MODEL_PATH, emoji, load_class_names, load_pil_image, predict_image, pretty

st.set_page_config(page_title="VehicleVision", page_icon="🚗", layout="centered",
                   initial_sidebar_state="collapsed")

CLASS_NAMES = load_class_names()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; visibility: hidden; }
.stApp {
  background:
    radial-gradient(900px 500px at 10% -10%, rgba(124,92,255,.28), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(0,200,255,.18), transparent 55%),
    linear-gradient(180deg, #0B1020 0%, #0E1530 100%);
  background-attachment: fixed;
}
.block-container { max-width: 1040px; padding: 2.2rem 1.2rem 3rem 1.2rem; }

/* hero */
.hero { text-align: center; padding: 1.2rem 0 1.6rem 0; animation: fadeUp .7s ease both; }
.badge { display: inline-block; padding: .28rem .8rem; border-radius: 999px; font-size: .78rem;
  font-weight: 600; letter-spacing: .04em; color: #B9A9FF; background: rgba(124,92,255,.14);
  border: 1px solid rgba(124,92,255,.35); margin-bottom: .9rem; }
.hero h1 { font-size: clamp(2.1rem, 5.5vw, 3.5rem); font-weight: 800; letter-spacing: -.03em;
  line-height: 1.05; margin: 0 0 .7rem 0;
  background: linear-gradient(90deg, #FFFFFF 0%, #B9A9FF 45%, #5CE1FF 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #A7B0CE; font-size: clamp(.98rem, 2.2vw, 1.15rem); margin: 0 auto; max-width: 560px; }
.chips { margin-top: 1.1rem; display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; }
.chip { padding: .35rem .8rem; border-radius: 999px; font-size: .85rem; color: #D5DBF2;
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); }

/* cards: style the two column containers */
[data-testid="stHorizontalBlock"] { gap: 1.4rem; align-items: stretch; }
[data-testid="column"] > div { height: 100%; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.035));
  border: 1px solid rgba(255,255,255,.10) !important; border-radius: 22px !important;
  box-shadow: 0 20px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);
  backdrop-filter: blur(14px); padding: .4rem .5rem; animation: fadeUp .8s ease both; }
.card-title { font-size: .8rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: #8E98BC; margin: .2rem 0 .8rem 0; }

/* uploader */
[data-testid="stFileUploaderDropzone"] { border: 2px dashed rgba(124,92,255,.55); border-radius: 18px;
  background: rgba(124,92,255,.07); padding: 1.6rem 1rem; transition: all .25s ease; }
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(124,92,255,.14); border-color: #9C85FF;
  transform: translateY(-1px); }
[data-testid="stFileUploaderDropzone"] button { border-radius: 10px; border: 1px solid rgba(255,255,255,.2); }
[data-testid="stFileUploaderFile"] { display: none; }

/* image preview */
[data-testid="stImage"] img { border-radius: 16px; box-shadow: 0 12px 30px rgba(0,0,0,.45);
  max-height: 340px; object-fit: contain; width: 100%; background: rgba(0,0,0,.25); }

/* predict button */
.stButton > button { width: 100%; border: 0; border-radius: 14px; padding: .95rem 1rem; font-weight: 700;
  font-size: 1.05rem; letter-spacing: .02em; color: #fff;
  background: linear-gradient(90deg, #7C5CFF 0%, #4C8DFF 55%, #2CD4FF 100%);
  box-shadow: 0 10px 28px rgba(92,120,255,.45); transition: transform .15s ease, box-shadow .15s ease, filter .15s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 34px rgba(92,120,255,.6); filter: brightness(1.07); color:#fff; }
.stButton > button:active { transform: translateY(0); }

/* result */
.result-top { display: flex; align-items: center; gap: 1.2rem; animation: fadeUp .6s ease both; }
.ring { --p: 0; --c: #7C5CFF; width: 112px; height: 112px; border-radius: 50%; flex: 0 0 auto; position: relative;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.10) 0); display: grid; place-items: center; }
.ring::before { content: ""; position: absolute; inset: 9px; border-radius: 50%; background: #121936; }
.ring span { position: relative; font-weight: 800; font-size: 1.35rem; color: #fff; }
.label-small { color: #8E98BC; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.label-big { font-size: clamp(1.7rem, 4vw, 2.3rem); font-weight: 800; color: #fff; line-height: 1.1; margin-top: .15rem; }
.conf-text { color: #A7B0CE; font-size: .95rem; margin-top: .25rem; }
.warn { margin-top: .9rem; padding: .6rem .8rem; border-radius: 12px; font-size: .88rem; color: #FFD58A;
  background: rgba(255,170,0,.10); border: 1px solid rgba(255,170,0,.25); }
.prob-title { margin: 1.3rem 0 .6rem 0; }
.prow { display: grid; grid-template-columns: 7.6rem 1fr 3.6rem; align-items: center; gap: .7rem; margin: .55rem 0; }
.plabel { color: #D5DBF2; font-size: .92rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pbar { height: 10px; border-radius: 999px; background: rgba(255,255,255,.08); overflow: hidden; }
.pfill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #4C5A8F, #6F7FBF);
  transform-origin: left; animation: grow .9s cubic-bezier(.2,.8,.2,1) both; }
.pfill.top { background: linear-gradient(90deg, #7C5CFF, #2CD4FF); box-shadow: 0 0 14px rgba(92,160,255,.55); }
.pval { text-align: right; color: #D5DBF2; font-size: .9rem; font-variant-numeric: tabular-nums; font-weight: 600; }
.empty { text-align: center; padding: 3.2rem 1rem; color: #8E98BC; }
.empty .big { font-size: 3rem; margin-bottom: .6rem; opacity: .9; }
.empty b { color: #D5DBF2; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(12px);} to { opacity: 1; transform: none; } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (max-width: 640px) {
  .result-top { flex-direction: column; text-align: center; }
  .prow { grid-template-columns: 6rem 1fr 3.2rem; }
  .block-container { padding-top: 1.2rem; }
}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_model():
    from tensorflow import keras  # imported lazily so the page renders immediately
    return keras.models.load_model(MODEL_PATH)


def render_result(label: str, conf: float, probs) -> str:
    order = sorted(range(len(CLASS_NAMES)), key=lambda i: -probs[i])
    rows = "".join(
        f'<div class="prow"><div class="plabel">{emoji(CLASS_NAMES[i])} {escape(pretty(CLASS_NAMES[i]))}</div>'
        f'<div class="pbar"><div class="pfill{" top" if k == 0 else ""}" style="width:{probs[i] * 100:.2f}%"></div></div>'
        f'<div class="pval">{probs[i] * 100:.1f}%</div></div>'
        for k, i in enumerate(order))
    warn = ('<div class="warn">⚠️ Low confidence - this image may not show one of the supported vehicles, '
            'or the view is unclear.</div>') if conf < 50 else ""
    return (f'<div class="result-top"><div class="ring" style="--p:{conf:.1f}"><span>{conf:.1f}%</span></div>'
            f'<div><div class="label-small">Predicted vehicle</div>'
            f'<div class="label-big">{emoji(label)} {escape(pretty(label))}</div>'
            f'<div class="conf-text">Confidence: <b>{conf:.2f}%</b></div></div></div>{warn}'
            f'<div class="label-small prob-title">Probability distribution</div>{rows}')


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    chips = "".join(f'<span class="chip">{emoji(c)} {escape(pretty(c))}</span>' for c in CLASS_NAMES)
    st.markdown(
        '<div class="hero"><div class="badge">CUSTOM CNN · TRAINED FROM SCRATCH</div>'
        '<h1>VehicleVision</h1>'
        '<p>Drop in a photo and our convolutional neural network will tell you what kind of vehicle it is.</p>'
        f'<div class="chips">{chips}</div></div>', unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.error("Trained model not found (`models/vehicle_classifier.keras`). Run `python train.py` first.")
        return

    left, right = st.columns(2, gap="large")
    img = None
    with left:
        with st.container(border=True):
            st.markdown('<div class="card-title">1 · Upload image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"],
                                        label_visibility="collapsed")
            key = None
            if uploaded is not None:
                key = f"{uploaded.name}:{uploaded.size}"
                try:
                    img = load_pil_image(uploaded)
                    preview = img.copy()
                    preview.thumbnail((900, 900))
                    st.image(preview)
                except ValueError:
                    st.error("That file could not be read as an image. Please upload a valid JPG, JPEG or PNG.")
                    img = None
            clicked = st.button("🔍  Predict", disabled=img is None, key="predict")

    if st.session_state.get("result_key") != key:      # new/removed upload -> clear old result
        st.session_state.pop("result", None)
        st.session_state["result_key"] = key
    if clicked and img is not None:
        with st.spinner("Analysing image..."):
            try:
                st.session_state["result"] = predict_image(get_model(), img, CLASS_NAMES)
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop("result", None)
                st.error(f"Prediction failed: {exc}")

    with right:
        with st.container(border=True):
            st.markdown('<div class="card-title">2 · Result</div>', unsafe_allow_html=True)
            res = st.session_state.get("result")
            if res is None:
                msg = ("Press <b>Predict</b> to classify your image." if img is not None
                       else "Upload an image to get started.")
                st.markdown(f'<div class="empty"><div class="big">🚦</div>{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(render_result(*res), unsafe_allow_html=True)


main()
