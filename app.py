"""
Image Diagnosis & Routing - a small demo in the spirit of BIMB V1.

Upload a photo (or pick a sample) and it measures blur, noise,
detail and JPEG blocking with plain OpenCV, then recommends which
enhancement the image should be routed to, with the reason spelled
out next to each score.

Run:  streamlit run app.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from diagnosis import diagnose

SAMPLES = Path(__file__).parent / "samples"
SAMPLE_LABELS = {
    "clean.jpg": "Sample: clean photo",
    "blurred.jpg": "Sample: blurred",
    "noisy.jpg": "Sample: noisy",
    "lowres.jpg": "Sample: low resolution",
    "compressed.jpg": "Sample: heavy JPEG compression",
    "grayscale.jpg": "Sample: black and white",
}

st.set_page_config(page_title="Image Diagnosis & Routing", layout="wide")

st.title("Image Diagnosis & Routing")
st.write(
    "Drop in a photo. The app measures what is wrong with it "
    "(blur, noise, missing detail, JPEG blocking) and routes it to "
    "the enhancement that fits, with a reason for every step. "
    "Diagnosis first, enhancement second, the way BIMB V1 does it."
)


def load_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


source = st.radio("Image source", ["Pick a sample", "Upload my own"], horizontal=True)

img = None
caption = ""
if source == "Pick a sample":
    names = [n for n in SAMPLE_LABELS if (SAMPLES / n).exists()]
    if not names:
        st.warning("No samples found. Run `python make_samples.py` first.")
    else:
        choice = st.selectbox("Sample", names, format_func=lambda n: SAMPLE_LABELS[n])
        img = cv2.imread(str(SAMPLES / choice))
        caption = SAMPLE_LABELS[choice]
else:
    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if uploaded is not None:
        img = load_bgr(uploaded.getvalue())
        caption = uploaded.name

if img is None:
    st.info("Pick a sample above to see the diagnosis instantly.")
    st.stop()

# Diagnose on a copy capped at 1600 px wide so big phone photos stay fast.
work = img
if img.shape[1] > 1600:
    scale = 1600 / img.shape[1]
    work = cv2.resize(img, None, fx=scale, fy=scale)

with st.spinner("Measuring..."):
    d = diagnose(work)

left, right = st.columns([1, 1])
with left:
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)
    st.caption(
        f"{d.width} x {d.height} px - {d.megapixels:.2f} MP - "
        f"faces detected: {d.faces} - "
        f"colour: {'black and white' if d.is_grayscale else 'colour'}"
    )

with right:
    st.subheader("Scores (0-100, higher is better)")
    c1, c2 = st.columns(2)
    c1.metric("Sharpness", f"{d.sharpness:.0f}")
    c2.metric("Cleanliness (low noise)", f"{d.cleanliness:.0f}")
    c1.metric("Detail", f"{d.detail:.0f}")
    c2.metric("Compression health", f"{d.compression:.0f}")

    st.subheader("What was measured")
    st.table({
        "Measurement": [
            "Laplacian variance (focus)",
            "Noise sigma, Immerkaer estimate",
            "High-frequency energy share",
            "JPEG blockiness ratio",
            "Mean saturation",
        ],
        "Value": [
            f"{d.laplacian_variance:.1f}",
            f"{d.noise_sigma:.2f}",
            f"{d.hf_share:.4f}",
            f"{d.blockiness_ratio:.2f}",
            f"{d.mean_saturation:.0f} / 255",
        ],
    })

st.subheader("Routing decision")
st.write(f"**Verdict:** {d.verdict}")
if d.recommendations:
    for i, r in enumerate(d.recommendations, 1):
        badge = "needs it" if r.severity == "strong" else "optional"
        st.markdown(
            f"{i}. **{r.action}** -> {r.agent} ({badge})  \n{r.reason}"
        )
else:
    st.write("No enhancement recommended. This one is fine as it is.")

st.divider()
st.caption(
    "How it works: sharpness is the variance of the Laplacian "
    "(Pech-Pacheco et al. 2000), noise is Immerkaer's Laplacian-mask "
    "estimate (1996), detail is the high-frequency share of the FFT "
    "spectrum, and blockiness compares pixel jumps across JPEG 8x8 "
    "block borders against jumps inside blocks. The router is plain "
    "rules on those numbers, no model involved."
)
