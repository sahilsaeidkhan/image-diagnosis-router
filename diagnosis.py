"""
Image diagnosis and routing core.

Given a photo, this measures four things that decide what kind of
enhancement it needs:

  sharpness   variance of the Laplacian (Pech-Pacheco et al., 2000).
              Blurred images lose edges, so the Laplacian response
              flattens and its variance drops.
  noise       Immerkaer's method (1996). Convolving with the Laplacian
              mask [[1,-2,1],[-2,4,-2],[1,-2,1]] mostly cancels smooth
              image content and leaves noise, whose sigma is recovered
              from the mean absolute response.
  detail      share of FFT energy in the high frequencies. A photo that
              was upscaled from a small original has almost no energy
              up there, whatever its pixel dimensions say.
  blockiness  JPEG compresses in 8x8 blocks. If pixel jumps across
              block borders are much bigger than jumps inside blocks,
              the image was squeezed hard. We report the ratio of the
              two (about 1.0 for a clean image).

On top of the measurements sits a small rule-based router that turns
them into an ordered enhancement plan, the same diagnosis-then-route
idea BIMB V1 is built around. No machine learning in here on purpose:
the rules are readable and every recommendation carries its reason.

Scores are 0-100 per axis, higher means better quality on that axis.
The raw values are always reported alongside, because the scores are
just a calibrated mapping of the raw values (see score functions).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------- metrics


def laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure. Higher = sharper. ~100 and below usually reads
    as blurry on ordinary photos, but it is content dependent."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer (1996) noise estimate, returned as a sigma in the
    same 0-255 units as the pixels. Validated in the test suite
    against synthetic noise of known sigma."""
    h, w = gray.shape
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, mask)
    inner = np.abs(conv[1:-1, 1:-1])  # drop the border, mask is 3x3
    sigma = np.sqrt(np.pi / 2.0) * inner.sum() / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def high_frequency_share(gray: np.ndarray) -> float:
    """Fraction of FFT energy outside the central low-frequency disc
    (radius = 40% of Nyquist). An image upscaled 4x has essentially
    no energy out there (its real content stops at 25%), while a
    natively sharp photo keeps a healthy share. Measured on the
    sample set: clean photo 0.0121, the same photo upscaled from
    240x180 just 0.00028."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    power = np.abs(f) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = min(cx, cy)
    total = power.sum()
    if total == 0:
        return 0.0
    high = power[r > 0.40 * r_max].sum()
    return float(high / total)


def blockiness_ratio(gray: np.ndarray) -> float:
    """Mean absolute pixel jump across 8x8 block borders divided by
    the mean jump everywhere else. ~1.0 means no visible blocking;
    heavily compressed JPEGs push this well above 1.3."""
    g = gray.astype(np.float64)

    def axis_ratio(diffs: np.ndarray, axis_len: int, axis: int) -> float:
        # diffs index i sits between pixel i and i+1; block borders are
        # between pixels 7|8, 15|16, ... so border indices are 7, 15, ...
        idx = np.arange(diffs.shape[axis])
        border_mask = (idx % 8) == 7
        if axis == 1:
            border = diffs[:, border_mask]
            inside = diffs[:, ~border_mask]
        else:
            border = diffs[border_mask, :]
            inside = diffs[~border_mask, :]
        inside_mean = inside.mean()
        if inside_mean < 1e-9:
            return 1.0
        return float(border.mean() / inside_mean)

    dv = np.abs(np.diff(g, axis=1))
    dh = np.abs(np.diff(g, axis=0))
    return (axis_ratio(dv, g.shape[1], axis=1) + axis_ratio(dh, g.shape[0], axis=0)) / 2.0


def colour_stats(bgr: np.ndarray) -> tuple[float, bool]:
    """Mean saturation (0-255) and whether the image is effectively
    black and white (channels nearly identical almost everywhere)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    channel_spread = bgr.max(axis=2).astype(int) - bgr.min(axis=2).astype(int)
    effectively_gray = float((channel_spread < 8).mean()) > 0.98
    return mean_sat, effectively_gray


_FACE_MODEL = None


def count_faces(bgr: np.ndarray) -> int:
    """Face count from OpenCV's YuNet detector (model file ships in
    models/, from the OpenCV Zoo, MIT-style licensed zoo model).
    Only used as a hint for the router (portrait-style enhancement),
    it is not a quality metric. Returns 0 if the model is missing."""
    global _FACE_MODEL
    from pathlib import Path

    model_path = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
    if not model_path.exists():
        return 0
    if _FACE_MODEL is None:
        _FACE_MODEL = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold=0.6
        )
    detector = _FACE_MODEL
    img = bgr
    if max(img.shape[:2]) > 800:
        scale = 800 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale)
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    return 0 if faces is None else len(faces)


# ---------------------------------------------------------------- scores
# Each score maps a raw measurement onto 0-100. The anchor points come
# from the published behaviour of these metrics on ordinary photos and
# were sanity-checked against the sample set in samples/ (see
# EVALUATION.md). They are calibration choices, not laws of nature.


def sharpness_score(lap_var: float) -> float:
    # log scale: var 10 -> 0, 100 -> ~43, 500 -> ~74, 2000 -> 100
    s = 100.0 * (np.log10(max(lap_var, 1.0)) - 1.0) / (np.log10(2000.0) - 1.0)
    return float(np.clip(s, 0, 100))


def noise_score(sigma: float) -> float:
    # sigma 0 -> 100, 5 -> ~66, 10 -> ~43, 25 -> ~12
    return float(100.0 * np.exp(-sigma / 12.0))


def detail_score(hf_share: float) -> float:
    # log scale, anchored on the sample set: share 0.0002 -> 0,
    # 0.0003 (upscaled) -> ~8, 0.0067 (heavy JPEG) -> ~81,
    # 0.0121 (clean photo) -> ~95, 0.015 -> 100
    lo, hi = np.log10(0.0002), np.log10(0.015)
    s = 100.0 * (np.log10(max(hf_share, 1e-6)) - lo) / (hi - lo)
    return float(np.clip(s, 0, 100))


def blockiness_score(ratio: float) -> float:
    # ratio 1.0 -> 100, 1.3 -> ~64, 1.8 -> ~4
    return float(np.clip(100.0 - (ratio - 1.0) * 120.0, 0, 100))


# ---------------------------------------------------------------- report


@dataclass
class Recommendation:
    action: str        # e.g. "denoise"
    agent: str         # closest BIMB-style identity, for display
    reason: str        # plain-English, cites the measured value
    severity: str      # "strong" or "mild"


@dataclass
class Diagnosis:
    width: int
    height: int
    megapixels: float
    laplacian_variance: float
    noise_sigma: float
    hf_share: float
    blockiness_ratio: float
    mean_saturation: float
    is_grayscale: bool
    faces: int
    sharpness: float
    cleanliness: float   # from noise
    detail: float
    compression: float   # from blockiness (higher = cleaner)
    recommendations: list = field(default_factory=list)
    verdict: str = ""


def diagnose(bgr: np.ndarray) -> Diagnosis:
    """Run the full diagnosis on a BGR image (as cv2.imread returns)."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lap_var = laplacian_variance(gray)
    sigma = estimate_noise_sigma(gray)
    hf = high_frequency_share(gray)
    block = blockiness_ratio(gray)
    sat, is_gray = colour_stats(bgr)
    faces = count_faces(bgr)

    d = Diagnosis(
        width=w,
        height=h,
        megapixels=w * h / 1e6,
        laplacian_variance=lap_var,
        noise_sigma=sigma,
        hf_share=hf,
        blockiness_ratio=block,
        mean_saturation=sat,
        is_grayscale=is_gray,
        faces=faces,
        sharpness=sharpness_score(lap_var),
        cleanliness=noise_score(sigma),
        detail=detail_score(hf),
        compression=blockiness_score(block),
    )
    d.recommendations = route(d)
    d.verdict = _verdict(d)
    return d


# ---------------------------------------------------------------- router

# Thresholds on the raw measurements. Deliberately conservative: the
# router only speaks up when a measurement is clearly off, and says
# "mild" when it is borderline.
BLUR_STRONG = 100.0      # Laplacian variance below this: clearly soft
BLUR_MILD = 300.0
NOISE_STRONG = 8.0       # estimated sigma above this: clearly noisy
NOISE_MILD = 6.0         # Immerkaer over-reads on very textured
                         # scenes (the clean sample reads ~4.8 from
                         # texture alone), so mild starts a bit higher
DETAIL_LOW = 35.0        # detail score below this: likely upscaled
BLOCK_STRONG = 1.45      # blockiness ratio above this: heavy JPEG
BLOCK_MILD = 1.25
LOW_MP = 0.6             # below this many megapixels: small file


def route(d: Diagnosis) -> list[Recommendation]:
    recs: list[Recommendation] = []

    # Colour is independent of the cleanup chain: a black and white
    # photo routes to the Painter-style step whatever else is wrong.
    if d.is_grayscale:
        recs.append(Recommendation(
            action="colourise",
            agent="Painter",
            reason="The image is effectively black and white "
                   f"(mean saturation {d.mean_saturation:.0f} out of 255), "
                   "so colourisation is the natural first step.",
            severity="strong",
        ))

    # Cleanup chain, in the order an enhancement pipeline should run:
    # denoise first because sharpening and upscaling both amplify
    # noise, then deblur, then upscale.
    if d.noise_sigma > NOISE_STRONG:
        recs.append(Recommendation(
            action="denoise",
            agent="Restoration",
            reason=f"Noise is high (estimated sigma {d.noise_sigma:.1f}). "
                   "Denoise before anything else, later steps amplify noise.",
            severity="strong",
        ))
    elif d.noise_sigma > NOISE_MILD:
        recs.append(Recommendation(
            action="denoise",
            agent="Restoration",
            reason=f"Some grain is present (estimated sigma {d.noise_sigma:.1f}). "
                   "A light denoise would clean it up.",
            severity="mild",
        ))

    if d.laplacian_variance < BLUR_STRONG:
        recs.append(Recommendation(
            action="deblur",
            agent="Restoration",
            reason=f"The image is soft (Laplacian variance {d.laplacian_variance:.0f}, "
                   "sharp photos usually read in the hundreds). "
                   "Route to deblurring / sharpening.",
            severity="strong",
        ))
    elif d.laplacian_variance < BLUR_MILD:
        recs.append(Recommendation(
            action="deblur",
            agent="Restoration",
            reason=f"Edges are a little soft (Laplacian variance {d.laplacian_variance:.0f}). "
                   "Mild sharpening would help.",
            severity="mild",
        ))

    if d.blockiness_ratio > BLOCK_MILD:
        sev = "strong" if d.blockiness_ratio > BLOCK_STRONG else "mild"
        recs.append(Recommendation(
            action="deblock / restore",
            agent="Restoration",
            reason=f"JPEG blocking is visible (blockiness ratio {d.blockiness_ratio:.2f}, "
                   "a clean image sits near 1.0). An artifact-removal pass "
                   "should run before enlarging.",
            severity=sev,
        ))

    # Thin detail only counts as an upscale signal when blur and
    # blocking do not already explain it (both eat high frequencies).
    thin_detail = (
        d.detail < DETAIL_LOW
        and d.laplacian_variance >= BLUR_STRONG
        and d.blockiness_ratio <= BLOCK_STRONG
    )
    if thin_detail or d.megapixels < LOW_MP:
        bits = []
        if d.megapixels < LOW_MP:
            bits.append(f"the file is small ({d.megapixels:.2f} MP)")
        if thin_detail:
            bits.append(f"high-frequency detail is thin (detail score {d.detail:.0f}/100), "
                        "which usually means it was already upscaled once")
        recs.append(Recommendation(
            action="upscale",
            agent="Digital Artist",
            reason="Route to upscaling because " + " and ".join(bits) + ".",
            severity="strong" if d.megapixels < 0.25 or (thin_detail and d.megapixels < LOW_MP) else "mild",
        ))

    if d.faces > 0 and not d.is_grayscale:
        recs.append(Recommendation(
            action="facial enhancement",
            agent="Makeup Artist",
            reason=f"{d.faces} face(s) detected, so a portrait / facial "
                   "enhancement pass fits this image.",
            severity="mild",
        ))

    return recs


def _verdict(d: Diagnosis) -> str:
    strong = [r for r in d.recommendations if r.severity == "strong"]
    if not strong:
        return "Looks healthy. No strong problems measured, at most minor touch-ups."
    names = ", ".join(r.action for r in strong)
    return f"Needs work: {names}."
