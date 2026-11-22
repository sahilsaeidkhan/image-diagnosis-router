"""
Generate the sample images in samples/ from one clean photo.

clean.jpg is a real photo (Lorem Picsum, picsum.photos/id/1015, an
Unsplash-sourced landscape, free to use). If it is missing, this
script downloads it. Every other sample is a damaged copy of it, so
the demo and the tests compare like with like:

  blurred.jpg     Gaussian blur, sigma 2.2
  noisy.jpg       additive Gaussian noise, sigma 18 (seeded RNG)
  lowres.jpg      the photo at 240x180, its true delivered size
  compressed.jpg  JPEG quality 6
  grayscale.jpg   black and white copy (for the colourise route)

Run:  python make_samples.py
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SAMPLES = Path(__file__).parent / "samples"
SOURCE_URL = "https://picsum.photos/id/1015/960/720"


def ensure_clean() -> np.ndarray:
    clean_path = SAMPLES / "clean.jpg"
    if not clean_path.exists():
        raw = SAMPLES / "_download.jpg"
        urllib.request.urlretrieve(SOURCE_URL, raw)
        img = Image.open(raw).convert("RGB")
        img.save(clean_path, quality=92)
        raw.unlink()
    img = cv2.imread(str(clean_path))
    if img is None:
        raise RuntimeError(f"could not read {clean_path}")
    return img


def save(img: np.ndarray, name: str, quality: int = 92) -> None:
    cv2.imwrite(str(SAMPLES / name), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print("wrote", name, img.shape[1], "x", img.shape[0])


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    clean = ensure_clean()
    print("clean.jpg", clean.shape[1], "x", clean.shape[0])

    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=2.2)
    save(blurred, "blurred.jpg")

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 18, clean.shape)
    noisy = np.clip(clean.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    save(noisy, "noisy.jpg")

    lowres = cv2.resize(clean, (240, 180), interpolation=cv2.INTER_AREA)
    save(lowres, "lowres.jpg")

    save(clean, "compressed.jpg", quality=6)

    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    save(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "grayscale.jpg")


if __name__ == "__main__":
    main()
