"""
Checks for the diagnosis core, run against the generated samples.

Every damaged sample must score worse than the clean photo on the
axis it damages, the noise estimator must roughly recover a known
synthetic sigma, and the router must name the right treatment for
each sample. Prints a metrics table (used for EVALUATION.md) and
PASS/FAIL lines; exits non-zero if anything fails.

Run from the project root:  python tests/test_diagnosis.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from diagnosis import diagnose, estimate_noise_sigma  # noqa: E402

SAMPLES = Path(__file__).parent.parent / "samples"

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(SAMPLES / name))
    assert img is not None, f"missing sample {name} - run make_samples.py"
    return img


def actions(d) -> list[str]:
    return [r.action for r in d.recommendations]


def main() -> int:
    diag = {}
    for name in ["clean.jpg", "blurred.jpg", "noisy.jpg",
                 "lowres.jpg", "compressed.jpg", "grayscale.jpg"]:
        diag[name] = diagnose(load(name))

    print("\n--- metrics table ---")
    print(f"{'sample':16s} {'lap_var':>9s} {'sigma':>6s} {'hf40':>8s} "
          f"{'block':>5s} {'sat':>5s}  routes")
    for name, d in diag.items():
        routes = ", ".join(f"{r.action}({r.severity})" for r in d.recommendations)
        print(f"{name:16s} {d.laplacian_variance:9.1f} {d.noise_sigma:6.2f} "
              f"{d.hf_share:8.5f} {d.blockiness_ratio:5.2f} "
              f"{d.mean_saturation:5.0f}  {routes}")
    print("---------------------\n")

    clean, blurred = diag["clean.jpg"], diag["blurred.jpg"]
    noisy, lowres = diag["noisy.jpg"], diag["lowres.jpg"]
    comp, gray = diag["compressed.jpg"], diag["grayscale.jpg"]

    # Blur: the blurred copy must collapse on the focus measure.
    check("blur: laplacian variance collapses",
          blurred.laplacian_variance < 0.10 * clean.laplacian_variance,
          f"clean={clean.laplacian_variance:.0f} blurred={blurred.laplacian_variance:.0f}")
    check("blur: sharpness score drops",
          blurred.sharpness < clean.sharpness - 50,
          f"clean={clean.sharpness:.0f} blurred={blurred.sharpness:.0f}")
    check("blur: router says deblur (strong)",
          any(r.action == "deblur" and r.severity == "strong"
              for r in blurred.recommendations))

    # Noise: sigma estimate must jump on the noisy copy.
    check("noise: sigma estimate jumps",
          noisy.noise_sigma > clean.noise_sigma + 6,
          f"clean={clean.noise_sigma:.2f} noisy={noisy.noise_sigma:.2f}")
    check("noise: router says denoise (strong) first",
          noisy.recommendations
          and noisy.recommendations[0].action == "denoise"
          and noisy.recommendations[0].severity == "strong")

    # Noise estimator accuracy: flat image + known Gaussian noise.
    rng = np.random.default_rng(7)
    flat = np.full((400, 400), 128.0)
    for true_sigma in (5.0, 15.0, 25.0):
        test_img = np.clip(flat + rng.normal(0, true_sigma, flat.shape), 0, 255)
        est = estimate_noise_sigma(test_img.astype(np.uint8))
        check(f"noise: Immerkaer recovers sigma={true_sigma:.0f}",
              abs(est - true_sigma) < 0.25 * true_sigma,
              f"estimated={est:.2f}")

    # Compression: blockiness ratio must jump on the quality-6 JPEG.
    check("jpeg: blockiness ratio jumps",
          comp.blockiness_ratio > clean.blockiness_ratio + 0.5,
          f"clean={clean.blockiness_ratio:.2f} q6={comp.blockiness_ratio:.2f}")
    check("jpeg: router says deblock (strong)",
          any(r.action.startswith("deblock") and r.severity == "strong"
              for r in comp.recommendations))

    # Low resolution: tiny file routes to upscale, and an image that
    # was upscaled back to full size must show thin detail.
    check("lowres: file is much smaller",
          lowres.megapixels < 0.1 * clean.megapixels,
          f"clean={clean.megapixels:.2f}MP lowres={lowres.megapixels:.2f}MP")
    check("lowres: router says upscale (strong)",
          any(r.action == "upscale" and r.severity == "strong"
              for r in lowres.recommendations))
    up = cv2.resize(load("lowres.jpg"), (960, 720), interpolation=cv2.INTER_CUBIC)
    d_up = diagnose(up)
    check("lowres: upscaled-back copy shows thin detail",
          d_up.detail < clean.detail - 40,
          f"clean detail={clean.detail:.0f} upscaled={d_up.detail:.0f}")

    # Colour: the grayscale copy must route to colourise.
    check("colour: grayscale detected",
          gray.is_grayscale and gray.mean_saturation < 5,
          f"sat={gray.mean_saturation:.1f}")
    check("colour: router says colourise (strong)",
          any(r.action == "colourise" and r.severity == "strong"
              for r in gray.recommendations))

    # Clean photo: the router must not invent strong work.
    strong_clean = [r.action for r in clean.recommendations if r.severity == "strong"]
    check("clean: no strong recommendations", strong_clean == [],
          f"strong={strong_clean}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
