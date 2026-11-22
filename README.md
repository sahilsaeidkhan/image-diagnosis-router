# Image Diagnosis & Routing

Upload a photo and this app tells you what is actually wrong with it:
how blurry, how noisy, how much real detail it has, and how badly
JPEG compression has chewed on it. Then it suggests an ordered
enhancement plan (denoise, deblur, deblock, upscale, colourise),
with the measured number behind every step. No model makes the
decision. Four classical measurements and a short list of rules do.

The idea comes from Adima AI's BIMB V1 launch post: diagnose first,
then route the image to the right fix, instead of throwing every
filter at every picture.

## Run it

```
pip install -r requirements.txt
python make_samples.py
streamlit run app.py
```

Two setup notes:

- The sample photos are not in this repo, they are generated.
  `make_samples.py` downloads one clean photo from Lorem Picsum
  and builds the five damaged copies from it. It needs the
  internet once, for that download.
- Face detection uses OpenCV's YuNet model, which is also not in
  the repo. To enable the face hint, download the file
  `face_detection_yunet_2023mar.onnx` from the OpenCV Zoo on
  GitHub (it lives in the `models/face_detection_yunet` folder
  there) and put it in this repo's `models/` folder. Without it
  the app still runs fine, it just reports zero faces.

Then pick a sample in the app: the blurred one routes to deblur,
the black and white one to colourise, and so on.

## What it measures

Sharpness, using the variance of the Laplacian (Pech-Pacheco et
al., 2000). Blurred photos lose their edges, the Laplacian response
goes flat, and the variance collapses. On the sample set the clean
photo reads 2785 and its blurred copy reads 8.

Noise, using Immerkaer's method (1996), which recovers an estimate
of the noise sigma from a Laplacian mask that cancels smooth image
content. Checked against synthetic noise of known sigma 5, 15 and
25, it estimates 4.98, 15.04 and 25.03.

Detail, using the share of FFT energy in the high frequencies
(beyond 40% of Nyquist). This catches the case dimensions cannot:
a small photo that was already upscaled once is big in pixels and
empty in detail. The clean sample scores 95/100 here, its
upscaled-from-240x180 twin scores 8/100.

JPEG blocking, by comparing pixel jumps across 8x8 block borders
with jumps inside blocks. A clean image sits near a ratio of 1.0.
The quality-6 sample reads 2.77.

It also checks whether the photo is effectively black and white
(routes to colourise, the Painter-style step) and, if the YuNet
model is present, whether it contains faces (routes to facial
enhancement, the Makeup Artist-style step).

## The router

The router is a short list of readable rules in `diagnosis.py`,
not a model. The order matters and follows how an enhancement
pipeline should actually run: colourise is independent, then
denoise (later steps amplify noise), then deblur, then deblocking,
then upscale. Every recommendation prints the measured value that
caused it, so you never have to take the app's word for anything.

## Results

```
python tests/test_diagnosis.py
```

16 checks on the metrics and the router, all passing as of
22 Nov 2026. Each damaged sample produces exactly one strong
recommendation, and it is the right one; the clean photo produces
none. The full metrics table is in `EVALUATION.md`.

## Files

- `app.py` the Streamlit interface
- `diagnosis.py` all measurements, scores and the router
- `make_samples.py` generates the sample set (see Run it)
- `samples/` one clean photo and five damaged copies, generated
- `tests/test_diagnosis.py` 16 checks on the metrics and the router
- `EVALUATION.md` the measured numbers and test results
- `models/` where the optional YuNet model goes (not included)

The clean sample photo is from Lorem Picsum (picsum.photos/id/1015,
an Unsplash-sourced landscape). The damaged samples are generated
from it locally: Gaussian blur, Gaussian noise with a fixed seed,
a true 240x180 downscale, JPEG quality 6, and a grayscale copy.

## Limitations

Honest list, because a diagnosis tool that oversells itself is
missing the point:

- The thresholds are calibrated on the sample photo and its copies.
  They behave well there, but they are calibration choices, not
  universal constants. A portrait with deliberate background blur
  will read as "soft" to the focus measure, because the metric
  cannot know the blur was intentional.
- The noise estimate over-reads on very textured scenes (grass,
  water, foliage). On the clean sample it reports sigma 4.8 where
  no noise was added. The router's thresholds leave room for this,
  but the raw number on a real photo is a ranking signal, not a
  lab measurement.
- Noise also inflates the sharpness number, which is why the
  router denoises before it trusts anything about sharpness.
- Blocking detection only understands JPEG's 8x8 grid. WebP
  artifacts, over-sharpening halos and banding are not covered.
- Face detection is a small on-device model and misses small or
  turned faces. It only adds a hint to the routing, never a score.
- Nothing here enhances anything. This is the diagnosis and
  routing half of the pipeline, the half that decides what should
  happen. The enhancement models are the other half.

## Why I built it

BIMB V1's launch post described an agent that diagnoses blur,
noise and compression and then routes the image to the right
model. I wanted to see how far plain, explainable measurements
get you toward that diagnosis step without any model at all.
Further than I expected: on a controlled set, four classical
metrics and a page of rules separate the failure modes cleanly.
Where they stop working is documented above, which is also kind
of the point.

Built by Sahil Saeid, 2nd-year B.Tech CSE student working on
AI/ML (RAG systems, PyTorch, FastAPI). This is a demo, shared for
feedback, not a product.
