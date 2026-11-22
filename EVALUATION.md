# Evaluation

Everything below was actually run on 22 Nov 2026 in this folder
(Python 3.12, opencv-python-headless 5.1.0, numpy 2.5.5). Reproduce
with:

```
python make_samples.py        # only needed once, samples are committed
python tests/test_diagnosis.py
```

## Test result

16 of 16 checks passed. The checks assert that each damaged sample
scores worse than the clean photo on the axis it damages, that the
noise estimator recovers known synthetic noise levels, and that the
router names the right treatment for every sample.

## Measured values on the sample set

All samples come from one clean photo (960x720), so the numbers are
comparable like for like.

| Sample | Laplacian var | Noise sigma | HF share (r=0.40) | Blockiness | Saturation | Router said |
|---|---|---|---|---|---|---|
| clean.jpg | 2784.5 | 4.83 | 0.01210 | 1.08 | 141 | nothing (no strong problems) |
| blurred.jpg | 8.1 | 0.28 | 0.00023 | 1.14 | 138 | deblur (strong) |
| noisy.jpg | 5687.0 | 14.39 | 0.01803 | 0.99 | 139 | denoise (strong) |
| lowres.jpg (240x180) | 2697.7 | 5.89 | 0.01176 | 1.01 | 139 | upscale (strong) |
| compressed.jpg (JPEG q6) | 941.2 | 1.04 | 0.00671 | 2.77 | 137 | deblock / restore (strong) |
| grayscale.jpg | 2642.7 | 4.65 | 0.01180 | 1.10 | 0 | colourise (strong) |

Each damaged sample produced exactly one strong recommendation, the
right one, and the clean photo produced none. That one-in-one-out
behaviour is the main thing this demo claims, and it holds on this
set.

## Noise estimator accuracy

Immerkaer's estimator was checked against flat images with known
Gaussian noise added:

| True sigma | Estimated |
|---|---|
| 5 | 4.98 |
| 15 | 15.04 |
| 25 | 25.03 |

On flat content it is nearly exact. On the real clean photo it reads
4.83 even though no noise was added: fine texture (water, foliage)
leaks into the estimate. The router's mild-noise threshold (6.0) was
set above that leakage for this reason. Treat the sigma as a
ranking signal on real photos, not a lab measurement.

## Detail metric check

The detail score uses FFT energy beyond 40% of Nyquist. The same
photo upscaled back from 240x180 to 960x720 scores 8/100 against
95/100 for the clean original, so the metric does catch the
"big file, no real detail" case, which plain dimensions cannot.

## Face detection check

YuNet (model in models/, from the OpenCV Zoo) was tried on two
portrait photos from Lorem Picsum: id 64 gave 1 face and the router
added the facial enhancement route; id 65 gave 0 faces (the face is
small and turned in that shot). Face detection is only a routing
hint here, not a quality score.

## App check

The Streamlit app was run two ways: through Streamlit's AppTest
harness (0 exceptions, all four score metrics rendered) and as a
real server (`streamlit run app.py`), which answered HTTP 200.

## Honest limitations

- Thresholds are calibrated on one photo and its damaged copies.
  They behave sensibly, but a different kind of photo (a portrait
  with shallow depth of field, a night shot) can fool individual
  metrics. The Laplacian variance in particular cannot tell
  intentional background blur from a blurry photo.
- Noise inflates the Laplacian variance (the noisy sample reads
  5687, higher than clean), so sharpness is only judged after the
  denoise step in the routing order. Judging sharpness of a noisy
  image directly would be wrong, and the router avoids it by
  ordering, not by correcting the number.
- The Immerkaer estimate over-reads on heavily textured scenes, as
  noted above.
- Blockiness only sees JPEG-style 8x8 blocking. PNG screenshots,
  WebP artifacts and over-sharpening halos are out of scope.
- The router is rules, not a learned model. That is deliberate for
  a demo (every decision can be explained in one sentence), but a
  production version would want the thresholds fitted per category
  of photo, or a small classifier on top of these features.
- The lowres sample is a genuinely small file. A screenshot that
  was upscaled before sharing is caught by the detail score
  instead, which the test suite covers separately.
