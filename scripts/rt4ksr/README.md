# RT4KSR

[RT4KSR](https://github.com/eduardzamfir/RT4KSR) — "Towards Real-Time 4K Image
Super-Resolution", Zamfir / Conde / Timofte, NTIRE Real-Time 4K SR workshop at
CVPRW 2023. Apache-2.0 code.

**Verdict: measured, works, fast enough to play live, and not adopted.** On this
project's five test clips it never beats the modes already in the app. The full
comparison is in `docs/GENERAL-VIDEO-UPSCALING.md`; the short version is below.

## What it is

A plain per-frame convolution stack. After reparameterization (upstream's
intended deployment form) the whole network is **14 convolutions, 44,521
parameters, ~12 GFLOP/frame at 1080p output**:

```
PixelUnshuffle(2)                     854x480 -> 427x240, 3 -> 12 channels
conv3x3 12 -> 24
[ LayerNorm2d -> conv3x3 24 -> 24 -> GELU ] x 4
LayerNorm2d -> conv3x3 24 -> 24
conv3x3 24 -> 48 -> PixelShuffle(4)   -> 1708x960, 3 channels
```

Two scales ship: **x2 and x3**. There is no x4 model, so a 4× upscale here is two
chained x2 passes — the same convention the app already uses for Anime4K — and
the tables say so rather than implying a single 4× network.

The LayerNorm is per-pixel across channels (no spatial reduction), and the
activation is exact `erf` GELU. Both matter for a Metal port; neither is hard.

## Speed: not the problem

`python3 scripts/rt4ksr/bench.py` on this machine (Apple M5 Pro), after folding:

| input | output | GFLOP | ms/frame | fps | fraction of a 30fps frame budget |
|---|---|---|---|---|---|
| 480×270 | 960×540 | 3.04 | 3.3 | 305 | 0.10× |
| 640×360 | 1280×720 | 5.41 | 2.5 | 399 | 0.08× |
| 854×480 | 1708×960 | 9.62 | 5.0 | 201 | 0.15× |
| 960×540 | 1920×1080 | 12.17 | 6.0 | 166 | 0.18× |

At a few milliseconds a frame the MPS timer is noisy: repeated runs of the same
case move by around 15% (540p→1080p measured between 6.0 and 7.1 ms across runs),
so read these as "a few milliseconds", not as three significant figures. The CPU
column of `bench.py` is ~20× slower and is not the interesting one.

For scale: Real-ESRGAN `general-x4v3` measures **84–153 ms/frame** (640×360 →
854×480 input) on the same machine, and `animevideov3` — the same architecture with
half the body — is **22.6 ms/frame at 480×270→1920×1080 = 44 fps**. *(An earlier
version of this line said 278 ms/frame at 0.12× realtime, from a single-layer
extrapolation that has since been retracted; see
`results/general/realesrgan_feasibility.txt` and
`results/general/realesrgan_bench.txt`.)* This model is still one to two orders of
magnitude cheaper than the general model and would be trivial to run live.

## Quality: why it was not adopted

Across `results/general/*`, RT4KSR never wins a column on any of the five clips:

| setup | best existing mode | RT4KSR | note |
|---|---|---|---|
| live action 540p→1080p | cnnDenoise 33.02 dB | 32.84 dB | below bilinear (32.95) |
| live action 270p→1080p | cnn 29.90 dB | **29.90 dB** | tie on PSNR; SSIM below bilinear; much softer (0.57× vs 0.77× acutance) |
| anime 540p→1080p | cnnDenoise 39.33 dB | 38.68 dB | below bilinear (39.42) |
| anime 270p→1080p | cnn 35.57 dB | 35.29 dB | SSIM below cnn |
| 1080p→4K (its home turf) | cnnDenoise 33.53 dB | 33.52 dB | 0.01 dB behind; SSIM below bilinear; less stable (1.229× vs 1.136×) |

Its acutance is 0.57–0.85× of the reference against the Anime4K CNN's 0.77–0.90×,
so it is consistently the softer option, and on the moving clips it is marginally
less temporally stable than the CNN. At 3–5× zoom on shirt and poster text it
looks like a cleaned-up bilinear — no detail the other modes do not already have.

This is what the model is designed to do. The repo "generates bicubically
downscaled LR images online" and tests on standard SR benchmarks; its published
gain over bicubic is +0.27 dB (33.92 → 34.19 dB on NTIRE23-RTSR x2). It has no
deblocking or denoise stage and was never trained on compressed video, which is
exactly what these SD test clips are.

## Files

| file | what it is |
|---|---|
| `rt4ksr.py` | the model: training-shaped class (so checkpoints load `strict=True`), the fold, and a frame-at-a-time upscaler. No `torchvision`, no mmcv. |
| `fetch_weights.py` | downloads the two checkpoints to `~/.cache/macvideoupscaler`, pinned by size **and** sha256. |
| `render.py` | offline render, ffmpeg decode → model → encode. Lossless by default. |
| `bench.py` | throughput per device and input size; counts the FLOPs too. |
| `verify_against_upstream.py` | proves the three claims below against a clone of upstream, on the real weights. |

## Three deliberate changes from upstream

Upstream's `test.py` **cannot run on this machine at all**, so the class bodies
are vendored with three fixes. All three are verified, not assumed:

1. **`reparameterize()` is reimplemented.** Upstream's allocates
   `torch.ones(...).cuda()` unconditionally and wraps everything in
   `DataParallel`, and `test()` calls `reparameterize()` with no arguments — the
   released inference path is broken as written. Ours does the same arithmetic
   (fold 1×1 → 3×3 → 1×1, then add the two identities) on any device.
2. **`torchvision.transforms.GaussianBlur` is dropped.** The high-frequency branch
   that uses it is **dead** in the released config: `forget` is hardcoded `False`,
   so `hf_feats` is computed and then never read. Dropping it changes the output
   by **exactly zero** (verified bit-for-bit), and saves 17% of the arithmetic.
3. **The `torchvision` dependency goes with it**, leaving `numpy` + `torch`.

## Verification

`python3 scripts/rt4ksr/verify_against_upstream.py --upstream /path/to/RT4KSR`
clones-and-compares against the real weights. Actual output:

```
x2  output (1, 3, 192, 256)   params 44,521
  training class vs upstream         OK    max|d|=0.000e+00
  folded vs training class           OK    max|d|=4.113e-06
  dead branch dropped vs folded      OK    max|d|=0.000e+00
x3  output (1, 3, 288, 384)   params 57,541
  training class vs upstream         OK    max|d|=0.000e+00
  folded vs training class           OK    max|d|=6.557e-06
  dead branch dropped vs folded      OK    max|d|=0.000e+00
```

The 4e-06 is float32 rounding accumulated by the fold (roughly 1/128 of an 8-bit
level), which is why the fold tolerance is 1e-5 rather than exact. The other two
are exact zero, not "small".

## Usage

```bash
python3 scripts/rt4ksr/fetch_weights.py
python3 scripts/rt4ksr/render.py --input in.mp4 --output out.mkv --scale 2   # lossless
python3 scripts/rt4ksr/render.py --input in.mp4 --output out.mp4 --scale 4 --crf 16
```

`--scale 4` chains two x2 passes. Input dimensions must be even (the network
unshuffles by 2).

## Licences

- **Code** (this directory is a reimplementation, but the arithmetic is
  upstream's): upstream is **Apache-2.0**.
- **Weights**: this repository does not vendor them; `fetch_weights.py` downloads
  them from upstream's own GitHub at pinned hashes. Upstream ships them in an
  Apache-2.0 repository without a separate weights licence.

Both are permissive and would not block shipping the model.

## Appendix: PiperSR, and why it is not an alternative

[PiperSR](https://github.com/ModelPiper/PiperSR) was proposed as a real-time
ANE-native alternative. It was checked rather than taken at its word, and the
published claims do not survive:

- **The GitHub repo cannot produce a working model.** `.gitignore` excludes
  `*.mlmodel`, which matches `Data/com.apple.CoreML/model.mlmodel` — the model
  *specification*. Cloning yields weights with no architecture, so the README's
  "Clone and run — no separate download needed" is not true. (The PyPI wheel does
  contain it, which is how it was tested here.)
- **Its own code cannot run.** `benchmark.py`, `inference.py` and `cli.py` all
  pass `compute_units=ct.ComputeUnit.CPU_AND_NEURAL_ENGINE`, which is not a
  coremltools enum member (it is `CPU_AND_NE`; the author appears to have used the
  Swift spelling `MLComputeUnits.cpuAndNeuralEngine`). They also pass a
  `{"input": array}` CHW float32 array to a model whose single input is
  `input_image`, an **image** type that coremltools requires to be a
  `PIL.Image`. So the published benchmarks cannot have been produced by the
  published benchmark script.
- **"Any resolution (CoreML flexible input shape)" is false.** The shipped spec
  declares a fixed 128×128 input and 256×256 output with `constant` dimensions.
  Feeding 320×240 fails with *"Image size 320 x 240 not in allowed set of image
  sizes"*. Everything must be tiled 128×128 with no halo, which is a seam
  problem for super-resolution.
- **The README contradicts its own model card.** The card leaves Set14, BSD100
  and Urban100 as "—" and lists only M2 Max; the README fills in all four
  datasets and a seven-row hardware table. The README's "M2 Max, full-frame 360p,
  44.4 FPS, 22.5 ms" row is numerically identical to its "M2, tiled 128×128
  (dynamic weights), 44.5 FPS, 22.5 ms" row, and `benchmark.py` defaults to
  320×240 — i.e. the "360p full-frame" figure looks like a relabelled 128×128
  tile measurement, which is also the only size the model can accept.
- **Its PSNR protocol is not comparable to the models it compares against.** The
  card claims 37.54 dB on Set5 x2 and the README frames that as "0.5 dB below
  SPAN-S (38.06)". But `compute_psnr` in `benchmark.py` measures **RGB with no
  border crop**, while every published figure in its comparison table is
  Y-channel with a border crop. RGB PSNR runs roughly 0.5–1.5 dB *higher*, so the
  claimed comparison is not established by its own harness.
- An "M1 Pro ~25–30 FPS" figure does not appear anywhere in the repo; the card
  explicitly leaves M1 unmeasured.

What is real: the architecture matches its description (14 convs, 6 SiLU, 7 adds,
1 pixel_shuffle — exactly head + 6×(2 conv) + tail), the parameter count is
consistent with it, and a 128×128 tile runs in **1.29 ms** on an M5 Pro. At that
per-tile cost a 960×540 frame costs 40 tiles ≈ 51 ms ≈ 19 fps, so tiling is
workable — but slower than RT4KSR, with seams to solve, on a model whose
quality claims have not been substantiated.

RT4KSR is the better bet of the two: peer-reviewed, an order of magnitude
cheaper, no tiling, and independently reproducible.
