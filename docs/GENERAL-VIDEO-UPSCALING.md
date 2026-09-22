# General (live-action) video upscaling — what was added, and what it measures

Asked for as "add the best one for general video upscaling to this app", against
a shortlist of MetalFX Spatial, Real-ESRGAN `general-x4v3`, Real-ESRGAN `x2plus`,
Anime4K and RealBasicVSR.

> **A wider shortlist followed** (SeedVR2, Video2X, FlashVSR, Real-ESRGAN
> AnimeVideo-v3, Real-CUGAN). Its classification — three of the five are not live
> modes at all, and the one that is was measured here — is in
> **[OFFLINE-UPSCALING.md](OFFLINE-UPSCALING.md)**.

**Added: MetalFX Spatial** (`--mode metalfx`, *MetalFX Spatial* in the Enhancement
menu). It is the only one of the five that can run inside this player.

**But the measurements do not make it an upgrade over what was already here.** It
is the sharpest of the four available modes and by far the cheapest, and it is
the *least* faithful and the *most* temporally active. On animation it is beaten
outright by the Anime4K CNN that was already shipped. Details below, including
the two measurements that decided it and the ones that qualify it.

## Why the other four are not options

> **Correction (2026-09-15).** The first version of this section called the whole
> Real-ESRGAN family 10–100× too slow to play, from a **projection** built on one
> hand-written layer benchmark. That was wrong by ~2.6×, and it was wrong in the
> direction that mattered: it ruled out a model that measures real-time. The
> measured numbers are below; `results/general/realesrgan_feasibility.txt` records
> both the projection and the correction.

The app is a **live player**: the contract is real-time playback from a 2–3 frame
buffer. A model that cannot hold ~30 fps is not a slower version of this feature,
it is a different feature (offline resampling), which this app does not have.

**`realesr-animevideov3` — the anime-video model — is fast enough, and is the one
member of the family that is.** `SRVGGNetCompact(num_feat=64, num_conv=16)`:
18 convolutions, 621,424 params, 0.62 M MACs per input pixel. Measured end to end
through PyTorch/MPS (`scripts/realesrgan/bench.py`, whole network, random frames):

| input | output | ms/frame | fps | of a 30 fps budget |
|---|---|---|---|---|
| 480×270 | 1920×1080 | **22.59** | **44.26** | **0.68×** |
| 640×360 | 2560×1440 | 41.39 | 24.16 | 1.24× |
| 854×480 | 3416×1920 | 75.00 | 13.33 | 2.25× |
| 960×540 | 3840×2160 | 89.13 | 11.22 | 2.67× |

That is already under a 30 fps frame time for the app's 270p→1080p case, before
any Metal port — the app's own 4× SD-only rule means that case is exactly the one
it would be used for. Achieved throughput is ~7.2 TFLOP/s.

`real-esr-general-x4v3` is the same architecture with twice the body:
`SRVGGNetCompact(num_feat=64, num_conv=32)`, 34 convolutions, **1.21 M MACs per
input pixel**. Measured the same way:

| input | output | ms/frame | fps | of a 30 fps budget |
|---|---|---|---|---|
| 480×270 | 1920×1080 | 47.58 | 21.02 | 1.43× |
| 640×360 | 2560×1440 | 84.12 | 11.89 | 2.52× |
| 854×480 | 3416×1920 | 152.54 | 6.56 | 4.58× |
| 960×540 | 3840×2160 | 170.46 | 5.87 | 5.11× |

So `general-x4v3` is near real time at 270p (21 fps) and out of budget beyond it.
The earlier projection put it at 277.7 ms/frame for a 640×480 input; the
measurement brackets that size at 84–152 ms, i.e. ~100–110 ms interpolated, and
the projection was ~2.6× too pessimistic. The reason is that the layer measured
(`scripts/bench_realesrgan_conv.swift`, 2.67 TFLOP/s achieved) was timed alone, so
its per-dispatch cost was charged once against a single layer's arithmetic, and
that rate was then scaled to a whole network that the framework can run at a
higher rate than one hand-written kernel reaches.

Where the family still loses:

- **CPU is genuinely unplayable** — 495 ms/frame at 270p and 2176 ms/frame at
  540p, 20× slower than MPS. Every "too slow" claim here is true of the CPU path
  and false of the GPU path, and none of it applies to a Metal port.
- **`x2plus`** is an RRDB network, roughly an order of magnitude heavier again
  than `general-x4v3`. Never measured; excluded by structure, not by benchmark.
- **RealBasicVSR** is rejected on structure rather than speed: it needs a sequence
  window and optical flow, i.e. a different pipeline, not a different scaler.
- **Anime4K** was already in the app and remains the animation choice.

What the correction does *not* do is retroactively justify animevideov3 on
quality: it is ported now, but only because the port turned out 3.1× cheaper than
the eager measurement, not because it got better on animation. See §"The
AnimeVideo-v3 question" below for the quality evidence and the port decision, and
`scripts/realesrgan/README.md` for the memory measurements taken against the
project's 2 GB model budget.

That leaves MetalFX Spatial, which is real-time with three orders of magnitude to
spare:

| output | MetalFX Spatial | Anime4K CNN |
|---|---|---|
| 640×480 → 1280×960 | **0.33 ms** | 0.95 ms |
| 640×480 → 2560×1920 (4×) | **0.67 ms** | ~2× the 2× pass |
| 1280×720 → 2560×1440 | **0.55 ms** | — |
| 1920×1080 → 3840×2160 | **0.99 ms** | 3.81 ms |

In-app it holds real time at every setting tested (frames delivered / clock, on
the 300-frame clips): 540p→1080p **0.997×**, 270p→1080p at 4× **0.992×**,
1080p→2160p **1.007×**, all 300/300 frames, state `finished`.

## What it is, and what it is not

MetalFX Spatial is an edge-adaptive *spatial filter*, not a generative network.
It cannot invent texture that is not in the source, and it has no temporal
feedback, so it cannot accumulate a ghost or a trail across frames and there is
no "waxy face" failure mode. That is a statement about its structure, not a
measured clean bill of health on motion: it is a per-frame filter, so anything it
does to the picture it does independently every frame, and the sharpening it
applies does amplify frame-to-frame change (see the flicker numbers below, and
the caveat about that metric). The spatial scaler also takes the output geometry
directly, so the app's 4× is **one** pass rather than two chained 2× passes —
measured better than chaining on the 480×270 → 1080p case (acutance 1.94× vs
2.18× relative to the truth, PSNR **29.76 vs 28.80 dB**), so the sharpening is
applied once instead of twice.

What it does have is a fixed and quite aggressive sharpening stage. That is the
whole story of the numbers below.

There is no quality knob: `MTLFXSpatialScaler` exposes only geometry, formats and
`colorProcessingMode`. That one was tested both ways on the real decoded frame —
`perceptual` (correct for display-ready gamma-encoded video, and what the app
uses) overshoots less than `linear` (acutance 1.69× vs 1.86×, PSNR 32.21 vs 32.07
dB) — so the shipped setting is the better of the two available.

The temporal scaler, `MTLFXTemporalScaler`, would be the higher-quality MetalFX
and the better answer on motion stability, but it requires per-pixel motion
vectors and depth, which an ffmpeg decode does not provide. It is not reachable
from this pipeline without a new source of motion vectors.

## Measurement method

`scripts/eval_general_upscaler.py`. It drives `playercheck`, so what is scored is
the pixel buffer the player actually handed to the display layer — no model is
re-implemented for the test.

The standard super-resolution proxy: take a high-resolution reference, downscale
and re-compress it to make the source, upscale the source back to the reference's
geometry with each method, and compare against the reference. Frame *N* of the
source and of the reference are the same instant, so a 2× (or 4×) output lands
exactly on the reference's grid.

| metric | what it says |
|---|---|
| `psnr`, `ssim` | fidelity to the reference. Higher is more faithful; a method that invents plausible detail is penalised by construction. |
| `acutance` | mean \|Laplacian\| of luma as a fraction of the reference's. 1.00 = as sharp as the truth. **Above 1 means the output has edge energy the original does not have** — ringing, halos, invented edges. |
| `smooth_rmse` | RMSE over the reference's flattest 40% of pixels: where blocking and banding show. Lower = better compression cleanup. |
| `flicker` | mean \|consecutive-frame difference\| over 8–12 consecutive frames, in absolute 8-bit levels. The harness also reports that as a ratio to the bypass (bilinear) run and to the source's own frame-to-frame difference. **All three are needed** — see the caveat below. |

Content: `test-media/liveaction/` (live action, both a soft 1080p original and a
sharp 4K one) and `test-media/general/` (anime, derived from the existing
1080p Naruto alignment). Live-action material was added for this work precisely
because the repo's existing clips are animation.

How much each clip actually moves, as mean \|frame-to-frame difference\|:

| clip | source flicker | what it means for the metrics |
|---|---|---|
| live action 540p / 270p | 17.9 / 17.8 | handheld, real motion. This is where temporal numbers are trustworthy. |
| anime 540p / 270p | 0.0015 / 0.0006 | **effectively frozen** — an animation timeline holding frames. |
| 1080p → 4K (metro) | 0.45 | a locked-off shot; nearly static. |

**Caveat one — the ratio needs a denominator.** `flicker` ratios divide one
frame-difference by another. On the anime clips the bypass differs from frame to
frame by 0.0015 levels out of 255, i.e. essentially nothing, so any model that
does not reproduce the previous frame bit-for-bit produces an enormous ratio for
no visible reason. Early runs of this harness reported "RealBasicVSR flickers
47×" and "RT4KSR flickers 0.02×" on those clips; both are arithmetic on a
near-zero denominator. The harness now prints the absolute figure first and
labels the ratio unreliable when the bypass is under 1.0 level. Read the absolute
number, and read it against the source's own flicker.

**Caveat two — sharper legitimately reads as less stable.** An image with more
high-frequency energy has larger frame-to-frame differences, so a sharper output
will look less stable whether or not it is crawling. Neither ratio can separate
"sharper" from "crawling", and this is the one metric here that should never be
read as a verdict on its own.

## Results

The `flicker` column is `absolute (ratio to bypass)`. On the anime clips the
ratio is meaningless — see caveat one above — so read the absolute number there.

The **RealBasicVSR** rows in the two 270p tables were measured before that model
was removed from the project (15 Sept). They are kept as the record of why it was
rejected, but they can no longer be re-rendered from this repository: its runner
and its 210 MB of weights are gone. Every other row is reproducible.

### Live action, soft source — 960×540 → 1920×1080 (2×)

| mode | PSNR | SSIM | acutance | smooth_rmse | flicker |
|---|---|---|---|---|---|
| Bypass (bilinear) | 32.95 dB | 0.9162 | 0.54× | 3.72 | 17.57 (1.000×) |
| Anime4K CNN | 32.96 dB | 0.9141 | 0.77× | 3.57 | 17.94 (1.021×) |
| Anime4K CNN + denoise | **33.02 dB** | 0.9158 | 0.73× | **3.54** | 17.91 (1.019×) |
| **MetalFX Spatial** | 32.44 dB | 0.8932 | **1.74×** | 3.81 | 18.47 (1.051×) |
| RT4KSR 2× | 32.84 dB | 0.9075 | 0.75× | 3.66 | 18.01 (1.025×) |

### Live action, soft source — 480×270 → 1920×1080 (4×)

| mode | PSNR | SSIM | acutance | smooth_rmse | flicker |
|---|---|---|---|---|---|
| Bypass | 29.28 dB | 0.8712 | 0.39× | 5.25 | 16.49 (1.000×) |
| Anime4K CNN | **29.90 dB** | **0.8719** | 0.77× | **4.77** | 17.93 (1.088×) |
| **MetalFX Spatial** | 29.54 dB | 0.8485 | **1.99×** | 5.04 | 18.08 (1.097×) |
| RealBasicVSR x4 | 27.44 dB | 0.8285 | 1.44× | 5.86 | 18.92 (1.148×) |
| RT4KSR 2× (chained) | **29.90 dB** | 0.8612 | 0.57× | 4.96 | 18.10 (1.098×) |

### Live action, sharp source — 1920×1080 → 3840×2160 (2×)

| mode | PSNR | SSIM | acutance | smooth_rmse | flicker |
|---|---|---|---|---|---|
| Bypass | 33.03 dB | 0.9330 | 0.56× | 2.62 | 0.412 (1.000×) |
| Anime4K CNN | 33.50 dB | 0.9377 | 0.90× | 2.60 | 0.479 (1.164×) |
| Anime4K CNN + denoise | **33.53 dB** | **0.9388** | 0.87× | **2.58** | 0.468 (1.136×) |
| **MetalFX Spatial** | 32.76 dB | 0.9190 | **1.54×** | 2.83 | 0.645 (1.566×) |
| RT4KSR 2× | 33.52 dB | 0.9326 | 0.82× | 2.69 | 0.506 (1.229×) |

### Anime — 960×540 → 1920×1080 (2×)

| mode | PSNR | SSIM | acutance | smooth_rmse | flicker |
|---|---|---|---|---|---|
| Bypass | **39.42 dB** | **0.9732** | 0.65× | 0.89 | 0.0015 (—) |
| Anime4K CNN | 39.22 dB | 0.9690 | 1.29× | 0.97 | 0.0017 (—) |
| Anime4K CNN + denoise | 39.33 dB | 0.9730 | 0.93× | **0.89** | 0.0017 (—) |
| **MetalFX Spatial** | 37.51 dB | 0.9552 | **2.18×** | 1.04 | 0.0026 (—) |
| RT4KSR 2× | 38.68 dB | 0.9655 | 1.02× | 0.96 | 0.0018 (—) |

### Anime — 480×270 → 1920×1080 (4×)

| mode | PSNR | SSIM | acutance | smooth_rmse | flicker |
|---|---|---|---|---|---|
| Bypass | 34.04 dB | 0.9472 | 0.48× | 1.29 | 0.0270 (—) |
| Anime4K CNN | **35.57 dB** | **0.9477** | 1.34× | 1.38 | 0.0277 (—) |
| **MetalFX Spatial** | 33.85 dB | 0.9288 | **2.56×** | 1.45 | 0.0458 (—) |
| RealBasicVSR x4 | 27.80 dB | 0.8748 | 2.93× | 2.09 | 0.716 (—) |
| RT4KSR 2× (chained) | 35.29 dB | 0.9368 | 0.85× | 1.46 | 0.0007 (—) |

Raw numbers: `results/general/*/metrics.json`; grids: the `grid_*.png` beside
them. `scripts/eval_all_general.sh` regenerates all five tables from scratch.

The five live modes are measured by driving the player, which dumps frames under
its own pacing: repeating a run moves those figures by about ±0.03 dB (the anime
270p run read 34.08/35.54 before and 34.04/35.57 after, on unchanged sources).
Differences of that size are not real. The offline models render
deterministically and repeat exactly.

## Reading the numbers

**MetalFX Spatial is the sharpest and the least faithful, on every clip.** Its
acutance is 1.54–2.18× the *ground truth* — it does not merely resolve detail the
source lost, it adds edge energy the original never had. That is what a halo is,
and it is visible: on the soft live-action clip's corrugated roof it exaggerates
the corrugation contrast well past the reference and stair-steps the diagonal
ridges (`grid_corrugated_roof.png`). Its PSNR and SSIM are the lowest of the four
modes in all three live-action runs, and on anime it scores **worse than plain
bilinear** (37.51 vs 39.42 dB). The request's expectation that MetalFX would be
the *conservative* option — "looks clean but not dramatically sharper" — is not
what is measured here; the opposite is.

**It looks good where the source is genuinely sharp.** On the 4K-sourced clip it
is competitive by eye with the Anime4K CNN on poster text and track hardware
(`grid_poster_text.png`, `grid_track_hardware.png`), where the ~0.7 dB PSNR
difference is not a visible one. This is the "clean HD" case, and it is where the
mode earns its place.

**Anime4K stays the better anime model, and it is not close.** Adding MetalFX's
sharpening to cel animation over-sharpens flat regions and inked outlines: 37.51
dB against the CNN's 39.22 and bilinear's 39.42, with 2.18× acutance and the
worst flicker of the set. The Anime4K CNN + denoise variant is the best-scoring
mode on anime by SSIM and `smooth_rmse`.

**Neither sharpening mode reaches 1.00× acutance.** Anime4K CNN leaves the
picture *soft* (0.73–0.90×) and MetalFX leaves it *hard* (1.54–2.18×). The truth
is in between and neither hits it; the errors are just in opposite directions,
and undershoot is the safer one — it does not draw edges that are not there.

**Compression cleanup: no mode is doing much.** `smooth_rmse` moves only 0.04 dB
between the best and worst mode on the sharp clip, and the Anime4K denoise
weights are the only ones aimed at it at all. MetalFX is a spatial filter with no
denoise stage and measured the worst of the four in every run.

## RT4KSR — measured, and not adopted as a live mode

[RT4KSR](https://github.com/eduardzamfir/RT4KSR) (Zamfir, Conde, Timofte; NTIRE
Real-Time 4K SR workshop, CVPRW 2023) is a genuinely appealing model for this
app's problem: 44,521 parameters, 12 GFLOP/frame at 1080p output, Apache-2.0 code,
and no recurrence or optical flow, so it is a plain per-frame convolution stack.

**It is fast enough, comfortably.** After the reparameterization the authors
intend — folding each 1×1 → 3×3 → 1×1 block into a single 3×3 convolution, which
upstream's `test.py` cannot actually do here because it hardcodes `.cuda()` — the
whole network is 14 convolutions. `scripts/rt4ksr/bench.py` on this machine:

| input | output | GFLOP | ms/frame | fps | × realtime @30fps |
|---|---|---|---|---|---|
| 480×270 | 960×540 | 3.04 | 3.3 | 305 | 0.10× |
| 640×360 | 1280×720 | 5.41 | 2.5 | 399 | 0.08× |
| 854×480 | 1708×960 | 9.62 | 5.0 | 201 | 0.15× |
| 960×540 | 1920×1080 | 12.17 | 6.0 | 166 | 0.18× |

So 540p → 1080p costs 18% of a 30 fps frame budget through PyTorch, and a hand
written Metal port would be cheaper again. Nothing about this model is too slow.
(The timings move by ~15% run to run at these durations — 540p→1080p measured
between 6.0 and 7.1 ms — so they are "a few milliseconds", not precise figures.)

**It just does not beat what is already here.** In all five tables above, across
five clips and two upscaling factors, RT4KSR never wins a column:

- It ties the best PSNR twice (29.90 dB at 270p→1080p with Anime4K CNN; 33.52 dB
  at 1080p→4K, 0.01 dB behind CNN + denoise) and loses the other three.
- It is **softer than Anime4K CNN on every clip** — 0.57–0.85× acutance against
  the CNN's 0.77–0.90× — so on cel animation and on poster text it resolves less.
- Its SSIM is below plain bilinear on three of the five setups.
- On the well-conditioned (moving) clips it is marginally *less* temporally stable
  than the CNN: 1.025× vs 1.021× at 540p, 1.098× vs 1.088× at 270p.

That matches what the model is: an efficient reconstruction network trained on
**bicubically downscaled clean images** (the repo generates its LR online and
tests on standard SR benchmarks), evaluated on a benchmark whose bicubic baseline
is 33.92 dB where RT4KSR scores 34.19 dB — a 0.27 dB gain over bilinear on clean
content. It has no deblocking or denoise stage, which is exactly what these
compressed SD sources need, and it is where the Anime4K denoise variant and the
CNN's stronger sharpening win.

**At 3–5× zoom the two grids tell the same story as the table.**
`grid_shirt_text_with_rt4ksr.png` (Japanese text on a shirt, soft live-action
source) and `grid_poster_text_with_rt4ksr.png` (Spanish poster text, sharp 1080p
source; note the 4K reference, so its crop is in 4K coordinates): Anime4K CNN and
CNN + denoise resolve visibly more letter definition, MetalFX over-sharpens with
halos, and RT4KSR sits just above plain bilinear — a cleaner, slightly sharper
bilinear, with no detail the other modes do not already have.

**Verdict: not adopted.** A live mode that measures at or below the ones already
present would be a menu entry, not an improvement, and it would cost a Metal port
plus a weight-baking step to maintain. (It was written when there were four live
modes; there are five now, and the argument is unchanged — it still wins no
column.) The model itself lives in
`scripts/rt4ksr/` and is verified against upstream, so the decision is cheap to
revisit — see `scripts/rt4ksr/README.md`. The two things that would change it are
a 4× model (this one is 2× and 3× only, so 4× here is two chained passes) and a
version trained on compressed video rather than clean bicubic downscales.

## The AnimeVideo-v3 question

`realesr-animevideov3` is the Real-ESRGAN model recommended for anime video, and
it is the one model here that is both **fast enough to play live** (22.59 ms at
270p→1080p, above) and **a genuine contender on picture quality**. It was scored
in the two 4× setups alongside the existing modes — and it is now one of them,
which is the subject of the port decision at the end of this section.

| | anime 270p→1080p | | | | live action 270p→1080p | | | |
|---|---|---|---|---|---|---|---|---|
| mode | PSNR | SSIM | acutance | smooth | PSNR | SSIM | acutance | smooth |
| reference | — | — | 1.000× | — | — | — | 1.000× | — |
| bypass | 34.08 | 0.9474 | 0.476× | 1.288 | 29.28 | 0.8712 | 0.390× | 5.247 |
| cnn | **35.57** | **0.9478** | 1.340× | 1.374 | **29.90** | **0.8719** | 0.768× | 4.770 |
| metalfx | 33.84 | 0.9289 | 2.558× | 1.459 | 29.54 | 0.8485 | 1.995× | 5.041 |
| rt4ksr | 35.29 | 0.9368 | 0.850× | 1.464 | 29.90 | 0.8612 | 0.567× | 4.960 |
| **realesrgan (live)** | 28.36 | 0.8959 | 1.972× | 1.912 | 27.79 | 0.8601 | **0.971×** | 5.364 |
| **realesrgan-general (live)** | 26.14 | 0.8866 | 2.399× | 1.748 | 26.35 | 0.8503 | 1.044× | 5.387 |
| general-x4v3 (offline render) | 26.18 | 0.8870 | 2.400× | 1.734 | 26.36 | 0.8504 | 1.044× | 5.370 |

The last two rows are the same checkpoint measured two ways, and they agree to
**0.04 dB** (26.35 against 26.36 live action, 26.14 against 26.18 anime). That
agreement is the point of keeping both: it is the cross-check that the live mode
runs the checkpoint rather than something adjacent to it, and it is what makes the
live and offline columns below comparable at all.

`reference_acutance` is 1.708 on the anime clip and 3.290 on the live-action one,
so the acutance column is expressed as a ratio to the truth in the table above.

**On live action it gets the sharpness right and everything else wrong.** Its
acutance is **0.971×** the ground truth — the only mode within 3% of correct —
against the CNN's 0.768× (under-sharp) and MetalFX's 1.995× (over-sharp with
halos). That is visible:
`results/general/liveaction_270p_to_1080p_4x/grid_shirt_text_with_animevideov3.png`
shows Japanese shirt text at 3× where animevideov3 is the most legible of the six
panels — the katakana strokes separate cleanly and `NALS` resolves to letter
shapes, where the CNN leaves them merged and MetalFX smears them. It pays for
that with the **worst fidelity of the modes that shipped before it** — PSNR
**27.79** and SSIM **0.8601**, below plain bilinear's 29.28/0.8712 — the worst
smooth-region error (5.364), and the most added temporal activity of those modes
(1.061× the source's own frame-to-frame change, against the CNN's 1.008×).
`realesrgan-general` below is worse on all of those; the distinction is that it
was added as a comparison target, and this mode is a recommendation.

**On anime it simply loses.** `grid_anime_with_animevideov3.png`, same clip
family at the same crop and zoom, shows the CNN closer to the reference and
animevideov3 posterising the smooth skin gradient — the flat-region error of
1.912 against the CNN's 1.374 and bilinear's 1.288 quantifies exactly what the
grid shows. Acutance 1.972× against a 1.000× truth means it is inventing detail
the source does not have, which is what a GAN is for and what an anime cel with
large flat areas punishes.

So the honest summary is narrower than "the best model tested": **animevideov3 is
the best mode here at restoring fine photographic detail to SD live action, and
it is the wrong choice for cel animation**, where the Anime4K CNN already shipped
beats it on every fidelity measure and looks better. The comparison view settles
it per clip in one keystroke, which is the point of having both.

**Should it be ported?** The port is cheap — `SRVGGNetCompact` is 18 convolutions
of 3×3 with PReLU, no attention, no recurrence, no optical flow, and unlike
Anime4K it is a single 2.4 MB checkpoint rather than a chain of passes. Against
that, it would be a *fifth* live mode whose win is confined to one kind of
content, and the app already covers that content with MetalFX (sharper) and CNN
(cleaner). The recommendation was **no port now**; the model, the renderer, the
bench and the scored frames are all in `scripts/realesrgan/` and `results/general/`
so the decision costs one afternoon to reverse if the blend of material changes
toward live action.

**It has since been reversed, on cost.** The mode is now
`ScalerKind.realESRGAN` — x4 only, SD-only — and the reason is worth recording
because it went against the obvious port. It does **not** use a hand-written Metal
kernel: measuring this GPU's primitives showed a kernel could not win. fp16 tops
out at 8.4 TFLOP/s and `simdgroup_matrix` is *slower* still (4.2) against 160.7
GFLOP of work, so even a perfect kernel lands at ~19 ms while MPS already achieves
~7.1 TFLOP/s effective. The win is fusion, not arithmetic — eager PyTorch
dispatches conv, bias and PReLU separately and writes 64 channels of intermediate
to memory between each pair, an MPSGraph executable does not — and it is **3.1×**:
**7.33 ms** at 270p→1080p against 22.59 ms for the same model on the same GPU,
i.e. 0.22× of a 30 fps budget instead of 0.68×.

**A user then reported that the shipped mode "looks bad on real video that's not
anime", and that is correct — the box above is a synthetic-downscale result.** Every
scored live-action clip in this project is a lanczos downscale of a known reference,
which is close to invertible; genuinely degraded video is not. Checked on native 480p
game footage, `animevideov3` replaces fine structure with smooth rounded blobs and
expands bright highlights (particle effects become balloons), while adding no
temporal instability and almost no tone shift — it is hallucination on out-of-domain
material, and the CNN and MetalFX keep the texture. So its live niche is **SD anime**,
and on non-anime video prefer the CNN or MetalFX. Evidence and numbers:
`scripts/realesrgan/README.md`, `results/general/yt_mhw_480p/`.

**The other Real-ESRGAN checkpoint was scored too, and it is offered as a mode
anyway — to be judged, not as a recommendation.** `general-x4v3` runs through the
same port at 13.76 ms (0.41× of a frame budget — real time at 270p, past it from
854×480 up, where it holds **0.63× real time** in the player against
`animevideov3`'s 0.92× in the same run), and it had never been measured on picture
before. It turns out to have **the lowest PSNR and SSIM of every mode tested on
both clips**, and to lose to `animevideov3` on both — including on live action,
which is the general model's whole purpose (26.36 dB against 27.79). By the numbers
it is slower *and* worse.

It is nonetheless `ScalerKind.realESRGANGeneral` / `--mode realesrgan-general`,
next to `animevideov3` in the menu, **with a hint that states it measured worse**.
The reason is the paragraph above this one: the scores it lost on come almost
entirely from *synthetic* downscales, so they are evidence rather than proof, and
the point of the mode is that a user asked to judge it on material this project has
not scored. `scripts/realesrgan/pair_grid.sh` puts one real frame through both
models side by side without opening the app. Numbers and the reversed verdict:
`scripts/realesrgan/README.md`.

The quality finding above is unchanged and ships with the mode: **animevideov3 is
the best mode here on SD live-action detail and the worst on cel animation**, where
the Anime4K CNN already beats it on every fidelity measure. It is a mode to reach
for on photographic material, not the default for anime. Verification (60 dB
against PyTorch, through the shipping code path), benchmarks and the live scoring
are in `scripts/realesrgan/README.md`.

## What this means for the app

- **Default is MetalFX Spatial** (changed from Anime4K CNN on request). It is the
  right default for a general player — cheapest by ~4×, real-time with three
  orders of magnitude to spare, and the sharper of the two on live action — but
  on anime, the app's purpose, the CNN measures closer to the truth (37.51 dB
  against bilinear's 39.42 for MetalFX; the CNN beats bilinear outright). The
  Enhancement menu switches in one click or one `--mode` flag, and switching does
  not restart playback.
- **`MetalFX Spatial` is the general-video / live-action option**, and the one to
  pick for clean 720p/1080p material where a crisper picture is wanted and some
  edge enhancement is acceptable. It is also the mode to pick when GPU headroom
  matters: 4× cheaper than the CNN at 1080p→4K.
- **`Off` plays the source at its own size** — no scaling, no GPU work beyond the
  copy, byte-identical output, for watching SD material as-is or for judging what
  a mode is adding. Magnification, and the comparison view, do not apply to it.
- **Decide per clip with the comparison view (⌥⌘C).** Any two models can be wiped
  against each other on the same frame, which is the only honest way to settle
  "which looks better on *this* video" — the mode is selectable on both halves.
- **AnimeVideo-v3 is a live mode** (`ScalerKind.realESRGAN`). It was the
  fastest member of the Real-ESRGAN family by 2×, and porting it through MPSGraph
  made it 3.1× faster again — 7.33 ms at 270p→1080p against the 22.59 ms measured
  eagerly, so 0.22× of a 30 fps budget. It is x4-only (no 2× checkpoint exists) and
  therefore SD-only (past 1024×576 the 4× output exceeds the encoder, and the mode
  refuses rather than clamping). On the synthetic clips it has the best-calibrated
  sharpness of every mode on live action (0.97× of the truth) and the worst
  fidelity of every mode on anime, where the CNN remains the better choice; on
  genuine degraded video it replaces texture rather than sharpening it.
  See "The AnimeVideo-v3 question" above and `scripts/realesrgan/README.md`.
- **General-x4 is a live mode too** (`ScalerKind.realESRGANGeneral`), and the one
  entry in this menu that exists for comparison rather than recommendation. It
  scored below `animevideov3` on every clip and column measured, at 1.84× the
  cost, and is real time at 270p/360p but not at 480p (0.63×). It is there because
  the user asked to judge it on their own footage, and it carries its own warning
  in the menu hint. `scripts/realesrgan/pair_grid.sh` for the side-by-side.
- **If quality matters more than playback and the model is heavier than that**,
  this app is still the wrong tool: `general-x4v3` is real time only at 270p/360p,
  `x2plus` is heavier again and unmeasured, and the diffusion models in the wider
  field are offline by construction. That is an offline resample path, which is a
  different feature and was not built.
- **Two models were evaluated for a mode and rejected on measurement, not
  on principle:** RT4KSR (fast enough, but softer than the CNN and no more
  faithful — see above) and RealBasicVSR (offline-only at ~1 fps, and its
  absolute frame-to-frame wobble on static content is 26× the bilinear run's,
  i.e. visible shimmer). Both remain measurable from `scripts/` if the material
  or the models change.

## Reproduce

The whole comparison, all five tables, in one command:

```bash
scripts/eval_all_general.sh /tmp/general-eval            # renders RT4KSR + animevideov3, scores every mode
```

It renders the offline models, then scores every mode. To run one setup by hand:

```bash
# soft live action, 2x
python3 scripts/eval_general_upscaler.py \
  --playercheck engine/MacVideoUpscaler/.build/release/playercheck \
  --source test-media/liveaction/tt_540p_c30.mp4 \
  --reference test-media/liveaction/tt_1080p_ref.mp4 \
  --scale 2 --modes bypass,cnn,cnnDenoise,metalfx,off \
  --out results/general/liveaction_540p_to_1080p

# 4x
... --source test-media/liveaction/tt_270p_c28.mp4 --scale 4 ...

# anime
... --source test-media/general/naruto_540p_c30.mp4 \
    --reference test-media/general/naruto_1080p_ref.mp4 --scale 2 ...

# sharp 4K-sourced clip: rebuild the material first, see
# test-media/liveaction/README.md

# Real-ESRGAN/animevideov3 throughput, both checkpoints (needs fetch_weights.py once)
python3 scripts/realesrgan/fetch_weights.py
python3 scripts/realesrgan/bench.py --variant animevideov3 --devices mps,cpu --iters 3
python3 scripts/realesrgan/bench.py --variant general-x4v3   --devices mps     --iters 3

# the single-layer microbenchmark the original (wrong) projection came from -
# keep it as a kernel number, never extrapolate it to a network
xcrun -sdk macosx swiftc -O scripts/bench_realesrgan_conv.swift \
    -o /tmp/convbench -framework Metal && /tmp/convbench

# one crop to look at
python3 scripts/make_eval_grid.py --dir results/general/liveaction_540p_to_1080p \
    --frame 105 --crop 950,290,240,160 --zoom 3 --cols 2 \
    --modes bypass,cnn,metalfx --out /tmp/roof.png
```
