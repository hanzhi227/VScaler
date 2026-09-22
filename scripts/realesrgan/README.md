# Real-ESRGAN (`SRVGGNetCompact`) — measured, and both checkpoints are live modes

Two official x4 checkpoints, both `SRVGGNetCompact` — a plain convolutional stack
with no attention, no recurrence and no optical flow, which is what makes them
portable at all:

| checkpoint | body convs | params | MACs/input px | mode | measured |
|---|---|---|---|---|---|
| `realesr-animevideov3` | 16 | 621,424 | 0.62 M | `--mode realesrgan` | 7.33 ms at 270p→1080p |
| `realesr-general-x4v3` | 32 | 1,213,296 | 1.21 M | `--mode realesrgan-general` | 13.76 ms, 1.84× |

Both are **x4 only**. There is no x2 checkpoint for either, so they fit the app's
SD cases (270p → 1080p, 540p → 1080p) exactly and cannot serve a 2× request at all.
Both run through one port (`engine/RealESRGANMetal`), selected by the
`RealESRGANModel` the pipeline is given, and both are now **shipped live modes**.

**`general-x4v3` is shipped to be judged, not because it scored well.** It loses
to its own anime-trained sibling on every column measured here — both content
types, both fidelity metrics, at 1.84× the cost. That is a real finding and it is
below. But this project's scoring set is almost entirely *synthetic* lanczos
downscales of known references, which flatters every reconstruction model; on the
one genuinely degraded clip the two models differ in ways the numbers do not
capture. So the checkpoint that lost the scoring is offered as a mode, with its
menu hint saying so, and this is the honest way to settle it on real material:

```bash
scripts/realesrgan/pair_grid.sh          # one frame through both + bilinear + source
```

**Read the next two sections before choosing either.** A user reported that this
family "looks bad on real video that's not anime", and they are right: on
genuinely degraded video the models replace texture rather than sharpen it. The
niche is narrower than the tables below imply.

## What it actually looks like on real, non-anime video

**A user reported that this mode "looks bad on real video that's not anime", and
they are right — in a way the numbers above do not capture.** Everything scored in
this project until now used `tt_270p_c28`, which is real *content* but a **synthetic
degradation**: a lanczos downscale of a known-good 1080p reference. That is close to
invertible, and inverting it is what a reconstruction model is for. Genuine video is
not a downscale of anything; it is natively degraded by noise, compression and
motion, and a GAN has no reference to be faithful to.

Checked on `test-media/yt/mhw_ascendance_480p.mp4` — genuine native 480p, non-anime,
854×480 so it fits the 4× encoder limit and the mode can actually run on it
(`results/general/yt_mhw_480p/`, `clip.json` has the full record):

| what was checked | result |
|---|---|
| is it temporal instability? | **no.** Over 40 consecutive frames with real motion, flicker was **1.001×** the source's own frame-to-frame change (bilinear 0.969×, general-x4v3 1.036×). Flat regions: 32.97 against the source's 32.95 |
| is it a tone shift? | **no.** Global luma mean 76.16 against the source's 75.60, saturation 39.10% against 41.15% — far too small for the visible difference |
| is it a port bug? | **no.** The port matches PyTorch to 59.99 dB on this model, and the live path matched the offline one to 53.18 dB on this exact frame. This is the model |
| so what is it? | **it replaces structure rather than sharpening it.** Mean abs difference from the source over those frames: bilinear 3.196, animevideov3 **5.124**, general-x4v3 5.482. On the blue particle effect in `grid_real_particles_animevideov3.png`, individual sparks become smooth rounded balloons and the blown specular highlight loses its internal structure; on the rock and foliage it turns texture into soft blobs. The Anime4K CNN and MetalFX keep it |

So the mechanism is generative hallucination on out-of-domain material — animevideov3
is trained on anime video, and game footage or film is not that. **This narrows the
claim made elsewhere in this file.** "Best on SD live-action detail" was an
*acutance* result on a synthetic downscale; it does not survive on real video, where
the CNN is at least as good on text and clearly better on texture. The honest
statement of this mode's niche is now **SD anime, its training domain** — and even
there the CNN beats it on every fidelity measure.

## The verdict, and why it changed

**Earlier in this project the decision was "not ported"** — not on speed, which
was already adequate, but on picture: the model's win is confined to photographic
detail, which MetalFX and the Anime4K CNN already cover between them on an app
whose stated purpose is animation. That quality argument is unchanged and is
re-derived in full below. What changed is the *cost*, and it changed enough to
make the mode cheap to offer:

| | measured | how |
|---|---|---|
| eager PyTorch/MPS, whole frame | 22.59 ms at 480×270 → 1080p | `scripts/realesrgan/bench.py` |
| **the port that shipped** | **7.33 ms** at the same size | `resrbench --bench` |

**3.1× faster than PyTorch through the same GPU**, with no change to the model.
The reason is fusion: PyTorch's MPS backend dispatches conv, bias-add and PReLU as
separate kernels and writes 64 channels of intermediate to memory between each
pair, 17 times over; an MPSGraph executable fuses them. The arithmetic is
identical and the output is verified to match (below).

So the mode is now offered, and the honest framing is the one the quality numbers
support: **it is the best mode here on SD live-action detail and the worst on cel
animation.** It is a mode to reach for on photographic material, not the default
for anime — the Anime4K CNN remains the cleaner and more faithful choice there.

## The port: MPSGraph, not a hand-written kernel

This is a measured decision and it went against the obvious one.

The obvious port was a hand-written Metal kernel, and the numbers say it cannot
win. `scripts/bench_alu_throughput.swift` measures this GPU's primitives:

| primitive | measured |
|---|---|
| fp32 FMA | 4.83–5.47 TFLOP/s |
| **fp16 FMA (`half4`)** | **8.47–9.32 TFLOP/s** |
| threadgroup memory | 2.57 TB/s |
| `simdgroup_matrix` 8×8×8 half | **4.23 TFLOP/s** |

`animevideov3` needs 160.7 GFLOP at 480×270, so against the fp16 ceiling a
*perfect* SIMT kernel lands at ~19 ms and a realistic one worse — and the matrix
units are **slower**, not faster, so there is no fast path above it. MPS reaching
7.1 TFLOP/s effective (22.59 ms) is already ~85% of that ceiling. A hand-written
kernel had nothing to win and would most likely have been slower than what the
framework already does, so the model is expressed as an MPSGraph executable and
Metal is used only for the two things the graph cannot do:

| pass | what it does |
|---|---|
| `frameToNHWC4` | decoded BGRA frame → the `[1,H,W,4]` fp16 tensor the graph takes |
| 18 × convolution + bias + PReLU | MPSGraph |
| `shuffleResidual` | the 48-channel output → the 4× picture, pixel-shuffled, with the network's nearest-neighbour residual of the source added in fp32 |

The input tensor is 4 channels rather than 3 so that the kernel and the graph are
one aligned `half4` per pixel; the first convolution's 4th weight column is zero,
so this costs one wasted multiply and changes nothing.

### Is it the same model?

`scripts/realesrgan/verify_metal.py` runs the shipping code path and compares
against two references, because one number cannot separate two different things:

| comparison | PSNR | max abs error | what a gap would mean |
|---|---|---|---|
| Metal vs **fp32 CPU** | **59.97 dB** | 0.0056 | the price of fp16 arithmetic |
| Metal vs **fp16 MPS** (PyTorch) | **59.99 dB** | 0.0028 | implementation disagreement — a layout, PReLU-index or shuffle bug |
| fp16 MPS vs fp32 | 75.98 dB | — | the floor fp16 alone imposes |

The max error against a different runtime doing the same arithmetic is **0.7 of
one 8-bit level** on an 8-bit output, i.e. rounding-order noise. The weights
themselves cost nothing: re-deriving the network from the exported fp16 blob
reproduces the fp32 model to **70.4 dB**, and the blob's 622,000 scalars match the
documented 621,424 parameters plus the padded input column exactly.

The port also reproduces the offline numbers through the live player, which is the
check that matters: the same frame scored live and offline gives acutance 0.971×
against 0.971× and PSNR 27.79 against 27.81 on live action, and 28.36 against
28.42 dB on anime. (The offline renders those come from were made before the live
mode replaced them in `scripts/eval_all_general.sh`; the offline rows the script
produces *now* are the two `general-x4v3` ones, which agree with their live mode
to 0.04 dB. Both comparisons say the same thing — the player is not the variable.)

The model is **deterministic across processes** — three separate `resrbench --dump`
runs on one input produce an identical sha256 — so the small wobble seen in the
player comes from playback pacing, not from the network. That is worth stating
because it bounds what any in-app comparison can claim: two `playercheck` runs of
the *same* mode dump frames that differ by up to ~25 grey levels at index 90, and
the pre-existing modes do the same (cnn 24, metalfx 56), so "byte-identical across
runs" is not a test this harness can pass. The GUI was therefore verified the way
that does work: launched from the installed bundle with
`MACVIDEOUPSCALER_MODE=realesrgan`, its dumped frame matched the harness's
Real-ESRGAN run to **54.14 dB**, while differing from MetalFX and the CNN by
27 dB — the same picture, and unmistakably not the default mode.

## Throughput

`resrbench --bench`, MPSGraph + 2 Metal kernels, whole frame, `results/general/realesrgan_metal.txt`:

| input | output | ms/frame | fps | of a 30 fps budget | activation |
|---|---|---|---|---|---|
| 480×270 | 1920×1080 | **7.33** | 136.8 | **0.22×** | 13.5 MB |
| 640×360 | 2560×1440 | 12.53 | 79.8 | 0.38× | 24.0 MB |
| 854×480 | 3416×1920 | 21.71 | 46.1 | 0.65× | 42.6 MB |
| 960×540 | 3840×2160 | 27.39 | 36.5 | 0.82× | 53.9 MB |
| 1280×720 | 5120×2880 | 50.45 | 19.8 | 1.51× | 95.8 MB |
| 1920×1080 | 7680×4320 | 112.63 | 8.9 | 3.38× | 215.7 MB |

The last two sizes cannot be used in the app at all — 5120×2880 and 7680×4320 are
past the hardware H.264 encoder's 4096×2304 limit, which is why the mode is
**SD-only**: sources up to 1024×576 (~0.2 MP) get 4×, and larger ones are refused
rather than silently clamped to a magnification the network cannot produce.

`general-x4v3` runs the same table at ~1.84× the cost (13.76 ms at 480×270, and
past the frame budget from 854×480 on); both tables are in
`results/general/realesrgan_metal.txt`.

Cost is linear in *input* pixel count, not output: the whole network runs at input
resolution and only the final 48→3 pixel shuffle and the residual touch the 4×
picture. That is why 270p→1080p is 16× cheaper than 1080p→4K rather than 4×.

## In the player, end to end

`playercheck --mode realesrgan --scale 4`, 4× output, from `results/general/realesrgan_metal.txt`:

| source | output | frames | fps | realtime | footprint |
|---|---|---|---|---|---|
| 480×270 | 1920×1080 | 178 | 30.21 | **1.007×** | 140 MB |
| 640×480 | 2560×1920 | 354 | 29.72 | **0.992×** | 237 MB |
| 960×540 | 3840×2160 | 296 | 29.73 | **0.991×** | 353 MB |
| 1920×1080 | (7680×4320) | 0 | — | **refused** | — |

It holds real time at every size up to the encoder limit, with roughly ±1%
run-to-run variance. The margin is not uniform and is worth reading off the
standalone table rather than the ratio, which is capped at 1.0 by playback pacing:
at 270p the model uses 22% of a frame period, at 540p it uses 82%. Above that
there is nothing left for decode, composite and encode, and that is where the
encoder limit happens to land anyway.

Two behaviours the mode has that no earlier mode needed:

- **Asked for 2×, it runs at 4×** and says so (`requestedScaleFactor: 2`,
  `scaleFactor: 4`). There is no 2× checkpoint, and silently showing half the
  magnification would be worse than saying it.
- **A source too large for 4× is refused with the reason**, not clamped:
  *"Real-ESRGAN AnimeVideo-v3 (4×) is 4× only, and 7680×4320 is past the H.264
  encoder's 4096×2304 limit. It runs on sources up to 1024×576 — choose another
  mode for this one."* Every other mode could satisfy any request by clamping; a
  model pinned to one factor cannot.

## Memory: far inside the 2 GB budget

The project's budget for an upscaling model is **2 GB**. For this family the
weights are irrelevant to it and the activations are the whole cost — the body
runs at *input* resolution with 64 channels and the output is 16× the input pixel
count — so the cost is linear in frame area, not in model size.

| input → output | animevideov3 (PyTorch/MPS) | general-x4v3 |
|---|---|---|
| 480×270 → 1920×1080 | 37.2 MB | 39.6 MB |
| 640×360 → 2560×1440 | 64.2 MB | — |
| 960×540 → 3840×2160 | 141.4 MB | 143.8 MB |

Measured as live allocation with the output frame still held
(`torch.mps.current_allocated_memory`). The port is *smaller*, because it holds
one fp16 tensor rather than a chain of fp32 intermediates: the tensors it owns are
**13.5 MB at 480×270** and **53.9 MB at 960×540**, and the whole player process
measures 140–353 MB resident while running it. That is **14–54× inside the budget
at 4K output**; the checkpoints themselves are 2.5 MB and 4.9 MB on disk.

One caution on how to read the PyTorch rows: `torch.mps.driver_allocated_memory()`
reports a flat **1.12 GB** at every size and for both variants, which looks like
the requirement and is not — it is the allocator's reserved pool. The live figure
is the one that scales with the frame. PyTorch's MPS backend exposes no
high-water-mark API, so this is the allocation with the output alive rather than
a true peak; for a whole-frame pass those are the same buffers.

## On picture, through the same harness as every other mode

| | anime 270p→1080p | | | | live action 270p→1080p | | | |
|---|---|---|---|---|---|---|---|---|
| mode | PSNR | SSIM | acutance | smooth | PSNR | SSIM | acutance | smooth |
| reference | — | — | 1.000× | — | — | — | 1.000× | — |
| bypass | 34.08 | 0.9474 | 0.476× | 1.288 | 29.28 | 0.8712 | 0.390× | 5.247 |
| cnn | **35.57** | **0.9478** | 1.340× | 1.374 | **29.90** | **0.8719** | 0.768× | 4.770 |
| metalfx | 33.84 | 0.9289 | 2.558× | 1.459 | 29.54 | 0.8485 | 1.995× | 5.041 |
| rt4ksr (offline) | 35.29 | 0.9368 | 0.850× | 1.464 | 29.90 | 0.8612 | 0.567× | 4.960 |
| **realesrgan (live)** | 28.36 | 0.8959 | 1.972× | 1.912 | 27.79 | 0.8601 | **0.971×** | 5.364 |
| **realesrgan-general (live)** | 26.14 | 0.8866 | 2.399× | 1.748 | 26.35 | 0.8503 | 1.044× | 5.387 |
| general-x4v3 (offline) | 26.18 | 0.8870 | 2.400× | 1.734 | 26.36 | 0.8504 | 1.044× | 5.370 |

`reference_acutance` is 1.708 on the anime clip and 3.290 on the live-action one,
so the acutance column is a ratio to the truth. These rows are produced by
`scripts/eval_all_general.sh`, which drives `playercheck` — the live modes are
scored through the player, against the same reference frames as the offline
renders, which is what makes the columns comparable. The two `general-x4v3` rows
are the same checkpoint measured those two ways, and they agree to **0.04 dB**,
which is the cross-check that the live mode is running the checkpoint.

**On live action it gets the sharpness right and everything else wrong.** (And see
the section above: on *real* video, not a synthetic downscale, it does not get even
that right.) Its acutance is **0.971×** the ground truth — the only mode within 3% —
against the CNN's 0.768× (under-sharp) and MetalFX's 1.995× (over-sharp with
halos). That is visible:
`results/general/liveaction_270p_to_1080p_4x/grid_shirt_text_with_animevideov3.png`
shows Japanese shirt text at 3× where animevideov3 is the most legible of the six
panels — the katakana strokes separate cleanly and `NALS` resolves to letter
shapes, where the CNN leaves them merged and MetalFX smears them. It pays for
that with the **worst fidelity of the modes that shipped before it** — PSNR
**27.79**, SSIM **0.8601**, both below plain bilinear's 29.28/0.8712 — the worst
smooth-region error among them (5.364), and the most added temporal activity
(1.061× the source's own frame-to-frame change, against the CNN's 1.008×).
`general-x4v3` is below it on every one of those; it was added as a comparison
target, not as a recommendation, and this mode is a recommendation.

**On anime it simply loses.** `grid_anime_with_animevideov3.png`, same clip
family at the same crop and zoom, shows the CNN closer to the reference and
animevideov3 posterising the smooth skin gradient — the flat-region error of
1.912 against the CNN's 1.374 and bilinear's 1.288 quantifies exactly what the
grid shows. Acutance 1.972× against a 1.000× truth means it is inventing detail
the source does not have, which is what a GAN is for and what an anime cel with
large flat areas punishes. The comparison view settles it per clip in one
keystroke, which is the point of having both.

### general-x4v3: the general model loses to the anime model, on both

This one had never been scored on picture — only timed — so "the other checkpoint"
was an open question rather than a known quantity. It is now measured, and the
answer is that it is dominated:

| | anime 270p→1080p | | | | live action 270p→1080p | | | |
|---|---|---|---|---|---|---|---|---|
| mode | PSNR | SSIM | acutance | smooth | PSNR | SSIM | acutance | smooth |
| **animevideov3** | **28.36** | **0.8959** | **1.972×** | 1.912 | **27.79** | **0.8601** | **0.971×** | 5.364 |
| **general-x4v3** | 26.18 | 0.8870 | 2.400× | **1.734** | 26.36 | 0.8504 | 1.044× | 5.370 |
| Anime4K CNN | 35.57 | 0.9478 | 1.340× | 1.374 | 29.90 | 0.8719 | 0.768× | 4.770 |
| bilinear | 34.08 | 0.9474 | 0.476× | 1.288 | 29.28 | 0.8712 | 0.390× | 5.247 |

It has **the lowest PSNR and SSIM of every mode tested, on both clips**, including
plain bilinear — and it is worse than `animevideov3` on every one of those eight
columns except anime smooth-region error, where it is marginally better
(1.734 against 1.912) while still losing to the CNN by a wide margin.

The `Anime4K CNN` and `bilinear` rows here are the live runs, so the comparison is
like for like; the two Real-ESRGAN rows are the offline render of each checkpoint.

Once `general-x4v3` became a mode, the same two clips were scored **through the
player**, which is what the shipped path actually renders. It reproduces the
offline checkpoint row to within 0.04 dB, which is the useful part of having both:
a live mode that disagrees with its own offline render is a wiring bug, not a
model result.

| 4×, via `playercheck` | anime 270p→1080p | | | live action 270p→1080p | | |
|---|---|---|---|---|---|---|
| mode | PSNR | SSIM | flicker | PSNR | SSIM | flicker |
| realesrgan (live) | **28.36** | **0.8959** | 0.0598 | **27.79** | **0.8601** | 18.873 |
| realesrgan-general (live) | 26.14 | 0.8866 | 0.0539 | 26.35 | 0.8503 | **20.170** |
| general-x4v3 (offline render) | 26.18 | 0.8870 | 0.0010 | 26.36 | 0.8504 | 20.169 |
| Anime4K CNN (live) | 35.57 | 0.9478 | 0.0427 | 29.90 | 0.8719 | 17.934 |

`flicker` is the mean absolute frame-to-frame difference over a run of consecutive
frames; on the live-action clip, where there is real motion, `general-x4v3` is the
**most temporally active mode measured — 20.17 against the bilinear run's 16.49,
i.e. 1.223× the source's own frame-to-frame change**, where `animevideov3` is
1.145×. The anime clip's numbers are meaningless as ratios (its bypass flicker is
0.02, under the floor the metric needs) so only the live-action column should be
read that way.

The result worth pausing on is the live-action column. `general-x4v3` is the
\*general-purpose\* checkpoint — the one trained across a wider degradation range
precisely so it can handle photographic material — and it is **beaten on live
action by the anime-specific model** (27.79 dB against 26.36, acutance 0.971×
against 1.044×). Its acutance is genuinely well-calibrated, second only to the
mode we ship, but everything else about it is worse. One plausible reading is that
this project's live-action clips are bicubically downscaled and re-compressed, a
degradation `animevideov3`'s training set matches closely and the general model's
broader range dilutes; that is a hypothesis, not a measurement, and the practical
conclusion does not depend on which explanation is right.

And it costs more. Measured back to back on the same machine, under the same load
(`results/general/realesrgan_metal.txt`):

| input → 1080p | animevideov3 | general-x4v3 | ratio | real time? |
|---|---|---|---|---|
| 480×270 | **7.50 ms** (0.22× of a 30 fps budget) | **13.76 ms** (0.41×) | 1.84× | both |
| 640×360 | 12.75 ms (0.38×) | 23.59 ms (0.71×) | 1.85× | both |
| 854×480 | 21.99 ms (0.66×) | 41.28 ms (1.24×) | 1.88× | only animevideov3 |
| 960×540 | 27.63 ms (0.83×) | 51.91 ms (1.56×) | 1.88× | only animevideov3 |

So it is slower *and* worse, which is the opposite of a trade-off — and its live
window is narrower, not wider: real time at 270p and 360p, where the shipped mode
already is, and past the frame budget from 480p up, where the shipped mode is not.
Measured in the player at 854×480→3416×1920 it holds **0.92× real time on the
first three seconds** and **0.63×** once the graph is warm, against
`animevideov3`'s 0.92× in the same conditions — so on a 480p source this mode
drops roughly a third of the frames, and the picture is what you are judging.

**Verdict: measured, ported, and shipped anyway — as a comparison target.**
`--mode realesrgan-general` is in the Enhancement menu next to `animevideov3`, with
a hint that states plainly that it measured worse. That is a deliberate decision
rather than a reversal of the numbers above:

* the scoring set this verdict rests on is **almost entirely synthetic** — lanczos
  downscales of known references, which are close to invertible and are exactly
  what a reconstruction model is good at. The one genuine native-degradation clip
  in the repo is 65 seconds of promotional video. A ranking derived from that set
  is evidence, not proof;
* the whole point of the mode is that the user asked to **judge it on their own
  material**, and a mode you can switch to in one keystroke is a better
  instrument than a table;
* the cost of being wrong about it is small: both graphs are built lazily per
  geometry, so the unselected mode costs nothing per frame.

The negative result is still one command to reproduce and one to revisit:

```bash
python3 scripts/realesrgan/export_weights.py --variant general-x4v3
engine/RealESRGANMetal/.build/release/resrbench --bench --variant general-x4v3
python3 scripts/realesrgan/render.py --input CLIP.mp4 --output OUT.mkv --variant general-x4v3
scripts/realesrgan/pair_grid.sh                    # both models on one real frame
```

One packaging note, because it cost a wrong claim before it was measured
properly, and because shipping this mode means **paying that cost on purpose**.
The generated weights used to live in the **`resrbench` target** so they could not
reach the app: Swift does *not* dead-strip a large literal from an unreferenced
public enum. Now that it *is* a mode, the literal is in the library the app links
and the app carries it. Re-measured here by building the same code with a stub
weights file in place of the real one — the only form of this check that cannot be
fooled by a stale artifact:

| app binary (installed, signed) | bytes |
|---|---|
| without `general-x4v3`'s blob | 3,162,480 |
| **with it** | **6,445,760** |
| cost of the option | **3,283,280 (3.28 MB)** |

The earlier version of this table said 3,160,784 → 6,464,752; the difference is
build-to-build noise in the rest of the binary, and both agree on what matters —
the literal costs its own size, 3.25 MB of base64 plus a little.

The first check of this said the opposite, and the reason is worth recording: the
build had **failed**, and the size and `strings` checks were pointed at the binary
left over from the previous successful build. A `strings` search for the literal
name and a single 80-character base64 probe both came back clean on a binary that
did not contain the code being tested. What settled it was a controlled
experiment — build with the file, build without it, compare.

A second lesson from redoing it today: **`strings` on this macOS binary reports
neither mode name nor the base64 blob, while both are demonstrably present** — a
byte-level search finds `Real-ESRGAN General-x4 (4×)`, `realesrgan-general` and
the full 3.25 MB literal. The binary grew by exactly the blob's size, which is the
measurement; `strings` was the instrument that was wrong, not the build.

### The quality numbers, re-derived outside the harness

The harness's anime figures are the decisive ones, so they were reproduced from
the PNGs, without `eval_general_upscaler.py`, on still frame 105 of
`test-media/general/naruto_270p_c28.mp4`:

```
method                     PSNR     SSIM   acutance  per-channel mean offset vs ref
bilinear 4x               34.53   0.9456    0.703 (0.38x)  R+0.05 G+0.03 B+0.10
bicubic 4x                35.45   0.9462    0.965 (0.52x)  R+0.06 G+0.04 B+0.11
animevideov3 x4           27.85   0.8804    3.693 (1.99x)  R+0.88 G+0.74 B+0.41
```

It agrees with the harness (28.42 dB / 1.97× offline, 28.36 / 1.97× live), so the
anime loss is the model and not the measurement path. The failure mode is the
acutance: it **doubles the edge energy** of a soft, compressed SD source (3.69
against the reference's 1.86), and the error is not confined to edges —
smooth-region RMSE 1.905 against the bilinear run's 1.29, which is the
posterisation of flat cel gradients visible in the grid. There is also a small
systematic lift of 0.4–0.9 grey levels in every channel: real, and much too small
to explain the PSNR gap by itself.

Both baselines beat it. Bicubic, which is what any ordinary resampler does, is
7.6 dB ahead of it on the anime clip.

## The correction this work produced

This module exists partly because it disproved a claim the project had already
published. `results/general/realesrgan_feasibility.txt` projected `general-x4v3`
at 277.7 ms/frame for a 640×480 input from a **single-layer** Swift Metal
microbenchmark (2.67 TFLOP/s achieved), concluding the Real-ESRGAN family was
10–100× too slow to play live. Measured end to end, the family achieves
**~7.2 TFLOP/s**, the same 640×480 case costs ~100–110 ms, and animevideov3 is
real-time. The microbenchmark paid per-dispatch overhead once against one layer's
arithmetic and then scaled that rate to a whole network. Both numbers are kept in
that file; the docs that cited the wrong one are corrected.

The port then repeated the lesson in a different direction: the same
microbenchmark family of reasoning said a hand-written kernel should beat a
framework, and measuring the GPU's ceilings (above) showed it could not. Two
projections, both wrong, both caught by measuring the whole thing.

## Files

| file | what it does |
|---|---|
| `srvgg.py` | the model, `load()`, and an `Upscaler` with overlapping-tile support for larger sources |
| `fetch_weights.py` | downloads both checkpoints to `~/.cache/macvideoupscaler`, pinned by size and sha256 (`--only` for one) |
| `bench.py` | PyTorch throughput and FLOPs, `--variant {animevideov3,general-x4v3}`, whole frame, per device |
| `render.py` | offline x4 render of a file, lossless by default so the output is measurable; `--variant` picks the checkpoint |
| `export_weights.py` | checkpoint → the fp16 blob, regenerating the Swift source; `--variant` or `--all`. `animevideov3` lands in the `RealESRGANMetal` library, `general-x4v3` in `resrbench` (see the packaging note above) |
| `verify_metal.py` | the port against both PyTorch references, through the shipping code path |
| `bench_metal.sh` | regenerates `results/general/realesrgan_metal.txt` end to end |

```bash
python3 scripts/realesrgan/fetch_weights.py
python3 scripts/realesrgan/bench.py --variant animevideov3 --devices mps,cpu --iters 3
python3 scripts/realesrgan/render.py --input CLIP.mp4 --output OUT.mkv   # x4, lossless
python3 scripts/realesrgan/render.py --input CLIP.mp4 --output OUT.mkv --variant general-x4v3

# the live mode
python3 scripts/realesrgan/export_weights.py      # only if the weights changed
swift build -c release --package-path engine/RealESRGANMetal
engine/RealESRGANMetal/.build/release/resrbench --bench
python3 scripts/realesrgan/verify_metal.py --size 480x270
scripts/realesrgan/bench_metal.sh
```

`render.py` takes no `--scale`: both checkpoints are x4, so there is nothing to
choose. Its output is scored against the app's live modes by
`scripts/eval_general_upscaler.py --extra-video`, which is how the
`rt4ksr` rows in `results/general/*/metrics.json` were produced — offline renders
and live player modes go through the same scoring code against the same reference
frames, which is what makes the columns comparable.

`export_weights.py` is only needed if a checkpoint changes: the generated
`engine/RealESRGANMetal/Sources/RealESRGANMetal/RealESRGANWeights.swift` carries
the 1.25 MB blob as one base64 literal, so the app bundle stays "the executable
and Info.plist" with no model file to install. It must stay a *single* literal —
the first version emitted `"..." + "..."` per line and swift-frontend sat at 100%
CPU for eleven minutes on the resulting 14,000-term expression before it was
killed.

### Measurement variance

Repeated runs of `bench.py` on this machine move by around **10%**: 480×270 came
out at 22.59 ms in the original run, 23.05 ms in a second and 25.27 ms in a
third, and the larger sizes move similarly. Read the PyTorch tables as "about
25 ms at 270p", not to four significant figures. The port is tighter — 7.31 and
7.33 ms in two runs — but the live `realtimeRatio` still moves ±1% run to run,
which is the number to trust for "does it keep up" and is why nothing here turns
on the third digit.

## Provenance

Upstream: <https://github.com/xinntao/Real-ESRGAN> (BSD-3-Clause). Weights from
the official v0.2.5.0 release. `realesr-animevideov3.pth` pinned at 2,504,012 bytes
/ `b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d`;
`realesr-general-x4v3.pth` at 4,885,111 bytes /
`8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`.

The released checkpoints confirm the architecture exactly — `strict=True` loads
with the shapes above, no missing or unexpected keys — so the class here is the
upstream one rather than a reimplementation.
