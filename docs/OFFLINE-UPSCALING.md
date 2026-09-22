# Beyond live playback — the five-model shortlist, classified

A shortlist was put to this project as "our picks by priority":

| if you care most about | the pick |
|---|---|
| maximum restoration quality | SeedVR2 |
| a practical end-to-end workflow | Video2X |
| consistency across moving frames | FlashVSR |
| faithful anime detail | Real-ESRGAN AnimeVideo-v3 |
| lighter anime batch processing | Real-CUGAN |

**The verdict: the table is sound as a list of offline options and wrong as a list
of alternatives to this app.** These are not five candidates for one choice — they
are three different kinds of thing. Four of them are what to run when you want
more than a player can give, and then play the result. **The fifth, Real-ESRGAN
AnimeVideo-v3, turned out to be a live mode after all, and is now one of the
app's** — a measurement that reversed a published decision, so it is documented
below rather than quietly amended.

Two of the five have been measured on this machine — Real-ESRGAN `animevideov3`
and SeedVR2 — and the other three have not, and are described from their own
documentation. That distinction is kept explicit throughout.

## The classification

| pick | what it actually is | runs here? | live? |
|---|---|---|---|
| **SeedVR2** | one-step **diffusion** restoration model, 3B params | **measured** — not on the CUDA build (H100-80G for 720p), but the 4-bit MLX build runs; see `scripts/seedvr2/` | no — 0.006–0.30× real time, and over the 2 GB budget |
| **Video2X 6.0** | a **framework**, not a model | not natively — ncnn + Vulkan, macOS only via a container | no, file processing |
| **FlashVSR** | diffusion-based one-step **streaming** VSR | no — A100-class CUDA, no MPS path | not on this hardware |
| **Real-ESRGAN AnimeVideo-v3** | `SRVGGNetCompact`, 18 convs, 621K params | **yes — ported; 7.33 ms at 270p→1080p** | **yes — a shipped live mode, SD 4× only** |
| **Real-CUGAN** | Cascade U-Nets for anime; PyTorch + a separate ncnn build | not natively — ncnn/Vulkan | no |

Two entries are worth separating out more sharply, because they are the two most
easily misread:

**Video2X is not a model and does not compete with any model.** It is a C/C++
framework that drives Anime4K v4, Real-ESRGAN, Real-CUGAN and RIFE through ncnn
and Vulkan, plus arbitrary MPV GLSL shaders, over files in and files out. If the
goal is "upscale my library", it is the answer, and the models inside it are the
same names on this list — it is a delivery mechanism, not a sixth opinion about
picture quality. Its ncnn/Vulkan backend is also why it is not a macOS-native
tool: on this platform it means a container image.

**"Consistency across moving frames" for FlashVSR is the strongest single claim on
the list**, because temporal consistency is exactly where diffusion VSR earns its
cost. It is also the one that assumes an A100: ~17 fps at 768×1408 on a single
A100, block-sparse attention tuned for A100/A800, and the project states that
RTX 40/50 behaviour is unknown. There is no MPS path. On this Mac it is not a
slower option, it is a non-option.

## What was actually measured here

**Real-ESRGAN** is the family that became live modes — both of its released x4
checkpoints, through one Metal port:

| | measured |
|---|---|
| throughput, eager PyTorch/MPS | 22.59 ms/frame at 480×270 → 1920×1080 = 44.26 fps = 0.68× of a 30 fps budget |
| throughput, **the shipped port** | **7.33 ms = 136.8 fps = 0.22× of a budget** — 3.1× faster, by fusing conv+bias+PReLU in one MPSGraph executable instead of dispatching them separately |
| in the player, 4× | 480×270→1080p **1.007× real time**, 640×480→2560×1920 **0.992×**, 960×540→3840×2160 **0.991×**; a 1080p source is refused, because 4× there is 7680×4320 and past the encoder |
| heavy sibling | `general-x4v3` (2× the MACs): 47.58 ms = 21.0 fps eagerly, **13.76 ms** through the port — but it loses to `animevideov3` on PSNR and SSIM on both clips at 1.84× the cost, and is real time only at 270p/360p. It is nevertheless **shipped as `--mode realesrgan-general`**, labelled in the menu as the weaker one, because the user asked to judge it on their own footage and the scores that reject it rest mostly on synthetic downscales |
| live GPU memory | **13.5 MB at 480×270, 53.9 MB at 960×540** of tensors held; 140–353 MB resident for the whole player |
| vs the truth, SD live action | acutance **0.97×** — the only mode within 3% (Anime4K CNN 0.77×, MetalFX 1.99×); most legible on fine text at 3×. **But this is on a synthetic lanczos downscale**, and on genuinely degraded non-anime video the model smooths fine texture into rounded blobs and expands highlights — reported by a user, reproduced, and documented in `scripts/realesrgan/README.md` |
| vs the truth, anime | PSNR **28.36** against the CNN's 35.57, SSIM 0.8959 vs 0.9478, smooth-region error 1.912 vs 1.374 — it posterises flat cel gradients |

So the recommendation "faithful anime detail" holds for **offline** anime work —
where the model sees a whole frame at full quality with no deadline, and where a
detail-recovering GAN is judged on what it resolves rather than on PSNR — and it
now also holds **live on SD live action**, which is the case the port was made for.
It does **not** hold for live anime, where the Anime4K CNN already beats it on
every fidelity measure and looks better. Both statements are about the same model;
the content decides which applies. The mode ships with that stated in its own menu
hint rather than only in this document.

**SeedVR2 was measured too, and it also runs — through MLX, in 4-bit, and only
just.** This is worth stating precisely because the classification above first
read "runs here? no", which was true of the released CUDA implementation and
false of the model: an MLX 4-bit conversion exists and executes on this GPU.
What it does on this machine, end to end (`scripts/seedvr2/mlx_probe.sh`, raw
output in `results/seedvr2/mlx.txt`):

| | measured |
|---|---|
| image 480×270 → 960×540, 2× | 6.75 s wall, of which ~5.5 s is the frame (1.28 s is model load) |
| video 1× restoration at **source** size | ~1.8 s/frame |
| video 2× **upscaling** | **refused** — "safe video mode only supports source-size restoration or smaller output" |
| peak RSS | **2.75–2.90 GB in every configuration** — over the 2 GB budget, and forcing a 1 GB MLX cache did not lower it |
| best real-time fraction | **0.006×** (2× upscale) to **0.017×** (1× restore), against 30 fps |

So SeedVR2 is not a live mode here for two independent reasons — throughput two
orders of magnitude short, and a footprint above the budget that quantisation
reduced 9× without bringing under it. Its 4-bit build is nonetheless the most
substantial thing on this list: it is the difference between "does not load" and
"runs", and it is why the row above could be measured at all rather than argued
from upstream's H100 guidance.

**And the shortlist corrected a claim this project had published.** Earlier work
here projected Real-ESRGAN-class models at 0.12× real time and called the family
10–100× too slow to play, from a single-layer Metal microbenchmark. Measured end
to end the family reaches ~7.2 TFLOP/s, `animevideov3` is faster than real time,
and that projection is retracted in `results/general/realesrgan_feasibility.txt`.
The lesson generalises to the other four rows of this table: **"too slow" is a
claim about a specific implementation on specific hardware, and it should not be
inherited from someone else's benchmark or extrapolated from a smaller one.**

## Reading the quality recommendations

The four "care most about" columns are not commensurable, and the table hides
that:

- **Maximum restoration quality (SeedVR2)** is a generative objective. It will win
  on perceptual measures and lose on PSNR/SSIM against *every* deterministic model
  here, by construction, because it is rewarded for plausible detail rather than
  for the pixels that were there. Judge it on crops, never on a fidelity metric.
- **Consistency across moving frames (FlashVSR)** optimises a property none of the
  other four row entries are even measured on in a like-for-like way. This project
  built a flicker metric for exactly this question and found it easy to get wrong —
  a ratio normalised by a near-frozen bypass explodes. Absolute frame-to-frame
  change, and change relative to the source's own, are the numbers that survive
  scrutiny; `docs/GENERAL-VIDEO-UPSCALING.md` records the bug and the fix.
- **Faithful anime detail (AnimeVideo-v3)** and **lighter anime batch (Real-CUGAN)**
  are the two rows that overlap: both are anime-targeted, both have denoise and
  conservative variants, and real-CUGAN additionally offers 2×/3×/4× where
  Real-ESRGAN's video model is 4× only. That 2×/3× flexibility is the concrete
  reason to pick Real-CUGAN when the source does not divide evenly into 4×, and
  the reason an "end-to-end workflow" (Video2X) that wraps both is the practical
  answer for a library rather than for one clip.

## What this means for the app

**One thing changed, and it was the one thing the measurements said could.** The
app now has **five** live modes: the original four plus **Real-ESRGAN
AnimeVideo-v3** (x4 only, SD only). It meets the player's contract on every axis
that can be measured — frame budget, memory budget, and being the same model the
offline renderer produces — so nothing about the port is a compromise or an
exception. The port decision turned on cost alone, and it was a measurement that
reversed it: the same network is 3.1× cheaper inside an MPSGraph executable than
in eager PyTorch, which moved it from 0.68× of a frame budget to 0.22×.

The other four still cannot hold the contract here. This is a **player**,
real-time from a 2–3 frame buffer with a **2 GB budget for any upscaling model**,
and SeedVR2 meets neither budget nor frame time; Video2X and Real-CUGAN are
ncnn/Vulkan file processors with no macOS-native path; FlashVSR assumes an A100.
Of those, only SeedVR2 was worth measuring rather than dismissing, and it was
measured through its 4-bit MLX build precisely so the answer would not rest on
someone else's H100 guidance.

What the app also remains is the *end* of the pipeline: it plays the source at its
own size with **`Off`** (pixel-identical to the decoded source, no scaling), or
scales it 2×/4× with MetalFX, Anime4K CNN, CNN + denoise, or Real-ESRGAN. For
anything else on this shortlist the workflow is still: run it offline, write a
file, play that file — which is what the `render.py` scripts in `scripts/rt4ksr/`
and `scripts/realesrgan/` do for the models this machine can run at all.

## What did *not* change: the quality judgement

Worth keeping separate from the cost reversal, because it is the part that is easy
to lose in a "we ported it" announcement. **AnimeVideo-v3 is the best mode here on
SD live-action detail and the worst on cel animation.** Its acutance is 0.97× of
the truth on live action — the only mode within 3% — and 1.97× on anime, where its
PSNR of 28.36 is below plain bilinear's 34.08 and 7.2 dB below the Anime4K CNN's.
A GAN that invents detail is rewarded on photographic texture and punished on flat
cel gradients, and this app's stated purpose is animation. The mode is offered, it
is not the default, and its menu hint says so.

That is also why the mode is a *port* rather than an endorsement: the measurements
support having the option, and the same measurements say which content to point it
at. The comparison view settles it per clip in one keystroke, which is the point of
having both.

## Sources

- SeedVR2 — <https://github.com/ByteDance-Seed/SeedVR>: "one-step diffusion-based",
  SeedVR2-3B, 1×H100-80G for 100×720×1280, batch `torchrun` inference. **Measured
  here** through the 4-bit MLX build
  (<https://huggingface.co/AbstractFramework/seedvr2-3b-4bit>, run by
  <https://github.com/lpalbou/mlx-gen>): see `scripts/seedvr2/README.md`,
  `results/seedvr2/bench.txt` and `results/seedvr2/mlx.txt`.
- Video2X — <https://github.com/k4yt3x/video2x>: "super resolution and frame
  interpolation framework", v6.0.0 C/C++ rewrite, drives Anime4K v4 / Real-ESRGAN /
  Real-CUGAN / RIFE via ncnn and Vulkan.
- FlashVSR — <https://github.com/OpenImagingLab/FlashVSR>: "diffusion-based one-step
  streaming framework", ~17 FPS at 768×1408 on a single A100, block-sparse
  attention tuned for A100/A800, RTX 40/50 unknown, no MPS.
- Real-CUGAN — <https://github.com/bilibili/ailab> (`Real-CUGAN`): "Real Cascade
  U-Nets for Anime Image Super Resolution", 2×/3×/4×, four denoise strengths at 2×
  and two at 3×/4× plus conservative variants, PyTorch code with a separate NCNN
  build for mobile and AMD.
- Real-ESRGAN AnimeVideo-v3 — measured here, and **ported as a live mode**:
  `engine/RealESRGANMetal/` (MPSGraph executable + 2 Metal kernels, weights baked
  in), `scripts/realesrgan/README.md`, `results/general/realesrgan_bench.txt`,
  `results/general/realesrgan_metal.txt` and `docs/GENERAL-VIDEO-UPSCALING.md`.
