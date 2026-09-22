# SeedVR2

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) — "One-Step Video Restoration
via Diffusion Adversarial Post-Training", ICLR 2026. Apache-2.0. Weights are used
here from the ComfyUI single-file distribution,
[`Comfy-Org/SeedVR2`](https://huggingface.co/Comfy-Org/SeedVR2) (Apache-2.0).

**Verdict: it cannot be a live mode in this app, and quantising it does not
change that.** The 3B checkpoint needs **7.6 GB** of GPU memory in the smallest
configuration that runs at all — 3.8× the project's 2 GB budget — and its most
aggressive CUDA quantisations do not run on this GPU. A **4-bit MLX build does
exist** and is genuinely better: it runs, at 2.75–2.90 GB of RSS, which is still
over budget, and it is ~9× slower per frame than the DiT floor measured here at
the same single-frame operating point. Its video mode **refuses upscaling**
altogether. Best throughput measured anywhere in this directory is **0.30× real
time** with upstream's configured windowed attention, and every output resolution
the app actually uses runs out of memory before it produces a frame.

Two independent runtimes were measured, because the answer should not depend on
one implementation: the released PyTorch/MPS shapes (below, and in
`results/seedvr2/bench.txt`) and the 4-bit MLX pipeline that is the only SeedVR2
that runs here (`results/seedvr2/mlx.txt`). They agree on the verdict and differ
by ~9× on cost, which is itself informative.

SeedVR2 is worth writing down anyway, because it is the strongest version of the
argument for a model like this: inference really is **one step** (no iterative
sampling, no classifier-free guidance — `sample_steps=1`, `cfg_scale=1.0` in the
released inference script), and it is the only model in this project's
"offline by construction" category whose cost could not be dismissed by
structure. It was measured rather than assumed, and it does not fit.

## The architecture, verified against the shipped weights

Every figure below is read off the safetensors headers in `Comfy-Org/SeedVR2`,
which carry the full tensor list — name, dtype, shape — in the first few
kilobytes of each file. `inspect_weights.py` fetches only that header, so the
architecture is checked against the released weights instead of inferred from a
config file.

That matters, because the config is misleading about it. `configs_3b/main.yaml`
says `num_layers: 32`, `expand_ratio: 4`, `mlp` swiglu and `mm_layers: 10`; the
weights say:

```
3B   32 blocks = 10 dual-stream (separate `txt` and `vid` attention + MLP)
                 + 22 vid-only (one stream, named `all` in the checkpoint)
     vid_dim 2560, MLP hidden 6912, heads 20 x 128, patch (1,2,2), 33 -> 16 channels
7B   36 blocks, all dual-stream
     vid_dim 3072, MLP hidden 8192, heads 24 x 128
```

`expand_ratio: 4` does not describe the released MLP width (it is 2.7 × dim), and
the dual-stream blocks are invisible in a config-derived model. Counting from
the headers reproduces the checkpoints exactly:

| | modelled | released |
|---|---|---|
| 3B DiT | 3.3908 B | 3.3915 B (13,566,090,228 B fp32 / 4) |
| 7B DiT | 8.2377 B | 8.2397 B |
| VAE | 250.7 M | 1,002,691,902 B / 4 |

The 3B figure is also what `bench.py` builds and reports as a cross-check, so the
timings below are timings of the right shape.

The VAE is a 3D causal autoencoder, spatial /8 and temporal /4, 16 latent
channels, 250.7 M parameters. It is **not** in the timings — see the note in
`seedvr2.py` — so every figure here is a floor on the real thing.

## Speed: measured, not projected

`scripts/seedvr2/bench.py` builds the block stack at those dimensions and times
real forward passes on the MPS GPU with random weights. A forward pass does not
depend on weight values, so the timing is the real one for the shape. This is
deliberately not a projection from a kernel benchmark: this project has already
been burned once by that (`results/general/realesrgan_feasibility.txt`, ~2.6×
too pessimistic — see `scripts/realesrgan/README.md`).

Best case, 640×480 output, MPS, fp16, upstream's windowed attention
(`results/seedvr2/bench.txt` is the full sweep):

| frames per pass | ms/frame | fps | × real time @30fps |
|---|---|---|---|
| 1 | 599–669 | 1.5 | **0.05×** |
| 4 | 151–166 | 6.0 | 0.20–0.22× |
| 9 | 109–125 | 8.0–9.2 | **0.27–0.31×** |
| 25 | 122 | 8.2 | 0.27× |

The cost per frame *falls* with clip length, because the DiT processes a temporal
window in one pass and one frame per pass pays the whole weight-read cost for a
single frame — that is what upstream's adaptive window attention buys. Even at
its best, 0.30× real time, it is **3.3× too slow** for this player, and that is
the floor before the VAE, which also runs per clip.

Every configuration the app actually uses fails earlier than it can be timed:

| configuration | result |
|---|---|
| 1280×960, 9 frames | 604–687 ms/frame = **0.049–0.055×** real time |
| 1920×1080, 9 frames | **out of memory** — and 1080 is *off the model's patch lattice* anyway |
| 3840×2160, 9 frames | **out of memory** |
| 1280×720, 100 frames — upstream's own example | **out of memory** |
| 7B, any size | **cannot load**: 16.01 GB of weights against a 15.10 GB ceiling |

The lattice point is a constraint worth knowing independently of speed: the VAE
divides each side by 8 and `patch_size` divides the spatial axes by 2 again, so
**every output dimension must be a multiple of 16**. 480, 540, 720, 960 and 2160
are; **1080 is not** (1080 / 8 = 135, which is odd). Upstream's own inference
defaults to 720p rather than 1080p, and `bench.py` reports the padding when a
request does not land on the grid — it rounds the latent up rather than refusing,
so the timing stays honest and the token increase is printed. A 1080p target
therefore costs more than 1080p, before it also runs out of memory.

For context, this GPU sustains 23.6–25.1 TFLOP/s on plain dense matmul; the
measured SeedVR2 passes achieve ~2.4 TFLOP/s of their nominal arithmetic, which
is what a windowed attention transformer with a 6912-wide MLP at these batch
sizes looks like in practice.

### The window is load-bearing, and it is not the whole story

Upstream windows every layer to `(4,3,3)`. Measured, that is not a free win at
small sizes — at 2,400 tokens global attention is actually *faster* (93.5 ms
against 109.5 ms per frame), because the window fold costs real padding and
reshape work. It is what makes the model tractable at all at scale: at 9,600
tokens global attention needs 3.45 GB more than is available and dies, while the
windowed version runs.

## Memory: the reason it is out, independent of speed

The project's budget for an upscaling model is **2 GB**. SeedVR2 does not come
close, at any quantisation:

| weights | size | fits 2 GB? | runs on this GPU? |
|---|---|---|---|
| `seedvr2_3b_fp16` | 6.78 GB | ✗ | ✓ (this is what was measured) |
| `seedvr2_3b_int8_convrot` | 3.46 GB | ✗ | ✗ |
| `seedvr2_3b_fp8_e4m3fn` | 3.39 GB | ✗ | ✗ |
| `seedvr2_3b_nvfp4` | **2.00 GB** | ✗ (weights alone) | ✗ |
| `seedvr2_7b_nvfp4` (smallest 7B) | 4.76 GB | ✗ | ✗ |
| `ema_vae_fp16` (always required) | 0.50 GB | — | ✓ |

The last column is the trap. The aggressive quantisations are the ones that look
like they would fit, and they are **CUDA/NVIDIA formats**: `nvfp4` is NVIDIA-only
by definition, and fp8 and int8 matmul are not implemented on MPS. Verified:

```console
$ python3 -c "import torch; a=torch.zeros(4,4,dtype=torch.float8_e4m3fn,device='mps'); a@a"
MPS device does not support mm for non-float inputs
$ ... dtype=torch.int8 ...
MPS device does not support mm for non-float inputs
```

So on this GPU there is exactly one usable format, fp16 at 6.78 GB, and the
measured working set at the smallest input that runs at all (640×480, **one**
frame) is **7.63 GB — 3.8× over budget**. At 640×480 and 25 frames it is
12.87 GB. Quantising the weights on a CUDA device would not rescue this: the
activation working set alone is already over the budget, and the always-required
VAE is another 0.5 GB.

## The 4-bit MLX build — the best available answer, measured

There is a real quantised path for Apple Silicon and it deserves the credit:
[`AbstractFramework/seedvr2-3b-4bit`](https://huggingface.co/AbstractFramework/seedvr2-3b-4bit)
is an **MLX** 4-bit conversion (q4 on the transformer and VAE attention linears,
bf16 for convolutions and norms) run by
[`mlx-gen`](https://github.com/lpalbou/mlx-gen), an mflux-derived Apple-native
runtime. Its model card reports 2.54 GiB of storage against the source's
13.57 GiB, and a Max RSS of 2.89 GiB against 25.49 GiB — a ~9× reduction, which
is the difference between "will not load" and "runs at all".

It replaces the PyTorch/CUDA path entirely, so it is a second, independent
measurement — and it is the only SeedVR2 that runs on this machine. Reproduce
with `scripts/seedvr2/mlx_probe.sh`; raw output in `results/seedvr2/mlx.txt`.

| what | wall | peak RSS |
|---|---|---|
| image 128×128 → 256×256, 2× (measures model load) | 1.28 s | 2.87 GB |
| image 480×270 → 960×540, 2× | 6.75 s | 2.82 GB |
| video 1× (restoration at **source** size), 3 frames | 6.73 s | 2.75 GB |
| video 1× with a 1 GB MLX cache | **fails** | 2.90 GB |
| video 2× (upscaling) | **refused** | — |

Four things follow, and they are all from mlx-gen's own output rather than
inference:

- **It is still over the 2 GB budget, and constraining the cache does not fix
  it.** Peak RSS is 2.75–2.90 GB in every configuration that runs, and forcing
  `--mlx-cache-limit-gb 1` *raised* it to 2.90 GB and broke the run. The
  footprint is the 4-bit weights plus the runtime, not a cache that can be
  trimmed.
- **Video upscaling is refused.** mlx-gen's own safety check:
  *"SeedVR2 safe video mode only supports source-size restoration or smaller
  output. Use 1x or override with --force-unsafe-video-memory."* The only
  supported video mode is 1× restoration — which is not what this app's
  `--scale` does.
- **Even 1× restoration is marginal.** It passed on the default cache and failed
  on a constrained one, against mlx-gen's own host-safe budget (peak 5.84 GB
  against a 5.79 GB budget); it auto-enables `--low-ram` for video, and reports
  a peak MLX allocation of **8.57 GB** even where the process RSS is 2.75 GB.
- **The throughput is two orders of magnitude off.** Subtracting the 1.28 s load
  baseline, a 2× frame costs **~5.5 s** and a 1× restore of a 480×270 frame costs
  **~1.8 s**. That is 0.006× and 0.017× real time at 30 fps, for one frame per
  invocation.

The comparison worth drawing is against the floor measured above. At the same
single-frame operating point — the floor's one-frame case, which is the worst case
for it — the DiT alone in fp16/MPS costs **599–667 ms/frame** at 640×480, while
the real 4-bit MLX pipeline costs **~5.5 s/frame** at 480×270→960×540, 1.7× the
token count: about **9× for comparable work**. The floor legitimately excludes
what accounts for the gap — the VAE (250.7 M parameters at pixel resolution,
absent from the DiT bench entirely), the dequantisation cost of 4-bit weights,
tone correction and I/O — which is exactly why it is labelled a floor, and why a
floor should never be quoted as a product's speed.

### Reproducing the MLX numbers

`mlx-gen` is not a project dependency and nothing in `engine/` imports it. It
lives in a throwaway venv:

```bash
python3 -m venv /tmp/mlxenv
/tmp/mlxenv/bin/pip install -U mlx-gen            # mlx 0.31.2, mlx-gen 0.37.0
/tmp/mlxenv/bin/mlxgen download --model AbstractFramework/seedvr2-3b-4bit   # ~2.6 GB
MLX_PY=/tmp/mlxenv/bin scripts/seedvr2/mlx_probe.sh
```

Note that the model card's own validation is a single **image** (133×113 →
658×560, 2.16 s, M5 Max 128 GB) — it is not a video figure and not this machine,
so it is not the number to hold against a player's frame budget.

## What it would take

Nothing about this is a criticism of the model — it is a 3B-parameter video
diffusion transformer, and upstream's own guidance is that "1 H100-80G can handle
videos with 100×720×1280", with 4×H100 for 1080p and 2K. A 26 GB laptop that
shares its memory with the OS is not the target. To play live here the model
would need to be roughly **3.3× faster and 4× smaller at the same time**, on a
GPU with no fp8 and no fp4 path, and upstream's released inference is
CUDA-only anyway (`device="cuda"` and `torch.autocast("cuda", ...)` are
hardcoded in `projects/inference_seedvr2_3b.py`; the MPS support that exists is
in a ComfyUI wrapper, not upstream).

It belongs in the same category as RealBasicVSR and `x2plus` in
`docs/GENERAL-VIDEO-UPSCALING.md`: offline restoration, which is a different
feature from this player.

## Files

| file | what it is |
|---|---|
| `seedvr2.py` | the cost model — architecture, parameter counts, token grids, FLOPs. No weights, no torch. |
| `bench.py` | builds the block stack at the verified dimensions and times real forward passes on MPS or CPU. |
| `bench_all.sh` | regenerates every figure in this file into `results/seedvr2/bench.txt`. |
| `inspect_weights.py` | reads a remote safetensors header by HTTP range request and prints the architecture. |
| `mlx_probe.sh` | drives the 4-bit MLX build through `mlx-gen` and records wall time and peak RSS: `results/seedvr2/mlx.txt`. |

## Usage

```bash
# what the network is, straight from the shipped weights (a few KB of download)
python3 scripts/seedvr2/inspect_weights.py \
    https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_3b_fp16.safetensors

# throughput on this machine
python3 scripts/seedvr2/bench.py --variant 3b --res 1280x960 --frames 9
python3 scripts/seedvr2/bench.py --variant 3b --res 640x480  --frames 25 --matmul
scripts/seedvr2/bench_all.sh
```

`bench.py` caps MPS allocation at 85% of the working set by default, so an
oversized configuration fails in seconds instead of swapping the machine — worth
knowing before running `--variant 7b --mem-fraction 1.0`, which will take the
whole system into swap rather than produce a number. That is not a hypothetical:
it is what happened while building this.

## Licence

Apache-2.0, both code and weights. Permissive; nothing here would block shipping
it, if it fit.
