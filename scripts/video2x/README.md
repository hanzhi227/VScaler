# Video2X

[Video2X](https://github.com/k4yt3x/video2x) — "a machine learning-based video
super resolution and frame interpolation framework". C++, AGPL-3.0, 21.6k stars,
v6.4.0 (Jan 2025).

**Verdict: measured, and there is nothing here to adopt.** Video2X is not a model
and not a candidate for the Enhancement menu. It is an orchestrator that drives
other people's upscalers over Vulkan, it has no native macOS build, and it is an
offline batch tool by design. Its two headline capabilities are already covered
here: the upscaler it drives by default is **Anime4K**, which is this project's
existing default model, and its other upscaler is **Real-ESRGAN**, which is
measured in `scripts/realesrgan/`. The frame-interpolation half is structurally
incompatible with a live player (see below).

This is a shorter note than `scripts/rt4ksr/README.md` because there is no model
to port and no throughput to measure that this project has not already measured
for the underlying networks.

## What it actually is

From upstream's own README, v6.4.0 supports:

- **Anime4K v4** and "all custom MPV-compatible GLSL shaders"
- **Real-ESRGAN**, **Real-CUGAN**, and **RIFE** "(all models)", via **ncnn** and
  **Vulkan**
- two modes: **filtering** (upscaling) and **frame interpolation**

So the upscaling it performs is the arithmetic of Anime4K, Real-ESRGAN or
Real-CUGAN, executed by ncnn on a Vulkan device. It contributes the pipeline
around them — frame extraction, model dispatch, re-encode — not a network.

That matters for the question actually asked of this project. "Can Video2X play
live?" decomposes into two questions this repository has already answered:

| Video2X backend | this project's position |
|---|---|
| Anime4K v4 | the **default model of the shipping app** (`--mode cnn`, the Metal port of Anime4K v3.2 `Upscale_CNN_x2_M`); measured at 3.81 ms/frame at 1080p→4K and **0.99× real time** |
| Real-ESRGAN | measured here: `animevideov3` x4 at **1.45× real time** at 480p→1080p on MPS, but it **loses badly on quality** — 28.42 dB against the Anime4K CNN's 35.53 dB on anime, 27.81 against 29.90 on live action (see `scripts/realesrgan/README.md`) |
| Real-CUGAN | not evaluated; a further waifu2x-family per-frame CNN, same shape of cost as Real-ESRGAN |
| RIFE | frame *interpolation* — changes the frame count, see below |

Adding Video2X as a live mode would therefore mean adding a Vulkan/ncnn dispatch
layer on top of models the app already runs natively on Metal (and, for Anime4K,
already ships). It cannot make those models faster than they already are; it can
only make them slower.

None of its backends is a memory problem — the project's **2 GB model budget**
rules out the models measured in `scripts/seedvr2/`, not these: Real-ESRGAN
measures 37–141 MB of live GPU memory here, and Anime4K is already shipped under
that budget. The objection to Video2X is structural, not one of size or speed.

## It cannot run on this machine as native code

Verified, not assumed:

```console
$ gh api repos/k4yt3x/video2x/releases --jq '.[] | .assets[].name' | sort -u
Video2X-x86_64.AppImage
video2x-linux-ubuntu2204-amd64.deb
video2x-linux-ubuntu2404-amd64.deb
video2x-linux-amd64.zip
video2x-qt6-windows-amd64-installer.exe
video2x-qt6-windows-amd64.zip
video2x-windows-amd64.zip
```

There is no macOS artifact in any release, and upstream's README says
"Cross-platform support for **Windows and Linux**". macOS appears only as a
**container**: "Video2X container images are available on the GitHub Container
Registry for easy deployment on Linux and macOS". Upstream also requires that
"the GPU must support Vulkan".

On this machine:

- no `video2x` formula in Homebrew, and no `video2x` on PyPI (`pypi.org/pypi/video2x` → 404) — v6 is a C++ rewrite, not the old Python package
- a Vulkan loader exists (`/opt/homebrew/lib/libvulkan.1.dylib`, 1.4.357 — MoltenVK), but Video2X has no macOS binary to load it
- `docker` is installed (29.7.2) but **the daemon is not running**, so the container path is not available either

And the container path would not answer the question anyway: Docker on macOS is a
Linux VM with no Metal or Vulkan GPU passthrough, so a containerised Video2X
would either fall back to CPU or fail on the Vulkan requirement — neither of
which is a measurement of what this pipeline's Metal path could do.

## Frame interpolation is not a live-player feature

The interpolation half of Video2X is not merely slow, it is the wrong shape.
RIFE converts N frames into 2N or 4N. This player's position is defined as
`frameIndex / fps` on a shared audio/video render clock
(`docs/../README.md`, "How playback works"), and its A/V sync is structural
because video and audio are stamped on the same session timeline. Inserting
frames changes the number of frames the clock must cover, so interpolation is an
**offline re-encode** operation: it has to finish, re-time and re-mux before
anything can be played. It is a different product from a live upscaler, and the
app's `--scale` is magnification, not frame multiplication.

## Licence

**AGPL-3.0** — a strong network copyleft. This project is not AGPL. Executing a
separately installed binary is one question; *bundling* Video2X into
`VScaler.app` would put the bundle under AGPL-3.0 obligations. That is a
reason not to vendor it, and it is worth recording next to the permissive
licences of everything actually adopted here (Anime4K MIT, RT4KSR Apache-2.0,
Real-ESRGAN BSD-3-Clause, SeedVR2 Apache-2.0).

## Reproducing the checks

```bash
gh api repos/k4yt3x/video2x --jq '"license=\(.license.spdx_id) lang=\(.language) stars=\(.stargazers_count)"'
gh api repos/k4yt3x/video2x/releases --jq '.[] | "\(.tag_name) \(.published_at)"'
gh api repos/k4yt3x/video2x/releases --jq '[.[].assets[].name] | unique | .[]'
curl -sS -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/video2x/json   # 404
brew info video2x                                                              # no formula
docker info                                                                    # daemon not running
```
