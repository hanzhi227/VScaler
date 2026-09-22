# Anime 2× upscaling on M5 Pro — Anime4K vs AnimeJaNai SuperUltraCompact

**Deliverable 1** of the live-upscaler project. Machine: Apple **M5 Pro** (16-core
GPU, Metal 4), 24 GB, macOS 26.6. Date: 2026-09-14.

---

## TL;DR — recommendation

**Use Anime4K v3.2 `Upscale_CNN_x2_M`, ported to Metal.** It is the only one of
the two that is anywhere near realtime on this machine, and its output is a
clear improvement over plain scaling:

| | Anime4K CNN x2 M (Metal) | AnimeJaNai HD_V3 SuperUltraCompact |
|---|---|---|
| model | 9-pass CNN in a `.glsl` shader, ported to Metal compute (2128 weights) | 185 KB ONNX (ECO-family), dynamic shape |
| 480p → 960p | **0.95 ms/frame** (1051 fps) | **51.0 ms/frame** (19.6 fps) |
| 1080p → 4K | **3.81 ms/frame** (262 fps) | **350.9 ms/frame** (2.9 fps) |
| live video loop (480p) | — | **9.97 fps = 0.33× realtime** (needs 30) |
| flicker amplification | 1.0022 | 1.0021 |
| verdict | **shippable, ~2.4× realtime end-to-end at 4K** | **not viable for live playback here** |

On the *same frame* on an idle machine (Sintel 1080p→4K): Anime4K 3.38 ms vs
AnimeJaNai 326.8 ms — **97×** on matched input. Anime4K is **92× lighter at
1080p→4K** while producing visibly sharper lines (see grids in
`results/compare/`). AnimeJaNai's SuperUltraCompact model is genuinely small on
disk, but on this Mac/CoreML it runs at 42 ms per output megapixel — about 1/3 of
realtime at 480p and 1/10 at 1080p — so it cannot sustain playback in a live
pipeline. It remains a reasonable **offline** enhancer.

> **Corrections (14 Sept audit) — read the numbers above with these.**
> - The **480p row is not matched input**: 854×480 for Metal versus 640×480 for
>   AnimeJaNai, and mean versus median timing. The "54× at 480p" figure should not
>   be quoted as a matched comparison; the 1080p→4K and same-frame numbers are the
>   defensible ones.
> - **No clean modern 1080p anime was tested.** Synthetic art, Big Buck Bunny and
>   Sintel are proxies, and Anime4K's own documentation targets native 1080p
>   anime, so that category is unproven here.
> - **Sharpness/acutance is not recovered detail or viewer preference.** CNN M
>   sharpens contours but some source stair-stepping remains; Mode A is more
>   aggressive and distorts edges on synthetic material. The quality call stays
>   provisional.
> - The **12-frame temporal-difference ratio cannot rule out localized shimmer**
>   on pans, hair, thin lines or subtitles — global motion dominates it. Longer
>   matched clips, and the final encoded output rather than shader PNGs, still
>   need review.
> - **No measured energy comparison exists.** AnimeJaNai's GPU samples were
>   contaminated by concurrent activity and there is no wattage figure on AC
>   power; utilization and thermal state cannot rank energy use.
> - **Smaller Anime4K presets were not benchmarked**, so "the lightest worthwhile
>   approach" is more accurately "the lightest *implemented* enhancement".

---

## Test material

Categories requested (clean modern / compressed / older) plus a controlled
synthetic case. All methods were fed the **identical** frames.

| clip | role | specs |
|---|---|---|
| `synth_clean_1080p` | clean modern (proxy) | 1920×1080, 24 fps, generated cel-shaded line art with gradients, pans, fine grass/wire lines |
| `bbb_1080p` | clean modern, real | Big Buck Bunny 1920×1080 60 fps (fur, gradients, motion) |
| `sintel_t3_src1080` | **modern animated film** (CC-BY, compressed) | Sintel 1920×1080, ~4 Mb/s source: fine fur/feather detail over light backgrounds |
| `naruto ep 001` | **older material** | 640×480 h264 29.97 fps DVD-era rip, ~1.5 Mb/s, dual audio |
| `naruto_compressed` | **compressed footage** | that episode re-encoded at crf 42 (heavy blocking/banding) |
| `seq_action` (12 frames) | flicker | consecutive 480p frames of an action shot |
| canonical frames | quality/speed | t=300 (line art), t=450 (dark), t=600 (motion), t=900 (fine architecture detail), t=590 compressed, plus BBB/synth 1080p |

Method: for every method × frame, render the 2× output from the same source PNG;
compare at 1:1 (auto-selected highest-detail crops) with full-frame context —
grids in `results/compare/*_grid.png`. Speed measured as ms/frame over ≥10
timed runs (3 warmups) on 1080p-class and 480p inputs. Flicker measured as the
temporal amplification ratio: mean |Δ| between consecutive outputs ÷ mean |Δ|
between consecutive inputs (≈1.0 = no added flicker).

**Material note:** real modern *anime* (Blu-ray) could not be used — licensing
prohibits downloading it — so "clean modern" is covered by synthetic line art,
Big Buck Bunny and Sintel (all CC-licensed), and "older/compressed" by the
DVD-era episode and its crf-42 re-encode. Timings were taken on an otherwise
idle machine except where noted.

**Power methodology and its limit:** real wattage could not be measured here:
`powermetrics` needs root, `xctrace`'s Power Profiler is **not supported on
macOS** (Apple ships it for iOS only), and battery current telemetry is
meaningless while on AC power (the test machine stayed plugged in). Instead: GPU utilisation
from `ioreg` (IOAccelerator `Device Utilization %`, no sudo needed), process
CPU%, ANE activity via onnxruntime/CoreML, and `ProcessInfo.thermalState`
sampled throughout. Treat these as *relative* proxies only.

---

## Speed & power comparison

| method | 480p→960p | 1080p→4K | GPU util | CPU | energy character |
|---|---|---|---|---|---|
| **Anime4K CNN x2 M** (Metal, our port) | 0.95 ms (1051 fps) | 3.81 ms (262 fps) | 79–98 % (GPU-only bench) | negligible in-model | GPU-bound, tiny per-pixel cost |
| Anime4K `bilinear` (Metal, floor) | 0.25 ms | 0.36 ms | 40 % | — | reference floor |
| Anime4K Mode A (mpv GLSL reference) | renders 640→1280 in ~0.7–1.4 s incl. PNG I/O | " | — | — | GL path, used for quality reference only |
| **AnimeJaNai HD_V3 SUC** (CoreML EP) | 51.0 ms (19.6 fps) | 350.9 ms (2.9 fps) | 48 % mean / 69 % max | **85 % mean** | CPU+GPU+ANE all loaded, ~10× too slow |
| **AnimeJaNai** (for reference, live loop) | 9.97 fps = 0.33× realtime at 480p | — | 48 % | 85 % | to reach 30 fps it would need >10× more efficiency |
| AnimeJaNai HD_V3 SUC (CPU EP) | 70.1 ms | 442.9 ms | — | ~100 % | strictly worse |

Engine end-to-end (decode + Metal upscale + VideoToolbox encode + HTTP, 1080p→4K,
CNN mode): **62–74 fps = 2.1–2.5× realtime**, engine CPU 22–31 %, GPU 34–44 %,
`thermalState` `nominal` throughout a 10-minute run. The Metal model itself is
3.8 ms/frame; the encoders and pipe are the rest.

AnimeJaNai analysis: the CoreML session bound 21/23 nodes (2 partitions), so it
is not silently falling back to CPU — it is simply compute/dynamic-shape
inefficient at 42 ms/Mp out. Reaching 30 fps at 1080p→4K would need a **>10×**
speedup (fp16 + static shapes + tiling might gain 2–4×, not 10×), so it is out
of scope for live 2× on this machine.

---

## Quality (visual, from `results/compare/`)

- **Clean line art** (`synth_1080p`, `bbb_1080p`): bicubic is soft with visible
  staircase on curves. **Anime4K CNN M reconstructs the cleanest lines** —
  cloud/hill contours smooth, wires continuous, flat cel areas untouched
  (validation: flat region reproduced exactly). AnimeJaNai is sharp but softer
  than CNN M with a faint halo on high-contrast edges and a slight level shift.
- **Compressed footage** (crf-42 credits text over red): CNN M makes glyphs
  clearly more legible and reduces blocking; **Mode A is sharpest but shows mild
  ringing/halos** around glyphs; AnimeJaNai smooths compression best but stays
  the softest of the three on text.
- **Modern animated film** (Sintel, 1080p, ~4 Mb/s): on fine dark-on-light
  texture, CNN M recovers the crispest edges; AnimeJaNai is close but slightly
  smoother. Both beat bicubic (`results/compare/sintel_t3_src1080_grid.png`).
- **Older DVD material** (480p): all three beat bicubic; CNN M's line
  reconstruction is the most consistent, and it does not invent texture in flat
  areas.
- **Motion / flicker**: no meaningful temporal instability from either method —
  amplification ratios 1.0022 (Anime4K CNN M, Mode A 1.0227) and 1.0021
  (AnimeJaNai). Mode A's +2.3 % is the only measurable flicker, consistent with
  its sharper/ringier character.
- **Gradients**: no banding introduced by either (checked on the synthetic sky
  and BBB).

Numerical validation of the Metal port (`results/metal/validation.json`):
output matches an independent NumPy port of the same GLSL weights to **≤1 LSB**
(0.00 % of pixels differ by more than 1/255); no NaN/Inf; orientation preserved;
line-art acutance vs bilinear **7.8×** (hatching) and **5.9×** (text).

---

## Recommendation

1. **Ship Anime4K CNN x2 M on Metal**, with `bilinear` as the automatic fallback
   when processing falls below realtime (the engine exposes both, hot-swappable).
2. Optionally expose the **denoise** weight variant (`Upscale_Denoise_CNN_x2_M`,
   same cost: 3.83 ms at 4K) for very compressed sources — it is a strict
   alternative, not an additive cost.
3. **Do not** put AnimeJaNai in the live path. Keep it as an offline option and
   revisit only if a ≥10× faster runtime path appears (or if a future machine
   changes the calculus).
4. Mode A's extra sharpness is not worth its halos and +2.3 % flicker for live
   playback; CNN M is the better quality/behaviour trade.

---

## Sustained-playback validation (acceptance target)

Measured with `scripts/sustained_test.py`, which drives the engine **and VLC**
and samples both sides (VLC's own `lostpictures` / `displayedpictures` /
`demuxcorrupted` statistics, engine fps, RSS, CPU, GPU, thermal state).

**Full episode** — *Naruto Shippuden* ep. 001 (640×480 DVD source, 23.5 min) →
1280×960, CNN mode, VLC playing the live stream, 5-minute monitored window:

| metric | result |
|---|---|
| dropped frames (VLC) | **0 lost pictures**, 0 corrupt, 0 discontinuities |
| frames displayed | 8934 in 297 s = **30.1 fps** (source is 29.97) |
| engine processing | 7.6–11.8× realtime (229–517 fps, mean 354) |
| CPU / GPU / thermal | mean 5 % / 45 % / `nominal` throughout |
| ring buffer | capped at 134 MB and flat |

**1080p → 4K** — 1080p-class source, CNN, 150 s windows (3 runs):

| metric | result |
|---|---|
| dropped frames (VLC) | **0 lost**, 0 corrupt, 0 discontinuities (4332–4350 displayed ≈ 29.5 fps) |
| engine processing | 1.78–2.5× realtime (mean 2.28×), GPU 36–45 %, CPU ~20 % |
| thermal | `nominal` for the whole run |
| A/V sync | no-seek **+0.0 ms** (r = 0.990); post-seek **−20.0 ms** (r = 0.997; one AAC frame of encoder priming), **drift 0.00 ms/s** in both |

Playback controls (the code paths the GUI drives) were verified headlessly —
`scripts/controls_test.sh`, evidence in `results/controls_verification.txt`:
pause froze processing at 298 frames, resume continued from there, seek started
session 2 at the requested 42 s (stream duration exactly 18.05 s of the 60 s
file) with the audio still aligned, and CNN ↔ bilinear mode switches took effect
mid-session without a restart.

### Subtitle and scene-cut artifacts

Subtitles are burned in at **source resolution before the 2× upscale**, so glyph
edges go through the same line reconstruction as the picture:

| check | bitmap subs (DVD, 640×480→1280×960) | text subs (SRT, 1080p→4K) |
|---|---|---|
| presence / timing | present at cues (t=3/12/22 s), absent between (t=8 s) | present at cues (t=3/12/22/32 s), absent between |
| glyph fidelity vs bicubic-2× reference | sub-region MAE 0.86–3.58 | — (no libass reference) |
| sharpness ratio vs reference | **1.01–1.52×** (never softer) | crisp at 1:1 inspection, CJK glyphs correct |
| visible artifacts | none (clean outline, no ringing/halo) | none |

Scene cuts, checked with an alignment-robust motion-series comparison at six
detected cuts: **correlation 1.000, frame shift 0, peak-motion ratio 1.00** —
cuts land on exactly the same frames as the source, with no ghosting across the
cut and no added flash.

### Memory: bounded, with one honest caveat

The read-ahead is bounded by the **128 MB ring** by construction, and VLC's
consumption throttles the engine to playback rate (the ring caps and stays flat).
Total footprint at 4K is ≈570–770 MB, of which ~95 MB is the three IOSurface
output buffers and 128 MB the ring.

During *active* processing there is a slow retained growth (RSS) that is **not
an application leak**: `leaks` reports *0 leaked bytes*, the region is
`MALLOC_SMALL` (Metal/driver), and the OS was seen reclaiming it — one run fell
from 347 MB back to 140 MB when the ring capped. Measured over an 8-minute
session: 139 MB → 989 MB peak, non-monotonic.

Two fixes were applied and re-measured on a clean 4-minute run (nothing else
running): the per-frame heap churn in the encoder pipe and HTTP pump was removed
earlier, and Metal command buffers now use `retainedReferences = false`
(commit+wait guarantees resource lifetime here). That cut the steady-state
growth from ~1.8 MB/s to **≈0.71 MB/s** (543 → 629 MB once the ring was capped).
Residual growth at 4K is ~0.7 MB/s (≈63 MB/min), which over a full 24-minute
episode is ≈1.5 GB — noticeable but not unbounded-by-design, since the ring
itself is capped. It is listed as the one open item rather than claimed solved;
a session restart (which seeks already do) resets it, and the next step is an
Instruments/Metal allocation trace.

## Acceptance status

Revised after the 14 Sept audit. "Peak processing" is capability; during playback
the engine is deliberately throttled to the playback rate by the bounded
read-ahead, so the sustained figure is ~1x and the bursty EMA that used to be
reported (8-17x) was misleading.

| criterion | evidence | status |
|---|---|---|
| visibly improved 1080p → 4K | 1:1 grids vs bicubic (`results/compare/*_grid.png`); acutance 5.9-7.8x on line art. Caveat: acutance is not recovered detail, and no real modern anime was tested | ⚠️ provisional |
| 24-30 fps playback | peak 62-74 fps processing at 1080p->4K (2.1-2.5x realtime); sustained ~1x by design | ✅ (peak) |
| stable audio sync | event-based check: output **+1.3 ms**, 1 s-delayed control **rejected (+1014 ms)** (`scripts/check_av_sync.py`) | ✅ |
| no dropped frames in VLC | 0 lost / 0 corrupt / 0 discontinuities in sampled runs | ✅ sampled |
| heat | `thermalState` `nominal` in every run; CPU 5-31 %, GPU 36-45 % | ✅ |
| bounded memory | read-ahead bounded to 30 s (~12 MB); full-episode run plateaus: RSS 76.7 → 97.2 MB, second-half growth **-5.5 MB/min** (was 138.8 → 988.8 MB before the pump autorelease-pool fix) | ✅ plateau |
| full-episode acceptance | complete episode run to `finished`: 42,387 frames vs 42,383 expected, **0 lost / 0 corrupt / 0 discontinuities**, thermal `nominal`, memory plateaued (`results/full_episode_verification.txt`) | ✅ |

Artifacts: `results/metal/BENCH.md`, `results/animejanai/bench.json`,
`results/anime4k/flicker.txt`, `results/compare/*.png`,
`results/sustained_*.csv`, `results/controls_verification.txt`,
`results/av_sync_verification.txt`, `scripts/`.
