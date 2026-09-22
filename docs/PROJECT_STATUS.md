# Project status — live 2×/4× anime upscaler (M5 Pro)

> **16 Sept — a user found a hole in yesterday's work, and it was a hole in the
> *test*, not the code.** Reported as "animevideov3 looks bad on real video thats
> not anime". Reproduced on genuine native 480p, and the report is correct.
>
> **Every live-action clip scored in this project is a synthetic degradation** — a
> lanczos downscale of a known-good reference. That is close to invertible, and
> inverting it is what a reconstruction model is built for. Real video is not a
> downscale of anything: it is natively degraded by noise, compression and motion,
> and a GAN has no reference to be faithful to. `test-media/liveaction/README.md`
> even says this trap out loud about its own 4K clip — *"a soft HD source
> downscaled to 540p is very nearly invertible by bilinear filtering, so it
> flatters methods that do not sharpen"* — and yesterday's "best on SD live-action
> detail" claim walked straight into it.
>
> **What it actually does**, on 854×480 non-anime game footage (the mode runs on
> it; 3416×1920 stays inside the encoder clamp):
>
> | hypothesis | measurement | verdict |
> |---|---|---|
> | temporal instability | flicker **1.001×** the source's own over 40 frames with motion (bilinear 0.969×, general-x4v3 1.036×); flat regions 32.97 vs the source's 32.95 | **no** |
> | tone / colour shift | luma mean 76.16 vs the source's 75.60; saturation 39.10% vs 41.15% | **no** |
> | a port or precision bug | matches PyTorch to 59.99 dB; live matched offline to 53.18 dB on the same frame | **no** |
> | **replacing structure instead of sharpening it** | mean abs difference from the source: bilinear 3.196, animevideov3 **5.124**, general-x4v3 5.482 | **yes** |
>
> On the blue particle effect, individual sparks become smooth rounded balloons and
> the blown specular highlight loses its internal structure; rock and foliage turn
> into soft blobs (`results/general/yt_mhw_480p/grid_real_particles_animevideov3.png`).
> The CNN and MetalFX keep it. That is generative hallucination on out-of-domain
> material — animevideov3 is trained on anime *video*, and a game trailer is not
> that.
>
> **What changed as a result, and what did not.** The mode stays, the port is
> unaffected, and nothing about it was wrong as code. What changed is the claim:
> "best on SD live-action detail" was an *acutance* number on a synthetic downscale,
> and it does not survive contact with real video. The menu hint now says so where
> the mode is chosen, and the claim is narrowed in `README.md`,
> `scripts/realesrgan/README.md`, `docs/OFFLINE-UPSCALING.md` and
> `docs/GENERAL-VIDEO-UPSCALING.md`. Its honest niche is now **SD anime** — and even
> there the CNN beats it on every fidelity measure.
>
> **The generalisable lesson: a test set you constructed can only flatter the thing
> it was constructed from.** Three corrections have now come from outside the
> harness — the retracted "Real-ESRGAN is 10–100× too slow" projection, the
> linker-strips-the-weights claim, and this. The first two were caught by measuring;
> this one was caught by a user watching a film, which is the only instrument that
> was ever going to catch it. **Add a natively-degraded clip to the scoring set
> before trusting a "looks good" claim again.**

> **16 Sept — `general-x4v3` is a live mode too, on request: "add general-x4v3 as
> an option so I can judge myself."** This reverses yesterday's verdict, and the
> reversal is deliberate rather than a discovery.
>
> **What did not change: it still loses.** Re-scored through the player it sits
> below `animevideov3` on both clips, as the offline render did — PSNR 26.18/26.36
> against 28.36/27.79, acutance further from the truth, 1.84× the cost. It is also
> genuinely *not real time* on a 480p source: **0.63× real time at
> 854×480→3416×1920** once the graph is warm, against `animevideov3`'s 0.92× in the
> same conditions. So it drops frames where the shipped mode does not.
>
> **What changed is what the verdict is based on.** Those scores come almost
> entirely from *synthetic* lanczos downscales of known references — the lesson
> three entries above. A mode that can be switched in one keystroke on the user's
> own material is a better instrument than a table built on clips the user has
> never watched. It ships as `ScalerKind.realESRGANGeneral` /
> `--mode realesrgan-general`, **with its menu hint stating that it measured
> worse**, so choosing it is deliberate rather than a surprise. The two graphs are
> built lazily per geometry, so the unselected mode costs nothing per frame.
>
> **The cost, paid on purpose this time.** Its 3.3 MB weight literal moves back
> into the library the app links: the binary is **6,464,752 bytes** again, from
> 3,161,376 with the checkpoint bench-only. That number was wrong once before
> because it was read off a binary left over from a failed build, so it was
> re-measured here on a build that succeeded. The `resrbench` copy of the weights
> file was deleted — two copies of the same public enum in one program would not
> compile, and the bench still reaches it through `import RealESRGANMetal`.
>
> **New tooling, because "judge it yourself" needs an artifact.** `scripts/
> realesrgan/pair_grid.sh` dumps one frame through bilinear and both checkpoints
> from the live path and composes 1:1 crops next to the source
> (`results/general/yt_mhw_480p/grid_real_*_esrgan_pair.png`: the blown specular
> highlight, foliage texture, and flat text on the rating card). It checks its own
> frame alignment rather than assuming it — the dumped frame has to beat its
> neighbour by 6 dB against the nearest-upscaled source, which is the same
> instrument that caught the off-by-one when the frame index was first used.
> `scripts/make_eval_grid.py` grew `--ref` and `--labels` so it can compose dumps
> from `playercheck` rather than only from a full eval run.
>
> **Both modes are now in the scored live-mode lists**, so the next
> `scripts/eval_all_general.sh` regenerates a `realesrgan-general` row through the
> player next to the offline `general-x4v3` row. Having both is the cross-check:
> the live mode should reproduce the offline render, as it does for
> `animevideov3`.

> **15 Sept — the app has five live modes: Real-ESRGAN `animevideov3` is ported and
> shipped.** Asked for as "can seedvr2 run here and real-esrgan? If so, integrate
> them into the live options." The answer splits cleanly, and both halves are
> measured rather than argued.
>
> **SeedVR2 cannot, and that is unchanged.** 0.30× real time at its best and
> 7.6 GB in the smallest configuration that runs; even the 4-bit MLX build, which
> is the only version that runs here at all, holds 2.75–2.90 GB — over the 2 GB
> budget — at 0.006–0.017× real time. Nothing about it is close. Details in
> `scripts/seedvr2/README.md`.
>
> **Real-ESRGAN could, and the reason it now ships is a measurement that reversed
> a decision this project had already published.** The earlier call was "not
> ported", on quality: `animevideov3` wins on photographic detail and loses on cel
> animation, which the Anime4K CNN already covers. That finding has **not**
> changed — it is re-derived below and ships with the mode. What changed is cost:
>
> | | measured | |
> |---|---|---|
> | eager PyTorch/MPS, whole frame | 22.59 ms at 480×270 → 1080p | 0.68× of a 30 fps budget |
> | **the port that shipped** | **7.33 ms** | **0.22× of a budget — 3.1× faster** |
> | in the player, 4× | 1.007× / 0.992× / 0.991× real time at 270p / 480p / 540p | 140–353 MB resident |
>
> **The port does not use a hand-written Metal kernel, and that was the
> measurement.** `scripts/bench_alu_throughput.swift` and
> `scripts/bench_matrix_throughput.swift` put this GPU's ceilings at **8.4 TFLOP/s
> fp16** and **4.2 TFLOP/s** for `simdgroup_matrix` — the matrix units are
> *slower* — against 160.7 GFLOP of work at 480×270. A perfect SIMT kernel would
> reach ~19 ms; a realistic one worse; MPS already achieves ~7.1 TFLOP/s effective.
> There was nothing for a kernel to win. The model is an **MPSGraph** executable
> and Metal does only the two things the graph cannot: BGRA frame → NHWC fp16
> tensor, and the 48-channel output → pixel-shuffled 4× picture with the network's
> nearest-neighbour residual. The 3.1× is fusion — eager PyTorch dispatches conv,
> bias and PReLU separately and writes 64 channels to memory between each of the
> 17 pairs; a graph executable does not.
>
> **It is verified rather than asserted.** Through the shipping code path the
> output matches PyTorch's fp16 MPS run to **59.99 dB** (max error 0.7 of one
> 8-bit level — rounding-order noise between two runtimes, not a layout bug) and
> the fp32 model to 59.97 dB; re-deriving the network from the exported fp16
> weights reproduces fp32 to 70.4 dB. The live mode also reproduces the offline
> render down to the metric: acutance 0.971× live against 0.971× offline on live
> action, PSNR 27.79 against 27.81; 28.36 against 28.42 dB on anime.
> `scripts/realesrgan/verify_metal.py`, `results/general/realesrgan_metal.txt`.
>
> **Two new behaviours, because this is the first model fixed at one scale.**
> `SRVGGNetCompact` for video is x4 only — there is no x2 checkpoint — so a 2×
> request runs at 4× and says so (`requestedScaleFactor` 2, `scaleFactor` 4) rather
> than silently showing half the magnification; and a source whose 4× output
> exceeds the encoder's 4096×2304 is **refused with the reason**, where every other
> mode could satisfy any request by clamping. That makes the mode SD-only, ≤1024×576.
> `ScalerFactory.resolveFactor` is shared by the player and the stream server so
> they cannot disagree.
>
> **The quality finding is unchanged, and this is the part that matters most.**
> On anime `animevideov3` is the **worst** mode measured: PSNR 28.36 against the
> Anime4K CNN's 35.56 (SSIM 0.8959 against 0.9475, smooth-region error 1.911
> against 1.383, acutance 1.971× against a 1.000× truth) — it posterises flat cel
> gradients, 5.7 dB below plain bilinear's 34.08. On SD live action it is the **best**:
> acutance 0.971×, the only mode within 3%, and the most legible on fine text at
> 3×. It is offered, it is not the default, and its Enhancement-menu hint says
> which content to point it at. The comparison view settles it per clip in one
> keystroke, which is the point of having both.
>
> **The generalisable lesson, now with two examples pointing opposite ways.** This
> module already existed because it disproved a published claim: a single-layer
> microbenchmark's 2.67 TFLOP/s was extrapolated to a whole network and produced
> "10–100× too slow", where the measured family reaches ~7.2 TFLOP/s. The port
> then repeated the mistake in the other direction — assuming a hand-written
> kernel would beat a framework, when measuring the hardware ceilings showed it
> could not. **Both errors were projections from a smaller measurement to a
> bigger one, and both were caught by measuring the whole thing.**
>
> **Follow-up the same day: the other Real-ESRGAN checkpoint, measured and
> rejected.** Asked as "what about the other esrgan? general-x4v3 480x270 ->
> 1920x1080 47.58 ms 21.02 fps". It had only ever been *timed*, never scored, so
> "the other one" was an open question. Both answers are now in:
>
> | | anime 270p→1080p | | live action 270p→1080p | | cost @270p |
> |---|---|---|---|---|---|
> | | PSNR | SSIM | PSNR | SSIM | |
> | animevideov3 (shipped) | **28.36** | **0.8959** | **27.79** | **0.8601** | **7.50 ms** |
> | general-x4v3 | 26.18 | 0.8870 | 26.36 | 0.8504 | 13.76 ms |
> | Anime4K CNN | 35.56 | 0.9475 | 29.90 | 0.8719 | — |
>
> It has **the lowest PSNR and SSIM of every mode tested on both clips**, and it
> loses to `animevideov3` on all four of those columns — **including on live
> action, which is the general-purpose checkpoint's entire reason to exist**
> (27.79 against 26.36 dB). Its acutance is genuinely well calibrated (1.036× of
> the truth, second only to the shipped mode's 0.971×) and everything else about
> it is worse. It is also 1.84× the cost, so its real-time window would be
> *narrower*: real time at 270p and 360p, past a 30 fps budget from 854×480 up,
> where the shipped mode is still inside it at 540p.
>
> **Slower and worse is not a trade-off, so it is not adopted** — but it is
> ported, since it is the same class at a different depth: the pipeline now takes
> a `RealESRGANModel`, the exporter takes `--variant`, and `resrbench --variant`
> measures either. Reproducing or revisiting the decision is two commands.
>
> **And packaging it wrong cost the app 3.3 MB, which is the kind of thing this
> log exists to record.** Its generated weights first went into the same library
> the app links, on the assumption that an unreferenced public enum would be
> dead-stripped. It is not: the app binary went from **3,160,784 to 6,464,752
> bytes**. They now live in the `resrbench` target and the app is back to
> 3,161,376. The lesson worth keeping is how nearly this was missed — the first
> check was a `strings` probe against a binary left over from a **failed build**,
> which reported the previous build's contents and said the weights were absent.
> **A stale artifact answers a question about the wrong binary**, and only a
> controlled build-with/build-without comparison is immune to it.
>
> **It also surfaced a real bug, of the quiet kind.** `srvgg.Upscaler` had no way
> to choose a checkpoint — `WEIGHTS_NAME` was hardcoded — so `render.py --variant
> general-x4v3` passed `num_conv=32` and loaded the *anime* file into it. That
> mismatch is silent at load: the two depths share every layer except the last,
> so it fails only at `body.34`, pointing at the final convolution when the fault
> was the file name. The variant table now lives in one place in `srvgg.py` and
> the paths come from it.

> **15 Sept — SeedVR2 measured, and a model budget written down: 2 GB.**
>
> **The budget first, because it decides the rest.** No upscaling model may hold
> more than 2 GB. It is stated in `README.md` now, and it is a hard constraint
> rather than a preference: 26 GB of unified memory is shared with the OS and the
> player keeps three frames of uncompressed BGRA in flight. The four shipped modes
> are far inside it — MetalFX has no weights at all, the Anime4K CNN's are
> kilobytes — so it had never needed saying until a model arrived that could not
> meet it.
>
> **SeedVR2 was the model worth checking properly, and it fails on two independent
> counts.** It is a **one-step** diffusion restoration transformer, so unlike a
> 50-step model it could not be dismissed by structure, and the shortlist ranked it
> first for restoration quality. Measured in two runtimes, both kept:
>
> | | PyTorch/MPS, released shapes | MLX 4-bit (`mlx-gen`) |
> |---|---|---|
> | best frame time | 110 ms/frame = **0.30× real time** (640×480; the DiT alone, 9 frames batched, VAE excluded — a floor) | ~1.8 s/frame at **1× restore**, ~5.5 s at 2× = **0.006–0.017× real time** |
> | memory | **7.63 GB** at 640×480 and *one* frame; 12.87 GB at 25; 7B **cannot load** (16.01 GB vs a 15.10 GB ceiling) | **2.75–2.90 GB** peak RSS in every configuration that runs |
> | resolutions the app uses | **1080p, 4K and upstream's own 100×720×1280 all run out of memory** before producing a frame | video **upscaling is refused**: "safe video mode only supports source-size restoration or smaller output" |
>
> The 4-bit MLX build is the substantial finding inside that: it is the only
> SeedVR2 that runs on this machine at all, and it cut storage 13.57 → 2.54 GiB
> (~9×). It is still over budget, and forcing a 1 GB MLX cache **raised** peak RSS
> to 2.90 GB and broke the run — the footprint is the weights plus the runtime, not
> a trimmable cache. **Not adopted on either count alone.**
>
> **The architecture was verified rather than read from the config**, which
> mattered: `configs_3b/main.yaml` says `expand_ratio: 4` and `mm_layers: 10`, and
> the shipped weights say the MLP is 2.7 × dim (hidden 6912) and that blocks 0–9
> are dual-stream text+video while 10–31 are video-only. A model built from the
> config would have been wrong in both directions. Counting from the safetensors
> headers — fetched by HTTP range request, a few KB of a 6.8 GB file —
> reproduces the released checkpoint to **0.02%** (3.3908 B modelled vs 3.3915 B
> shipped). Tools: `scripts/seedvr2/`.
>
> **Video2X is not a model and was never a candidate**, which the shortlist
> obscured by listing it beside models: it is an AGPL-3.0 C/C++ **framework** that
> drives Anime4K v4, Real-ESRGAN, Real-CUGAN and RIFE through ncnn/Vulkan, and its
> default upscaler is the Anime4K CNN this app already ships. No macOS binary
> exists in any release (macOS is container-only, and the daemon is not running
> here), and its frame-interpolation mode changes the frame count, which a clock
> defined as `frameIndex / fps` cannot use. Structural, not a speed or size
> question — `scripts/video2x/README.md`.
>
> **Real-ESRGAN also got the memory measurement it was missing**: **37–141 MB** of
> live GPU memory from 480×270→1080p to 960×540→4K, weights 2.5 MB and 4.9 MB. So
> it clears the budget by 14–54× and it holds real time, and it is *still* not
> adopted — on picture, which is the entry below. It is the only model in this
> round that passes both hard bars and fails only on taste.
>
> **One measurement worth reusing.** `torch.mps.driver_allocated_memory()` reports
> a flat 1.12 GB for Real-ESRGAN at every input size; that is the allocator's
> reserved pool, not the requirement. `current_allocated_memory` is the number
> that scales with the frame. Reporting the first one as "memory used" would have
> overstated this family by ~8×.

> **15 Sept — a published claim of ours was wrong, and the model that disproved it
> is the best one we have measured on live action.** Raised as a five-model
> shortlist ("our picks by priority": SeedVR2, Video2X, FlashVSR, Real-ESRGAN
> AnimeVideo-v3, Real-CUGAN). Classification and the reads on the other four are in
> `docs/OFFLINE-UPSCALING.md`; the part that changes this project is AnimeVideo-v3.
>
> **We had published that Real-ESRGAN-class models are 10–100× too slow to play
> live. They are not.** The claim in `results/general/realesrgan_feasibility.txt`
> came from a **single-layer** Swift Metal microbenchmark:
> `scripts/bench_realesrgan_conv.swift` times one 3×3 64→64 fp32 PReLU layer at
> 640×480, achieving 2.67 TFLOP/s, and the network's cost was then projected from
> that rate times layer count. Measured end to end through PyTorch/MPS
> (`scripts/realesrgan/bench.py`), the same family reaches **~7.2 TFLOP/s**, and
> the numbers are not close:
>
> | | projected | measured |
> |---|---|---|
> | `realesr-general-x4v3`, 640×480 in | 277.7 ms/frame | ~100–110 ms (bracketed by 84 ms at 640×360 and 152 ms at 854×480) |
> | `realesr-animevideov3`, 480×270 → 1920×1080 | (not projected) | **22.59 ms/frame = 44.26 fps = 0.68× of a 30 fps budget** |
>
> The error was in the extrapolation, not the kernel measurement: the layer is one
> of 18–34, it ran alone, and the per-dispatch overhead was therefore charged once
> against a single layer's arithmetic before being multiplied up. The file now
> carries both the retracted projection and the correction, and the three documents
> that cited it (`README.md`, `docs/GENERAL-VIDEO-UPSCALING.md`, this file) are
> fixed. **The generalisable lesson: "too slow" is a claim about one implementation
> on one machine, and it does not transfer — to a whole network, to a different
> runtime, or to the next model on the list.**
>
> **AnimeVideo-v3 is real time and it is content-specific.** `SRVGGNetCompact`,
> 18 convolutions of 3×3 with PReLU, 621,424 params, one 2.4 MB checkpoint, both
> released checkpoints loading `strict=True` into the vendored class. At 480×270 →
> 1920×1080 it is **44.26 fps**, so the app's SD 4× case fits with room to spare
> before any Metal port. On picture it splits cleanly by content:
>
> * **SD live action: the best mode tested.** Acutance **0.971×** of the ground
>   truth — the only mode within 3% (Anime4K CNN 0.768×, MetalFX 1.995×, RT4KSR
>   0.567×, bypass 0.390×) — and at 3× zoom on Japanese shirt text it is visibly the
>   most legible of six panels
>   (`results/general/liveaction_270p_to_1080p_4x/grid_shirt_text_with_animevideov3.png`).
>   It pays for it: worst PSNR (27.81) and SSIM (0.8603) of every mode including
>   bilinear, and the most added temporal activity of the model modes (1.060× the
>   source's own frame-to-frame change, against the CNN's 1.008×).
>   *(16 Sept: "of every mode including bilinear" was true when this was written
>   and is not any more — `general-x4v3`, scored the same day, is worse still at
>   26.36 dB. These are the offline render's figures; the live mode measures
>   27.79/0.8601 and 1.061×, which is the same picture.)*
> * **Anime: it loses to what is already shipped.** PSNR 28.42 against the CNN's
>   35.53, SSIM 0.8964 against 0.9473, smooth-region error 1.905 against 1.381 —
>   it posterises flat cel gradients, which is exactly what
>   `grid_anime_with_animevideov3.png` shows and what a GAN at 1.972× acutance
>   against a 1.000× truth is doing by design.
>
> **Not ported**, and that is a judgement the measurements support rather than
> force: it would be a fifth live mode whose win is confined to photographic
> detail, which MetalFX (sharper) and the CNN (cleaner) already cover between them,
> on an app whose stated purpose is animation. The model, the renderer, the bench
> and the scored frames are all kept (`scripts/realesrgan/`,
> `results/general/realesrgan_bench.txt`), so reversing it costs an afternoon.
>
> *Superseded later the same day — see the entry above. It was ported, at 7.33 ms
> instead of 22.59, because an MPSGraph executable fuses the conv/bias/PReLU chain
> that eager PyTorch dispatches as separate passes. The quality judgement in this
> entry is unchanged and still stands.*
>
> **Also newly measured: `general-x4v3` at 47.58 ms/frame = 21.0 fps** at the same
> 1080p output — near real time, not real time, and 2× the MACs by design. Both
> checksums and both sizes are pinned in `fetch_weights.py`, which now fetches
> whichever checkpoint the bench wants (`--only`).
>
> **One honest limit on the numbers above.** They are PyTorch/MPS: a hand-written
> Metal port would be faster, so 44 fps is a floor on animevideov3 and 21 fps is a
> floor on `general-x4v3`, not a ceiling. And all of it is fps for the *model*
> alone — playback also decodes, composites, encodes and plays audio, which is why
> the budget column in `results/general/realesrgan_bench.txt` is the honest way to
> read it.

> **15 Sept — an `Off` mode, and MetalFX becomes the default.** Asked for as "add a
> mode where it is off. Also please default to MetalFX."
>
> **`Off` is a true passthrough, not a slow path.** A new `copyTexel` kernel copies
> the decoded frame to the output texture and nothing else; output geometry is the
> source's own (`ow = kind.magnifies ? sw * factor : sw`), and the result is
> **pixel-identical to the decoded source** — verified rather than asserted: a
> dumped `Off` frame matches ffmpeg's decode of the same source frame exactly, all
> 921,600 channel values, max difference 0. (The dump is 1-based, so
> `frame_0090.png` is source frame 89; the first comparison, against frame 90,
> showed a mean difference of 3.8 and was an off-by-one in the check, not the
> player.) That makes it usable as a control as
> well as a viewing mode: it is the only way to see exactly what a mode is adding.
> Magnification does not apply to it — the 2×/4× toggles, the Comparison View
> toggle and both comparison side menus are disabled while it is selected, and
> `applyComparison` refuses rather than compositing two different pixel grids. The
> refusal is explicit in the status line: "The comparison needs a mode that scales;
> Off plays at the source's own size."
>
> **Switching to or from Off restarts the session; switching among the scaling
> modes stays a hot swap.** Output dimensions change across the magnify boundary,
> so the swap cannot be done in place; `setKind` detects that case and restarts at
> the requested position plus a tenth of a second, while `cnn → cnnDenoise`
> continues to swap in place with no rebuffer. Verified both ways: metalfx → off
> gives 960×540, off → metalfx gives 1920×1080, cnn → cnnDenoise stays hot.
>
> **MetalFX is now the default** (`ScalerKind.defaultKind = .metalfx`) in the app,
> the embedded player and the stream CLI, and `ScalerKind.parse` accepts
> `off|none|native|unscaled` for the new mode (`bypass|bilinear|original` still
> mean bilinear, so an old command line does not silently change meaning). The
> default was verified byte-identical to an explicit `--mode metalfx`.
>
> **And verified in the installed bundle, not just the build tree.** Launched from
> `/Applications/MacVideoUpscaler.app` with no mode override and again with
> `MACVIDEOUPSCALER_MODE=metalfx`, both dumped frame 90 as the **same 1280×960 frame
> with the same sha256** (`7152314e…`); launched with `MACVIDEOUPSCALER_MODE=off` the
> GUI produced 640×480 and that frame was pixel-identical to ffmpeg's decode of the
> source. So the passes-through path and the default both survive bundling,
> signing and install, and the `Off` entry is reachable from the shipped menu.
>
> **Note the scope of that choice.** The measurements in this file are unchanged
> by it: on anime the Anime4K CNN is still the more faithful mode (37.51 dB against
> bilinear's 39.42 for MetalFX), and this app is for anime. MetalFX is the default
> because it is ~4× cheaper and the safer general-purpose choice for material that
> is not animation — not because it won a quality comparison. Switching is one
> click, or `--mode cnn`, and does not restart playback.

> **15 Sept — RT4KSR evaluated, and not adopted.** Raised as "we may want to test
> out RT4KSR and support as well", after realbasicvsr's ~0.5–1 fps settled that a
> recurrent model cannot play live. [RT4KSR](https://github.com/eduardzamfir/RT4KSR)
> is the opposite case and worth stating plainly: it is *fast enough to adopt* and
> *not good enough to adopt*, and the two conclusions come from different
> measurements.
>
> **It is fast.** 44,521 parameters; after folding each 1×1 → 3×3 → 1×1 block into
> one 3×3 convolution the whole network is 14 convolutions, and
> `scripts/rt4ksr/bench.py` measures **3.3–6.0 ms/frame** (480×270→960×540 through
> 960×540→1920×1080) on MPS — 0.08–0.18× of a 30 fps budget, against 278 ms/frame
> for Real-ESRGAN `general-x4v3` at 480p. *[The 278 ms figure is **retracted**: the
> measured cost is ~100–110 ms, and `animevideov3` is 22.6 ms. See the 15 Sept
> correction entry at the top of this file.]* A Metal port would be cheaper again, so
> the cost argument that ruled out every previous candidate does not apply here.
>
> **It is not better.** Scored on all five clips by the existing harness, it never
> wins a column: it ties the best PSNR twice (29.90 dB at 270p→1080p, 33.52 dB at
> 1080p→4K) and loses the other three, is softer than the Anime4K CNN on every clip
> (0.57–0.85× acutance against 0.77–0.90×), and has lower SSIM than plain bilinear
> on three of five. At 3–5× zoom on shirt and poster text it reads as a cleaned-up
> bilinear. The reason is the training data, not the implementation: it is trained
> on **bicubically downscaled clean images** and has no deblocking stage, and its
> published gain over bicubic is +0.27 dB. A fifth mode measuring at or below the
> four already there would be a menu entry, not an improvement, so none was added.
> It stays in `scripts/rt4ksr/` — verified, runnable and reproducible — so that
> decision is cheap to revisit if a 4× model or a compression-trained version
> appears.
>
> **Vendoring, not depending.** Upstream's `test.py` cannot run on this machine as
> written (it hardcodes `.cuda()`, wraps everything in `DataParallel`, and calls
> `reparameterize()` with no arguments), so the class bodies are vendored with
> three changes, each verified rather than assumed: the fold is reimplemented
> device-agnostically; `torchvision.transforms.GaussianBlur` is dropped, because
> the high-frequency branch that consumes it is **dead** in the released config
> (`forget` is hardcoded `False`, so `hf_feats` is computed and never read); and
> the `torchvision` dependency goes with it. Against upstream on the real weights:
> the training-shaped class is **bit-identical**, the fold is within 4.1e-06, and
> dropping the dead branch changes the output by **exactly 0**.
>
> **A metric in this project was wrong, and the new clips exposed it.** `flicker`
> was reported only as a ratio to the bypass run. The anime clips are effectively
> *frozen* — the source differs frame to frame by 0.0006–0.0015 levels out of 255,
> and frames 100–102 of the 270p source are bit-identical — so that ratio was
> dividing by essentially zero. It had produced "RealBasicVSR flickers 47×" and
> "RT4KSR flickers 0.02×" on those clips, both arithmetic on a near-zero
> denominator rather than measurements. The harness now prints the **absolute**
> frame-to-frame difference first (and as a ratio to the source's own), labels the
> ratio unreliable when the bypass is under 1.0 level, and the docs carry a table
> of how much each clip actually moves. RealBasicVSR's instability on static
> content is real in absolute terms — 0.716 against the bilinear run's 0.027 —
> but "47×" was not the number to quote.
>
> **PiperSR checked and rejected.** Proposed as a real-time ANE-native
> alternative. Its repository `.gitignore` excludes `*.mlmodel`, which matches the
> CoreML model *specification*, so a clone yields weights with no architecture and
> cannot run; its own `benchmark.py`/`inference.py` call
> `ct.ComputeUnit.CPU_AND_NEURAL_ENGINE`, which is not a coremltools member, and
> pass a `{"input": array}` float array to a model whose input is `input_image`, an
> image type requiring a `PIL.Image` — so its published numbers cannot have come
> from its published harness. Despite "any resolution (CoreML flexible input
> shape)" the shipped model declares a **fixed 128×128** input (320×240 is
> rejected outright), which makes its "360p full-frame, 44.4 FPS" row impossible
> and identical to its own 128×128 tile row. Its card also contradicts its README
> (Set14/BSD100/Urban100 are "—" on the card, filled in on the README), and its
> PSNR is RGB with no border crop while every model it compares against is
> Y-channel with a border crop. Tested from the PyPI wheel, which does contain the
> spec: a 128×128 tile runs in 1.29 ms, so tiling 960×540 costs ~51 ms (~19 fps) —
> workable, slower than RT4KSR, with seams, on unsubstantiated quality claims.
>
> **What was added:** `scripts/rt4ksr/` (`rt4ksr.py`, `fetch_weights.py`,
> `bench.py`, `render.py`, `README.md`), `scripts/eval_all_general.sh` (the whole
> five-table comparison in one reproducible command), the flicker-metric fix in
> `scripts/eval_general_upscaler.py`, an RT4KSR row in all five tables in
> `docs/GENERAL-VIDEO-UPSCALING.md`, and `grid_shirt_text_with_rt4ksr.png` /
> `grid_poster_text_with_rt4ksr.png` crops including it. `results/general/*/metrics.json` were
> regenerated and now carry `source_flicker` and `flicker_ratio_reliable`. One
> mishap worth recording: `results/general/liveaction_540p_to_1080p/grid_shirt_text.png`
> was overwritten in place before its parameters were known, and was rebuilt from
> the dumped frames at the sibling grids' convention (crop 240×160, zoom 3, 2
> columns). Same size and content class, but the exact crop origin of the original
> is not recoverable, so that one file is a reconstruction rather than the
> original. The 5-model version is `grid_shirt_text_with_rt4ksr.png`.
>

> **15 Sept — RealBasicVSR removed.** Asked for as "Then remove support for
> realbasicvsr", after the paused-comparison work. Gone: the `ExportController`
> and the whole *Enhancement ▸ Export* path in the app (menu item, ⌥⌘E, progress
> badge, `_EXPORT` launch hook), `scripts/realbasicvsr/` (`enhance.py`, `rbvsr.py`,
> `fetch_weights.py` and the four MPS investigation scripts),
> `docs/REALBASICVSR.md`, the bundle's copy of the Python runner in
> `make_app.sh`, the `--realbasicvsr` mode of `scripts/eval_all_general.sh`, and
> the 210 MB `~/.cache/macvideoupscaler/RealBasicVSR_x4.pth`.
>
> **The player is self-contained again, which is the point.** The export was the
> one feature that needed something outside the bundle — a Python with PyTorch and
> the weights — and the only one that could not be a playback mode, because the
> network is recurrent. What is removed is the code path, not the record: its
> measurements and the reasoning behind rejecting it as a live mode stay in
> `docs/GENERAL-VIDEO-UPSCALING.md` and the dated entries below, and existing
> renders under `results/general/*` are untouched.
>
> Nothing replaced it: the live modes (Anime4K CNN / CNN + denoise / MetalFX /
> bypass) are unchanged, and an offline ×4 restore is no longer offered from the
> app at all. Checked by grepping the repo for the name (only historical
> evaluation text and result files match) and by rebuilding and launching the app,
> whose Enhancement menu now ends at the comparison items.
>
> **15 Sept — the comparison view, on a paused frame.** Reported as "There is a
> comparison view. It doesn't really work when I pause the video. Please make it
> work when I pause." Correct, and it was structural rather than a bug in the
> compositor: the overlay is applied per frame by the video thread, and pausing
> *parks* that thread. So moving the line or switching a model updated the status
> snapshot and the UI — the percentage, the slider, the captions all moved — while
> the picture could not change, because no frame was produced to change it.
> Measured: with the player paused at 3.8 s, a split change and a left-model change
> left `framesProcessed` at 114 and produced no output frame at all.
>
> **Why this needed more than a redraw call.** The frame on screen has already been
> handed to the renderer, and the player does not keep it: the decoder's output is
> uploaded into one reusable source texture and the frames ahead of the clock are
> sitting in the display layer's queue. A frame cannot replace the image on screen
> by being enqueued at its own timestamp either — that queue is ordered, and the
> frames in it are *later*. So the player now keeps the sources of the last
> `bufferFrames + 1` frames (a `SourceRing`) **once the comparison has been used in
> a session**, and a change that only affects how the picture looks wakes the
> parked producer, which flushes the display layer (keeping the image on screen),
> re-renders the frame the clock has reached *and* the frames queued behind it, and
> re-queues them all at their original timestamps. The first one presents because
> its timestamp has been reached; the ones behind it are restored rather than
> skipped, so resume is unchanged. Two things were measured first in isolation
> rather than assumed: a frame enqueued at or below a *stopped* clock's time is
> presented immediately, and a flush followed by a re-enqueue of lower timestamps
> works and leaves the queue consistent for the resume.
>
> **It covers both halves and the enhancement, not just the line.** Switching the
> left model, the right model (the Enhancement menu), or the comparison itself
> takes effect on the frozen frame too — so the "a mode swap while paused is
> deferred until resume" note in the README is now wrong and has been corrected.
> The redraw is exactly the play path's work: the accepted frame is a real
> scaler's output and the composite is the same kernel, so a half is still what
> that model produces. Verified from the **presented** image, not from what was
> rendered — the layer's `displayedPixelBuffer` read back through a new `readback`
> harness verb, because a queued frame is not proof of what is on screen. At split
> 25% on a paused frame the presented left 320 columns are byte-identical to a
> bypass run and the right 958 to a CNN run, divider exactly 2 white columns at
> x=320–321; repeated redraws reproduce the same frame byte-for-byte; the render
> clock and `framesProcessed` do not move through eight such changes; and resuming
> continues at full rate. Full evidence:
> `results/paused_comparison_verification.txt`.
>
> **The one case that is not exact, and why.** With the comparison switched on
> *while paused* and nothing retained (it was off for the whole session) there is
> no history to redraw from. The still then lands on the newest frame the producer
> has and the clock moves onto it — +0.066 s in the measurement — after which it is
> stable, and the next pause after any playback is exact. That is a real caveat,
> not a rounding: it is the only case where the frame you stopped on is not the
> frame you are looking at. The alternative was rebuilding the session at the
> paused position, which costs ~0.6 s of "Preparing…" for a slider drag and would
> have been worse.
>
> **A pre-existing bug this turned up: the divider was never drawn when both sides
> were the same model.** The README claimed CNN-vs-CNN "draws only the divider",
> but there were no white columns at all — `dividerSplit` declared its output at
> texture index 0 where the dispatcher binds the output at index 1 (correct for
> the composite kernel), and its thread grid spanned only the columns beside the
> split while the kernel tests absolute columns, so the pass painted into the wrong
> place and then matched nothing. Both fixed; the same-model case now differs from
> a plain CNN frame in exactly the 2 divider columns and nothing else. Found only
> because the paused path made me look at what is *presented* rather than at what
> the compositor was asked to do.
>
> **Cost.** The ring is allocated on first use, not on playback: nothing is paid
> by anyone who never opens the comparison, and it is `bufferFrames + 1` source
> frames thereafter — 5 MB for a 480p source, 33 MB for 1080p — with a byte cap so
> a 4K source gets fewer entries rather than a bigger budget. The redraw itself
> allocates a frame or two outside the pool threshold, because it is re-queueing
> frames the flush released.
>
> The app needed one hook to verify this end to end: the split presets in the menu
> carry no key equivalent and a synthetic click cannot drag the line, so
> `_SPLIT_AT="5:0.1,7:0.6"` moves the divider through the same
> `AppModel.setCompareSplit` the slider calls. Driven that way, the app paused by a
> click on the picture renders the paused frame again with the new split — divider
> at x=768 for 60% where the playing-phase frames had it at x=320 — with the clock
> frozen. One thing that made the first attempt at that run look like a failure: a
> click within 18 pt of the line grabs the divider instead of pausing, and with the
> split at 50% the centre of the picture *is* the line, so the click that was meant
> to pause moved the divider instead. That is the documented behaviour of the grab
> zone, not a new bug, but it is worth knowing when scripting the app.
>
> **15 Sept — RealBasicVSR, as an offline export.** Asked for as "I want
> RealBasicVSR", after the previous entry concluded that no Real-ESRGAN-class model
> could play live. It is in now — *Enhancement ▸ Export with RealBasicVSR…* (⌥⌘E)
> and `scripts/realbasicvsr/enhance.py` — and it is **an export, not a mode**:
> the network is recurrent (SPyNet flow + 20 residual blocks propagated in each
> direction, plus up to three cleaning passes per frame), so a frame cannot be
> produced from the player's 2–3 frame buffer. Measured **0.30 s/frame** at
> 480×270→1080p and **1.02 s/frame** at 540p→4K on MPS, against 0.9–0.99 ms for
> the live modes — about 300× the per-frame cost. A 24-minute SD episode is ~3½
> hours. fp16 is 2.5× faster and was rejected: 44/255 peak deviation.
>
> **The model is the published one, not a re-implementation.** MMagic's network
> classes are vendored verbatim and only the imports that pulled in
> mmcv/mmengine/mmagic were replaced, so `RealBasicVSR_x4.pth` loads with
> `strict=True` — 6.3 M parameters, `generator_ema` weights, 20 cleaning and 20
> propagation blocks per direction, all confirmed against the checkpoint.
>
> **Its quality is the most interesting result so far: best by eye, worst by
> metric.** On live-action 480×270→1080p it scores PSNR **27.40 dB / SSIM 0.8284**
> — *below plain bilinear's 29.28*, and the lowest of the four models — while the
> crop is visibly the cleanest (crisp kanji strokes where bilinear is mush, the
> CNN is blobby and MetalFX rings). Before believing that, I checked the usual
> explanation: a sub-pixel shift. PSNR is maximised at zero shift for every mode,
> so the score is real. The reason is the model: RealBasicVSR is trained with GAN
> and perceptual losses, i.e. to look real, and PSNR/SSIM are precisely the
> metrics that punish that. On anime it is the wrong tool outright — 27.83 dB
> against the Anime4K CNN's 35.55 — so Anime4K stays the anime model.
>
> **It also shimmers on held frames.** The flicker window for the anime clip lands
> on a shot where the source is byte-identical frame to frame. The other models
> produce essentially nothing (0.02–0.04 mean |Δ| on 0–255); RealBasicVSR produces
> **0.97**, repeating with the chunk period, because the output depends on a
> frame's position inside its chunk. The 47× ratio against a near-zero
> denominator is a bad summary and is reported as such; the absolute number is the
> real finding. The official implementation carries no recurrent state across
> chunks, so the exporter matches it as a baseline but defaults to two frames of
> look-behind — measured to cut the shimmer 31% (0.72→0.50) and the encoded file
> 41%, for 25% more compute.
>
> **Three MPS bugs turned up, all of which produced silently wrong pictures rather
> than errors.** Frame buffers arrive NHWC, and the NCHW view of one keeps a
> channels-last stride; on that layout MPS `grid_sample` returned an identity-warp
> error of **3.4 on a 0–1 image** against 5e-07 on CPU, which is inside SPyNet, so
> the optical flow came out with a max of **119 where the CPU's own max was 6.6**
> and the whole output was garbage — it took a per-level pyramid trace to see that
> the divergence started at the *coarsest* level, where the flow is zero and the
> warp is an identity. Forcing a contiguous NCHW view fixed it. MPS also has no
> `border` padding for `grid_sample`, which SPyNet uses at every level (border =
> clamp the coordinate to the edge, verified bit-identical to clamping the grid),
> and `Tensor.type(x.type())` cannot parse `'torch.mps.FloatTensor'`, so upstream
> `flow_warp` raises on Apple GPUs at all. After the fixes, flow agrees between
> backends to **107 dB**, cleaning to **145 dB** and the final output to **33 dB** —
> that residual is arithmetic in the propagation blocks being amplified by the
> recurrence, and is why no bit-exactness is claimed.
>
> **One consequence worth stating: this is the first feature that is not
> self-contained.** Playback still needs nothing but the bundle; an export needs a
> Python with PyTorch and 210 MB of weights. `make_app.sh` copies the runner into
> `Contents/Resources/realbasicvsr`, the app probes for an interpreter preferring
> versioned ones (`/opt/homebrew/bin/python3` here is 3.14 while torch lives in the
> 3.11 site-packages, so probing `python3` alone finds a Python that cannot run the
> model), and it reports what is missing rather than failing silently. The whole
> path is verified through the installed bundle with `_EXPORT=<path>`, and the
> bundle's output is byte-identical to the development binary's.
>
> **15 Sept — a general-video upscaler, and the measurement that refused to
> flatter it.** Asked for as "add the best one for general video upscaling",
> against a shortlist of MetalFX Spatial, Real-ESRGAN `general-x4v3`, `x2plus`,
> Anime4K and RealBasicVSR. **MetalFX Spatial is now a mode** (`--mode metalfx`,
> *MetalFX Spatial* in the Enhancement menu, selectable on either side of the
> comparison view). It is the only one of the five that can run here — and the
> measurement says it is *not* an upgrade over what was already shipped, which is
> the part worth recording.
>
> **Four of the five are excluded by arithmetic, not taste.** The app is a live
> player, so a model that cannot hold ~30 fps is a different feature, not a slower
> one. `general-x4v3` is 32 convolutions of 64×64 channels at input resolution,
> 1.21 M MACs per input pixel. Rather than assert that was too slow, I benchmarked
> one such layer with a register-blocked fp32 Metal kernel (2.67 TFLOP/s achieved,
> texture-cached reads, each fetch reused across 8 output channels — deliberately
> the *favourable* case) and projected the network: **277.7 ms/frame at 480p =
> 0.12× real time**, 0.04× at 720p, 0.02× at 1080p. The Anime4K CNN already in the
> app does 1080p→4K in **3.81 ms**. `x2plus` is an order of magnitude heavier
> again, and RealBasicVSR needs optical flow and a sequence window, which is a
> different pipeline. MetalFX Spatial takes the whole thing in **0.33 ms** at
> 640×480 → 1280×960 and **0.99 ms** at 1080p→4K.
>
> *[**The Real-ESRGAN half of that paragraph is retracted.** The projection was
> ~2.6× too pessimistic, `general-x4v3` measures ~100–110 ms at 640×480 = ~0.45×
> real time, and `animevideov3` is 22.6 ms at 270p→1080p = 44 fps, i.e. faster than
> real time. The Anime4K, MetalFX and RealBasicVSR figures in it stand. See the
> 15 Sept correction entry at the top of this file and
> `results/general/realesrgan_bench.txt`.]*
>
> **The finding that matters: MetalFX is the sharpest mode and the least
> faithful.** It carries a fixed, aggressive sharpening stage with no knob to
> dial it back (`MTLFXSpatialScaler` exposes geometry, formats and
> `colorProcessingMode` — that last one I A/B'd, and the shipped `.perceptual`
> beats `.linear`, 1.69× vs 1.86× acutance). Measured against a high-resolution
> reference on four clips, its `acutance` is **1.54–2.18× the ground truth** — not
> "resolved detail the source lost" but edge energy the original never had — and
> it has the lowest PSNR and SSIM of the four modes in every live-action run. On
> anime it is **worse than plain bilinear** (37.51 vs 39.42 dB). The expectation
> going in was that MetalFX would be the *conservative* choice ("looks clean but
> not dramatically sharper"); the opposite is what is measured. Meanwhile Anime4K
> CNN lands at 0.73–0.90× — *under*-sharp — so neither hits 1.00, and undershoot
> is the safer error because it does not draw edges that are not there.
>
> **The method was built to be checkable, and one number is weaker than the
> rest.** `scripts/eval_general_upscaler.py` drives the player and scores the
> pixel buffers it actually handed to the display layer, on a
> downscale-then-recompress proxy against a high-resolution reference; re-running
> it reproduces every published figure to the last digit. The `flicker` ratio
> (frame-to-frame difference against a bilinear run) is the weak one and is
> flagged as such in the report: it divides one frame difference by another, so a
> sharper image legitimately reads as less stable whether or not it is. It shows
> MetalFX amplifying the most (1.05–1.71×), and it cannot distinguish that from
> "sharper".
>
> **The repo had no live-action material at all** — every clip was anime or CG —
> which is exactly the content this request is about, so two Wikimedia Commons
> clips were added (`test-media/liveaction/`, CC BY-SA, provenance and rebuild
> commands in its README; the 112 MB 4K one is fetched rather than committed).
> The sharp 4K source matters: a soft HD original downscaled to 540p is very
> nearly invertible by bilinear, which flatters methods that do not sharpen, so
> both a soft and a sharp original were measured.
>
> **Where it earns its place.** On the 4K-sourced clip it is competitive with the
> Anime4K CNN by eye and ~0.7 dB behind numerically, at a quarter of the GPU
> cost — so it is the mode for clean 720p/1080p live action and for tight GPU
> headroom. It also turned out that 4× is *one* pass rather than two chained 2×
> passes (MetalFX takes the output geometry directly), which I checked rather than
> assumed: single-pass measured better on both axes, acutance 1.94× vs 2.18× and
> PSNR 29.76 vs 28.80 dB. **Anime4K stays the default.** *[Superseded: MetalFX
> became the default on request later the same day — see the entry at the top of
> this file. Anime4K is still the better mode on animation, which is what this
> paragraph establishes.]* Two claims that would
> have been convenient did not survive: there is no MetalFX quality knob, and the
> temporal scaler — which *would* be the better answer on motion — needs motion
> vectors and depth that an ffmpeg decode does not produce. One deviation is
> stated rather than hidden: Apple documents the MetalFX output texture as
> `storageMode.private`, and this pipeline hands it an IOSurface-backed pooled
> buffer so frames can go straight to the display layer.
>
> **Verification.** Real time at 0.992–1.007× across 4× SD, 2× 540p and 2×
> 1080p→2160p (300/300 frames each); a mid-movie CNN→MetalFX switch leaves frame
> 60 byte-identical to a pure-CNN run and frame 120 byte-identical to a pure
> MetalFX run; the comparison view's left half is byte-identical to a pure MetalFX
> run and the right half to a pure CNN run; and the GUI launched with
> `_MODE=metalfx` produced a frame byte-identical to `playercheck --mode metalfx`.
>
> **15 Sept — formats: what plays, and the two things that did not.** Asked for as
> "add support for mp4 and other video formats". mp4 already played; the work was
> finding out what that claim was worth and where it was false. A matrix generator
> (`scripts/make_format_matrix.sh`) now builds one file per container/codec
> combination and `scripts/check_formats.sh` plays each: **36 of 36 play**, from
> mp4/m4v/mov/mkv through webm (VP8/VP9), ts/m2ts/mts, flv, avi, divx, mjpeg,
> mpeg-1, mpeg-2 (.mpg/.vob/.mxf), wmv, asf, 3gp and RealVideo, with aac, opus,
> vorbis, mp3, mp2, ac3, wmav2 and PCM audio, all at 0.98–1.07× real time. Also
> measured: 10-bit H.264 and 10-bit HEVC (Hi10P anime converts down to BGRA), a
> 641×481 source (odd dimensions), 5.1 audio, and the user's own 11-minute 480p
> mp4 (0.996× at 2×, 0.997× at 4×, seeks to 300 s, three 10-s skips).
>
> **Two failures, both "opens but shows the wrong thing".** *Cover art is a video
> stream*: `-map 0:v:0` on a file whose poster frame comes first decoded one still
> image and stalled (1 frame in 5 s). The probe now lists every video stream, drops
> the ones flagged `attached_pic`, and among the rest takes the longest, which also
> covers a muxer that forgets to flag the poster. A file with no video at all
> (audio plus cover) now reports "has no video — only cover art" rather than
> playing the artwork. *Rotation was ignored*: a phone clip carries a display
> matrix rather than rotated pixels, so a portrait recording played on its side —
> a documented gap that turned out to matter as soon as "mp4 support" was taken
> literally. The probe reads `side_data_list.rotation` (or the legacy `rotate`
> tag), swaps the geometry, and the decoder transposes explicitly with
> `-noautorotate`. The filters were picked by measuring against ffmpeg's own
> autorotation and came out **bit-identical** at 90°/180°/270°, so a rotated file's
> output now matches that reference at 41–42 dB PSNR where a wrong direction scores
> ~12 dB. Text subtitles are drawn upright on the rotated frame; a bitmap
> subpicture is composited before the turn, because its position is a window
> defined against the source frame.
>
> **Acceptance was part of the problem.** The bundle declared four content types,
> so every other container was invisible to *Open With*, and the Open panel named
> those same four — greying out files the engine reads happily. The bundle now
> declares 20 types (every extension resolving to a system UTI, Matroska excepted
> and imported) and the panel takes `[.movie]`. `.ogv` is declared with no sample
> because this ffmpeg has no Theora encoder: stated rather than quietly counted.
>
> **Measurement notes.** Two mistakes of mine are worth recording because they
> nearly became wrong claims: a rotated test file built with `-metadata rotate=90`
> was not rotated at all (modern ffmpeg drops it — `-display_rotation` is an input
> option), and two "rotated subtitle" files built with `-c copy` had no subtitle
> stream at all (ffmpeg's default mapping omits subtitles), which made me believe
> bitmap subtitles were being lost by the rotation when the file simply had none.
> Both were caught by checking the test material itself instead of only the code
> under test.
>
> **15 Sept — comparison view: two models, one picture, a movable split.** Asked
> for as "a split filter to show original vs post pipeline", then extended to
> "select which model including the original video" and "a slider to fully cover
> the page left or right". ⌥⌘C splits the picture on a thin white line: the left
> half is one model, the right half is the enhancement, and the slider (or a drag
> on the line itself, or three menu presets) moves the boundary anywhere from 0%
> to 100% — so either model can cover the whole picture.
>
> **Each half is a real model's output, not an approximation.** The first cut
> redrew the left half with a bilinear kernel of its own; that was both weaker (the
> left could only ever be the *original*) and slightly dishonest (at 2× it matched
> the bypass path to ±1 LSB, and at 4× it was a single 4× pass where bypass chains
> two 2× passes — 49 dB PSNR, max delta 17). The left side is now produced by an
> actual `FrameScaler` into a scratch texture and composited, so **any** pair of
> models can be compared and each half is bit-for-bit what that model outputs.
> Verified at 50% (left Original, right CNN): the left 640 columns are
> byte-identical to a bypass run and the right 634 to a CNN run, while each half
> differs from the other model's version of it. Same model on both sides skips the
> second pass entirely — CNN vs CNN is byte-identical to a plain CNN frame apart
> from the divider.
>
> **The extremes are exact, and 0% is free.** Split 0% is byte-identical to a plain
> enhancement frame because the pass is skipped outright (there is nothing to
> composite); split 100% — Original covering the picture — is byte-identical to a
> bypass frame. So "show only the enhancement" costs and changes nothing.
>
> **The grab zone is the only part that could have gone wrong.** The picture is also
> the window's drag handle and the play/pause control. A press within 18 pt of the
> line moves the line; anywhere else still drags the window, a click still pauses, a
> double click still goes full screen. Verified with a synthetic
> press-drag-release on the line: it lands at 0.000, 1.000 and back to exactly
> 0.500 with playback uninterrupted, and a click afterwards still pauses.
>
> **Cost.** The second pass is real GPU work, but the frame loop is paced by the
> renderer rather than by the GPU, so it did not cost frame rate here: two
> *different* CNNs held 0.995× realtime at 480p→1280×960 and 1.000× at 4×→2560×1920,
> against 0.997×/0.999× for one model. Memory is the tighter constraint — a second
> CNN brings its own weights and intermediates (~305 MB in flight at 4× against
> ~205 MB). A slower GPU or a larger output could turn that GPU work into dropped
> frames, which is why this reads as a comparison tool rather than a playback mode.
>
> Two diagnostics were added while testing this, both to stop a harness run from
> doing something it should not: `_MUTE=1` (a test run should not play the film out
> loud — an earlier attempt to silence it by pointing the audio ordinal at a
> nonexistent track silently fell back to track 0 and played anyway, because an
> unknown ordinal is deliberately clamped to the first track), and
> `_COMPARE`/`_COMPARE_LEFT`/`_SPLIT`. The GUI's own path is verified by launching
> the app with those hooks and comparing its frame 120 byte-for-byte against
> `playercheck` at the same settings — they match.
>
> **The shortcut is ⌥⌘C, not the ⇧⌘C it should have been.** ⇧⌘C is the natural
> sibling of ⇧⌘S (stats), and it is what the item declared — until it turned out to
> be unverifiable and possibly dead. Two separate things showed up on the way: the
> toggle had been gated with `.disabled(state == .idle)`, and SwiftUI builds the menu
> while the player is still idle, so the item held a stale *disabled* state that
> claimed-and-swallowed its key equivalent (the ungated ⌃⌘F worked through the very
> same path, which is what isolated it). With the gate removed the item is correctly
> enabled with a real action — but no synthetic ⇧⌘ press reaches any SwiftUI item in
> this app, for any letter (⇧⌘C and ⇧⌘X both do nothing), while the ⌥⌘ form is
> delivered and fires repeatedly. Because ⌥ does not change the *character*, the ⌥⌘
> test is faithful to a real press and the ⇧⌘ one cannot be trusted either way, so
> shipping a keystroke that might do nothing was worse than shipping an unusual one.
> ⇧⌘C is a one-line change if the pair matching ⇧⌘S is preferred. Either way the
> feature is reachable without any shortcut: the transport-bar button and the
> Enhancement menu both toggle it.
>
> **15 Sept — fast skipping, and the freeze that came with it.** Reported as
> "skipping forward fast multiple times doesn't work". Three separate things were
> wrong, and the short 60 s test clip hid one of them:
>
> **Rebuild per press.** Every press tore the session down and rebuilt it (~0.6 s of
> ffprobe + ffmpeg respawn), so a burst queued a string of rebuilds and the picture
> lagged far behind the presses. Skips are now **debounced 500 ms and summed**: four
> quick "forward 10" presses are one rebuild landing at +40 s (verified: one rebuild,
> landing at 44.7 from 4.7), presses a second apart still step 10 s at a time, and a
> running "+30s" is drawn over the picture as the presses arrive so they are
> acknowledged while the debounce waits. (The badge was first written as a view
> that was inserted and removed with a transition while its *number* changed, which
> cross-faded the digits — "+20s" and "+30s" smeared into each other — and the
> animation sat on the enclosing container, so a skip starting also smeared the
> "Preparing…" text under it. It is now a stable view whose *opacity* alone is
> animated, and the digits swap instantly.)
>
> **A main-thread deadlock.** Seeks ran synchronously on the main thread, and inside
> them `Process.waitUntilExit()` ran a *nested run loop*. A second seek delivered by
> that nested loop re-entered `restart()` and blocked on the mutex the outer call was
> holding, which could then never unwind — the app froze, spinning at 100 % CPU with
> the picture stuck (sampled: `restart → FFmpegAudioDecoder.stop → waitUntilExit →
> __CFRunLoopRun → dispatch_main_queue_drain`). Fixed in three parts: a `Subprocess.wait`
> helper replaces every `waitUntilExit()` (all 8 call sites) with a non-pumping reap;
> session rebuilds run on a serial queue, so seeks no longer block the UI and cannot
> re-enter each other; and skips accumulate in a `pendingSeek` target so a rapid
> second press can no longer read the clock mid-rewind and move backwards.
>
> **A pause lost during a rebuild.** Pausing while a seek was still rebuilding was
> silently undone when that rebuild started the clock (`pausedFlag` was cleared to let
> the new session prime). The viewer's intent is now tracked separately
> (`pauseRequested`) and read at the moment the clock starts, so a pause always wins;
> `isPaused` reports the intent, not whether producers happen to be parked. Also found
> and fixed: the "Preparing…" overlay is drawn over the centre of the picture on every
> seek and was swallowing clicks aimed at the video (`allowsHitTesting(false)`).
>
> The 60 s test clip is genuinely part of the report: five presses from ~7 s reach the
> end, and a target past the end clamps to the last half second, so further presses
> appear to do nothing. Skipping is verified on the 10-minute clip.
>
> **15 Sept — double-click toggles full screen.** The obvious conflict is that the
> first click of a double click is delivered as an ordinary click, so acting on it
> immediately would pause the film every time someone went full screen. The single
> click is therefore held back for up to **300 ms** (the mpv/VLC figure — macOS's own
> `doubleClickInterval` is 500 ms, which felt sluggish for play/pause, and a faster
> system setting is still honoured) before toggling playback; a second click inside
> that window cancels it and toggles full screen instead. Verified with the in-app
> click hook: a double click flips `full=0 → 1 → 0` across two presses while the
> state stays `processing` and frames (148 → 358) and audio (6.06 → 13.08 s) keep
> advancing — the double click does not touch playback — and it works while already
> full screen, which also confirms clicks still reach the surface there. The single
> clicks around it still pause and resume. The View menu item now reads *Enter* or
> *Exit Full Screen* to match the window, tracked from the window's own enter/exit
> notifications rather than from the button presses, since the green button, ⌃⌘F and
> a swipe can all change it.
>
> **15 Sept — shipped as an app bundle.** `scripts/make_app.sh` builds
> `MacVideoUpscaler.app` (release binary + `AppBundle/Info.plist` + a generated icon),
> ad-hoc signs it, and installs it to `/Applications` (`--out DIR` to assemble
> without installing); it assembles in a temp directory and moves it into place, so
> only one copy of the app ever exists. The first version also left a second bundle
> in `dist/`, and the user found two apps in Spotlight — both copies registered with
> LaunchServices, which is exactly the confusion the move avoids. installed, it
> launches from Spotlight, the Dock or the Finder like anything else. The bundle is
> self-contained by construction — both Metal libraries are compiled from source
> strings inside the binary and ffmpeg/ffprobe are absolute paths — so nothing else
> had to be copied in, and the app can be moved anywhere. Opening a video is now the
> Finder's job: `application(_:open:)` was wired up (with the URL parked if the
> Apple Event beats the SwiftUI scene) and an **imported UTI for Matroska** declared,
> since macOS has no `.mkv` type for the Finder to match on — verified with `mdls`
> reporting `org.matroska.mkv` conforming to `public.movie`. `LSHandlerRank` is
> *Alternate*, so the app appears under Open With without hijacking double-click.
> A cold Finder launch of the installed app was verified end to end: `open -a
> /Applications/MacVideoUpscaler.app clip.mkv` played it (frames 21 → 261, audio 1.83 →
> 9.81 s, both renderers `.rendering`) with the decoders confirmed reading that path.
> The diagnostics hooks were made independent of each other and moved to `NSLog`, so
> a bundle launched from the Finder can be observed with `open --stderr`.
>
> **15 Sept — pause/resume silence, misleading rate, click-to-pause.** User report:
> playback was fine until the first pause, after which it showed **0.17× realtime
> and no audio**. Both were real, and independent.
>
> The silence: the audio producer gave up on a timer. `audioLoop` waited for
> `AVSampleBufferAudioRenderer.isReadyForMoreMediaData` and `break`-ed out of the
> loop after ~4.5 s of not-ready; a *paused* renderer holds a full queue for the
> whole pause, and since the audio thread spends most of its life in that wait, the
> first pause of any length killed audio **permanently** — the audio path has no
> restart, so nothing ever refilled it. Reproduced headlessly (pause at 6 s, resume
> at 16 s): audio feeding froze at frame 174 / playback 5.75 s, exactly the pause,
> and never advanced again through 10 s of continued playback. It now parks while
> paused and only returns on EOF or a renderer that has actually failed, which is
> reported. Same run after the fix: 6.91 s → 14.25 s of PCM fed, continuous across
> two pauses and a seek.
>
> The number: `realtimeRatio` was `framesProcessed / wall time since session
> start`, so every paused second counted as decode time and dragged the average
> down for the *rest of the session* — a realtime bypass run read **0.62×**, and the
> app raised its "will stutter" warning off that. It is now a sliding window keyed
> to the **render clock** rather than wall time, so a pause adds nothing to either
> delta (the clock is stopped too) instead of averaging the silence in, and the
> priming burst before the clock starts is excluded rather than baked in. Verified
> at 0.990–1.006× across four concurrent 1080p→4K denoise runs, each within 1% of
> an independent `frames/fps ÷ playback` computation.
>
> Also added: **clicking the picture toggles play/pause**. The window has no title
> bar, so the view now handles the press itself — a press that moves drags the
> window (`performDrag`), a press that does not is a click — because AppKit's
> background drag would otherwise swallow it. Verified twice over: through the real
> `PlayerView.mouseDown/mouseUp → onSurfaceClick` path in `playercheck`
> (`--script "…:click"`), which pauses at 9.71 s and resumes on a second click; and
> in the **running GUI**, where `_CLICK_AT` sends a synthetic click through
> `NSApp.sendEvent` so window hit-testing decides the target — it logs
> `hit=PlayerView` and the state walks processing → paused → processing, which is
> the one thing a harness cannot check (whether the SwiftUI overlay stack above the
> surface eats the click). The stats overlay gained an "audio fed" row, so a silent
> audio path is visible rather than inferred.
>
> **15 Sept — in-app player.** The GUI no longer streams to VLC. It plays the
> upscaled frames itself (`PlayerController` + `EmbeddedPlayer`: decode → Metal
> upscale → `CVPixelBufferPool` → `AVSampleBufferDisplayLayer`, with audio on a
> shared `AVSampleBufferRenderSynchronizer`). This addresses three user-reported
> playback problems at once: the MPEG-TS stream was not seekable (so VLC's
> scrubber could not work), every run required launching a separate owned VLC
> instance, and the read-ahead had to be 30 s to keep that pull-based player fed.
> The buffer is now 2–3 frames (~100 ms), seeking restarts the decoders and
> lands on the target, and VLC is no longer involved. The encode → MPEG-TS → HTTP
> → VLC path is unchanged and still shipped as the `macvideoupscale` CLI. Evidence
> and the honest gaps (no window screenshot, 4K footprint) are in `README.md`;
> the decision record is updated in `docs/VLC-INTEGRATION-DECISION.md`.
>
> **15 Sept — UI and blank-video follow-up.** The window is now video-only with
> all controls in the menu bar, a fading transport strip for scrubbing, a
> toggleable stats overlay (⇧⌘S) and real full screen; the display layer is
> hosted as an aspect-fitted sublayer so letterboxing is correct at any size.
> Two real bugs were found and fixed while investigating the "blank video"
> report: the first open clamped any requested start offset to 0 (duration not
> yet known), and 4× on HD was allowed although it measured **0.61× real time**
> with an **883 MB** buffer (now clamped to 2×). The blank the user saw was the
> source's own ~3 s fade-in plus a cold start with no first frame yet — the
> player now shows "Preparing…" until the first frame exists. Subtitles were
> verified to survive the 4× double pass for both bitmap and text tracks by
> dumping the rendered frames.
>
> **15 Sept — "stuck on loading video".** Diagnosed from `ps`/`sample`/`lsof` on a
> live instance: the ffmpeg children were in state **`T` (stopped)**, not sleeping.
> They inherited the terminal on **stdin**, and ffmpeg reads stdin for interactive
> keyboard commands; that read raised **SIGTTIN** because the app's process group
> was not the terminal's foreground group, so the kernel stopped the decoders and
> playback hung on "Preparing…" with no error. `SIGCONT` un-stopped them, which
> confirmed it. All seven spawned subprocesses (video/audio decoders, ffprobe,
> subtitle extraction, muxer, VLC) now set stdin explicitly — `/dev/null` except
> the muxer, whose stdin is its rawvideo pipe. A 12 s watchdog now fails the
> session with a message if a decoder produces nothing, so this class of hang can
> never be silent again. Reproduced and verified under a pseudo-terminal with the
> app backgrounded, on the user's own Episode 003.
>
> **15 Sept — mid-movie model/scale switching audited.** Mode changes are a hot
> scaler swap with no restart; verified by pixel equality against reference runs
> (the frame before a CNN→bypass swap is byte-identical to pure CNN, the frame
> after byte-identical to pure bypass, against a 0.6–1.2% difference between the
> modes). Scale changes restart at the playback position: 2×→4× and 4×→2× both
> verified mid-movie at ~0.97× real time, and both preserve the pause state.
> Two gaps were closed: a **failed** scaler build used to be swallowed (the old
> scaler kept running while the mode label claimed otherwise) — the kind is now
> only committed on a successful build, and a failure is reported; and the menu
> now shows the *effective* scale when the source clamps 4× to 2×.

> **14 Sept audit round.** An independent evaluation (`docs/EVALUATION.md`) found six
> playback-correctness defects and evidence gaps. All six were reproduced and fixed;
> see `docs/EVALUATION-RESPONSE.md` for the point-by-point account and
> `scripts/check_playback_correctness.py` to re-run them. Claims that the audit
> showed to be premature (full-episode acceptance, matched 480p comparison,
> quality/power conclusions) are corrected below rather than restated.

Deliverables: **(1)** the comparison + recommendation → `docs/COMPARISON.md`;
**(2)** the working local-stream prototype → `engine/` (how to run: `README.md`).

Verification evidence is listed per item. Everything below was executed on this
machine (Apple M5 Pro, macOS 26.6) — no claims rest on intent.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Test Anime4K **and** AnimeJaNai SuperUltraCompact on the same clips, across clean / compressed / older material | done | Identical canonical frames fed to both; `results/compare/*_grid.png`; clips described in `docs/COMPARISON.md` |
| 1 | Compare image quality | done | 1:1 detail grids (4 methods × 7 frames); acutance 5.9–7.8× vs bicubic; `results/metal/validation.json` |
| 1 | Compare flicker during motion | done | Temporal amplification ratios: Anime4K 1.0022 (Mode A 1.0227), AnimeJaNai 1.0021 — `results/anime4k/flicker.txt`, `results/animejanai/bench.json` |
| 1 | Compare processing speed | done | Anime4K Metal 0.95 ms (480p) / 3.81 ms (1080p→4K); AnimeJaNai 51.0 ms / 350.9 ms; live loop 9.97 fps = 0.33× RT — `results/metal/BENCH.md`, `results/animejanai/bench.json` |
| 1 | Compare power use | done (proxies; wattage unavailable) | GPU util, CPU %, ANE activity, thermal state. Reasons wattage is missing: `powermetrics` needs root, `xctrace` **"Power Profiler" is not supported on macOS** (Apple only ships it for iOS), and battery current telemetry is meaningless on AC power (the machine was on AC). Command for a real measurement on battery: `sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 10` |
| 1 | Choose the lightest worthwhile approach | done | Anime4K CNN x2 M on Metal (54–92× lighter, ≥2× realtime at 4K) |
| 2 | Small Mac app that decodes, processes on accelerated hardware, outputs 2× | done | `engine/MacVideoUpscaler` (Core + CLI + SwiftUI app) + `engine/Anime4KMetal`; release build smoke-tested: 1920×1080 → 3840×2160 H.264 + AAC |
| 2+ | Magnification above 2× (user request) | done | 2×/4× selectable (`--scale`, GUI picker). 4× chains two 2× CNN passes (no native 4×/3× kernel exists; 3× is deliberately not offered). 640×480 → **2560×1920 at 111 fps = 3.7× realtime**, A/V drift 0.00 ms/s. Clamped to 2× above the hardware encoder's 4096×2304 limit (verified: 4096×2304 OK, 5120×2880 fails −12903), with the reason surfaced in the UI |
| 2 | Metal for Anime4K | done | v3.2 `Upscale_CNN_x2_M` (+ denoise) ported; weights extracted mechanically; matches an independent NumPy port to **≤1 LSB** |
| 2 | SDR, 24–30 fps, up to 1080p, aspect ratio + audio timing preserved | done | 640×480→1280×960 and 1080p→4K verified; A/V lag +0.0 ms (no seek) / −20 ms (seek, one AAC frame) with **0.00 ms/s drift** — `results/av_sync_verification.txt` |
| 3 | Hardware-encode and serve a local stream VLC can open | done | `h264_videotoolbox` → MPEG-TS → local HTTP; VLC connected, decoded via `videotoolbox` at 3840×2160 |
| 3 | Playback starts after a short buffer while processing continues | done | ~1.5 s prebuffer gate; engine runs 2.4× ahead into a bounded ring, then throttles to playback rate |
| 4 | File selection, enhancement on/off, status display | done | SwiftUI panel rebuilt into labelled sections: NSOpenPanel picker, enhancement picker (CNN / CNN-denoise / Bypass), scale picker (2×/4×), **audio-track picker**, **subtitle-track picker**, status grid, Open-in-VLC. Verified: the app binary launches as a **Foreground** app (LaunchServices `ApplicationType=Foreground`, after fixing a background-only registration bug) and its embedded engine answers `/status.json` on port 8090 — so the windowed app genuinely initialises the pipeline. Visual/layout confirmation still needs a human run (Screen Recording permission denied to this helper) |
| 4 | Coordinate pause and seek in the app | done | **Arbitrary-time timeline** (scrubber + "go to" box accepting `90`/`1:30`/`1:02:03`), plus pause/resume and ±10 s. `scripts/controls_test.sh`: pause froze at 284 frames, resume continued, seek → session 2 at 42 s, post-seek A/V in sync (corr 0.998, drift 0.00 ms/s). `scripts/check_tracks.py`: seek to 600 s observed restarting at exactly 600.0 s. **Seek latency ≈0.5 s** (was 5–8 s under full-ring backpressure: teardown held the pipeline lock while joining a worker that needed it, and a pipe-blocked ffmpeg ignored SIGTERM for the full grace period) |
| 4+ | Audio + subtitle track selection (user report: "subtitle built in and dual audio tracks") | done | Tracks probed and published in `status.json` (`audioTracks`, `subtitleTracks`); picking one restarts the session on the new mapping. Verified: `--audio-index 0` matches source `a:0` at corr **0.999** vs 0.387 for `a:1`, and `--audio-index 1` matches `a:1` at **0.998** vs 0.391 (chosen on the segment where the two tracks differ most, r=0.394) — `scripts/check_tracks.py` |
| 4 | Bounded buffering; offer a lighter preset if enhancement can't sustain | done | 128 MB ring caps and stays flat; app warns when RT ratio < 0.95 and Bilinear is hot-swappable mid-session (verified) |
| F1 | Short clips could never play (prebuffer gate waited for a drained ring) | done | Gate now releases when the producer finishes; prebuffer clamped to capacity. 1 s clip: 200/0 bytes → **215,636 bytes / 24 frames**, also with a 512 MB prebuffer on a 64 MB ring |
| F2 | Controls used the processing position, not playback | done | Engine owns its VLC (coordinated pause; auto-reopen after seek); relative seeks and all restarts resume from playback; read-ahead bounded to 30 s so the frontier stays near the picture. Verified: −90 s seek lands at 620 s where the playback target is 618 s (frontier target 647 s) |
| F3 | Display aspect ratio lost | done | Anamorphic sources use hardware HEVC (h264_videotoolbox cannot carry aspect in MPEG-TS — verified). 720×480 SAR 32:27 → **1440×960 SAR 32:27 DAR 16:9**; square-pixel sources unchanged |
| F4 | A/V check accepted a 1 s-desynchronized stream | done | Rewritten event-based (each stream aligned via its own PTS): output **+1.3 ms**, 1 s-delayed control **rejected at +1014 ms**; `--expect-fail` makes the negative control part of the suite |
| F5 | Subtitle enable/disable did not update config | done | `config.burnSubtitles` is persisted before restarting. Verified off → on → off across sessions |
| F6 | Session ownership/threading risks | done | Scaler resolved under the lock and held for the frame (no skipped frames); teardown resets the ring and kills children before joining; socket closed exactly once by its owner; serve loop checks session generation; SIGINT/SIGTERM and app-quit now stop the owned VLC |
| F7 | Status counters not authoritative; harness killed unrelated processes | done | Frames/buffer flushed after the pump drains (`frames=24` not 3), `memoryFootprintMB` reports `phys_footprint`, speed is *sustained* throughput not a bursty EMA; sustained harness uses a release binary, its own ports/PIDs, and fails if player metrics are missing |
| 5 | Full-episode test: dropped frames, audio drift, memory growth, heat | done | 23.5-min episode: **0 lost / 0 corrupt / 0 discontinuities**, 30.1 fps displayed, thermal `nominal`; drift 0.00 ms/s. Memory measured, characterised and reduced (139 MB → 989 MB peak over 8 min; steady-state growth cut from ~1.8 MB/s to **0.71 MB/s** by removing per-frame allocations plus `retainedReferences=false`); residual is driver-level (`leaks` = 0) and is the one open item below |
| 5 | Artifact check: pans, gradients, fine textures | done | 1:1 grids vs bicubic (`results/compare/`) + flicker ratios 1.0022 / 1.0021 |
| 5 | Artifact check: **scene cuts** | done | Alignment-robust motion-series check: at all 6 detected cuts **correlation 1.000, shift 0 frames, peak-motion ratio 1.00** — no ghosting/flash, exact 1:1 frame mapping (`results/scene_cut_verification.txt`, `scripts/check_scene_cuts.py`) |
| 5 | Artifact check: **subtitles** | done | Burn-in implemented and verified for both kinds: bitmap DVD subs (ffmpeg overlay) and text/SRT (native CoreText). Presence/timing correct (present in cues at t=3/12/22, absent between), glyph fidelity vs a bicubic reference MAE 0.9–3.6 with sharpness ratio **1.01–1.52×** (never softer); visual inspection shows clean outlines, no ringing (`results/subtitle_verification.txt`) |
| 5 | Playback speed independent of processing speed (user report) | done | Measured VLC's own playback position against wall clock with dummy outputs (no window): **0.997×** while the enhancement frontier ran 3.30× ahead — i.e. playback is real time and the lead is buffered, not skipped. Ground truth: video PTS spacing 33.367 ms = 29.970 fps, container duration = frames/fps. The measurement method was validated first on a local file (1.006×). `results/playback_rate_verification.txt`, `scripts/check_playback_rate.py`. The app's position readout was relabelled "Enhanced to" because VLC's read-ahead is not observable from the engine |
| 5 | Acceptance: improved 1080p→4K at 24–30 fps with stable audio | **met** | Full episode (1414.4 s) ran to completion: 42,387 frames (42,383 expected), **0 lost / 0 corrupt / 0 discontinuities**, thermal `nominal`, sustained ~1.08× with peak capability 2.1–2.5×; memory plateaued (RSS 76.7 → 97.2 MB, second-half growth −5.5 MB/min) — `results/full_episode_verification.txt`. The audit's finding that no *prior* run covered an episode was correct and the earlier claim is withdrawn in favour of this measurement |
| 6 | Decide on native VLC integration | done | `docs/VLC-INTEGRATION-DECISION.md` (options, effort, verified facts, recommendation). Resolved by building an in-app player instead — no per-frame GPU↔CPU round trip (VLC 3 plugin), no unreleased ABI (VLC 4), and it fixes seek/launch ergonomics that a plugin would not have touched |
| 7 | Add the best upscaler for **general (live-action) video** to the app | done | MetalFX Spatial added as `--mode metalfx` and in the Enhancement menu; real time at 0.992–1.007×, byte-identical halves in the comparison view, hot-swappable mid-movie, GUI and harness frames byte-identical. Report, method, chips and caveats: `docs/GENERAL-VIDEO-UPSCALING.md`; raw metrics `results/general/*/metrics.json` |
| 7 | …and establish what the alternatives would cost here | done, **corrected** | Originally: `general-x4v3` at **0.12× real time at 480p** from a projected single-layer conv. **Retracted** — the projection was ~2.6× pessimistic. Measured end to end: `general-x4v3` ~100–110 ms at 640×480 (≈0.45× real time) and **`animevideov3` 22.59 ms at 270p→1080p = 44.26 fps = faster than real time**. `x2plus` heavier again and unmeasured; RealBasicVSR's objection is structural (optical flow + sequence window), not speed. None is a live mode here, but the reason is no longer "too slow" — `results/general/realesrgan_bench.txt`, `results/general/realesrgan_feasibility.txt` |
| 7 | …and say honestly whether it is an improvement | done | **It is not, on fidelity.** Acutance 1.54–2.18× the ground truth (over-sharpened, visible halos), lowest PSNR/SSIM of the four modes in every live-action run, and worse than plain bilinear on anime (37.51 vs 39.42 dB). The Anime4K CNN remains the more faithful mode and the better choice on animation; MetalFX is the default anyway, as the cheaper general-purpose option, and is one click or one `--mode` away from being replaced. Stated with the one soft metric (flicker) flagged as soft |
| 8 | RealBasicVSR in the app (user request) | **removed** | Offline **export**, not a playback mode: *Enhancement ▸ Export with RealBasicVSR…* (⌥⌘E) + `scripts/realbasicvsr/enhance.py`. Official weights load `strict=True` into verbatim-vendored MMagic classes. 0.30 s/frame at 480p→1080p, 1.02 s/frame at 540p→4K; cannot play live (recurrent, ~300× the CNN's per-frame cost). Report: `docs/REALBASICVSR.md` — built and verified as above, then **removed 15 Sept at the user's request** (see the entry at the top). |
| 8 | …verified through the installed app | **removed** | The bundle carries the runner, probes for a Python with torch (versioned interpreters first), shows progress as a badge, and exports a file **byte-identical** to the development binary's — driven by `_EXPORT=<path>` with `_EXPORT_MAX_FRAMES` — built and verified as above, then **removed 15 Sept at the user's request** (see the entry at the top). |
| 8 | …with the MPS faults found rather than worked around | **removed** | Three, each producing silently wrong pixels: channels-last `grid_sample` (identity-warp error 3.4 vs 5e-07, which made SPyNet's flow garbage), missing `border` padding, and `Tensor.type()` rejecting `'torch.mps.FloatTensor'`. After fixes flow agrees CPU↔MPS to 107 dB, cleaning to 145 dB, output to 33 dB — built and verified as above, then **removed 15 Sept at the user's request** (see the entry at the top). |

## Gaps / open items (stated, not hidden)

1. **Memory**: two fixes, both evidence-backed. The read-ahead is bounded to 30 s
   (~12 MB) instead of a 128 MB ring holding minutes; and the stdout pump had no
   autorelease pool, so its per-chunk `Data` objects accumulated on a bare thread
   (the audit flagged exactly this). Adding the pool cut engine RSS growth from
   ~45 MB/min to ~7 MB/min and halved the starting footprint (151.7 → 76.7 MB).
   Whether the residual plateaus is what the running full-episode test measures;
   an Instruments/Metal allocation trace is the next step if it does not.
2. **Playback position is an estimate**, good to roughly the read-ahead (30 s),
   not frame-exact. VLC's absolute clock carries an unknown base (it reported
   638 s before a second of a session starting at 600 s had played) and headless
   VLC can run it at input rate, so it is published for diagnostics only.
3. **Color metadata** is still not preserved through the rawvideo interface
   (aspect ratio is, and rotation now is too — see the formats entry above). A
   BT.709/BT.601 mistagging in the source is therefore not reflected in the output;
   since the pipeline is BGRA throughout and the display layer does the final
   conversion, this shows up as slightly shifted colour on mis-tagged SD sources
   rather than a hard error. `.ogv` is declared for the Finder but has no test
   sample: this ffmpeg has no Theora encoder.
4. **Subtitles**: burn-in works for bitmap and native text tracks, but this
   Homebrew ffmpeg has **no libass**, so styled ASS/SSA cannot be rendered.
5. **GUI**: ships as a signed bundle (`scripts/make_app.sh`), launches from the
   Finder/Spotlight/Dock, and opens files handed to it by the Finder. Its engine
   side is verified headlessly (playback, controls, the click and double-click paths
   through real window hit-testing, the Finder open event). What is still **not**
   verified is how it *looks*: Screen Recording and Accessibility are not granted to
   this helper, so the window has never been screenshotted — no screenshot, no visual
   review of the layout, the transport strip, the stats overlay or the icon in situ.
   A human glance is the remaining check.
6. **Comparison caveats** (accepted from the audit): no real clean modern 1080p
   anime; the 480p speed comparison is not matched-input (854x480 vs 640x480,
   mean vs median); the flicker ratio cannot rule out localized shimmer; smaller
   Anime4K presets were not benchmarked; no measured energy comparison exists.
   See the corrections block in `docs/COMPARISON.md`.
7. **AnimeJaNai** was measured with the dynamic-shape ONNX model on CoreML
   (21/23 nodes on CoreML, not a CPU fallback). Static-shape / fp16 / tiled
   variants were not attempted; even a 4x gain would leave it below realtime.
8. **Anime4K Mode A/B/C** GLSL speeds were not measured (the agent was stopped
   because mpv windows were disruptive); the shipped Metal CNN x2 M is measured.
9. **Native VLC integration remains not started** — deliberately, and now moot:
   the app plays in-process, so there is no VLC to integrate with. The stream
   pipeline is retained for external players via the CLI.
10. **The in-app player is verified by dumped frames and telemetry, not by a
    window screenshot.** `playercheck` and the GUI's opt-in env hooks confirm
    real-time frame delivery, a 3-frame buffer, working pause/seek/scale/mode
    controls, clean EOF, subtitle burn-in surviving 2× and 4×, both renderers
    reporting `.rendering` with `window=true`, audio staying fed across pauses and
    seeks, and click-to-pause through the real mouse path (in the GUI as well as
    the harness); the frames handed to the
    display layer were dumped as PNGs and inspected. Screen Recording permission is
    not available to this environment, so a human should still glance at the window
    itself. 4× on HD is now measured (0.61× / 883 MB) and clamped; the
    remaining trade is the uncompressed-BGRA buffer footprint at 4K (~307 MB).
11. **The audio path has no restart.** Pause/resume and seeks are fine, but if the
    output device is removed or the renderer fails, audio stops for the rest of the
    session (reported as "Audio stopped" in the status) while video keeps playing;
    recovering would mean building a new `AVSampleBufferAudioRenderer` and
    re-anchoring it to the synchronizer mid-session.
12. **Comparison view limits.** Both halves share one output geometry, so the two
    sides cannot be shown at different scale factors (a 2×-vs-4× comparison is not
    possible in one frame). There is no freeze-frame A/B: both sides always show the
    same live frame, so a moving shot is compared by eye across the line rather than
    as two stills. The grab zone is invisible — when the line sits at either edge
    (0% or 100%) nothing is drawn there, so the way back is the slider or a press
    near the edge. And the second model's cost is GPU work that is currently hidden
    by renderer pacing rather than by headroom being proven: on a slower GPU or a
    larger output it would show up as dropped frames, and the realtime warning in the
    status area is the only signal for that. `Off` is excluded from both halves by
    design (see the `Off` entry above) — it has no shared pixel grid to composite on.
13. **The AnimeVideo-v3 port is an open judgement, not a settled measurement.**
    It is 44 fps at 270p→1080p through PyTorch/MPS and its sharpness is the best
    calibrated of any mode tested on SD live action (0.971× of the truth), but it
    loses to the shipped Anime4K CNN on anime and the app is an anime upscaler, so
    it was not ported. Two things would change that, and neither has been measured:
    a hand-written Metal port's actual speed (the 22.59 ms is a PyTorch floor, not a
    Metal figure), and how it behaves on clean anime rather than the compressed SD
    sources in `test-media/`. The same caveat applies to the corrected
    `general-x4v3` figure.
14. **The corrected Real-ESRGAN numbers are MPS-only.** No Metal implementation of
    `SRVGGNetCompact` exists in this repo, so every figure in
    `results/general/realesrgan_bench.txt` is a framework number. Any future claim
    about what these models cost *in this app* needs a port first — which is exactly
    the extrapolation error the correction entry at the top of this file records.

## Reproduce everything

```bash
swift build -c release --package-path engine/MacVideoUpscaler
./scripts/make_app.sh                    # build, sign, install to /Applications
engine/MacVideoUpscaler/.build/release/MacVideoUpscalerApp    # the same app, unbundled
engine/MacVideoUpscaler/.build/release/macvideoupscale --file <video> --mode cnn --port 8090 --open-vlc

python3 scripts/check_av_sync.py --reference <src> --stream <captured.ts> --file-start 0 --dur 16
python3 scripts/check_tracks.py --file <video> --seek 600
python3 scripts/check_playback_rate.py --file <video> --scale 4 --port 8430
python3 scripts/check_playback_correctness.py     # the six audit fixes
python3 scripts/full_episode_test.py --file <episode> --minutes 30
python3 scripts/sustained_test.py --file <video> --mode cnn --port 8098 --seconds 300
./scripts/controls_test.sh <video> 42
python3 scripts/make_compare.py

# general-video upscaler evaluation: fidelity, sharpness and temporal stability
python3 scripts/eval_general_upscaler.py \
    --playercheck engine/MacVideoUpscaler/.build/release/playercheck \
    --source test-media/liveaction/tt_540p_c30.mp4 \
    --reference test-media/liveaction/tt_1080p_ref.mp4 \
    --scale 2 --modes bypass,cnn,cnnDenoise,metalfx,off \
    --out results/general/liveaction_540p_to_1080p

# Real-ESRGAN/animevideov3: both checkpoints, end to end (results/general/realesrgan_bench.txt)
python3 scripts/realesrgan/fetch_weights.py
python3 scripts/realesrgan/bench.py --variant animevideov3 --devices mps,cpu --iters 3
python3 scripts/realesrgan/bench.py --variant general-x4v3   --devices mps     --iters 3

# the single-layer kernel number the retracted projection came from - do not
# extrapolate it to a network, which is the mistake results/general/realesrgan_feasibility.txt records
xcrun -sdk macosx swiftc -O scripts/bench_realesrgan_conv.swift \
    -o /tmp/convbench -framework Metal && /tmp/convbench

# the Off mode's passthrough claim: pixel-identical to the decoded source.
# NOTE the dump is 1-based (`frame_0090.png` is source frame 89).
engine/MacVideoUpscaler/.build/release/playercheck \
    --file test-media/segments/naruto_t290.mkv --mode off --seconds 5 --no-audio \
    --dump-dir /tmp/offdump --dump-at "90"
ffmpeg -v error -i test-media/segments/naruto_t290.mkv -vf "select=eq(n\,89)" \
    -fps_mode passthrough -frames:v 1 -pix_fmt rgb24 /tmp/src89.png -y
python3 -c "
import numpy as np; from PIL import Image
a=np.asarray(Image.open('/tmp/offdump/frame_0090.png').convert('RGB')).astype(int)
b=np.asarray(Image.open('/tmp/src89.png').convert('RGB')).astype(int)
print('max diff', np.abs(a-b).max())"   # prints 0
```
