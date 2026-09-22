# Project evaluation — 14 September 2026

**Verdict: a functioning technical prototype with promising throughput; the initial playback acceptance target is not yet validated.** Keep Anime4K CNN x2 M on Metal as the provisional default. Fix playback correctness and strengthen the evidence before native VLC integration or further features.

This evaluation covers the existing implementation against the six-item brief. It adds an audit report and test evidence, without changing application code. The workspace has no Git repository, and source files changed during the review, so this is not an immutable commit review. End-of-review source hashes and the individual runtime logs are saved in `results/evaluation-2026-09-14/`. One initial HD test received SIGTERM from an unidentified source; its partial output is excluded from performance conclusions. The repeat completed successfully.

## What exists and works

- Both Swift executables build in release configuration on this Apple M5 Pro with 24 GB RAM. Remaining build warnings are SwiftUI deprecations. The build used temporary module caches because the sandbox cannot write the normal user caches.
- The implementation really decodes with FFmpeg, processes frames with Metal, hardware-encodes H.264 with VideoToolbox, and serves MPEG-TS on loopback. It is more than a mockup or an offline conversion wrapper.
- A fresh synthetic 1080p24 smoke test delivered a complete 3840×2160 stream in **2.96 seconds of HTTP capture**, for roughly eight seconds of source material. This is about 66 output frames/s including capture startup, with an unrestricted HTTP consumer. It is a short throughput check, not sustained VLC playback. FFprobe counted 194 source frames and 195 output frames, so this run is not evidence of exact frame preservation.
- The old logs contain VLC playback samples with zero reported lost pictures. The shader has saved NumPy-reference validation, and same-frame comparison grids exist.
- File selection, enhancement modes, scale selection, track selection, status, and session restart controls are implemented. The byte ring has a fixed allocation and applies backpressure.

## Recommendation and comparison confidence

| Approach | Saved 1080p → 4K measurement | Assessment |
|---|---|---|
| Anime4K CNN x2 M, Metal | 3.81 ms/frame mean, shader benchmark | Strong provisional choice; end-to-end smoke test also exceeds 24–30 fps |
| AnimeJaNai HD V3 SuperUltraCompact, ONNX Runtime CoreML EP | 350.9 ms/frame median, inference benchmark | Tested configuration is far below realtime; exclude it from the prototype's live path |

These are prior benchmark results, not fresh model benchmarks. The AnimeJaNai log records 21/23 graph nodes assigned to CoreML, which supports that the provider was active but does not establish where every operation executed within CoreML. Do not generalize this result to every possible optimized Mac runtime. Static-shape deployment was not demonstrated.

The recommendation is well supported on speed. The report's stronger quality, power, and “lightest” conclusions need qualification:

- I inspected the saved Naruto and synthetic comparison grids. CNN M sharpens contours relative to bicubic, but some source stair-stepping remains. Mode A is more aggressive; the synthetic ellipse shows edge distortion. Sharpness/acutance is not a measure of recovered detail or overall preference.
- There is no real clean modern 1080p anime sample. Synthetic art, Big Buck Bunny, and Sintel are useful proxies but cannot complete that category. This matters particularly because [Anime4K's upstream documentation](https://github.com/bloc97/Anime4K) identifies native 1080p anime as its intended material and describes limitations on older SD footage.
- A twelve-frame global temporal-difference ratio cannot rule out localized shimmer on pans, moving hair, thin lines, or subtitles. Real motion and scene changes dominate that metric. Inspect longer matched clips and compare the final encoded output as well as shader PNGs.
- No measured wattage or energy-per-frame comparison exists. The AnimeJaNai JSON explicitly says its GPU-utilization samples were contaminated by concurrent activity. Utilization and thermal state cannot establish which pipeline consumes less energy.
- The headline 480p comparison uses **854×480 for Metal versus 640×480 for AnimeJaNai**, and mean versus median timing. Do not present its exact speedup as a matched-input benchmark.
- CNN M is the lightest *implemented enhancement* in this prototype, but smaller Anime4K variants were not benchmarked. The claim that it is the lightest worthwhile preset remains unproven.

## Findings requiring correction

### P1 — Prebuffering can prevent short clips from ever playing

`StreamServer.serveStream` waits for the prebuffer threshold unless `ring.isEOS` becomes true. `RingBuffer.isEOS` requires both producer completion **and an empty ring**. A completed stream smaller than the threshold contains unread bytes, so neither condition can become satisfied while the server waits.

Reproduced with a one-second 320×180 clip: the default configuration returned HTTP headers but **zero video bytes before a five-second timeout**, although the engine had finished. With `--prebuffer-mb 0` (internally clamped to 64 KiB), the same clip delivered all 24 enhanced frames, 215,260 bytes. A configured prebuffer larger than ring capacity can also deadlock.

Fix: distinguish producer completion from fully drained EOF; release remaining bytes when the producer finishes, and clamp prebuffer to capacity. Add short-file and oversized-prebuffer regression checks.

### P1 — Playback controls use the processing position instead of VLC's position

`pause()` only stops the frame worker. It does not pause VLC or stop draining already encoded content. `seek(by:)`, audio changes, and scale changes use `startAt + frameIndex / fps`, which is the production frontier. The 128 MiB ring can hold tens of seconds or minutes of content, so these operations can skip far beyond what the viewer is watching. Enhancement changes also take effect behind the queued output.

The GUI's restart callback only asks the user to press “Open in VLC” again. It does not reconnect VLC automatically. These are processing controls, not the coordinated playback controls required by the brief.

Fix: manage an owned VLC session and read its playback clock; coordinate pause/resume and reopen after seek. Use the playback clock for relative seeks and track/preset restarts, and expose queued latency separately from processing progress.

### P1 — Display aspect ratio is lost

`FFmpegDecoder.Meta` carries width, height, fps, and duration, but no sample aspect ratio. Raw BGRA loses that metadata, and the muxer never restores it.

Reproduced with 720×480, SAR 32:27, DAR 16:9: output is correctly doubled to 1440×960 but has **no SAR/DAR metadata**, leaving a 3:2 pixel geometry instead of the source's 16:9 display geometry. This affects legitimate DVD-era inputs. The existing `subs_text.mkv` fixture also has non-square pixels: 1920×1080 with SAR 3:4 and DAR 4:3.

Fix: preserve SAR through probing and encoding while retaining the exact doubled raster dimensions. Add an anamorphic playback/output-metadata check. Also define handling for rotation and color metadata, which the rawvideo interface does not preserve automatically.

### P1 — The synchronization test accepts deliberately desynchronized video

`scripts/check_av_sync.py` compares source audio to output audio, prints video timestamps without evaluating them, and calls `sys.exit` before the optional video check. Thus `--video-t` never runs. It does not establish audio-to-video synchronization.

Negative control: I delayed the video stream by one second while leaving its audio unchanged. FFprobe reported video start 2.421 s and audio start 1.400 s. The checker still returned **“OK (in sync, no drift)”**, correlation 0.9967 and drift 0.00 ms/s. Evidence: `av_checker_negative_control.txt`.

Fix: compare corresponding visual and audio events on a common output timeline, including a deliberately offset failure case. Measure near the beginning, middle, and end of full episodes and after seeks. Report the method's resolution; a 20 ms envelope analysis cannot establish exact zero drift.

### P2 — Subtitle enable/disable does not update the configuration

`PipelineController.setSubtitles` computes whether the enabled value changed, but never assigns it to `config.burnSubtitles`. Reproduced using the existing text-subtitle fixture: `subs-on:0` restarted the session but `subtitlesEnabled` remained false and the subtitle status remained “off”. Opening a file with subtitles initially enabled follows a different path and does not test this bug.

Fix: persist the enabled value before restarting, and test both directions after opening a file.

### P2 — Session ownership and threading need hardening

Source inspection found additional correctness risks that were not stress-tested during this audit:

- `setKind` mutates the scaler while the worker reads and calls it outside that lock; the optional `scaler?.scale` can skip processing if a switch clears it at the wrong time.
- Restart holds `ctrl` while joining a worker that also needs `ctrl`, then proceeds after a timeout without proving the old worker exited. Resources and worker state are shared across sessions.
- `dropClients` closes the descriptor while `serveStream` will close it again; descriptor reuse can make this affect a later socket. The serving loop also needs an explicit session/generation boundary.

Fix: establish one owner for session resources, frame-boundary control changes, and socket closure. Verify repeated seeks, opening another file while paused, disconnect/reconnect, and mode changes under backpressure.

## Acceptance evidence still missing

| Brief item | Evaluation |
|---|---|
| Comparison and recommendation | Useful provisional deliverable; real modern anime, robust motion review, matched power testing, and smallest-preset comparison remain |
| Accelerated exact 2× engine | Implemented and fast on a short 1080p test; aspect ratio and timestamp/color handling require work |
| Local stream to VLC | Implemented; historical VLC samples support basic playback; short-stream startup is broken |
| Basic coordinated controls and bounded buffering | Partial; fixed byte ring exists, but player coordination and session correctness are incomplete |
| Full-episode playback acceptance | **Not established** |
| Native VLC decision | Defer implementation until the external prototype passes acceptance |

I recovered and preserved nine historical sustained-test CSVs from `/tmp` under `results/evaluation-2026-09-14/historical/`. The file called `sustained_full_ep.csv` has 60 samples ending at **299.5 seconds**. The longest retained run ends at **451.1 seconds**. None documents a complete 23.5-minute episode. Short zero-drop samples should retain that description.

Memory remains unresolved: the 451.1-second run grows from 138.8 MB to 988.8 MB. A fixed stream ring does not bound all retained application or driver allocations. A clean `leaks` result does not prove driver ownership or eventual stabilization. The stdout pump also uses Foundation `availableData` repeatedly on a bare thread without an explicit per-iteration autorelease pool; investigate with allocation traces before attributing the growth. Measure the engine, FFmpeg children, and VLC together.

The status counters are not authoritative end-of-stream metrics: the successful HD output contains 195 frames while the finished status reports 187, because progress is sampled and not flushed at completion. `memoryFootprintMB` remains zero. Use actual stream frame counts and process measurements for acceptance.

The current sustained harness also defaults to the debug executable and terminates matching engines and VLC globally. Before rerunning it, make it use an explicit release binary, dedicated ports, and only processes it owns. Preserve telemetry and fail the test when player metrics are unavailable.

## Next milestone

1. Correct prebuffer completion, playback-clock coordination, SAR, subtitle toggles, and session ownership. Add focused regression checks for the reproduced failures.
2. Replace the A/V acceptance check with an event-based check that rejects the negative control. Make sustained testing reproducible and save complete logs.
3. Compare final encoded 2× output on clean modern anime, older material, and a compressed derivative of the same clips. Include a smaller Anime4K preset and a matched ordinary-scaling baseline.
4. Run complete 24 fps and 30 fps episodes in VLC, recording displayed/lost frames, actual audio/video offset over time, total memory, thermal state, and energy where measurable. Observe whether memory plateaus rather than extrapolating a short run.
5. Revisit native VLC only after this evidence shows the enhancement is worth keeping. The existing claim that a CPU-facing native filter must be slower is not demonstrated: the current external pipeline already copies frames through CPU-visible buffers and adds encoding. Compare actual costs if pursuing that route.

The architecture is worth continuing. Throughput is not the immediate blocker; correctness and trustworthy validation are.
