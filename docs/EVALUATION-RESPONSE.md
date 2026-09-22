# Response to the evaluation of 14 September 2026

`docs/EVALUATION.md` reviewed the prototype and found six defects plus gaps in the
evidence. Every finding was reproduced here before being fixed; nothing below is
claimed on the basis of inspection alone. Each item gives the cause, the fix, and
the command that now checks it.

Reproduce the whole set:

```bash
swift build -c release --package-path engine/MacVideoUpscaler
python3 scripts/check_playback_correctness.py          # the six fixes below
python3 scripts/check_av_sync.py --reference <src> --stream <ts> --file-start 300
```

The audit's overall verdict — keep Anime4K on Metal, fix playback correctness
before native VLC integration — is accepted and is what this work did. No native
VLC filter work was started.

## P1 — Prebuffering could prevent short clips from ever playing

**Reproduced.** A 1 s 320×180 clip with the default 4 MB prebuffer returned
HTTP 200 with **0 bytes** and stayed that way.

**Cause**, precisely: `StreamServer` gated on `ring.isEOS`, and `RingBuffer.isEOS`
is `producerFinished && count == 0`. A stream shorter than the prebuffer leaves
bytes unread forever, so neither condition could be satisfied. There was a second
half to it: production was never the problem — with `ZU_DEBUG_PUMP=1` the pump
logs `markEOS total=215636`, i.e. all 24 frames were in the ring. The status
readout said `bufferBytes: 0` only because no status update ran after the pump
wrote, which is the "counters are not authoritative" point the audit also made.

**Fix.** `RingBuffer.producerDone` (finished, drained or not) is the gate now;
`prebufferBytes` is clamped to capacity so an oversized request cannot deadlock;
and `finishSession` joins the pump and then publishes final counters.

**Evidence** (`check_playback_correctness.py`, checks 1–2):

| case | before | after |
|---|---|---|
| 1 s clip, default prebuffer | 200 / 0 bytes | 215,636 bytes, **24 frames** |
| prebuffer 512 MB into a 64 MB ring | never served | 215,636 bytes, 24 frames |

Final status now reports `frames=24` for that clip (was `frames=3`).

## P1 — Playback controls used the processing position

**Reproduced.** `seek(by: -90)` with playback at content ~617 s jumped to the
frontier-based target instead: the session restarted near 0, and separately near
the frontier rather than the picture.

**Cause.** Three things: `seek(by:)` anchored on `startAt + frameIndex/fps`
(the production frontier); the audio/scale/subtitle restart paths used the same
frontier; and `pause()` only stopped the frame worker while VLC kept playing out
whatever was already encoded. `Open in VLC` also launched an unowned instance, so
nothing could be coordinated or reconnected.

**Fix.**

- The engine now **launches and owns** VLC (`VLCController`) and drives it over
  its HTTP interface: pause/resume are coordinated with the player, and after a
  seek the engine reopens the stream itself on the new session URL (the GUI no
  longer has to ask the user to do it).
- Relative seeks and every session-restarting control resolve their target from a
  `resumePosition()` anchored to playback, not to production.
- The **read-ahead is now bounded in time** (`leadSeconds`, default 30 s) instead
  of only by the 128 MB byte ring. This is the deeper fix: the ring used to hold
  minutes of content, so *any* frontier-relative control was minutes wrong. It
  also cuts the buffer to ~12 MB at 1080p-class bitrates and throttles the
  pipeline to playback rate instead of encoding 3× more than is watched.

**Evidence** (`check_playback_correctness.py`, check 6): lead stays at the 30 s
cap, and a −90 s relative seek lands at **620 s** where the playback-anchored
target is 618 s — the frontier-based target would have been 647 s. With the old
128 MB ring the same test was off by 275 s.

**Remaining limitation (stated, not hidden).** The position is an *estimate*
accurate to roughly the read-ahead, not a frame-exact player position. VLC's
absolute clock carries a base we do not control — on a session starting at 600 s
it reported 638 s before a second had played — and with dummy outputs it can run
at input rate rather than real time. It is therefore published as a diagnostic
(`vlcReportedSeconds`) and not used for control. Getting frame-exact position
would need VLC's `time` to be trustworthy, which is not established.

## P1 — Display aspect ratio was lost

**Reproduced.** 720×480 SAR 32:27 → output 1440×960 with `SAR N/A, DAR N/A`, i.e.
square-pixel 3:2 instead of 16:9.

**Cause.** `Meta` carried no aspect ratio and raw BGRA cannot. Restoring it is
not just a flag: **hardware H.264 cannot express it** — verified directly,
`-aspect 16:9` with `h264_videotoolbox` into MPEG-TS still probes as `N/A`, while
`libx264` and `hevc_videotoolbox` both carry it (32:27 / 16:9).

**Fix.** `Meta.displayAspect` is probed (derived from SAR, reduced) and stamped on
the output. Anamorphic sources are encoded with **`hevc_videotoolbox`**, which is
still hardware-accelerated; square-pixel sources keep H.264 unchanged, and the
reason for the switch is surfaced in the status message.

**Evidence** (`check_playback_correctness.py`, check 3):
`hevc 1440×960 SAR 32:27 DAR 16:9`. Square-pixel control: the Naruto source still
comes out `h264` with no aspect stamped.

**Not addressed:** rotation metadata and color metadata are still not preserved
through the rawvideo interface. Documented in `README.md`.

## P1 — The synchronization test accepted deliberately desynchronized video

**Reproduced.** With the video delayed 1 s (audio untouched), the old checker
returned "OK (in sync, no drift)" — correlation 0.9967.

**Cause**, two independent mistakes: it compared audio to audio and never
evaluated the video, returning before the optional video check; and its lag
search covered only ±300 ms, so a 1 s error could not be found. A third cause
surfaced when my first rewrite still passed the control: comparing *frame content
order* is blind to a timestamp shift, because delaying PTS does not reorder
frames.

**Fix.** `scripts/check_av_sync.py` was rewritten to place each stream's events on
a common content timeline using that stream's own presentation timestamps — video
per-frame motion stamped with frame PTS, audio 20 ms RMS envelope stamped from the
audio start — and then to compare the independently measured video lag with the
audio lag. A delay on one stream shows up directly, and the resolution is the PTS
step (one frame, 33 ms), not the 20 ms envelope window.

**Evidence** (`check_playback_correctness.py`, check 5; `results/av_*`):

| stream | A/V error | verdict |
|---|---|---|
| engine output | **+1.3 ms** (video corr 1.000, audio corr 0.997) | OK |
| video delayed 1 s | **+1014.4 ms** | FAIL (correctly rejected) |

The checker now has an explicit `--expect-fail` mode so this negative control is
part of the suite rather than a manual experiment.

## P2 — Subtitle enable/disable did not update the configuration

**Reproduced, with the audit's cause confirmed exactly:** `setSubtitles` computed
whether the value changed but never assigned `config.burnSubtitles`, so the
restart re-read the old value. Neither direction worked.

**Fix.** Persist before restarting. **Evidence** (check 4): session 1 `off`,
session 2 `on` (`subrip #2 (native text render)`), session 3 `off` again.

## P2 — Session ownership and threading

- **Scaler race:** the worker read `self.scaler` unlocked, so a mode switch
  clearing it mid-frame made `scaler?.scale` a silent no-op (a skipped frame).
  Now the scaler is resolved under the same lock that `setKind` uses and held in
  a strong local for the whole frame; a missing scaler fails the session instead
  of skipping.
- **Restart joining a worker that needs the same lock:** fixed earlier by the
  `restartMirror` atomic, and teardown now resets the ring and kills the ffmpeg
  children *before* joining, so the joins complete instead of timing out.
- **Double close:** `dropClients` closed the descriptor and the serving thread
  closed it again; after fd reuse that could close an unrelated socket. The
  serving thread now owns the descriptor and closes it exactly once
  (`dropClients` only shuts it down), and the serve loop checks the session
  generation each iteration.

## Evidence and harness gaps

- **The residual memory growth was the stdout pump.** The audit's hypothesis was
  right: the pump thread calls Foundation `availableData` in a loop on a bare
  thread with no autorelease pool, so each chunk's autoreleased `Data` accumulated
  for the life of the session. The worker had a pool; the pump did not. Wrapping
  each iteration in `autoreleasepool` (in the pump and in the new VLC clock poll
  thread) cut measured engine RSS growth from **~45 MB/min to ~7 MB/min** and
  halved the starting footprint (151.7 MB → 76.7 MB at 30 s, same episode, same
  configuration). Whether the remainder plateaus is what the running full-episode
  test is measuring.
- **Orphaned VLC on SIGTERM.** Found while setting up the sustained run: killing an
  engine skipped its cleanup, leaving the owned VLC alive holding its HTTP port —
  and the next session's clock queries were then answered by that stale instance,
  reporting a dead stream. Fixed three ways: SIGINT *and* SIGTERM are handled in
  the CLI, the app stops the engine on quit, and the VLC HTTP port is now chosen
  free-at-startup and published in `status.json` (`vlcHttpPort`) so tools never
  assume the default.
- **Status counters are now authoritative at end of session:** frames and buffer
  are flushed after the pump drains, and `memoryFootprintMB` reports
  `phys_footprint` instead of always being 0. `processFps` / `realtimeRatio` now
  report *sustained* throughput; the old instantaneous EMA
  read 8–17× while the true sustained rate was ~1×, which made the UI indicator
  (and the "2.1–2.5× realtime" headline) misleading under bounded read-ahead.
  *(Superseded 15 Sept for the embedded player: "since the session began" counted
  paused wall time as decode time, so the metric is now a sliding window keyed to
  the render clock — see `PROJECT_STATUS.md`.)*
- **Sustained harness rewritten** (`scripts/full_episode_test.py`): explicit
  release binary, dedicated ports, terminates only the processes it started (no
  global `pkill`), requires the player metrics (fails if absent), samples engine
  RSS + phys_footprint + VLC RSS + ffmpeg children RSS + thermal + player
  counters, and reports whether memory plateaus (second-half growth rate versus
  first-half).
- **Full-episode acceptance: measured, and it passes.** A complete episode
  (Naruto ep 1, 1414.4 s, 640×480 → 1280×960 CNN x2) ran to `finished` with the
  harness above: 44 samples over 1323 s, 42,387 frames produced against 42,383
  expected for the duration, **0 lost / 0 corrupt / 0 discontinuities**, thermal
  `nominal` throughout, and sustained throughput settling at ~1.08× (throttled by
  the bounded read-ahead; the 2.1–2.5× figure is peak capability).
  Memory: engine RSS 76.7 → 97.2 MB (peak 165.6), and the growth rate went from
  +7.1 MB/min in the first half to **−5.5 MB/min in the second — it plateaus**,
  which is the answer the audit asked for and the opposite of the 138.8 → 988.8 MB
  it measured before. Evidence: `results/full_episode_verification.txt` plus the
  CSV. The earlier "full episode" claim is withdrawn as the audit asked; this run
  replaces it. Note the audit was right that no *prior* run covered an episode —
  the historical CSVs ending at 299.5 s and 451.1 s stand as they are.
- **Comparison caveats accepted.** `docs/COMPARISON.md` now carries a corrections
  section: the 480p comparison was not matched-input (854×480 vs 640×480, mean vs
  median), no real clean modern 1080p anime was tested, the 12-frame
  temporal-difference ratio cannot rule out localized shimmer, the smallest
  Anime4K preset was not benchmarked, and there is no measured energy comparison
  (contaminated GPU samples; no wattage on AC power). "Lightest worthwhile
  approach" is downgraded to "lightest implemented enhancement".

## What is still open

1. Full-episode playback acceptance (running).
2. Frame-exact playback position (currently an estimate good to the read-ahead).
3. Rotation and color metadata preservation.
4. Real modern 1080p anime, longer motion review, matched power measurement,
   smallest-preset comparison.
5. Native VLC integration remains deliberately not started.
