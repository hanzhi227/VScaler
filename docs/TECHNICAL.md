# VScaler technical notes

This is the development and measurement record. Some tables below describe
earlier builds. The [README](../README.md) describes the current setup and
limits. In particular, the embedded player's current default is a five second
rendered-frame target capped at 1.5 GB, not the three frame buffer used in early
tests.

Decodes a video, upscales every frame **on the GPU**, and plays it **inside the
app** on a shared audio/video render clock. It is built for real footage of any
kind — live action, animation, games, old rips — with the model chosen per file:
a cheap general scaler for anything you have not inspected yet, a CNN tuned for
animation, and two photograph-trained generative networks for judging on your
own material. Five models and two non-model positions are selectable from the
Enhancement menu:

| mode | what it is |
|---|---|
| **MetalFX Spatial** (`--mode metalfx`) | **the default**, and the general-video/live-action option: by far the cheapest, sharpens harder than the source warrants — see [docs/GENERAL-VIDEO-UPSCALING.md](GENERAL-VIDEO-UPSCALING.md) |
| CNN / + denoise (`cnn`, `cnnDenoise`) | Metal CNN 2× (or 4× chained); strong on animation and clean edges |
| **Real-ESRGAN AnimeVideo-v3** (`--mode realesrgan`) | `SRVGGNetCompact`, an 18-convolution **generative** network, 4× only and SD-only; best on SD live-action detail *in the synthetic test*, worst on cel animation, and on real (non-anime) video it replaces texture rather than sharpening it — see [scripts/realesrgan/README.md](../scripts/realesrgan/README.md) |
| Real-ESRGAN General-x4 (`--mode realesrgan-general`) | the same network at twice the depth, trained on general photographs instead of anime. **Offered so you can judge it on your own footage, not because it measured well**: it loses to AnimeVideo-v3 on every clip and column scored here, at 1.8× the time |
| Bypass (`bypass`) | plain bilinear: scaled, not enhanced — what another player would show |
| Off (`off`) | no scaling and no model: the decoded frame at its own resolution, a true passthrough |

MetalFX is the default because it is ~4× cheaper than the CNN and the safer thing
to do to a file nobody has looked at yet; the measurements behind that, including
where it over-sharpens, are in the general-video report.

**Model budget: no upscaling model may hold more than 2 GB.** That is the bar a
new mode has to clear, alongside holding real time, and it is a hard constraint
rather than a preference — the player shares 26 GB of unified memory with the OS
and buffers rendered BGRA frames separately. All five model modes
shipped here are far inside it (MetalFX has no weights at all; the Anime4K CNN's
are kilobytes; the two Real-ESRGAN checkpoints are 1.25 MB and 2.44 MB of weights,
with 13.5–53.9 MB of activations, measuring 139–301 MB resident end to end). It
is also why the models evaluated under `scripts/` are judged on two axes and not
one: `scripts/seedvr2/` measures **2.75–12.9 GB** and fails the budget before
quality is reachable at all, where Real-ESRGAN clears it by two orders of
magnitude and was decided on quality instead — and then re-decided, when the port
turned out to be 3.1× cheaper than the measurement that had argued against it.

The window is nothing but video. All controls live in the **menu bar**
(Playback / Enhancement / Audio / Subtitles), and a slim transport strip fades in
over the picture when you move the mouse — so the video fills the window, and
takes the whole screen in **full screen** (green button or ⌃⌘F). Stats are off by
default and toggle with **⇧⌘S** as an overlay, which is where they belong now
that the picture has the window.

Use **⌘O** to select multiple videos or **⇧⌘O** to select a folder. Videos in a
folder and its subfolders are added in filename order. New selections append to
the queue; the first video starts immediately and the next starts after the last
frame of the current video plays. The transport's list button shows the queue,
where you can choose or remove an item, add more videos, or clear it. The previous
and next buttons also appear in the Playback menu.

There is no re-encode and no external player in this path: the upscaled frames go
straight to an `AVSampleBufferDisplayLayer`, audio is decoded to PCM and
scheduled on the same `AVSampleBufferRenderSynchronizer`. A bounded `FrameStore`
holds rendered frames ahead of the display layer. Seeking restarts the decoders at
the new position and flushes the renderers, so scrubbing acts immediately.

The original stream pipeline (encode → MPEG-TS → local HTTP → VLC) is still
available from the CLI (`macvideoupscale`) for serving an external player; it is
unchanged and still validated.

```
embedded player (the GUI)
┌──────────────┐  BGRA   ┌───────────────────┐  BGRA  ┌───────────────────────┐
│ ffmpeg decode│────────▶│ Metal 2×/4× upscale│───────▶│ CVPixelBufferPool      │
│ (any container│ rawvideo│ CNN / MetalFX /    │        │ bounded FrameStore     │
│  VT hwaccel) │  pipe   │ (+ bilinear mode)  │        └──────────┬────────────┘
└──────────────┘         └───────────────────┘                    │ CMSampleBuffer
                                                                   ▼
                                                        ┌───────────────────────┐
   ffmpeg audio ── f32 PCM ──▶ AVSampleBufferAudioRenderer ─▶│ AVSampleBufferRender- │
                                                             │ Synchronizer (clock)  │
                              AVSampleBufferDisplayLayer ◀───│ video + audio renderers│
                                                             └───────────────────────┘

stream pipeline (CLI, for an external player)
   decode → Metal upscale → h264_videotoolbox → bounded byte ring → HTTP → VLC
```

## Requirements

- Apple Silicon Mac (developed on **M5 Pro**), macOS 14+
- Homebrew `ffmpeg` (decode + audio; must be at `/opt/homebrew/bin/ffmpeg`)
- Xcode command line tools (Swift 5.10+)
- Only the optional stream path needs VLC (`/Applications/VLC.app`)

## Build & run

```bash
# Build the app bundle and install it (then launch it like any other app:
# Spotlight, the Dock, or double-click / "Open With" / drag a video onto it).
# It is assembled in a temp dir and moved into place, so there is only ever one
# copy of the app: two bundles with the same identifier make LaunchServices show
# both in Spotlight and pick either for "Open With".
./scripts/make_app.sh                  # → /Applications/VScaler.app
./scripts/make_app.sh --out ./dist     # assemble elsewhere, do not install
./scripts/make_app.sh --install ~/Applications
./scripts/make_app.sh --run            # build, install, launch
```

Once installed, opening a video is the Finder's job: **⌘O** or the menu bar if the
app is already up, otherwise select the file and *Open With ▸ VScaler*. The
bundle is self-contained — the Metal kernels are compiled from source strings
inside the binary and ffmpeg is referenced by absolute path — so it can be moved
or copied anywhere. It is ad-hoc signed, not notarized: fine locally, but a copy
that travels through a download would need **right-click ▸ Open** the first time.

The rest of these commands are the development paths (and how the player was
verified):

```bash
# GUI app straight from the build directory, no bundle: video-only window,
# controls in the menu bar (Open ⌘O, Play/Pause space, ±10 s ←/→, ±1 min ⌘←/⌘→,
# enhancement/scale/audio/subtitles menus, Show Stats ⇧⌘S, full screen ⌃⌘F).
# Clicking the picture toggles play/pause; double-clicking it toggles full screen.
engine/MacVideoUpscaler/.build/release/MacVideoUpscalerApp

# headless verification of the embedded player (status JSON every 0.5 s)
engine/MacVideoUpscaler/.build/release/playercheck \
    --file "/path/to/episode.mkv" --mode cnn --scale 2 --seconds 30
engine/MacVideoUpscaler/.build/release/playercheck \
    --file ep.mkv --script "5:pause,8:resume,12:seek:60,20:mode:bypass"
# ...and inspect exactly what it renders (no window / no screen-recording needed):
engine/MacVideoUpscaler/.build/release/playercheck \
    --file ep.mkv --sub-index 0 --scale 4 --seek 12 --dump-at 0 --dump-dir /tmp/frames

# headless CLI that serves a local MPEG-TS stream to an external player
engine/MacVideoUpscaler/.build/release/macvideoupscale \
    --file "/path/to/episode.mkv" --mode cnn --scale 2 --port 8090 --open-vlc
```

The GUI plays in-app as soon as you open a file — there is no "open in VLC" step.
The CLI's `--open-vlc` is the old path, where VLC opens
`http://127.0.0.1:8090/stream.ts?session=1` and `http://127.0.0.1:8090/status.json`
returns the status snapshot.

### Options

| Flag | Applies to | Default | Meaning |
|---|---|---|---|
| `--file` | both | — | input video (MKV, MP4, anything ffmpeg reads) |
| `--mode` | both | `metalfx` | `off` (no scaling, no model — the decoded frame at its own size), `bypass` (bilinear, scaled but unenhanced), `cnn` (CNN), `cnnDenoise` (denoise weights), `metalfx` (MetalFX Spatial, general/live-action video), `realesrgan` (Real-ESRGAN AnimeVideo-v3, 4× only and SD-only), `realesrgan-general` (Real-ESRGAN General-x4, the same constraint) |
| `--scale` | both | `2` | output magnification: `2` or `4` (see below) |
| `--no-audio` | both | off | drop audio |
| `--seconds` | playercheck | 30 | how long to run before printing the summary |
| `--seek` | playercheck | 0 | start offset in seconds |
| `--sub-index N` / `--audio-index N` | playercheck | off / `0` | burn subtitle track N (0-based) / use audio track N |
| `--dump N` / `--dump-at "0,60"` / `--dump-dir D` | playercheck | — | write output frames as PNGs (first N, or specific indices) |
| `--swiftui` | playercheck | off | host the surface through `NSHostingView`, like the GUI |
| `--script "5:pause,8:resume,12:seek:60,20:click"` | playercheck | — | headless control scripting; emits `{"event":…}` JSON lines. Verbs: `pause`, `resume`, `click` (a mouse click on the picture), `seek:<s>`, `seekrel:<s>`, `mode:<off\|bypass\|cnn\|denoise\|metalfx\|realesrgan\|realesrgan-general>`, `scale:<2\|4>`, `audio:<ordinal>`, `compare:<on\|off>`, `compareleft:<mode>`, `split:<0..1>`, `dragdivider:<viewFraction>` (a synthetic press-and-drag on the line), `readback` (write the frame the renderer is *showing* to `<dump-dir>/displayed_NNN.png` — the paused redraw changes what is presented, which a dump of what was rendered does not prove on its own). The stream CLI has its own set (`play`, `subtitles`, `subs-on/off`, …) |
| `--window` | playercheck | off | show the playback window (otherwise the layer is driven offscreen) |
| `--port` | CLI | `8090` | local HTTP port |
| `--start` | CLI | `0` | start offset in seconds |
| `--ring-mb` | CLI | `128` | bounded stream buffer (memory cap for the read-ahead) |
| `--prebuffer-mb` | CLI | `4` | bytes buffered before serving starts (~1.5 s at 22 Mb/s) |
| `--lead-seconds` | CLI | `30` | how far the producer may run ahead of playback |
| `--open-vlc` | CLI | off | launch VLC on the stream after start |
| `--audio-index` / `--subtitles` / `--sub-index` | CLI | `0` / off / `0` | audio and subtitle-track selection for the stream |

The embedded player's buffer target is `PlayerController.Config.bufferSeconds`,
which defaults to 5 seconds. `bufferBudgetBytes` caps its rendered-frame store at
1.5 GB, with a minimum of four frames. At 3840×2160, that budget holds 45 frames.
`playercheck --buffer-seconds` can override the time target for a test.

The GUI process itself can be driven headlessly for verification (this is how the
player was checked without Screen Recording permission): `MACVIDEOUPSCALER_OPEN=<file>`
opens a file shortly after launch, `_MODE` / `_SCALE` / `_SUB` / `_AUDIO` override
the pickers, `_DUMP_AT="0,150"` + `_DUMP_DIR` write output frames as PNGs,
`_STATS=1` starts with the stats overlay on, `_COMPARE=1` /
`_COMPARE_LEFT=<mode>` / `_SPLIT=<0..1>` start the comparison view,
`_MUTE=1` silences a diagnostic run by setting the volume to 0 — the film is still
decoded and fed to the renderer, so the audio path is exercised without playing it
out loud (`playercheck --mute` is the same thing),
`_CLICK_AT="5,8"` /
`_DBLCLICK_AT="12"` click the centre of the picture at those seconds through the
app's own event dispatch, `_SPLIT_AT="5:0.1,7:0.6"` moves the comparison divider
at those seconds through the same setter the slider and the line drag call (the
one way to drive the paused comparison in the app, since the split presets carry
no key equivalent), and `_DIAG=1` logs status, audio, full-screen and layer
state each second. Each hook stands alone,
and for a bundle launched from the Finder the log has to be captured explicitly:

```bash
open --env MACVIDEOUPSCALER_DIAG=1 --stderr /tmp/run.log -a /Applications/VScaler.app clip.mkv
```

### Magnification: 2× or 4×

Anime4K ships its CNN at **2×** only. The project's 4× is two chained 2× passes
through a reused intermediate, which is what `--scale 4` does. A 3× mode would
need a different model and is deliberately not offered.

**`off` is the one mode that does not magnify**, so the setting does not apply to
it: the frame is presented at the source's own geometry and the display layer does
whatever scaling is needed to fit the window. It is a true passthrough — verified
pixel-identical to the decoded source, all 921,600 channel values of a dumped
frame matching ffmpeg's decode of the same frame exactly — and it is deliberately
not the same thing
as `bypass`, which still scales bilinearly to 2×/4× (so it does GPU work and is
what invents the interpolated pixels). Switching between `off` and a magnifying
mode **restarts the session** at the current position rather than hot-swapping,
because the output geometry itself changes; switching between two magnifying modes
is still the hot swap described below. The comparison view needs two models on one
pixel grid, so it is unavailable while `off` is selected.

The two playback paths differ in what limits 4×:

- **Embedded player**: no video encoder is involved, so the old 4096×2304 H.264
  session limit is not the reason here — **throughput and memory are**. 4× is two
  chained 2× passes, so the second pass runs at 4× the output pixels. Measured:
  640×480 → 2560×1920 holds **0.98× real time** at ~207 MB, but 1080p → 7680×4320
  only reaches **0.61×** (constant stutter) with an ~883 MB in-flight buffer. So
  4× is allowed while the output stays within 4096×2304 and is otherwise clamped to
  2×, with the reason shown in the UI and the stats overlay.
- **Stream CLI**: still `h264_videotoolbox`, which refuses anything above
  4096×2304, so 4× there remains SD-only.

### Source tracks

The engine probes the file's streams and reports them in the status snapshot
(`audioTracks`, `subtitleTracks`), which is what the GUI pickers list. Changing
the audio track, scale or subtitles restarts the session at the current position.

Subtitles are **burned into the picture at source resolution** before upscaling
(bitmap tracks via ffmpeg's overlay filter; text cues rendered natively with
CoreText because this Homebrew ffmpeg lacks libass), so they cannot be toggled
independently of the video — use the picker.

## How playback works

- **Rendered-frame buffer.** A `FrameStore` holds frames ahead of the display
  layer. Capacity is the smaller of the five second target and the 1.5 GB byte
  budget, with a four frame minimum. The producer waits when the store is full.
  The stream path has a separate 128 MB compressed byte ring.
- **The position is the render clock**, not an estimate. The slider and readout
  come from the synchronizer's `currentTime`, so they are frame-exact rather than
  "good to roughly the read-ahead".
- **A/V sync is structural.** Video and audio are stamped on the same session
  timeline (both decoders are started at the same `-ss`; video PTS is
  `frameIndex / fps`, audio PTS is `samples / 48000`) and handed to two renderers
  attached to one `AVSampleBufferRenderSynchronizer`. There is no drift
  correction to tune.
- **Seeking** tears the session down (kill decoders → join producers → flush
  renderers), rebuilds it at the target, and starts the clock once the first
  samples are queued. A seek made while paused presents the new frame and stays
  paused. It happens **off the main thread**, so the window and the menu stay
  responsive while the decoders respawn (~0.6 s on the test clips, dominated by
  ffprobe + ffmpeg), and seeks queue in order instead of interleaving.
- **Mode switch is hot**: CNN ↔ bilinear ↔ CNN+denoise swaps between frames with
  no restart and no rebuffer. **Scale** (and audio/subtitle-track changes) restart
  at the current position, because the output geometry or the source mapping
  changes; both preserve the pause state.
- **Controls are the picture and the menu bar.** Clicking the video toggles
  play/pause and double-clicking it toggles full screen (a press that moves instead
  drags the window, since the window has no title bar). **← / → skip 10 s**
  (⌘← / ⌘→ skip a minute): presses **add up** and are debounced, so five quick
  presses become one rebuild 40 s forward rather than five rebuilds — a running
  "+30s" appears over the picture as you press, so the presses are acknowledged
  immediately, and the skip happens half a second after the last one. Because the first click of a
  double click arrives as an ordinary click, the single click is held back for up to
  **300 ms** to see whether a second one follows — the same trade mpv and VLC make
  (macOS's own interval is 500 ms, which felt sluggish for pausing). Everything else
  is in the menu bar, so full screen is nothing but video. Moving the mouse reveals a
  transport bar that hides itself after 2.5 s of stillness, and never while paused.
- **Playback runs at real time.** Measured **0.99–1.01×** while the GPU keeps up.
  The percentage shown is a sliding-window measurement — content seconds produced
  per second of *render clock*, over the last full second of playing. Keying it to
  the clock rather than wall time is what keeps a pause from reading as a
  slowdown (see the paused-session row in Validation).
- **Aspect ratio is preserved without a second codec.** The player view hosts the
  display layer and aspect-fits it to the source's display ratio inside its
  bounds, so the picture is letterboxed correctly (and anamorphic sources are
  stretched to their DAR) with no HEVC fallback.
- **Cold starts say so.** Until the first frame is on screen the window shows a
  "Preparing…" indicator rather than black, because decoders, the scaler and the
  first frame take a moment to come up — and a black screen with no explanation
  is indistinguishable from a broken player.

### Comparing two models on one picture

**⌥⌘C** splits the picture so two models can be judged against each other on the
same frame, at the same instant, at the same output size. The left half shows the
model picked on the left, the right half shows the enhancement (the enhancement
menu's model), and a thin white line marks the boundary.

- Either side can be any model — **CNN, CNN + denoise, MetalFX Spatial,
  Real-ESRGAN, or Original**, where Original is the bypass path: the source
  upscaled with a plain bilinear filter, which is what any other player would show.
  Real-ESRGAN is 4× only, so it can only be a half of a 4× session; asking for it
  at 2× is refused with that reason rather than silently dropping the half.
- The **slider** (or the menu's three presets) moves the line anywhere from 0% to
  100%, so either model can cover the whole picture. At 0% the comparison costs
  and changes nothing at all — the frame is byte-identical to normal playback.
- **Drag the line on the picture** to wipe between them. The line is grabbable
  within 18 pt; everywhere else a press-and-drag still moves the window, a click
  still pauses, and a double click still goes full screen.
- Both halves are produced by real scalers and composited, so a half is *exactly*
  what that model outputs — verified bit-for-bit against single-model runs (see
  Validation). Picking the same model for both sides skips the second pass
  entirely and draws only the divider.
- The second pass is real GPU work but did not cost frame rate in measurement
  (two different CNNs: 0.995× at 480p→1280×960 and 1.000× at 4×→2560×1920,
  against 0.997×/0.999× for one model). Memory is the tighter constraint, since a
  second CNN allocates its own weights: ~305 MB in flight against ~205 MB at 4×.
- **It works on a stopped frame, which is the point of a comparison view.** While
  paused, moving the line, switching either side, or turning the comparison on and
  off re-renders the frame on screen straight away rather than waiting for playback
  to resume. Once the comparison has been used, the player keeps the sources of the
  last few frames, so the still stays the frame you stopped on and its halves are
  exactly what the chosen models output on it. Pausing first and comparing second —
  the only way to judge a single frame properly — is now the ordinary way to use it.
  See Validation for what was measured, and
  [results/paused_comparison_verification.txt](../results/paused_comparison_verification.txt).

## Layout

```
engine/VideoUpscaleMetal/ Metal CNN 2× pipeline (Anime4K v3.2 weights), vupbench CLI
engine/MacVideoUpscaler/     the app: MacVideoUpscalerCore + macvideoupscale (CLI) + MacVideoUpscalerApp (SwiftUI)
                          + playercheck (embedded-player harness)
  MacVideoUpscalerCore/      PlayerController + EmbeddedPlayer (in-app), PipelineController (stream),
                          FFmpegDecoder/FFmpegAudioDecoder, Upscaler, MetalFXScaler, StreamServer,
                          VLCController
  AppBundle/Info.plist    bundle metadata for the .app (document types, icon, version)
scripts/                  test-material generation, benchmark, sync-check and sustained-test tooling
                          + make_app.sh (builds/signs/installs the .app) and its icon generator
                          + eval_general_upscaler.py / make_eval_grid.py (upscaler quality)
                          + bench_realesrgan_conv.swift (one conv layer in isolation -
                            a kernel number, never extrapolate it to a network)
                          + realesrgan/ (SRVGGNetCompact: weights, bench, x4 render)
                          + rt4ksr/ (NTIRE Real-Time 4K SR: weights, bench, x4 render)
results/                  benchmark outputs, quality grids, validation JSON
test-media/               generated + derived test clips and canonical frames
                          liveaction/ = the only live-action material, see its README
docs/                     comparison report, general-video report, offline shortlist,
                          evaluation notes, VLC integration research
```

## Validation (M5 Pro)

### Embedded player (historical measurements)

| check | result |
|---|---|
| **`off` is a true passthrough** | frame 60 dumped from `--mode off` is **byte-identical** to the source decoded by ffmpeg (`max\|d\| = 0`, 0 of 518400 pixels differing) |
| **`off` does not scale** | output is the source's own geometry, `960x540` for a 540p source, and `--mode off --scale 4` still gives `960x540` — the magnification setting does not apply |
| **`off` is not bypass** | bypass still magnifies (`960x540 → 1920x1080`) and does the bilinear work; off does neither |
| **crossing to/from `off` restarts, it does not hot-swap** | the output geometry itself changes, so switching mid-playback restarts at the current position: metalfx→off ends at `960x540`, off→metalfx at `1920x1080`, both `state=processing`. Two *magnifying* modes still hot-swap (cnn→cnnDenoise: 148 frames in 5 s, no restart) |
| **`off` refuses the comparison view** | `compare:on` in off mode leaves `comparing=false` with "The comparison needs a mode that scales; Off plays at the source's own size", rather than compositing a 1:1 frame against a 2× model on a grid they do not share. A left half set to `off` by any route is refused the same way, and the menu hides it |
| **the default is MetalFX** | with no `--mode`, the dumped frame is **byte-identical** to an explicit `--mode metalfx` run and differs from cnn (max 66) and bypass (max 47) |
| **the installed bundle defaults to MetalFX too** | launching `/Applications/MacVideoUpscaler.app` with no mode override and with `_MODE=metalfx` produces **the same 1280×960 frame, identical sha256** (`7152314e…`); the GUI honours `_MODE=off` the same way, giving 640×480 pixel-identical to the source frame |
| **`off` correctness re-checked independently** | `--mode off --dump-at 90` (the dump is 1-based, so this is source frame 89) matched ffmpeg's decode with **max difference 0 across all 921,600 channel values**. The first attempt compared against frame 90 and showed a mean difference of 3.8 — an off-by-one in the check, not a defect |
| app bundle | `make_app.sh` builds, ad-hoc signs (`codesign --verify --strict` OK) and installs `MacVideoUpscaler.app`; `open -a` on the **installed** app with no file gives `state=idle` and a real window |
| open a file from the Finder | cold launch via Apple Event (`open -a /Applications/MacVideoUpscaler.app clip.mkv`) plays it: frames 21 → 261 (playback 0.61 → 8.61 s), audio 1.83 → 9.81 s, `video=rendering audio=rendering window=true`, with two ffmpeg children decoding that exact path |
| `.mkv` becomes an openable type | the imported UTI registers: `mdls` reports `kMDItemContentType = org.matroska.mkv` conforming to `public.movie`, so the app shows under *Open With*; `LSHandlerRank = Alternate` leaves double-click with the system player |
| playback advances at real time | 602 frames / 20 s = **0.998×** on 480p CNN; 451 / 15 s = 0.990× on 1080p→4K; `playercheck` |
| buffer in this early build | **3 frames ≈ 100 ms** (2 above ~9 MP); memory 103 MB at 480p→960p, 307 MB at 1080p→4K. The current buffer is larger; see the opening note. |
| renderers actually render | `renderStatus` = `video=rendering audio=rendering`, `window=true`; frames dumped from the real GUI process |
| the picture itself | dumped pixel buffers show upscaled anime content; a 4× dump at t=12 shows the burned-in subtitle |
| pause / resume | clock freezes on pause, continues on resume (event script) |
| pause / resume, **audio** | audio stayed fed across a 10 s pause and a second one: 6.91 s → 14.25 s of PCM, continuous. It used to **stop at the first pause** and stay silent for the rest of the movie, because the audio thread spent its "waiting for the renderer" budget (~4.5 s) against a paused, deliberately-full queue and then gave up — the audio path has no restart. It now parks while paused and only stops on EOF or a failed renderer |
| click the picture | a synthetic click on the surface pauses (clock frozen at 9.71 s) and a second one resumes, through `PlayerView.mouseDown/mouseUp → onSurfaceClick`. In the **real GUI** too, where window hit-testing decides it: `_CLICK_AT` logs `hit=PlayerView` and the state walks processing → paused → processing, so the click is not swallowed by the SwiftUI overlays |
| double-click the picture | toggles full screen **without touching playback**: `_DBLCLICK_AT="5,9"` flips `full=0 → 1 → 0` while state stays `processing` and frames (148 → 358) and audio (6.06 → 13.08 s) keep advancing, and the same works *while already full screen*. The single clicks around it still pause and resume — the 300 ms deferral keeps the two apart |
| rate after a pause | reads **1.00×** on the same run that reports 0.616× when the metric counts paused wall time as decode time; four concurrent 1080p→4K denoise runs report 0.990–1.006×, each agreeing with an independent `frames/fps ÷ playback` computation to within 1% |
| seek | lands exactly on the target (seek to 120 → 120.0 s; seek to 20 → 20.0 s) |
| seek latency | a seek returns to the caller immediately (event lag **+0.00 s**, was +0.74…+2.73 s when the rebuild ran on the main thread) and the picture lands ~0.6 s later |
| skip accumulation | four "forward 10" presses inside 600 ms → **one** rebuild, landing at +40 s; two presses a second apart → two rebuilds (+10, then +20); forward-forward-back → one rebuild at +10. Presses are debounced 500 ms and summed, and the running total is shown as "+30s" while it waits |
| rapid skipping doesn't wedge | the app survives a burst of arrow presses that used to freeze it: the main thread keeps running (status keeps updating) and playback resumes; the freeze was `Process.waitUntilExit()` pumping the run loop inside a seek that held the controller's mutex, so the next seek delivered by that nested loop deadlocked on it |
| seek while paused | presents the new frame and **stays paused** |
| start offset | fixed: the first open used to clamp a requested offset to 0 because the duration was not yet known |
| stalled decoders | fixed: ffmpeg inherited the terminal on **stdin**, and its interactive keyboard read raised **SIGTTIN**, which *stops* the child when the app runs in a background process group — playback then hung on "Preparing…" with no error. Every spawned child now gets `/dev/null` stdin, and a 12 s watchdog reports a stalled decoder instead of hanging |
| mode switch, mid-movie | **hot, no restart**: the frame before the swap is byte-identical to a pure-CNN run (0.00%) and the frame after to a pure-bypass run (0.00%), while CNN and bypass differ by 0.6–1.2%. A swap made while paused is applied *there and then*: the frozen frame is re-rendered with the new model (0.00% vs a pure-bypass run, read back from the presented image, clock frozen). A **failed** scaler build now keeps the old mode and says so, instead of silently ignoring the change |
| scale switch, mid-movie | restarts at the playback position (+0.1 s): 2× → 4× gives 2560×1920 and 4× → 2× gives 1280×960, both at ~0.97× real time; made while paused it stays paused, presents the new-scale frame, and resumes from the same position |
| 4× clamp | 1080p clamps to 2× (307 MB, 0.97×); unclamped 8K measured **0.61× / 883 MB** — the reason for the clamp |
| subtitles, 2× **and** 4× | bitmap (`dvd_subtitle`) and text (`subrip`, CoreText) both survive the double pass — burn-in happens at source resolution, before scaling |
| end of file | 10 s clip: state `finished`, 600/600 frames delivered |
| comparison: each half is its own model | at 50% (left Original, right CNN) the left 640 columns are **byte-identical** to a bypass run's, and the right 634 are byte-identical to a CNN run's, while each half differs from the *other* model's version — so both sides are exactly what those models output, not a resampling of it |
| comparison: the extremes are pure models | split 0% is byte-identical to a plain CNN frame (the pass is skipped entirely, so "show only the enhancement" costs nothing), and split 100% — Original covering the picture — is byte-identical to a bypass frame |
| comparison: same model both sides | CNN vs CNN leaves the picture untouched: both halves are byte-identical to a plain CNN run, and only the 2 divider columns differ. No second pass is run. (The divider was in fact missing here until 15 Sept — the divider kernel declared its output at texture index 0 where the dispatcher binds index 1, and its thread grid spanned only the columns beside the split while the kernel tests absolute columns, so it painted nothing. Caught while verifying the paused view, fixed, re-measured) |
| comparison: the divider | exactly 2 output columns of pure white (`max(2, width/640)`, so 6 at 4× 4K) sitting at the split; the pixels either side are the two models' |
| **comparison: while paused** | moving the line, switching either model, or turning the comparison on/off **changes the picture on a paused frame**: the still is re-rendered and re-presented with the clock frozen. At split 25% (left Original, right CNN) the presented image's left 320 columns are **byte-identical** to a bypass run and its right 958 to a CNN run, with the divider exactly 2 white columns at x=320–321; the same holds at 50% (x=640) and 75% (x=960). Switching the left model to CNN (both sides the same) leaves everything but the divider byte-identical to a plain CNN frame, and an enhancement-model change while paused is byte-identical to a pure-bypass frame. Repeated redraws are stable (same setting reproduces the frame byte-for-byte), `framesProcessed` and the render clock do not move through eight such changes, and resuming is unaffected (the producer refills its 3-frame queue, then holds the source rate: 84→146 frames within 2 s of the resume and 206 by 4 s). Divider drags on the picture behave the same way (the drag path calls the same setter) |
| comparison: while paused, with no history | with the comparison switched **on** while paused and no frames retained (it was off for the whole session), the still lands on the newest frame the producer has and the clock moves onto it (measured +0.066 s) — the one case where the still is not exactly the frame the viewer stopped on. Afterwards it is stable, and once playback has run with the comparison on, the still is the frame that was on screen, with no clock movement (found by readback: divider at x=768–769 for split 60%, clock frozen at 6.786 s) |
| comparison: what "the picture" means | the checks above read the presented image back from the renderer (`AVSampleBufferVideoRenderer.displayedPixelBuffer`, macOS 14.4+), not the last frame rendered: a queued frame is not proof of what is on screen. Exposed as `PlayerController.displayedFrame` and the harness's `readback` verb |
| comparison: dragging the line | a synthetic press on the divider dragged to the view's left edge lands at 0.000, to the right edge at 1.000, and back to the middle at 0.500, with playback uninterrupted (state stays `processing`, clock keeps advancing) — so a divider drag does not move the window and does not pause. A plain click afterwards still pauses (state reads `paused`, clock frozen at 9.62 s), and click/double-click were re-checked with the comparison on |
| comparison: cost | two *different* CNNs ran at 0.995× realtime (480p→1280×960) and 1.000× (4×→2560×1920) against 0.997×/0.999× for a single model — no frame rate lost, though a second CNN costs ~100 MB more in flight at 4× |
| comparison: the app and the harness agree | the GUI launched with `_COMPARE=1 _COMPARE_LEFT=bypass _SPLIT=0.25` produced frame 120 **byte-identical** to `playercheck --compare --compareleft bypass --split 0.25` frame 120, and the dump has white at exactly x=320–321 |
| stream path regression | CLI still serves `/status.json` and produces frames |
| format matrix | **36 of 36** container/codec files play (mp4, m4v, mov, mkv, webm VP8/VP9, ts, m2ts, mts, flv, avi, divx, mjpeg, mpeg-1, mpeg-2, vob, mxf, wmv, asf, 3gp, rv20) with aac/opus/vorbis/mp3/mp2/ac3/wmav2/PCM audio, 0.98–1.07× real time — `scripts/check_formats.sh` against `scripts/make_format_matrix.sh` |
| 10-bit and odd sizes | 10-bit H.264 and 10-bit HEVC decode and convert to BGRA; a 641×481 source produces a 1282×962 output without a texture-size problem |
| cover art | a poster frame in front of the film is skipped in all three layouts (film first, poster first flagged `attached_pic`, poster first unflagged): 145–146 frames instead of **1 frame then a stall**, which is what `-map 0:v:0` did. A file with only a cover (audio + mjpeg) reports `has no video — only cover art` |
| rotation | a display matrix at 90°, 180° and 270° is applied: output is 960×1280 for a 640×480 source at 90°/270° and 1280×960 at 180°, and the picture matches ffmpeg's own autorotation at **41.4–42.1 dB** (a wrong direction scores ~12 dB). The filters were checked bit-identical to autorotation first, so the two can be compared this way |
| rotation + subtitles | a rotated file with text subtitles burns them in upright after the turn; with bitmap (dvd_subtitle) subtitles the subpicture is composited before the turn and rotates with the picture. The unrotated bitmap path is byte-identical to before the change |
| a real-world mp4 | the 11-minute 480×360 SpongeBob file plays at **0.996×** real time at 2× (960×720) and **0.997×** at 4× (1920×1440), seeks to 300 s, takes three 10-second skips, and plays out to `finished` at 665.4/665.5 s |
| declared document types | `lsregister -dump` shows all 20 types bound to the installed bundle, and the app plays a `.webm`, `.ts`, `.wmv`, `.rm`, `.mxf`, `.divx` and a rotated `.mp4` opened through LaunchServices (`open -n -a`) |

### General (live-action) video — MetalFX Spatial

Added against a shortlist of MetalFX Spatial, Real-ESRGAN `general-x4v3`,
`x2plus`, Anime4K and RealBasicVSR. Full report and method:
**[docs/GENERAL-VIDEO-UPSCALING.md](GENERAL-VIDEO-UPSCALING.md)**. Summary:

| check | result |
|---|---|
| it runs | `--mode metalfx`, *MetalFX Spatial* in the Enhancement menu and in both halves of the comparison view; the menus and both CLIs pick it up from `ScalerKind` |
| real time | 540p→1080p **0.997×**, 270p→1080p at 4× **0.992×**, 1080p→2160p **1.007×** — 300/300 frames, `finished` |
| cost | **0.33 ms** at 640×480→1280×960 and **0.99 ms** at 1080p→4K, against 3.81 ms for the Anime4K CNN at the latter — the cheapest mode by ~4× |
| mode switch is hot | switching CNN→MetalFX mid-movie delivers frame 60 **byte-identical** to a pure-CNN run and frame 120 byte-identical to a pure-MetalFX run (against a 1.45–1.65 mean difference between the modes), state stays `processing` through the swap |
| comparison view | at 50% (left MetalFX, right CNN) the left 960 columns are byte-identical to a pure MetalFX run and the right half to a pure CNN run, divider white |
| app and harness agree | the GUI launched with `_MODE=metalfx` produced frame 90 **byte-identical** to `playercheck --mode metalfx` frame 90 |
| 4× is one pass, not two | MetalFX takes the output geometry directly, so 4× is a single encode rather than chained 2× passes — measured better: acutance **1.94× vs 2.18×** of the truth and PSNR **29.76 vs 28.80 dB** on 480×270→1080p |
| the knob that exists | `colorProcessingMode` is the only quality setting; `.perceptual` (correct for gamma-encoded video, and shipped) measured better than `.linear`: acutance 1.69× vs 1.86×, PSNR 32.21 vs 32.07 dB |
| quality vs the truth | **it is the sharpest mode and the least faithful.** Acutance **1.54–2.18×** the ground truth (Anime4K CNN: 0.73–0.90×), lowest PSNR/SSIM of the four modes in all three live-action runs, and on anime **37.51 dB — worse than plain bilinear's 39.42** |
| the other four | **corrected 2026-09-15.** `general-x4v3` is **47.6 ms/frame at 270p→1080p = 21 fps** measured end to end, not the 277.7 ms a single-layer microbenchmark projected — that projection understated the family's throughput ~2.6× and is retracted in `results/general/realesrgan_feasibility.txt`; see `results/general/realesrgan_bench.txt`. Through the port it is **13.76 ms**, and it **loses to `animevideov3` on PSNR and SSIM on *both* anime and live action** (26.18/26.36 dB against 28.36/27.79) at 1.84× the cost, so its live window is narrower as well as worse — real time at 270p/360p, but **0.63× real time at 854×480→3416×1920**. It is nonetheless **offered as `--mode realesrgan-general`**, on the same port, so it can be judged on real footage rather than on the scores; see the Real-ESRGAN section below. `scripts/realesrgan/README.md` `realesr-animevideov3` is **faster than real time** and is the best mode tested on SD live-action detail (acutance 0.97× vs the CNN's 0.77×) but loses to the CNN on anime (PSNR 28.36 vs 35.57). `x2plus` is heavier again and unmeasured; RealBasicVSR needs optical flow and a sequence window, which is a structural objection rather than a speed one. **Both `animevideov3` and `general-x4v3` have since been ported and are live modes** — 7.33 ms instead of 22.59 for the first, `--mode realesrgan` and `--mode realesrgan-general` — written up in `docs/GENERAL-VIDEO-UPSCALING.md` §"The AnimeVideo-v3 question" and `scripts/realesrgan/README.md` |
| reproducibility | re-running `scripts/eval_general_upscaler.py` reproduces every published figure to the last digit |

### RT4KSR — measured, works, not adopted

[RT4KSR](https://github.com/eduardzamfir/RT4KSR) (NTIRE Real-Time 4K SR, CVPRW
2023) was evaluated as a fifth mode: **44,521 parameters**, 14 convolutions after
folding, no recurrence. Full report: `scripts/rt4ksr/README.md`.

| check | result |
|---|---|
| it is fast enough | **3.3–6.0 ms/frame** from 480×270→960×540 to 960×540→1920×1080 through PyTorch/MPS — 0.08–0.18× of a 30 fps budget, so nothing about it is too slow to play |
| it is verified | folded model checked against upstream on the real weights: training-shaped class **bit-identical**, folding within 4.1e-06, and the high-frequency branch it drops changes the output by **exactly 0** (that branch is dead in the released config) |
| quality | **it never wins a column.** Ties the best PSNR twice (29.90 dB at 270p→1080p, 33.52 dB at 1080p→4K) and loses the other three; softer than the Anime4K CNN on every clip (0.57–0.85× acutance vs 0.77–0.90×); SSIM below plain bilinear on three of five setups |
| why | trained on bicubically downscaled **clean** images with no deblocking stage, against sources that are compressed SD video. Published gain over bicubic is +0.27 dB |
| verdict | **not adopted** — a live mode measuring at or below the ones already present would be a menu entry, not an improvement, and a Metal port plus weight baking is real maintenance. (Written when there were four model modes; there are five now and the argument is unchanged — it still wins no column.) Kept in `scripts/` so the decision is cheap to revisit |

Also checked and rejected: **PiperSR**, proposed as a real-time ANE-native model.
Its GitHub `.gitignore` excludes the CoreML model *specification*, so a clone
cannot run; its own `benchmark.py`/`inference.py` call a non-existent coremltools
enum and pass the wrong input name and type, so its published benchmarks cannot
have come from its published harness; and despite "any resolution (flexible input
shape)" the shipped model accepts **only 128×128**, which makes its headline
"360p full-frame, 44.4 FPS" impossible. Details in `scripts/rt4ksr/README.md`.

### Real-ESRGAN — two shipped live modes, and where not to point them

The Real-ESRGAN family is `SRVGGNetCompact`, and both of its released video
checkpoints now play live through one port: `animevideov3`, trained on anime video
(`ScalerKind.realESRGAN` / `--mode realesrgan`), and `general-x4v3`, twice as deep
and trained on general photographs (`ScalerKind.realESRGANGeneral` /
`--mode realesrgan-general`). Full report: `scripts/realesrgan/README.md`.

| check | result |
|---|---|
| it is real time | **7.33 ms/frame at 480×270→1920×1080 = 136.8 fps = 0.22× of a 30 fps budget** for `animevideov3`, whole network. That is **3.1× faster than the 22.59 ms** the same model measures in eager PyTorch on this GPU, because an MPSGraph executable fuses conv+bias+PReLU where eager dispatch writes 64 channels of intermediate to memory between each pair. `general-x4v3` is 13.76 ms at the same size — 1.84× |
| in the player, end to end | `animevideov3`: 480×270→1080p **1.007× real time**, 640×480→2560×1920 **0.992×**, 960×540→3840×2160 **0.991×** — 140–353 MB resident. `general-x4v3`: **0.92× real time at 854×480→3416×1920** and **0.63×** in the same run once the graph is warm — the deeper network stops holding 30 fps before the smaller one does. A 1080p source is **refused** by both, not clamped: 4× there is 7680×4320, past the encoder's 4096×2304 |
| they are inside the 2 GB model budget | `animevideov3` holds **13.5 MB at 480×270, 53.9 MB at 960×540** of tensors; `general-x4v3` holds the same tensor shapes plus 2.4 MB more weights. Measured end to end, whole player resident: **139 MB** at 480×270→1080p (`animevideov3`) and **146 MB** (`general-x4v3`); **295 MB** and **301 MB** at 854×480→3416×1920. PyTorch's eager 37–141 MB is the comparison; the port holds one fp16 tensor rather than a chain of fp32 intermediates |
| they are verified, not asserted | the shipping code path matches PyTorch to **59.99 dB** (max error 0.7 of one 8-bit level) and the fp32 model to 59.97 dB; re-deriving the network from the exported fp16 weights reproduces fp32 to 70.4 dB (`animevideov3`) and 68.9 dB (`general-x4v3`), which is the round trip that would catch a wrong checkpoint — that bug did happen, and loading `general-x4v3`'s depth against `animevideov3`'s file fails only at `body.34`, pointing at the last convolution instead of at the filename. `scripts/realesrgan/verify_metal.py` |
| they are the right architecture | `SRVGGNetCompact`, 3×3 convolutions with PReLU: 18 and 621,424 parameters for `animevideov3`, 34 and 1,213,296 for `general-x4v3` — no attention, no recurrence, no optical flow. Both released checkpoints load `strict=True`, no missing or unexpected keys |
| `animevideov3` best on SD live-action detail — **on a synthetic downscale only** | acutance **0.971×** of the ground truth (the only mode within 3%; Anime4K CNN 0.768×, MetalFX 1.995×) and most legible on fine text at 3× — `grid_shirt_text_with_animevideov3.png`. **This does not survive on real video:** reported as looking bad on non-anime material, checked on genuine native 480p, and it smooths fine texture into rounded blobs and expands highlights where the CNN and MetalFX keep it — `results/general/yt_mhw_480p/grid_real_particles_animevideov3.png`. The test that flattered it was a lanczos downscale of a known reference, which is close to invertible |
| `animevideov3` worst on anime | PSNR **28.36** vs the CNN's 35.57, SSIM 0.8959 vs 0.9478, smooth-region error 1.912 vs 1.374 — it posterises flat cel gradients, visible in `grid_anime_with_animevideov3.png` |
| **`general-x4v3` loses on every column measured**, including to its anime-trained sibling | anime 270p→1080p: PSNR **26.18** against `animevideov3`'s 28.36 and bilinear's 34.07; live action: **26.36** against 27.79 and 29.28. Acutance 2.400×/1.044× against a 1.000× truth means it invents more edge energy than either. The live mode reproduces this — scored through the player it lands at **26.14 and 26.35 dB**, within 0.04 dB of the offline render, which is also the cross-check that the mode runs the checkpoint and not something else. On the real 480p clip its flicker is **1.223× the bilinear run's own frame-to-frame change**, the most temporally active of any mode there, against `animevideov3`'s 1.145× |
| why ship it then | **because a user asked to judge it themselves, and that is a better test than this project's scoring set.** Every clip scored here that flattered these models is a synthetic lanczos downscale; the one genuine native-degradation clip is a single 65-second promotional video. On the user's own material the ordering may differ, and switching modes in the comparison view costs one keystroke. It is offered with its menu hint stating plainly that it measured worse, so choosing it is a deliberate act rather than a discovery |
| what it costs to carry | the second checkpoint's base64 literal adds **3,283,280 bytes to the app binary** (3,162,480 → 6,445,760, measured by building the same code with a stubbed-out weights file), because Swift does not dead-strip a large literal from a public enum. Both graphs are built lazily, so the mode that is not selected costs nothing at runtime — but the weights load with the image either way |
| 4× only, and they say so | there is no x2 checkpoint for either variant, so a 2× request runs at 4× rather than silently showing half the magnification. Every other mode could satisfy any request by clamping; a model pinned to one factor cannot |
| they also correct the record | twice. They disproved the published claim that Real-ESRGAN-class models are 10–100× too slow (a single-layer microbenchmark's 2.67 TFLOP/s extrapolated to a whole network; measured end to end the family reaches ~7.2), and then the port disproved the reverse assumption that a hand-written kernel would beat a framework — measured GPU ceilings (fp16 8.4 TFLOP/s, `simdgroup_matrix` *slower* at 4.2) say it could not |
| verdict | **both shipped; neither is the default for anything.** `animevideov3`’s quality finding that first argued against a port is unchanged and ships with the mode: it wins on photographic detail *in synthetic tests* and loses on cel animation, where the Anime4K CNN is cleaner and more faithful. `general-x4v3` is kept as a comparison target, not a recommendation |

To see for yourself, without opening the app — this prints nothing and writes
three crops under `results/general/yt_mhw_480p/`, left to right: source, bilinear,
AnimeVideo-v3, General-x4v3.

```bash
scripts/realesrgan/pair_grid.sh
```

For the wider shortlist this model came from — SeedVR2, Video2X, FlashVSR, Real-CUGAN
— see **[docs/OFFLINE-UPSCALING.md](OFFLINE-UPSCALING.md)**, where this model is
now the entry that *is* a live mode.

### SeedVR2 — measured, runs here only in 4-bit, and fails on both budget and frame time

The highest-quality option on that shortlist, and the one that was worth checking
properly: a **one-step** diffusion restoration transformer (no iterative sampling,
no CFG), so unlike a 50-step model it could not be dismissed by structure. Two
independent runtimes were measured. Full report: `scripts/seedvr2/README.md`.

| check | result |
|---|---|
| the architecture is verified | read off the shipped safetensors headers, not the config: **32 blocks, of which 10 are dual-stream (text+video) and 22 video-only**, MLP hidden 6912 — which `expand_ratio: 4` does not describe. The model built from those shapes matches the released 3B checkpoint to **0.02%** (3.3908 B vs 3.3915 B) |
| the CUDA build does not fit | 3B fp16 needs **7.63 GB** at the smallest input that runs (640×480, *one* frame) — 3.8× over the 2 GB budget; 12.87 GB at 25 frames. 7B cannot load at all (16.01 GB against a 15.10 GB GPU ceiling). The sub-2 GB quantisations (`nvfp4`, `fp8`, `int8`) **do not run on MPS** at all — verified, `MPS device does not support mm for non-float inputs` |
| its best frame time | **110 ms/frame = 0.30× real time** at 640×480, and that is the *floor*: the DiT alone, batched 9 frames, random weights, VAE excluded. **Every output resolution the app uses runs out of memory before it produces a frame** (1080p and 4K, and upstream's own 100×720×1280 example) |
| the 4-bit MLX build really is better | [`AbstractFramework/seedvr2-3b-4bit`](https://huggingface.co/AbstractFramework/seedvr2-3b-4bit) runs through `mlx-gen`: storage 13.57 → 2.54 GiB. It is the only SeedVR2 that runs on this machine, and it is why this row could be measured rather than argued from H100 guidance |
| …and still fails both bars | peak RSS **2.75–2.90 GB** in every configuration that runs — over budget, and forcing a 1 GB MLX cache **raised** it and broke the run. ~5.5 s for one 480×270→960×540 frame, ~1.8 s/frame for 1× restore = **0.006–0.017× real time**. Video **upscaling is refused outright** by mlx-gen's own safety check ("safe video mode only supports source-size restoration or smaller output") |
| verdict | **not adopted** — two independent reasons, either sufficient: throughput two orders of magnitude short, and a footprint quantisation reduced 9× without bringing under 2 GB. Kept in `scripts/seedvr2/` with both benches, so the answer is reproducible |
| also checked | **Video2X** is not a model at all but a C/C++ **framework** (AGPL-3.0) that drives Anime4K v4, Real-ESRGAN, Real-CUGAN and RIFE over ncnn/Vulkan — its default upscaler is the Anime4K CNN this app already ships. No macOS build exists in any release and it is batch by construction; frame interpolation changes the frame count, which a clock defined as `frameIndex / fps` cannot use. `scripts/video2x/README.md` |

### Stream pipeline (VLC)

| check | result |
|---|---|
| 1080p → 4K CNN processing (peak) | 3.8 ms/frame GPU-only; **62–74 fps = 2.1–2.5× realtime** |
| 480p → 4× (2560×1920) | **111 fps = 3.7× realtime** on the SD Naruto source |
| full-episode playback (VLC) | 0 lost pictures, 0 corrupt, 0 discontinuities |
| A/V sync (event-based) | **+1.3 ms** output; a 1 s-delayed control is **rejected** (+1014 ms) — `scripts/check_av_sync.py` |
| playback rate | **0.997× = real time** while enhancing 3.3× ahead (`scripts/check_playback_rate.py`) |
| seek latency | **~0.5 s** with the ring full (`--script` event timing) |
| audio/subtitle tracks | selection verified against the source tracks (`scripts/check_tracks.py`) |
| aspect ratio | anamorphic 720×480 SAR 32:27 → 1440×960 SAR 32:27 DAR 16:9 via hardware HEVC |
| memory | read-ahead bounded to ~30 s (≈12 MB at 1080p-class bitrates) plus fixed buffers |
| thermal / CPU / GPU | `nominal` throughout; 5–31 % CPU, 36–45 % GPU |
| playback controls | pause/resume/seek/mode-switch verified headlessly (`scripts/controls_test.sh`) |

Reproduce:

```bash
# embedded player, with controls
engine/MacVideoUpscaler/.build/release/playercheck \
    --file test-media/segments/naruto_t290.mkv --mode cnn --seconds 18 \
    --script "3:pause,6:seek:15,9:resume,13:scale:4"

# format support: build a file per container/codec, then play every one of them
scripts/make_format_matrix.sh /tmp/format-matrix
scripts/check_formats.sh /tmp/format-matrix 5

# the comparison view, split 25% with the original on the left
engine/MacVideoUpscaler/.build/release/playercheck \
    --file <video> --compare --seconds 10 --mute \
    --script "1:compareleft:bypass,1.1:split:0.25,5:dragdivider:1"

# A/V sync + pixel fidelity of a stream
python3 scripts/check_av_sync.py --reference <src.mkv> --stream <captured.ts> --dur 20 --video-t 5

# sustained playback with telemetry (needs VLC)
python3 scripts/sustained_test.py --file <video> --mode cnn --port 8098 --seconds 300

# playback controls (pause/resume/seek/mode switch) + post-seek A/V sync
./scripts/controls_test.sh <video> [seek-target-seconds]

# playback rate: timestamps + VLC's real playback speed
python3 scripts/check_playback_rate.py --file <video> --scale 4 --port 8430

# full-episode sustained run: player + memory + thermal telemetry
python3 scripts/full_episode_test.py --file <episode> --minutes 30
```

## Formats

The demuxing and decoding are **ffmpeg's**, not AVFoundation's — that is why the
player reads Matroska at all — so container support is not a list this project
maintains by hand. What it does maintain is:

- **What the Finder will hand over.** `AppBundle/Info.plist` declares 20 content
  types: `org.matroska.mkv`, `public.mpeg-4`, `com.apple.m4v-video`,
  `com.apple.quicktime-movie`, `public.avi`, `org.videolan.divx`,
  `org.webmproject.webm`, `public.mpeg-2-transport-stream`,
  `public.avchd-mpeg-2-transport-stream`, `com.adobe.flash.video`,
  `com.microsoft.windows-media-wmv`, `com.microsoft.advanced-systems-format`,
  `public.3gpp`, `org.videolan.vob`, `public.mpeg`, `public.mpeg-2-video`,
  `org.xiph.ogg-video`, `org.smpte.mxf`, `com.real.realmedia` and
  `com.real.realmedia-vbr`. Every one resolves to a UTI macOS already declares
  except Matroska, which has no system UTI and is imported by the bundle.
  `LSHandlerRank` stays *Alternate*, so the app shows up under *Open With*
  without taking double click away from a system player.
- **What the Open panel offers.** `[.movie]` — anything the system types as a
  movie. It used to name four containers, which greyed out everything else that
  actually plays.
- **What was measured.** `scripts/make_format_matrix.sh` builds one file per
  container/codec combination and `scripts/check_formats.sh` plays each:
  **36 of 36 play** — mp4, m4v, mov, mkv, webm (VP8 and VP9), ts, m2ts, mts, flv,
  avi, divx, mjpeg, mpeg-1, mpeg-2 in .mpg/.vob/.mxf, wmv, asf, 3gp and rv20 —
  with **aac, opus, vorbis, mp3, mp2, ac3, wmav2 and PCM** audio, all at
  0.98–1.07× real time on a 480p source. Also covered: 10-bit H.264 and 10-bit
  HEVC (Hi10P anime decodes and converts down to BGRA), a 641×481 source (odd
  dimensions → 1282×962 output), 5.1 audio (downmixed to stereo), and an
  11-minute file (seek to 300 s, three 10-second skips, seek to the last 5 s).

Two things needed fixing rather than declaring, both cases where the file opened
but showed the wrong thing:

- **Cover art is a video stream.** `-map 0:v:0` on a file whose poster frame comes
  first decoded *one still image* and then stalled (measured: 1 frame in 5 s).
  The probe now lists every video stream, skips the ones flagged `attached_pic`,
  and among the rest takes the longest — so a poster in front of the film is
  ignored even when a muxer forgets to flag it. A file with no video at all
  (audio plus a cover) now says so instead of playing the artwork. All three
  layouts are in the matrix: film-first, poster-first-and-flagged, and
  poster-first-unflagged.
- **Rotation.** A phone clip carries a display matrix rather than rotated pixels,
  and the app used to ignore it, so a portrait recording played on its side. The
  probe reads the rotation (`side_data_list.rotation`, or the legacy `rotate`
  tag), swaps the raster geometry, and the decoder applies an explicit transpose
  with `-noautorotate` so it can never be applied twice. The filters chosen were
  verified **bit-identical to ffmpeg's own autorotation** at 90°, 180° and 270°,
  and a rotated file's output matches that reference at 41–42 dB PSNR (a wrong
  direction scores ~12 dB). Text subtitles are drawn upright on the rotated
  frame; a bitmap subpicture is composited *before* the turn, because its
  position is a window defined against the source frame — so the subtitle turns
  with the picture instead of being misplaced by the rotation. Note that this
  combination is synthetic: rotation metadata comes from cameras, bitmap
  subtitles from discs.

`.ogv` is the one declared type with no sample, because this ffmpeg has no Theora
encoder: the type is there for the Finder but is unverified.

## Known limitations

- **MetalFX Spatial over-sharpens on anything but clean HD.** It adds edge energy
  the source does not contain — measured acutance **1.54–2.18×** the ground truth
  across live-action and anime clips, with visible halos and stair-stepped
  diagonals on soft material (see the corrugated-roof crop in
  `results/general/liveaction_540p_to_1080p/`). It is also the *least* faithful
  mode by PSNR/SSIM in every run, and on animation it scores below plain bilinear.
  It is nonetheless **the default** — a perceived-sharpness and GPU-cost choice,
  not a fidelity one, and one click away from being replaced by the Anime4K CNN,
  which remains the better mode on animation. `MTLFXSpatialScaler` exposes no knob
  to dial it back.
- **MetalFX's temporal scaler is not reachable here.** `MTLFXTemporalScaler` is
  the higher-quality MetalFX and the better answer on motion, but it needs
  per-pixel motion vectors and depth, which the ffmpeg decode path does not
  produce.
- **The app wants the MetalFX output texture private, and hands it an
  IOSurface-backed one.** Apple's documentation for `MTLFXSpatialScaler.outputTexture`
  says the texture should be `storageMode.private`; this pipeline's output comes
  from a `CVPixelBufferPool` so the frames can go straight to the display layer.
  It encodes and completes without error and the frames dump correctly, but this
  is a documented deviation rather than an endorsed configuration.
- SDR, up to 1080p input, 24–30 fps targets (VFR sources are treated as CFR).
- **The app is ad-hoc signed, not notarized, and not sandboxed.** Notarization
  needs a Developer ID; sandboxing would block the Homebrew ffmpeg it spawns by
  absolute path. Locally that is invisible; a copy that arrived as a download
  needs right-click ▸ Open once, and ffmpeg must stay at `/opt/homebrew/bin`.
- **Embedded player: rendered frames take substantial memory.** The current
  store targets five seconds and caps its frames at 1.5 GB. A 1080p source
  scaled to 4K held 45 frames and used about 1.69 GB total in a short check.
  The stream path's compressed ring uses a separate 128 MB cap.
- **4× is SD-only**, for a new reason: the encoder limit is gone, but 4× chains two
  2× passes so 1080p→8K measured **0.61× real time** with an **883 MB** buffer.
  It is clamped to 2× above 4096×2304 rather than allowed and left to stutter.
- **A source can open on black.** The reference episode fades in from black, so
  the first ~3 s are genuinely black even though the player is working; the
  "Preparing…" indicator only covers the time before the first frame exists.
- **The app window itself was not screenshotted.** Screen Recording permission is
  not available to the build environment. The evidence is (a) the exact pixel
  buffers handed to the display layer, dumped as PNGs from the real GUI process,
  (b) `AVQueuedSampleBufferRenderingStatus == .rendering` with `window=true`, and
  (c) real-time frame delivery — plus a human glance at the window.
- Subtitle **burn-in** only; styled ASS/SSA is not supported (this Homebrew ffmpeg
  has no libass). Burn-in is at source resolution, so subtitles are upscaled with
  the picture and survive the 4× double pass.
- Audio is decoded at 48 kHz stereo via ffmpeg and resampled there; multi-channel
  layouts are downmixed to stereo. There is no audio *restart*: pause/resume and
  seeks are fine, but if the output device disappears the renderer fails and audio
  stops for the rest of the session (the status message says so) while video
  keeps playing.
- The stream path's limitations still apply to the CLI: it owns the VLC it
  launches, the playback position it reports is an estimate, and 4× is SD-only.
- The decoder pipe carries **BGRA** (≈33 MB/frame at 4K); a Metal RGB→YUV pass
  would cut pipe traffic and CPU for the stream path, and a compressed or planar
  display format would shrink the player's buffer at the cost of a conversion.

## What is and is not in the repository

Playable media stays out of the repository. The `.gitignore` keeps it out on purpose: test
clips are either **generated** (`scripts/gen_synth_anime.py`,
`scripts/make_format_matrix.sh`) or **fetched** from documented sources
(`test-media/liveaction/README.md`, `test-media/yt/README.md`, which also state
each clip's license), `third_party/` holds upstream archives and weights that
are re-downloadable, and `results/` keeps the measurements — the verification
`.txt` files, benchmark `.json`/`.csv`, harness scripts — but not the seven
hundred-odd PNGs they were derived from. TV episodes and other third-party
footage used during development are copyrighted material and are never part of
the tree.

Committing a fresh clone leaves you with the full source, the docs, and the
measurement record; a few commands regenerate or fetch everything else. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the build and the verification workflow.

## License

[MIT](../LICENSE). The models embedded in the binary carry their own licenses, restated in full in
[THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md): the Anime4K CNN weights
(MIT, © bloc97) and the two Real-ESRGAN checkpoints (BSD 3-Clause, © Xintao
Wang). ffmpeg and VLC are external programs this project spawns, not
libraries it links or ships — install your own.

## Acknowledgements

- **[Anime4K](https://github.com/bloc97/Anime4K)** by bloc97 — the CNN this
  project ports to Metal, and the reason the player exists.
- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** by Xintao Wang and
  contributors — the `SRVGGNetCompact` checkpoints behind both generative
  modes.
- **Apple's MetalFX** — the spatial scaler that makes the default mode
  nearly free.
- **ffmpeg** — every container and codec this player reads, it reads because
  of ffmpeg.
- The measured-evaluation upstreams under `scripts/` (RT4KSR, SeedVR2,
  Video2X, AnimeJaNai) — rejected for shipping, useful for knowing why.
