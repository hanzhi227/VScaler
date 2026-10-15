# Contributing

VScaler is a macOS video player that upscales frames during playback. Start with
the [README](README.md) for setup and modes. The [technical notes](docs/TECHNICAL.md)
hold the playback design, measurements, and verification commands.

## Environment

- Apple Silicon Mac, macOS 14+
- Xcode command line tools (Swift 5.10+)
- Homebrew `ffmpeg`/`ffprobe` at `/opt/homebrew/bin` (the decoders are spawned
  by absolute path, not linked)
- Optional: VLC, only for the external-player stream path

## Building

Three SwiftPM packages under `engine/`, no external Swift dependencies:

```bash
swift build --package-path engine/MacVideoUpscaler      # core + all three executables
swift build --package-path engine/VideoUpscaleMetal     # standalone, has its own bench CLI
swift build --package-path engine/RealESRGANMetal       # standalone, has its own bench CLI
```

To build and install the app in `/Applications`:

```bash
scripts/make_app.sh            # --out DIR assembles without installing; --run launches
```

The executables, all built from `engine/MacVideoUpscaler`:

| binary | what it is |
|---|---|
| `MacVideoUpscalerApp` | the GUI player |
| `macvideoupscale` | headless CLI that serves an upscaled MPEG-TS stream to an external player |
| `playercheck` | headless verification harness for the embedded player — the tool most changes are validated with |

## Verifying changes

This project's convention is that **behavior claims come with a measurement**.
`playercheck` drives the player pipeline offscreen and can dump the frames the
renderer presents:

```bash
engine/MacVideoUpscaler/.build/release/playercheck \
    --file <clip> --mode cnn --seconds 10 \
    --script "3:pause,6:seek:15,9:resume"      # controls as events
engine/MacVideoUpscaler/.build/release/playercheck \
    --file <clip> --mode metalfx --dump-at 90 --dump-dir /tmp/frames
```

The GUI also accepts `MACVIDEOUPSCALER_*` environment variables for opening a
file, selecting a mode and scale, synthetic clicks, and frame dumps. See
"Options" in the technical notes. The `scripts/check_*.py` tools verify A/V sync,
playback rate, subtitles, formats and scene cuts against a source file.

Run `python3 scripts/check_pause_resume.py` after a release build to check long
pauses, audio/video resumption, seeking while paused, and natural end of playback.
It generates a short synthetic clip and runs the embedded player with muted audio.

Run `python3 scripts/check_chrome.py` to check idle controls, mouse movement,
pause/resume, and fullscreen entry/exit through the GUI's diagnostic hooks.
It opens a silent synthetic clip for 35 seconds. Cursor rendering still needs
a manual check with the pointer over the video.

Run `python3 scripts/check_external_subtitles.py` to verify external SRT, VTT,
ASS, and SSA files against rendered pixels, including track switching,
invalid files, paused loading, seeking, and clearing subtitles on a new video.

Run `python3 scripts/check_vobsub.py` to verify external `.idx`/`.sub` pairs,
including rendered pixels when seeking into a cue, blank intervals, Off,
missing companions, and a new video. It requires the local bitmap subtitle
fixture at `test-media/segments/subs_bitmap.mkv`, or its path as an argument.

If your change alters the picture or playback, include the command you ran and
its measurements. The technical notes show the format used for earlier runs.

## Test material

**Nothing that plays is committed.** TV episodes, downloaded streams and
screenshots of them stay out of the repository — generate or fetch clips with
the documented commands instead:

- `scripts/gen_synth_anime.py` — synthesized animation-style clips, no
  third-party content
- `scripts/make_format_matrix.sh` — one file per container/codec
- `test-media/liveaction/README.md`, `test-media/yt/README.md` — fetched
  clips, their licenses and how to rebuild them
- `results/` keeps measurements (`.txt/.json/.csv/…`), not pixels; the
  `.gitignore` enforces all of this

## Generated files

The weight blobs are generated and carry their provenance in the first lines —
do not edit them by hand:

- `engine/VideoUpscaleMetal/Sources/VideoUpscaleMetal/CNNWeights.swift` ← `Scripts/extract_weights.py`
- `engine/RealESRGANMetal/Sources/RealESRGANMetal/*Weights.swift` ← `scripts/realesrgan/export_weights.py`

If you regenerate or add weights from an upstream project, keep the upstream
attribution in the header and add the project to
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Style

Match the surrounding code: comments state constraints and the reasons behind
non-obvious decisions, not a narration of the next line; validation claims in
docs are written the same way — what was measured, how, and where the number
came from.
