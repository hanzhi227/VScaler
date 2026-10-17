# VScaler

VScaler plays video on an Apple Silicon Mac while upscaling each frame on the GPU. It opens local files in its own window, keeps audio and video on a shared playback clock, and lets you switch enhancement modes during playback. The default is MetalFX Spatial. A CNN upscaler and two Real-ESRGAN checkpoints are also included.

This is a source release. Build the app on your Mac; there is no signed download in this repository.

## Requirements

- Apple Silicon Mac running macOS 14 or later
- Xcode command line tools with Swift 5.10 or later
- Homebrew `ffmpeg` and `ffprobe` at `/opt/homebrew/bin`
- VLC only if you want to use the optional command line stream player

VScaler calls the Homebrew tools as separate processes. The app bundle does not contain ffmpeg or VLC.

## Build and open a video

```bash
git clone https://github.com/hanzhi227/VScaler.git
cd VScaler
./scripts/make_app.sh --out ./dist
open ./dist/VScaler.app
```

The build script compiles a release build, assembles `dist/VScaler.app`, and signs it locally with an ad hoc signature. To install the app in `/Applications`, run `./scripts/make_app.sh` without `--out`. It replaces an existing VScaler app only when the bundle identifier matches this project.

Open a file with **⌘O**, drag it onto the app, or use **Open With** in Finder. **⇧⌘O** adds a folder recursively. New selections join the queue in filename order. Playback controls are in the menu bar and in a transport strip that appears when you move the pointer over the video. **⇧⌘S** toggles playback stats.

Use **Subtitles → Load Subtitle File…** (**⌥⌘O**) to select an SRT, VTT, ASS, or SSA
file, or either file from a VobSub `.idx`/`.sub` pair, for the current video. Keep
the matching `.idx` and `.sub` files together. The picker starts in a neighboring
`subs` folder when one exists. The loaded filename appears beside the embedded tracks
in the Subtitles menu; choose **Off** to disable it. Switching videos clears the
loaded subtitle file. ASS/SSA text is supported without its styling.

The app is not notarized. macOS may ask you to use **right click → Open** the first time you launch a copy downloaded from elsewhere.

## Enhancement modes

| Mode | Use |
|---|---|
| MetalFX Spatial | Default. Fast general scaling, though it can sharpen soft footage too much. |
| CNN | A 2× CNN upscaler. The 4× setting runs it twice. |
| CNN + denoise | The denoise variant of the same CNN. |
| Real-ESRGAN AnimeVideo-v3 | A 4× model for SD sources. Check its results on your own material; it can replace texture. |
| Real-ESRGAN General-x4 | A second 4× model included for comparison. It was slower and scored worse than AnimeVideo-v3 in this project's tests. |
| Bypass | Bilinear scaling without enhancement. |
| Off | Shows the decoded frame at its native resolution. |

Choose a mode from **Enhancement**. The comparison view shows two modes on one frame with a movable divider. The app limits 4× output to 4096×2304; larger outputs use 2×. Both Real-ESRGAN modes require an SD input and 4× output.

The numbers behind these choices, the playback design, and the verification commands are in [technical notes](docs/TECHNICAL.md). The [general video evaluation](docs/GENERAL-VIDEO-UPSCALING.md) discusses the default's sharpness tradeoff. Measurements were made on an M5 Pro and will vary with the machine and source.

## Develop and verify

There are three local Swift packages and no external Swift package dependencies. The main package builds the GUI, a stream CLI, and `playercheck`, which drives the embedded player without opening a window.

```bash
swift build -c release --package-path engine/MacVideoUpscaler
swift build --package-path engine/VideoUpscaleMetal
swift build --package-path engine/RealESRGANMetal

engine/MacVideoUpscaler/.build/release/playercheck \
  --file /path/to/video.mkv --mode cnn --scale 2 --seconds 30
```

The optional stream CLI serves an upscaled MPEG-TS stream to a separate player:

```bash
engine/MacVideoUpscaler/.build/release/macvideoupscale \
  --file /path/to/video.mkv --mode cnn --scale 2 --port 8090 --open-vlc
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the verification workflow. The repo includes measurement records in `results/` and scripts to generate test material. Playable clips, downloaded upstream archives, rendered frames, and build output are excluded from Git.

## Current limits

- Playback targets SDR sources up to 1080p at 24 to 30 fps. Variable frame rate input is treated as constant frame rate.
- Subtitles are drawn over the picture, so the upscaler does not redraw the lettering. Styled ASS/SSA subtitles are not supported with the required Homebrew ffmpeg build.
- Audio is decoded to 48 kHz stereo. If the output device disappears during playback, audio stops for that session while video continues.
- Rendered frames can use up to 1.5 GB before the display queue and model allocations. A 1080p source scaled to 4K used about 1.69 GB total in a short playback check.
- MetalFX Spatial can produce halos and over-sharpen soft material. Switch to another mode if the result looks wrong.
- The app expects ffmpeg and ffprobe at `/opt/homebrew/bin` and is not sandboxed or notarized.

The [technical notes](docs/TECHNICAL.md#known-limitations) cover the rest, including 4× throughput and the separate stream path.

## License and credits

VScaler is licensed under [MIT](LICENSE). The embedded Anime4K weights and Real-ESRGAN checkpoints have their own licenses and attribution in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Those notices are copied into the app bundle by the build script. ffmpeg and VLC are installed separately by the user.
