# Downloaded test clip (YouTube)

`mhw_ascendance_480p.mp4` — modern game footage, which none of the other test
material covers: the existing clips are anime, CG film and one live-action
handheld video. This one has real-time rendering detail, dense foliage and rock
texture, particle effects, and HUD/title text, all at a genuine 480p with a
known-good provenance.

## What it is

| | |
|---|---|
| source | <https://www.youtube.com/watch?v=Y8FOt75mbng> |
| title | *Monster Hunter Wilds: Ascendance — Long Sword \| Boosted Action Breakdown* |
| uploader | Monster Hunter (the publisher's official channel) |
| downloaded | `yt-dlp 2026.08.19 -f 135+140 --merge-output-format mp4` |
| result | 854×480, 29.97 fps, H.264 + AAC 44.1 kHz stereo, 65.1 s, 1949 frames, 9.5 MB |

Format `135` is the 480p video-only rendition and `140` the audio; they are muxed
into one file so the app can play it with sound. The 480p renditions of this
video are 30 fps even though the source is 60 fps.

**This is third-party, copyrighted promotional material.** It is here for local
testing only and should not be redistributed with the project. It is deliberately
not referenced by any automated test — every measurement in the repository that
uses it says so explicitly.

## What was checked with it

- **Playback and live upscaling hold real time** at both scales, which is the
  point of a 480p source: 2× → 1708×960 at 1.000× (MetalFX) and 1.013×
  (Anime4K CNN); 4× → 3416×1920 at 1.001× and 0.998×. 4× is allowed here because
  3416×1920 stays inside the 4096×2304 clamp.
- **The offline export handles a width that is not a multiple of 32**: 854 becomes
  864 going into the model and the output is cropped back to exactly 3416×1920,
  with the audio copied through. It runs at ~1.0 s/frame, so the whole 65-second
  clip is about 32 minutes.
- **The export is aligned.** Export frame *N* is source frame *N* (frame 45
  matches at 30.8 dB against 17.0 dB for its neighbours) and the best spatial
  alignment is (0, 0) for every mode — checked because an earlier crop looked
  shifted, which turned out to be the sharper image reading differently rather
  than a geometry fault.

Grids and the numbers are in `results/general/yt_mhw_480p/`. On the title-card
crop RealBasicVSR is the clear winner: "RATING PENDING to TEEN" is fully legible
where bilinear is mush, while the live MetalFX pass is close behind and costs
1.0 ms a frame instead of 1.0 s.
