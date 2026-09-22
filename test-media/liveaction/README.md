# Live-action test material

The rest of `test-media/` is animation or CG. These clips are live action, which
is the content the general-video upscaler path is for, and they exist because
the two things that matter about a general upscaler — how much real detail it
recovers, and how it behaves in motion — cannot be judged on cel-shaded sources.

## Provenance

| file | what it is |
|---|---|
| `table_tennis_1080p.webm` | the downloaded original: 1920x1080 VP8, ~30 fps, 23 s. Wikimedia Commons, *Table tennis* by Evworo, **CC BY-SA 3.0** — attribution required, share-alike. <https://commons.wikimedia.org/wiki/File:Table_tennis.webm> |
| `tt_1080p_ref.mp4` | ground truth: 1920x1080, 30 fps CFR, 300 frames, x264 CRF 12 |
| `tt_540p_c30.mp4` | test source: `tt_1080p_ref` downscaled (lanczos) to 960x540 and re-encoded CRF 30 — the 2x case |
| `tt_270p_c28.mp4` | test source: `tt_1080p_ref` downscaled to 480x270, CRF 28 — the 4x case |

The degraded sources are derived from the reference, so frame *N* of a source and
frame *N* of the reference are the same instant and the output of a 2x (or 4x)
upscale lands exactly on the reference's grid. That is what makes the fidelity
numbers in `results/general/` meaningful.

Rebuild:

```bash
cd test-media/liveaction
curl -sL -o table_tennis_1080p.webm \
  "https://upload.wikimedia.org/wikipedia/commons/2/2e/Table_tennis.webm"
ffmpeg -y -i table_tennis_1080p.webm -an -vf "fps=30" -frames:v 300 \
  -c:v libx264 -crf 12 -preset slow -pix_fmt yuv420p tt_1080p_ref.mp4
ffmpeg -y -i tt_1080p_ref.mp4 -an -vf "scale=960:540:flags=lanczos" \
  -c:v libx264 -crf 30 -preset slow -pix_fmt yuv420p tt_540p_c30.mp4
ffmpeg -y -i tt_1080p_ref.mp4 -an -vf "scale=480:270:flags=lanczos" \
  -c:v libx264 -crf 28 -preset slow -pix_fmt yuv420p tt_270p_c28.mp4
```

## The sharp-content test is not committed

`results/general/liveaction_sharp_1080p_to_2160p/` was measured on a 4K clip,
which is 112 MB and so is fetched rather than stored:

```bash
curl -sL -o /tmp/metro_4k.webm \
  "https://upload.wikimedia.org/wikipedia/commons/1/13/Tren_del_Metro_de_Quito_llegando_a_la_Estaci%C3%B3n_Carolina_%28junio_2025%29.webm"
# 3840x2160 VP9, 30 fps. CC BY-SA; see the Commons file page for the author.
ffmpeg -y -i /tmp/metro_4k.webm -an -vf "fps=30" -frames:v 210 \
  -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p /tmp/metro_2160p_ref.mp4
ffmpeg -y -i /tmp/metro_2160p_ref.mp4 -an -vf "scale=1920:1080:flags=lanczos" \
  -c:v libx264 -crf 30 -preset slow -pix_fmt yuv420p /tmp/metro_1080p_c30.mp4
scripts/eval_general_upscaler.py \
  --playercheck engine/MacVideoUpscaler/.build/release/playercheck \
  --source /tmp/metro_1080p_c30.mp4 --reference /tmp/metro_2160p_ref.mp4 \
  --scale 2 --stills 45,105,165 --seq 60:8 --out /tmp/metro-eval
```

The original matters here: a soft HD source downscaled to 540p is very nearly
invertible by bilinear filtering, so it flatters methods that do not sharpen.
The 4K clip is genuinely sharp, which is the fair test of whether an upscaler
recovers detail or invents it.
