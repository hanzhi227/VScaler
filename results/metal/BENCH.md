# Anime4K Metal (M5 Pro) benchmark

- Device: **Apple M5 Pro**
- Harness: `vupbench`, synchronous `commit()` + `waitUntilCompleted()` per frame
- Timing: wall-clock ms/frame, first iteration excluded as warmup; mean over all timed iterations, with median/min because the odd scheduler spike can inflate the mean at small sizes
- Source frames: synthetic asymmetric anime frames `assets/test_480p.png` (854x480) and `assets/test_1080p.png` (1920x1080)
- 1080p rows: `--repeat-to-1080` re-rasterises to 1080p-class height (identity here, since the source is already 1920x1080)
- Arithmetic: fp32 in all kernels; intermediates `rgba16Float`

| source | mode | input | output | iters | ms/frame | min | median | FPS | GPU util (mean/max) |
|---|---|---|---|---|---|---|---|---|---|
| 480p | `bilinear` | 854x480 | 1708x960 | 201 | **0.247** | 0.142 | 0.195 | 4042.0 | n/a |
| 480p | `cnnX2M` | 854x480 | 1708x960 | 201 | **0.951** | 0.760 | 0.830 | 1051.1 | n/a |
| 480p | `cnnDenoiseX2M` | 854x480 | 1708x960 | 201 | **0.924** | 0.756 | 0.821 | 1081.8 | n/a |
| 1080p | `bilinear` | 1920x1080 | 3840x2160 | 101 | **0.361** | 0.267 | 0.312 | 2770.1 | 39.5% / 79% |
| 1080p | `cnnX2M` | 1920x1080 | 3840x2160 | 101 | **3.812** | 3.258 | 3.648 | 262.3 | 78.8% / 98% |
| 1080p | `cnnDenoiseX2M` | 1920x1080 | 3840x2160 | 101 | **3.833** | 3.264 | 3.646 | 260.9 | 91.5% / 98% |

## Validation summary

See `validation.json`. Key results:
- Output dimensions exactly 2x; zero NaN/Inf in the fp32 output
- Metal output matches an independent NumPy port of the same GLSL weights to **<= 1 LSB** (0.00% of pixels differ by more than 1/255)
- Orientation preserved (forward row-profile correlation 0.998 vs mirrored -0.905)
- Line-art acutance vs bilinear: hatching **7.8x**, text **5.9x**
- Flat cel region reproduced exactly (200.0 -> 200.0)
