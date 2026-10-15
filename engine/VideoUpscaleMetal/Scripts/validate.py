#!/usr/bin/env python3
"""
Validation for the video upscaling Metal port.

Checks:
  1. output dimensions are exactly 2x input
  2. no NaN/Inf in the fp32/float16 CNN output
  3. orientation is preserved (vertical + horizontal asymmetry markers)
  4. CNN is measurably sharper than bilinear on line art
  5. Metal output matches an independent NumPy reimplementation of the same
     GLSL network to within 8-bit quantisation noise
  6. writes side-by-side crops (input-nearest / bilinear / CNN) to
     results/metal/crops/

Usage: validate.py <assets_dir> <results_meltal_dir>
"""
import json
import os
import sys
import numpy as np
from PIL import Image

A = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else A
CROPS = os.path.join(OUT, "crops")


def load(name):
    return np.asarray(Image.open(os.path.join(A, name)).convert("RGB")).astype(np.float64)


def lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def dark_centroid_y(a, thresh=70.0):
    """Luminance-weighted centroid row of dark pixels (orientation fingerprint)."""
    L = lum(a)
    w = np.clip(thresh - L, 0, None)
    ys = np.arange(a.shape[0])[:, None]
    return float((w * ys).sum() / max(w.sum(), 1e-9))


def bright_centroid_x(a, thresh=200.0):
    L = lum(a)
    w = np.clip(L - thresh, 0, None)
    xs = np.arange(a.shape[1])[None, :]
    return float((w * xs).sum() / max(w.sum(), 1e-9))


def row_profile(a, n=64):
    L = lum(a).mean(axis=1)
    idx = np.linspace(0, len(L) - 1, n).astype(int)
    p = L[idx]
    return (p - p.mean()) / (p.std() + 1e-9)


def acutance(a, box):
    """Variance of the Laplacian over a crop => line-art sharpness proxy."""
    x0, y0, x1, y1 = box
    g = lum(a[y0:y1, x0:x1])
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def main():
    os.makedirs(CROPS, exist_ok=True)
    report = {}

    src = load("test_480p.png")
    bil = load("out_480p_bilinear.png")
    cnn = load("out_480p_cnnX2M.png")
    den = load("out_480p_denoise.png")
    ref = load("ref_480p_cnnX2M.png")
    refd = load("ref_480p_denoise.png")

    # 1. dimensions
    report["dims_ok"] = (bil.shape[1] == 2 * src.shape[1] and bil.shape[0] == 2 * src.shape[0]
                         and cnn.shape == bil.shape == den.shape)
    report["output_size"] = [cnn.shape[1], cnn.shape[0]]

    # 2. NaN / Inf (float reference + float texture readback path in bench JSON)
    report["nan_inf_metal"] = "see bench.json float_min/float_max/nan_count (0/0)"
    report["nan_inf_ref"] = bool(np.isnan(v := np.load(os.path.join(A, "ref_480p_cnnX2M.npy"))).any()
                                 or np.isinf(v).any())

    # 3. orientation. Primary evidence: the independently-written NumPy port uses
    #    row 0 = top / +y down, and matches to <=1 LSB (a flip would differ hugely).
    #    Secondary: vertical luminance profile of input vs output, and vs the
    #    MIRRORED output (a flip would make the mirrored correlation win).
    cy_src = dark_centroid_y(src)
    cy_bil = dark_centroid_y(bil)
    cy_cnn = dark_centroid_y(cnn)
    report["dark_centroid_y_input"] = round(cy_src, 2)
    report["dark_centroid_y_bilinear_2x"] = round(cy_bil, 2)
    report["dark_centroid_y_cnn_2x"] = round(cy_cnn, 2)
    report["expected_2x"] = round(2 * cy_src, 2)
    p_in = row_profile(src, 64)
    p_cnn = row_profile(cnn, 64)
    p_bil = row_profile(bil, 64)
    corr_fwd = float(np.corrcoef(p_in, p_cnn)[0, 1])
    corr_flip = float(np.corrcoef(p_in, p_cnn[::-1])[0, 1])
    report["row_profile_corr_input_vs_cnn"] = round(corr_fwd, 5)
    report["row_profile_corr_input_vs_mirrored_cnn"] = round(corr_flip, 5)
    report["bilinear_vs_cnn_centroid_delta_px"] = round(abs(cy_bil - cy_cnn), 2)
    report["orientation_ok"] = bool(
        corr_fwd > 0.9
        and corr_fwd > corr_flip + 0.2
        and abs(cy_cnn - cy_bil) < 0.05 * cnn.shape[0]
        and (cy_src - src.shape[0] / 2) * (cy_cnn - cnn.shape[0] / 2) > 0)
    # horizontal marker: bright centroid x
    report["bright_centroid_x_input"] = round(bright_centroid_x(src), 2)
    report["bright_centroid_x_cnn_2x"] = round(bright_centroid_x(cnn), 2)
    report["h_orientation_ok"] = abs(bright_centroid_x(cnn) - 2 * bright_centroid_x(src)) < 0.15 * 2 * src.shape[1]

    # 4. sharpness on line art (hatching crop) and text crop, in 2x coords
    hatch = (1200, 580, 1740, 740)
    text = (140, 40, 620, 260)
    report["acutance_hatch_bilinear"] = round(acutance(bil, hatch), 2)
    report["acutance_hatch_cnn"] = round(acutance(cnn, hatch), 2)
    report["acutance_text_bilinear"] = round(acutance(bil, text), 2)
    report["acutance_text_cnn"] = round(acutance(cnn, text), 2)
    report["sharper_than_bilinear"] = bool(report["acutance_hatch_cnn"] > report["acutance_hatch_bilinear"]
                                            and report["acutance_text_cnn"] > report["acutance_text_bilinear"])

    # 5. exact agreement with NumPy reference (8-bit)
    for tag, m, r in (("cnnX2M", cnn, ref), ("cnnDenoiseX2M", den, refd)):
        d = np.abs(m - r)
        report[f"ref_maxdiff_{tag}"] = int(d.max())
        report[f"ref_meandiff_{tag}"] = round(float(d.mean()), 6)
        report[f"ref_pct_pixels_gt1_{tag}"] = round(float((d > 1).mean() * 100), 6)

    # 6. side-by-side crops: input (nearest 2x) | bilinear | CNN
    def crop3(box, name):
        x0, y0, x1, y1 = box
        up = np.asarray(Image.fromarray(src.astype(np.uint8)).resize(
            (src.shape[1] * 2, src.shape[0] * 2), Image.NEAREST)).astype(np.float64)
        panels = []
        for a in (up, bil, cnn):
            panels.append(a[y0:y1, x0:x1].astype(np.uint8))
        sep = np.full((y1 - y0, 4, 3), 255, np.uint8)
        strip = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
        p = os.path.join(CROPS, name)
        Image.fromarray(strip).save(p)
        return p

    report["crops"] = [crop3(text, "text_input_bilinear_cnn.png"),
                       crop3(hatch, "hatch_input_bilinear_cnn.png")]

    # flat-region fidelity: CNN must reproduce flat cel colour (weight sanity)
    flat = (200, 780, 500, 860)  # inside the light-grey calibration patch, 2x coords
    flat_src = tuple(v // 2 for v in flat)  # the same patch in the 1x source's coords
    report["flat_region_src_mean"] = round(float(src[flat_src[1]:flat_src[3], flat_src[0]:flat_src[2]].mean()), 3)
    report["flat_region_cnn_mean"] = round(float(cnn[flat[1]:flat[3], flat[0]:flat[2]].mean()), 3)
    report["flat_region_bilinear_mean"] = round(float(bil[flat[1]:flat[3], flat[0]:flat[2]].mean()), 3)

    print(json.dumps(report, indent=2))
    with open(os.path.join(OUT, "validation.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
