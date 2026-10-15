#!/usr/bin/env python3
"""
Independent NumPy reimplementation of the Anime4K v3.2 Upscale-CNN-x2-(M)
network, used to cross-validate the Metal port.

It parses the SAME GLSL weights with the SAME parser used to generate the Swift
constants, but implements the pass graph independently in NumPy (row 0 = top,
+y down, clamp-to-edge). Agreement with the Metal output therefore validates:
  * weight extraction & ordering
  * GLSL mat4 column-major semantics
  * neighbour offset order / clamp semantics
  * depth-to-space channel->pixel mapping (no vertical flip)

Usage:
  numpy_reference.py <input.png> <out_prefix> [--denoise]
Writes <out_prefix>.png plus <out_prefix>.npy (float32 HxWx3, pre-quantisation).
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_weights import parse_shader  # noqa: E402

OFF = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GLSL_DIR = os.path.join(ROOT, "third_party", "Anime4K", "Anime4K_v3.2")


def shift(a, ox, oy):
    """Clamp-to-edge neighbour fetch of HxWxC array."""
    H, W = a.shape[:2]
    ys = np.clip(np.arange(H, dtype=np.int64) + oy, 0, H - 1)
    xs = np.clip(np.arange(W, dtype=np.int64) + ox, 0, W - 1)
    return a[np.ix_(ys, xs)]


def apply_mats(acc, v, mats):
    # GLSL column-major: out[c] = sum_j m[4j+c] * v[j]  =>  mat[j, c] = m[4j+c]
    for m in mats:
        acc += np.einsum("hwj,jc->hwc", v, m, optimize=True)
    return acc


def conv3x3_1to4(x, flat):
    H, W = x.shape[:2]
    mats = flat[: 9 * 16].reshape(9, 4, 4)
    acc = np.empty((H, W, 4), np.float32)
    acc[...] = flat[144:148]
    for k in range(9):
        acc += np.einsum("hwj,jc->hwc", shift(x, *OFF[k]), mats[k], optimize=True)
    return acc


def conv3x3_8to4(x, flat):
    mats = flat[: 18 * 16].reshape(18, 4, 4)
    acc = np.empty(x.shape, np.float32)
    acc[...] = flat[288:292]
    for k in range(9):
        v = shift(x, *OFF[k])
        acc += np.einsum("hwj,jc->hwc", np.maximum(v, 0), mats[k], optimize=True)
        acc += np.einsum("hwj,jc->hwc", np.maximum(-v, 0), mats[9 + k], optimize=True)
    return acc


def conv1x1_56to4(feats, flat):
    mats = flat[: 14 * 16].reshape(14, 4, 4)
    acc = np.empty(feats[0].shape, np.float32)
    acc[...] = flat[224:228]
    for idx, f in enumerate(feats):
        acc += np.einsum("hwj,jc->hwc", np.maximum(f, 0), mats[2 * idx], optimize=True)
        acc += np.einsum("hwj,jc->hwc", np.maximum(-f, 0), mats[2 * idx + 1], optimize=True)
    return acc


def bilinear2x(a):
    """2x bilinear, clamp-to-edge: source coord = (o + 0.5)/2 - 0.5."""
    H, W = a.shape[:2]
    oy = np.arange(2 * H)
    ox = np.arange(2 * W)
    cy = (oy + 0.5) * 0.5 - 0.5
    cx = (ox + 0.5) * 0.5 - 0.5
    fy = np.floor(cy)
    fx = np.floor(cx)
    ty = (cy - fy)[:, None, None]
    tx = (cx - fx)[None, :, None]
    y0 = np.clip(fy, 0, H - 1).astype(np.int64)
    y1 = np.clip(fy + 1, 0, H - 1).astype(np.int64)
    x0 = np.clip(fx, 0, W - 1).astype(np.int64)
    x1 = np.clip(fx + 1, 0, W - 1).astype(np.int64)
    r0 = a[np.ix_(y0, x0)] * (1 - tx) + a[np.ix_(y0, x1)] * tx
    r1 = a[np.ix_(y1, x0)] * (1 - tx) + a[np.ix_(y1, x1)] * tx
    return (r0 * (1 - ty) + r1 * ty).astype(np.float32)


def run_cnn(x_rgb, weights, denoise=False):
    p = weights
    # GLSL MAIN is a vec4 (RGB, alpha=1). The 1st conv's 4th weight column is 0,
    # so alpha is irrelevant, but the operand needs 4 channels.
    ones = np.ones(x_rgb.shape[:2] + (1,), np.float32)
    x4 = np.concatenate([x_rgb, ones], axis=2)
    f = []
    f.append(conv3x3_1to4(x4, p[0]))
    for i in range(1, 7):
        f.append(conv3x3_8to4(f[i - 1], p[i]))
    last = conv1x1_56to4(f, p[7])
    H, W = x_rgb.shape[:2]
    out = np.zeros((2 * H, 2 * W, 3), np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            out[dy::2, dx::2] = last[:, :, dy * 2 + dx][:, :, None]
    out += bilinear2x(x_rgb)
    return out


def load_weights(denoise):
    name = "Anime4K_Upscale_Denoise_CNN_x2_M.glsl" if denoise else "Anime4K_Upscale_CNN_x2_M.glsl"
    passes = parse_shader(Path(GLSL_DIR) / name)
    flat = []
    for p in passes:
        arr = []
        for m in p["mats"]:
            arr.extend(m)
        arr.extend(p["bias"])
        flat.append(np.array(arr, np.float32))
    return flat


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    denoise = "--denoise" in sys.argv
    if len(args) < 2:
        raise SystemExit(__doc__)
    inp, out_prefix = args[0], args[1]

    img = Image.open(inp).convert("RGB")
    x = (np.asarray(img, np.float32) / 255.0).astype(np.float32)
    weights = load_weights(denoise)
    y = run_cnn(x, weights, denoise)

    nan = int(np.isnan(y).sum())
    yc = np.clip(y, 0.0, 1.0)
    out8 = np.rint(yc * 255.0).astype(np.uint8)
    Image.fromarray(out8).save(out_prefix + ".png")
    np.save(out_prefix + ".npy", y)
    print(f"ref {out_prefix}.png size={img.size[0]*2}x{img.size[1]*2} "
          f"nan={nan} min={float(np.nanmin(y)):.5f} max={float(np.nanmax(y)):.5f} "
          f"range=[{float(y.min()):.4f},{float(y.max()):.4f}]")


if __name__ == "__main__":
    main()
