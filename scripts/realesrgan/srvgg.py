"""Real-ESRGAN's `realesr-animevideov3` as a small, self-contained module.

Upstream: https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause).
Weights: `realesr-animevideov3.pth` from the official v0.2.5.0 release.

The architecture is `SRVGGNetCompact`: a pure convolutional stack, no attention,
no recurrence, no optical flow. `num_feat=64, num_conv=16, upscale=4`, which the
released checkpoint confirms exactly — 18 convolutions, 17 PReLU layers, 621,424
parameters (620,336 weights + 17 x 64 PReLU), and **0.62 M MACs per input pixel**
against `general-x4v3`'s 1.21 M, i.e. half the arithmetic of the general model
because it carries half the body convolutions.

It is the "AnimeVideo-v3" variant: trained on anime video rather than stills, so
it is the Real-ESRGAN model whose degradation model is closest to this app's
material. x4 only — there is no x2 checkpoint — so it fits a 270p -> 1080p source
exactly and does not fit the app's 540p -> 1080p case at all.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_DIR = os.path.join(os.path.expanduser("~/.cache"), "macvideoupscaler")
WEIGHTS_NAME = "realesr-animevideov3.pth"
NUM_FEAT = 64
NUM_CONV = 16
SCALE = 4

# Both released video checkpoints, as (body convolutions, file name). Kept in one
# place because they have to agree: a mismatch is silent at load time — the wrong
# file loads into a model of the right shape until the *final* convolution, which
# is the first layer whose channel counts differ between the two depths. That is
# exactly how it presented (`body.34.weight` 48 vs 64), pointing at the last layer
# when the fault was the file name.
VARIANTS = {
    "animevideov3": (16, "realesr-animevideov3.pth"),
    "general-x4v3": (32, "realesr-general-x4v3.pth"),
}


def variant_path(name):
    return os.path.join(CACHE_DIR, VARIANTS[name][1])


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=NUM_FEAT,
                 num_conv=NUM_CONV, upscale=SCALE):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
                nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                     nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.body(x)
        out = self.upsampler(out)
        # The released model adds a nearest-neighbour magnification of the input:
        # the network only has to produce the correction, not the whole picture.
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def _strip(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load(path=None, device="cpu", num_conv=NUM_CONV):
    """Loads an official checkpoint. Returns the model in eval mode."""
    path = path or os.path.join(CACHE_DIR, WEIGHTS_NAME)
    blob = torch.load(path, map_location="cpu")
    sd = _strip(blob.get("params", blob.get("params_ema", blob)))
    model = SRVGGNetCompact(num_conv=num_conv)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit("checkpoint does not match SRVGGNetCompact: "
                         "missing=%s unexpected=%s" % (missing[:4], unexpected[:4]))
    model.eval()
    return model.to(device)


class Upscaler:
    """Frame-at-a-time x4 inference, [0, 1] float32 NCHW in and out.

    `tile` splits the frame into overlapping tiles so a 1080p source can be
    processed in bounded memory; 0 means one pass over the whole frame.
    """

    def __init__(self, device="mps", weights=None, tile=0, overlap=16, num_conv=NUM_CONV):
        self.device = torch.device(device)
        self.model = load(weights, self.device, num_conv=num_conv)
        self.scale = SCALE
        self.tile = tile
        self.overlap = overlap

    @torch.no_grad()
    def upscale(self, x):
        if not self.tile:
            return self.model(x).clamp_(0.0, 1.0)
        n, c, h, w = x.shape
        out = torch.empty((n, c, h * self.scale, w * self.scale),
                          dtype=x.dtype, device=x.device)
        step = self.tile - 2 * self.overlap
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                x1, y1 = min(x0 + self.tile, w), min(y0 + self.tile, h)
                # Overlap so the 3x3 receptive field of a tile's edge sees the
                # same neighbours it would in a whole-frame pass; the crop below
                # keeps only the part that is not another tile's edge.
                sx0, sy0 = max(0, x0 - self.overlap), max(0, y0 - self.overlap)
                patch = x[:, :, sy0:y1, sx0:x1]
                y = self.model(patch).clamp_(0.0, 1.0)
                cx0, cy0 = (x0 - sx0) * self.scale, (y0 - sy0) * self.scale
                cw, ch = (x1 - x0) * self.scale, (y1 - y0) * self.scale
                out[:, :, y0 * self.scale:y1 * self.scale,
                    x0 * self.scale:x1 * self.scale] = y[:, :, cy0:cy0 + ch, cx0:cx0 + cw]
        return out

    @torch.no_grad()
    def upscale_np(self, frame):
        """uint8 HWC RGB -> uint8 HWC RGB at 4x."""
        x = torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True)).permute(2, 0, 1)
        x = x.float().div_(255.0).unsqueeze(0).to(self.device)
        y = self.upscale(x)[0].mul_(255.0).round_().clamp_(0, 255)
        return y.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
