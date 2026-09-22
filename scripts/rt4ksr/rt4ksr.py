"""RT4KSR (Zamfir, Conde, Timofte — CVPRW 2023) as a plain, self-contained module.

Upstream: https://github.com/eduardzamfir/RT4KSR (Apache-2.0).
Paper: "Towards Real-Time 4K Image Super-Resolution", NTIRE 2023 Real-Time 4K SR.

Why this file exists rather than a git dependency: upstream's `test.py` cannot run
here at all. It hardcodes `.cuda()` inside `reparameterize()`, wraps everything in
`DataParallel`, and calls `reparameterize()` with no arguments — so the released
inference path is CPU/MPS-hostile and, as written, broken. The class bodies below
are upstream's, with three deliberate changes:

  1. `torchvision.transforms.GaussianBlur` is dropped. In the released config
     (`forget=False`) the high-frequency branch that consumes it is computed and
     then thrown away, so the module is dead weight — see `drop_dead_branch`.
  2. `reparameterize()` is reimplemented device-agnostically (upstream's version
     allocates `torch.ones(...).cuda()`).
  3. The training-time blocks are kept only so the released checkpoints load with
     `strict=True`; inference uses the folded single-conv equivalents.

Two scales ship: x2 (1080p -> 4K) and x3 (720p -> 4K). There is no x4 model, so a
4x upscale here is two chained x2 passes — the same convention the app already
uses for Anime4K, and it is reported as such rather than as a single network.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_DIR = os.path.join(os.path.expanduser("~/.cache"), "macvideoupscaler")
DEFAULT_CFG = dict(feature_channels=24, num_blocks=4, act_type="gelu")


def activation(act_type):
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "lrelu":
        return nn.LeakyReLU(0.05, inplace=True)
    if act_type == "gelu":
        return nn.GELU()
    raise NotImplementedError("activation layer [%s] is not found" % act_type)


class LayerNorm2d(nn.Module):
    """Channel-wise normalization at every pixel (no spatial reduction)."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class ResBlock(nn.Module):
    """Reparameterizable residual block: 1x1 -> 3x3 -> 1x1, plus two identities."""

    def __init__(self, n_feats, ratio=2):
        super().__init__()
        mid = int(ratio * n_feats)
        self.expand_conv = nn.Conv2d(n_feats, mid, 1, 1, 0)
        self.fea_conv = nn.Conv2d(mid, mid, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(mid, n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        b0 = self.expand_conv.bias
        out = F.pad(out, (1, 1, 1, 1), "constant", 0)
        out[:, :, 0:1, :] = b0.view(1, -1, 1, 1)
        out[:, :, -1:, :] = b0.view(1, -1, 1, 1)
        out[:, :, :, 0:1] = b0.view(1, -1, 1, 1)
        out[:, :, :, -1:] = b0.view(1, -1, 1, 1)
        out = self.fea_conv(out) + out_identity
        return self.reduce_conv(out) + x


class RepResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        return self.rep_conv(x)


class SimplifiedNAFBlock(nn.Module):
    def __init__(self, in_c, act, layernorm=True, residual=False):
        super().__init__()
        self.residual = residual
        self.conv1 = ResBlock(in_c, 2)
        self.act = act
        self.norm = LayerNorm2d(in_c) if layernorm else None

    def forward(self, x):
        res = x
        if self.norm is not None:
            x = self.norm(x)
        x = self.conv1(x)
        if self.act is not None:
            x = self.act(x)
        return x + res if self.residual else x


class SimplifiedRepNAFBlock(nn.Module):
    def __init__(self, in_c, act, layernorm=True, residual=False):
        super().__init__()
        self.residual = residual
        self.conv1 = RepResBlock(in_c)
        self.act = act
        self.norm = LayerNorm2d(in_c) if layernorm else None

    def forward(self, x):
        res = x
        if self.norm is not None:
            x = self.norm(x)
        x = self.conv1(x)
        if self.act is not None:
            x = self.act(x)
        return x + res if self.residual else x


class RT4KSR(nn.Module):
    """Structure of the released checkpoints.

    `is_train=True` matches the saved state dict key-for-key (that is the only
    reason the gaussian/high-frequency branch is here at all). `is_train=False`
    builds the folded single-conv blocks, which is what actually runs.
    """

    def __init__(self, scale, is_train, drop_dead_branch=False, **kw):
        super().__init__()
        cfg = dict(DEFAULT_CFG)
        cfg.update(kw)
        num_feats = cfg["feature_channels"]
        num_blocks = cfg["num_blocks"]
        act = activation(cfg["act_type"])
        self.scale = scale
        self.is_train = is_train
        self.drop_dead_branch = drop_dead_branch

        # `forget` is hardcoded False upstream, which makes the whole
        # high-frequency branch below dead: its output is computed into
        # `hf_feats` and then never read. Kept for the training-shaped load,
        # skipped at inference.
        self.forget = False
        self.gamma = nn.Parameter(torch.zeros(1))

        self.down = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)
        self.head = nn.Sequential(nn.Conv2d(3 * 4, num_feats, 3, padding=1))

        block = ResBlock if is_train else RepResBlock
        hfb = [block(num_feats, 2) if is_train else block(num_feats)]
        hfb.append(act)
        self.hfb = nn.Sequential(*hfb)

        naf = SimplifiedNAFBlock if is_train else SimplifiedRepNAFBlock
        self.body = nn.Sequential(*[naf(num_feats, act) for _ in range(num_blocks)])

        tail = [LayerNorm2d(num_feats),
                ResBlock(num_feats, 2) if is_train else RepResBlock(num_feats)]
        self.tail = nn.Sequential(*tail)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, 3 * ((2 * scale) ** 2), 3, padding=1),
            nn.PixelShuffle(2 * scale),
        )

    def forward(self, x):
        deep = self.body(self.head(self.down(x)))
        if self.drop_dead_branch:
            return self.upsample(self.tail(deep))
        # `hfb` consumes the high-frequency branch. In the released config
        # `forget` is False, so this result is computed and then discarded —
        # reproducible, but it does not reach the output.
        hf_feats = self.hfb(self.head(self.down(x - _gaussian_blur(x))))
        if self.forget:
            deep = self.gamma * deep + hf_feats
        return self.upsample(self.tail(deep))


def _gaussian_blur(x):
    """torchvision's GaussianBlur(kernel_size=5, sigma=1), done without torchvision.

    Only the dead high-frequency branch calls this; it is here so the training
    architecture can be checked against upstream rather than merely assumed.
    """
    r = torch.arange(5, dtype=x.dtype, device=x.device) - 2.0
    phi = torch.exp(-r.pow(2) / 2.0)
    phi = phi / phi.sum()
    w = (phi[:, None] * phi[None, :]).view(1, 1, 5, 5).expand(x.shape[1], 1, 5, 5)
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"),
                    w.contiguous(), groups=x.shape[1])


def merge_rep(train_sd, scale, **kw):
    """Fold expand(1x1) -> fea(3x3) -> reduce(1x1) + identities into one 3x3 conv.

    Same arithmetic as upstream's `reparameterize`, minus the `.cuda()`.
    """
    rep = RT4KSR(scale=scale, is_train=False, **kw)
    rsd = rep.state_dict()
    for k in list(rsd.keys()):
        if "rep_conv.weight" not in k:
            if k in train_sd:
                rsd[k] = train_sd[k]
            continue
        k0 = train_sd[k.replace("rep", "expand")]
        k1 = train_sd[k.replace("rep", "fea")].clone()
        k2 = train_sd[k.replace("rep", "reduce")]
        bs = k.replace("weight", "bias")
        b0, b1, b2 = (train_sd[bs.replace("rep", n)] for n in ("expand", "fea", "reduce"))

        mid, n_feats = k0.shape[:2]
        for i in range(mid):                       # drop the middle identity
            k1[i, i, 1, 1] += 1.0
        merged_k = F.conv2d(k1, k0.permute(1, 0, 2, 3))
        ones = torch.ones(1, mid, 3, 3, dtype=b0.dtype, device=b0.device)
        merged_b = F.conv2d(b0.view(1, -1, 1, 1) * ones, k1, b1)
        merged_k = F.conv2d(merged_k.permute(1, 0, 2, 3), k2).permute(1, 0, 2, 3)
        merged_b = F.conv2d(merged_b, k2, b2).view(-1)
        for i in range(n_feats):                   # drop the global identity
            merged_k[i, i, 1, 1] += 1.0
        rsd[k] = merged_k.float()
        rsd[bs] = merged_b.float()

    rep.load_state_dict(rsd, strict=True)
    return rep


def _strip(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load(scale, path=None, device="cpu", **kw):
    """Loads and folds a released checkpoint. `scale` is 2 or 3."""
    path = path or os.path.join(CACHE_DIR, "RT4KSR_x%d.pth" % scale)
    blob = torch.load(path, map_location="cpu")
    sd = _strip(blob["state_dict"] if "state_dict" in blob else blob)

    train = RT4KSR(scale=scale, is_train=True, **kw)
    missing, unexpected = train.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit("checkpoint does not match the model: missing=%s unexpected=%s"
                         % (missing, unexpected))

    model = merge_rep(sd, scale, **kw)
    model.eval()
    return model.to(device)


class Upscaler:
    """Frame-at-a-time inference, in [0, 1] float32 NCHW."""

    def __init__(self, scale=2, device="mps", weights=None, models=None, **kw):
        if scale not in (2, 3, 4):
            raise ValueError("scale must be 2, 3 or 4")
        self.scale = scale
        self.device = torch.device(device)
        if scale == 4:
            self.models = [load(2, None if models is None else models.get(2), self.device, **kw),
                           load(2, None if models is None else models.get(2), self.device, **kw)]
        else:
            self.models = [load(scale, None if models is None else models.get(scale),
                                self.device, **kw)]

    @torch.no_grad()
    def upscale(self, x):
        for m in self.models:
            x = m(x).clamp_(0.0, 1.0)
        return x

    @torch.no_grad()
    def upscale_np(self, frame):
        """uint8 HWC RGB -> uint8 HWC RGB, same geometry x `scale`."""
        # from_numpy needs a writable array; the decode buffer and the caller's
        # frame both arrive read-only.
        x = torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True)).permute(2, 0, 1)
        x = x.float().div_(255.0).unsqueeze(0).to(self.device)
        y = self.upscale(x)[0].mul_(255.0).round_().clamp_(0, 255)
        return y.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
