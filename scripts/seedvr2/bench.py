#!/usr/bin/env python3
"""What one SeedVR2 forward pass costs on *this* machine.

`seedvr2.py` says how much arithmetic a SeedVR2 pass contains; this says how
long that arithmetic takes here. The two are deliberately separate, because this
project has already been burned by projecting a model's speed from a
microbenchmark: the Real-ESRGAN figure in `results/general/realesrgan_feasibility.txt`
projected 277.7 ms/frame at 480p from a hand-written Metal convolution kernel,
and the real measured cost at that input size is about 2.6x lower (~100-110 ms
at 640x480; the 22.6 ms figure quoted elsewhere is `animevideov3` at 480x270,
a smaller input with half the convolutions, so it is not the same comparison)
-- `scripts/realesrgan/README.md`. So this does not project from a
kernel either: it builds the transformer block stack at the dimensions the
shipped weights actually have -- including which blocks are dual-stream -- and
times a real forward pass through it.

Weights are random. Nothing in a forward pass depends on their values, so the
timing is the real one for the shape; it is the shape that costs.

  python3 scripts/seedvr2/bench.py                        # 3B, 1280x960, 25 frames
  python3 scripts/seedvr2/bench.py --variant 7b
  python3 scripts/seedvr2/bench.py --res 1920x1080 --frames 100
  python3 scripts/seedvr2/bench.py --attention full       # the other bracket

Real time here means 30 fps, matching the rest of the project. The figure this
prints is a *floor*: it is the DiT alone, and the 3D causal VAE's encode and
decode are not in it (see `seedvr2.vae_note`).
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

import seedvr2

FPS = 30.0


class RMSNorm(nn.Module):
    """Upstream's `fusedrms`: per-token RMS normalisation, no mean subtraction."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight


class SwiGLU(nn.Module):
    """`proj_in` / `proj_in_gate` / `proj_out`, as the checkpoints name them."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.proj_in = nn.Linear(dim, hidden, bias=False)
        self.proj_in_gate = nn.Linear(dim, hidden, bias=False)
        self.proj_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))


def windowed_attention(q, k, v, shape, window, extra=None):
    """Attention inside `window`-sized latent windows.

    q/k/v are (n, t*h*w, heads, head_dim) over the token grid `shape`. Queries
    attend only to tokens sharing a window, so with the configured (4,3,3) each
    query sees at most 36 keys instead of the whole clip.

    `extra` is (k, v) for tokens that are global rather than windowed -- the text
    stream's tokens, which every video query may see. They are appended as
    additional keys and values inside every window, which is what a windowed
    MMDiT with a global text stream does.

    The windows are folded into the *batch* dimension rather than kept as a 5-D
    tensor: MPSGraph's `scaled_dot_product_attention` fails to build a graph for
    the 5-D form ("incompatible dimensions", on operands it has already
    transposed), while the equivalent 4-D (windows-as-batch) call runs. That is a
    limitation of this backend, not of the model.
    """
    n, s, heads, hd = q.shape
    t, h, w = shape
    wt, wh, ww = window
    pt, ph, pw = (-t) % wt, (-h) % wh, (-w) % ww
    nt, nh, nw = (t + pt) // wt, (h + ph) // wh, (w + pw) // ww
    nwin, win = nt * nh * nw, wt * wh * ww

    def fold(x):
        # (n, t h w, heads, hd) -> (n*nwin, heads, win, hd), zero-padded to whole windows
        x = x.view(n, t, h, w, heads, hd)
        x = F.pad(x, (0, 0, 0, 0, 0, pw, 0, ph, 0, pt))
        x = x.view(n, nt, wt, nh, wh, nw, ww, heads, hd)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8)   # n, nt, nh, nw, heads, wt, wh, ww, hd
        return x.reshape(n, nwin, heads, win, hd).reshape(n * nwin, heads, win, hd)

    qf, kf, vf = fold(q), fold(k), fold(v)
    if extra is not None:
        ek, ev = extra
        # (n, s_extra, heads, hd) -> (n*nwin, heads, s_extra, hd)
        ek = (ek.permute(0, 2, 1, 3).unsqueeze(1)
                .expand(n, nwin, heads, -1, hd).reshape(n * nwin, heads, -1, hd))
        ev = (ev.permute(0, 2, 1, 3).unsqueeze(1)
                .expand(n, nwin, heads, -1, hd).reshape(n * nwin, heads, -1, hd))
        kf = torch.cat([kf, ek], dim=2)
        vf = torch.cat([vf, ev], dim=2)

    y = F.scaled_dot_product_attention(qf, kf, vf)
    y = y.reshape(n, nt, nh, nw, heads, wt, wh, ww, hd)
    y = y.permute(0, 1, 5, 2, 6, 3, 7, 4, 8)       # n, nt, wt, nh, wh, nw, ww, heads, hd
    y = y.reshape(n, nt * wt, nh * wh, nw * ww, heads, hd)
    return y[:, :t, :h, :w].reshape(n, s, heads, hd)


class Stream(nn.Module):
    """One attention+MLP stream: `vid`, `txt`, or the `all` stream of a vid-only block."""

    def __init__(self, dim, heads, head_dim, hidden, window, attention):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.window = window
        self.attention = attention
        self.norm_q = RMSNorm(head_dim)
        self.norm_k = RMSNorm(head_dim)
        self.proj_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=True)
        self.mlp = SwiGLU(dim, hidden)

    def qkv(self, x):
        n, s, _ = x.shape
        q, k, v = self.proj_qkv(x).view(n, s, 3, self.heads, self.head_dim).unbind(2)
        return self.norm_q(q), self.norm_k(k), v

    def attend(self, q, k, v, shape, extra=None):
        """Attention + this stream's output projection.

        `extra` is (k, v) for tokens that are global rather than windowed -- the
        text stream's tokens, which every video query may see.
        """
        n, s = q.shape[0], q.shape[1]
        if self.attention == "full":
            qt, kf, vf = (z.transpose(1, 2) for z in (q, k, v))
            if extra is not None:
                ek, ev = extra
                kf = torch.cat([kf, ek.transpose(1, 2)], dim=2)
                vf = torch.cat([vf, ev.transpose(1, 2)], dim=2)
            y = F.scaled_dot_product_attention(qt, kf, vf).transpose(1, 2)
        else:
            y = windowed_attention(q, k, v, shape, self.window, extra)
        return self.proj_out(y.reshape(n, s, -1))


class VidBlock(nn.Module):
    """A vid-only block: one stream (`all` in the checkpoint's naming)."""

    def __init__(self, dim, heads, head_dim, hidden, window, attention):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.stream = Stream(dim, heads, head_dim, hidden, window, attention)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, shape):
        x = x + self.stream.attend(*self.stream.qkv(self.norm1(x)), shape)
        return x + self.stream.mlp(self.norm2(x))


class DualBlock(nn.Module):
    """A dual-stream block: one joint attention over text + video, two MLPs.

    Both streams' tokens go through a single softmax -- that is what makes it
    MMDiT rather than two independent blocks. The video queries are windowed and
    the text tokens are global keys for them; the text queries (58 of them) see
    everything, which costs nothing next to the video path.
    """

    def __init__(self, dim, heads, head_dim, hidden, window, attention):
        super().__init__()
        self.vid = Stream(dim, heads, head_dim, hidden, window, attention)
        self.txt = Stream(dim, heads, head_dim, hidden, window, attention)
        self.norm1_vid, self.norm1_txt = RMSNorm(dim), RMSNorm(dim)
        self.norm2_vid, self.norm2_txt = RMSNorm(dim), RMSNorm(dim)

    def forward(self, vid, txt, shape):
        qv, kv, vv = self.vid.qkv(self.norm1_vid(vid))
        qt, kt, vt = self.txt.qkv(self.norm1_txt(txt))

        vid = vid + self.vid.attend(qv, kv, vv, shape, extra=(kt, vt))

        q = torch.cat([qt, qv], dim=1).transpose(1, 2)
        k = torch.cat([kt, kv], dim=1).transpose(1, 2)
        v = torch.cat([vt, vv], dim=1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)[:, :txt.shape[1]]
        txt = txt + self.txt.proj_out(y.reshape(y.shape[0], y.shape[1], -1))

        vid = vid + self.vid.mlp(self.norm2_vid(vid))
        txt = txt + self.txt.mlp(self.norm2_txt(txt))
        return vid, txt


class NaDiT(nn.Module):
    """The DiT at the shipped block plan: 10 dual + 22 vid-only for 3B, 36 dual for 7B."""

    def __init__(self, variant, attention="window"):
        super().__init__()
        v = seedvr2.VARIANTS[variant]
        dim, hidden = v["vid_dim"], v["mlp_hidden"]
        ph, pw = v["patch"][1], v["patch"][2]
        patch_volume = v["patch"][0] * ph * pw
        self.patch = v["patch"]
        self.vid_in = nn.Linear(v["vid_in_channels"] * patch_volume, dim, bias=True)
        self.txt_in = nn.Linear(seedvr2.TXT_DIM, dim, bias=True)
        blocks = []
        for n, kind in v["blocks"]:
            for _ in range(n):
                cls = DualBlock if kind == "dual" else VidBlock
                blocks.append(cls(dim, v["heads"], v["head_dim"], hidden,
                                  seedvr2.WINDOW, attention))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = RMSNorm(dim)
        self.vid_out = nn.Linear(dim, v["vid_out_channels"] * patch_volume, bias=True)

    def forward(self, latents, txt):
        n, c, t, h, w = latents.shape
        pt, ph, pw = self.patch
        # patchify: fold the patch axes into channels, as upstream's patch_size does
        x = latents.view(n, c, t // pt, pt, h // ph, ph, w // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(n, -1, c * pt * ph * pw)
        vid = self.vid_in(x)
        txt = self.txt_in(txt)
        shape = (t // pt, h // ph, w // pw)
        for blk in self.blocks:
            if isinstance(blk, DualBlock):
                vid, txt = blk(vid, txt, shape)
            else:
                vid = blk(vid, shape)
        return self.vid_out(self.out_norm(vid))


def sync(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()


def matmul_ceiling(device, dtype, dim=4096, iters=5):
    a = torch.randn(dim, dim, device=device, dtype=dtype)
    b = torch.randn(dim, dim, device=device, dtype=dtype)
    for _ in range(2):
        a @ b
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync(device)
    return 2.0 * dim ** 3 / ((time.perf_counter() - t0) / iters) / 1e12


def peak_memory_gb(device):
    if str(device).startswith("mps"):
        return torch.mps.driver_allocated_memory() / 1e9
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="3b", choices=sorted(seedvr2.VARIANTS))
    ap.add_argument("--res", default="1280x960", help="output WxH, as upstream's res_w/res_h")
    ap.add_argument("--frames", type=int, default=25,
                    help="frames per forward pass (upstream's example is 100)")
    ap.add_argument("--attention", default="window", choices=["window", "full"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--matmul", action="store_true", help="also report the matmul ceiling")
    ap.add_argument("--mem-fraction", type=float, default=0.85,
                    help="cap MPS allocation as a fraction of the GPU working set "
                         "(which macOS reports as ~17.7 GB on this 26 GB machine, "
                         "below physical RAM). The default leaves the OS room and "
                         "still admits the largest 3B case that fits; under it, an "
                         "oversized configuration fails cleanly instead of swapping "
                         "the whole system -- 7B fp16 weights alone are 16.5 GB. Set "
                         "1.0 to see what a dedicated device would attempt.")
    a = ap.parse_args()

    width, height = (int(v) for v in a.res.lower().split("x"))
    device = torch.device(a.device)
    dtype = getattr(torch, a.dtype)
    if str(device).startswith("mps") and a.mem_fraction < 1.0:
        torch.mps.set_per_process_memory_fraction(a.mem_fraction)

    t_lat, h_lat, w_lat = seedvr2.latent_grid(width, height, a.frames, a.variant)
    n_tokens = t_lat * h_lat * w_lat

    print(seedvr2.describe(a.variant))
    print("output %dx%d, %d frames -> latent grid %dx%dx%d = %s tokens, %s attention, %s"
          % (width, height, a.frames, t_lat, h_lat, w_lat, "{:,}".format(n_tokens),
             a.attention, a.dtype))
    print()

    if a.matmul:
        for dt in (torch.float16, torch.float32):
            tf = matmul_ceiling(device, dt)
            print("  dense matmul 4096^3 on %s %-9s %6.1f TFLOP/s"
                  % (a.device, str(dt).split(".")[-1], tf))
        print()

    model = NaDiT(a.variant, attention=a.attention).to(device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    # The DiT's input is 33 channels, not the VAE's 16: `vid_in.proj` is a
    # [dim, 132] weight, and 132 = 33 * patch_volume. The 33 is the noisy latent
    # (16) concatenated with the conditioning latent (16) and a mask (1) -- the
    # SR task is conditioned, so both are in the same tensor.
    #
    # The grid has to land on the model's patch lattice: the VAE divides by 8 and
    # patch_size divides by 2 again, so each output dimension must be a multiple
    # of 16. 1080 is not (1080/8 = 135, which is odd), which is why upstream's
    # own default output is 720p and not 1080p. Pad up to the next valid size
    # rather than refusing: the token count and so the timing stay honest, and
    # the padding is reported.
    v = seedvr2.VARIANTS[a.variant]
    ph, pw = v["patch"][1], v["patch"][2]
    t_lat, h_lat, w_lat = seedvr2.vae_latent_shape(width, height, a.frames)
    h_pad, w_pad = (-h_lat) % ph, (-w_lat) % pw
    if h_pad or w_pad:
        print("NOTE  %dx%d is off SeedVR2's patch lattice (each side must be a "
              "multiple of 16:\n      VAE /8 then patch /2). Padding the latent "
              "grid to %dx%d, i.e. an effective\n      output of %dx%d; the token "
              "count rises by %.1f%%.\n"
              % (width, height, w_lat + w_pad, h_lat + h_pad,
                 (w_lat + w_pad) * 8 * pw, (h_lat + h_pad) * 8 * ph,
                 100.0 * ((h_lat + h_pad) * (w_lat + w_pad) / (h_lat * w_lat) - 1)))
        h_lat, w_lat = h_lat + h_pad, w_lat + w_pad

    # Token count after padding, so the arithmetic context below describes the
    # pass that was actually timed rather than the unpadded request.
    n_tokens = (t_lat // v["patch"][0]) * (h_lat // ph) * (w_lat // pw)

    latents = torch.randn(1, v["vid_in_channels"], t_lat, h_lat, w_lat,
                          device=device, dtype=dtype)
    txt = torch.randn(1, seedvr2.TXT_TOKEN_COUNT, seedvr2.TXT_DIM,
                      device=device, dtype=dtype)

    try:
        with torch.no_grad():
            for _ in range(1):
                model(latents, txt)
            sync(device)
            t0 = time.perf_counter()
            for _ in range(a.iters):
                model(latents, txt)
            sync(device)
            dt = (time.perf_counter() - t0) / a.iters
    except RuntimeError as e:
        msg = str(e)
        if "memory" in msg.lower():
            print("ran out of memory before producing a timing:")
            print("  %s" % msg.split("\n")[0])
            print("  reduce --res or --frames. Per-frame cost is what matters and it "
                  "does not fall.")
        else:
            raise
        return

    ms_window = dt * 1000
    ms_frame = ms_window / a.frames
    fps = 1000.0 / ms_frame
    print("  model built here: %.4f B params   (released: %.4f B)"
          % (n_params / 1e9, seedvr2.params_from_checkpoint(a.variant) / 1e9))
    print("  one forward pass over %d frames:  %9.1f ms" % (a.frames, ms_window))
    print("  per frame:                        %9.1f ms   %6.2f fps   %.4fx realtime @30fps"
          % (ms_frame, fps, fps / FPS))
    print("  peak MPS allocation:              %9.2f GB" % peak_memory_gb(device))
    print()
    lin = seedvr2.flops_linear(a.variant, n_tokens) / a.frames
    att = seedvr2.flops_attention(a.variant, n_tokens, width, height, a.frames,
                                  windowed=(a.attention == "window")) / a.frames
    print("  arithmetic context (the timing above is measured, this is not a projection):")
    print("    linear      %9.2f TFLOP/frame" % (lin / 1e12))
    print("    attention   %9.2f TFLOP/frame  (%s)" % (att / 1e12, a.attention))
    print("    sum         %9.2f TFLOP/frame -> %8.1f TFLOP/s needed to hold 30 fps"
          % ((lin + att) / 1e12, (lin + att) * FPS / 1e12))
    print("    achieved    %9.2f TFLOP/s of the above, measured"
          % ((lin + att) / dt / 1e12))


if __name__ == "__main__":
    main()
