#!/usr/bin/env python3
"""Throughput of Real-ESRGAN's animevideov3 per device and input size.

The question this answers is the only one that matters for the app: can it play
live? Frames are random, so nothing but the network is being timed.

  python3 scripts/realesrgan/bench.py [--devices mps,cpu] [--iters 5] [--tile N]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import srvgg

SIZES = [(480, 270), (640, 360), (854, 480), (960, 540)]

# The two official x4 checkpoints share this architecture and differ only in the
# number of body convolutions, so measuring both here is what makes the family's
# real cost a measurement rather than a projection.
VARIANTS = {
    'animevideov3': dict(num_conv=16, weights=os.path.join(
        srvgg.CACHE_DIR, 'realesr-animevideov3.pth')),
    'general-x4v3': dict(num_conv=32, weights=os.path.join(
        srvgg.CACHE_DIR, 'realesr-general-x4v3.pth')),
}


def gflops(models, w, h):
    from torch.utils.flop_counter import FlopCounterMode
    up = srvgg.Upscaler(device='cpu', **models)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        up.upscale(torch.rand(1, 3, h, w))
    return fc.get_total_flops() / 1e9


def devices(requested):
    out = []
    for name in (d.strip() for d in requested.split(',')):
        if not name:
            continue
        if name == 'mps' and not torch.backends.mps.is_available():
            print('mps not available, skipping', file=sys.stderr)
            continue
        out.append(name)
    return out


def mps_memory(device):
    """(driver allocation in GB after the run, device working-set ceiling in GB).

    Reported because the project's budget for an upscaling model is 2 GB, and
    for this family it is the activation working set that spends it, not the
    weights: animevideov3 is 621k parameters (2.5 MB at fp32), but its body runs
    at input resolution with 64 channels and its output is 16x the input pixel
    count, so its cost is linear in frame area rather than in model size.

    This is not a peak: PyTorch's MPS backend exposes no high-water-mark API
    (only `current_allocated_memory`, `driver_allocated_memory` and
    `recommended_max_memory`). It is what is still allocated once the frame's
    output tensor is alive, which for a whole-frame pass is the same set of
    buffers the peak was made of.
    """
    if not str(device).startswith("mps"):
        return float("nan"), float("nan")
    return (torch.mps.driver_allocated_memory() / 1e9,
            torch.mps.recommended_max_memory() / 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--devices', default='mps,cpu')
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--variant', default='animevideov3', choices=sorted(VARIANTS))
    ap.add_argument('--weights', default=None)
    ap.add_argument('--tile', type=int, default=0,
                    help='tile size in source pixels (0 = whole frame at once)')
    a = ap.parse_args()

    v = VARIANTS[a.variant]
    kw = {'weights': a.weights or v['weights'], 'tile': a.tile, 'num_conv': v['num_conv']}
    if a.weights:
        raise SystemExit('--weights and --variant are alternatives; use --variant')
    print('variant: %s  (%d body convs)' % (a.variant, v['num_conv']))
    cost = {sz: gflops(kw, *sz) for sz in SIZES}

    for dev in devices(a.devices):
        try:
            up = srvgg.Upscaler(device=dev, **kw)
        except FileNotFoundError as e:
            raise SystemExit('weights not found (%s)\n'
                             'run scripts/realesrgan/fetch_weights.py first' % e)
        for (w, h) in SIZES:
            x = torch.rand(1, 3, h, w, device=torch.device(dev))
            with torch.no_grad():
                for _ in range(a.warmup):
                    up.upscale(x)
                if dev == 'mps':
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    up.upscale(x)
                if dev == 'mps':
                    torch.mps.synchronize()
                dt = (time.perf_counter() - t0) / a.iters
            used, ceiling = mps_memory(dev)
            print('%-4s %4dx%-4d -> %4dx%-4d  %8.2f ms/frame  %6.2f fps  '
                  '%.2fx of a 30fps budget  %6.2f GFLOP  %s'
                  % (dev, w, h, w * up.scale, h * up.scale, dt * 1000, 1 / dt,
                     dt * 30, cost[(w, h)],
                     ('GPU %5.2f of %5.2f GB' % (used, ceiling)) if ceiling == ceiling
                     else 'GPU n/a'))
        print()


if __name__ == '__main__':
    main()
