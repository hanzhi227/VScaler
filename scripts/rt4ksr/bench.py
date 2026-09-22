#!/usr/bin/env python3
"""Throughput of RT4KSR per device and input size, on this machine.

Frame-independent, so this is a straight loop over random frames — no chunks, no
state to carry. The numbers in docs/GENERAL-VIDEO-UPSCALING.md come from here.

  python3 scripts/rt4ksr/bench.py [--scale 2] [--devices mps,cpu] [--iters 5]

Read the timings with the FLOP count in mind: at 960x540 -> 1920x1080 one frame
is 12.17 GFLOP, so ~7 ms is ~1.7 TFLOP/s of actual work. That is well under this
machine's matmul peak, and that is what a 24-channel network at this size looks
like. It is a small model being timed, not a saturated GPU.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import rt4ksr

SIZES = [(480, 270), (640, 360), (854, 480), (960, 540)]


def gflops(scale, models, w, h):
    """FLOPs for one frame, counted on CPU.

    The counter instruments the ops, and the graph is the same on any device, so
    counting once on CPU keeps device-specific timing loops free of hooks.
    """
    from torch.utils.flop_counter import FlopCounterMode
    up = rt4ksr.Upscaler(scale=scale, device='cpu', models=models)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=int, default=2, choices=[2, 3, 4])
    ap.add_argument('--devices', default='mps,cpu')
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--weights', default=None, help='directory with RT4KSR_x*.pth')
    a = ap.parse_args()

    models = None
    if a.weights:
        s = 3 if a.scale == 3 else 2
        path = os.path.join(a.weights, 'RT4KSR_x%d.pth' % s)
        if not os.path.exists(path):
            raise SystemExit('no weights at %s' % path)
        models = {s: path}

    cost = {sz: gflops(a.scale, models, *sz) for sz in SIZES}

    for dev in devices(a.devices):
        try:
            up = rt4ksr.Upscaler(scale=a.scale, device=dev, models=models)
        except FileNotFoundError as e:
            raise SystemExit('weights not found (%s)\n'
                             'run scripts/rt4ksr/fetch_weights.py first' % e)
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
            print('%-4s %4dx%-4d -> %4dx%-4d  %8.2f ms/frame  %7.1f fps  '
                  '%.2fx of a 30fps budget  %6.2f GFLOP'
                  % (dev, w, h, w * a.scale, h * a.scale, dt * 1000, 1 / dt,
                     dt * 30, cost[(w, h)]))
        print()


if __name__ == '__main__':
    main()
