#!/usr/bin/env python3
"""Check this project's RT4KSR against upstream, on the real released weights.

Three claims, in order, each of which the docs make:

  1. the training-shaped class is a faithful copy of upstream's, so loading with
     strict=True means something;
  2. folding expand/fea/reduce into one 3x3 is numerically sound on the released
     weights, not just on paper;
  3. skipping the high-frequency branch changes nothing — it is dead when
     `forget=False`, which is hardcoded in upstream's `rt4ksr_rep`.

Usage:
  git clone --depth 1 https://github.com/eduardzamfir/RT4KSR /tmp/RT4KSR
  python3 scripts/rt4ksr/verify_against_upstream.py --upstream /tmp/RT4KSR

Exits non-zero if any claim fails. Needs the checkpoints, which
`fetch_weights.py` puts in ~/.cache/macvideoupscaler.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import rt4ksr


class Cfg:
    """Upstream's model config, as its argparse defaults define it."""


def cfg(scale, is_train):
    c = Cfg()
    c.scale = scale
    c.is_train = is_train
    c.feature_channels = rt4ksr.DEFAULT_CFG['feature_channels']
    c.num_blocks = rt4ksr.DEFAULT_CFG['num_blocks']
    c.act_type = rt4ksr.DEFAULT_CFG['act_type']
    return c


def rel(a, b):
    d = (a - b).abs().max().item()
    return d, d / max(b.abs().max().item(), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', required=True, help='path to a clone of RT4KSR')
    ap.add_argument('--scales', default='2,3')
    ap.add_argument('--weights', default=None, help='directory with RT4KSR_x*.pth')
    ap.add_argument('--tol', type=float, default=1e-5,
                    help='tolerance for the fold; it is float32 rounding, not exact')
    a = ap.parse_args()

    code = os.path.join(a.upstream, 'code')
    if not os.path.isdir(os.path.join(code, 'model')):
        raise SystemExit('no code/model under %s' % a.upstream)
    ckpt = os.path.join(a.upstream, 'code', 'checkpoints')
    sys.path.insert(0, code)
    import model as upstream  # noqa: E402  (needs the path above)

    torch.manual_seed(0)
    bad = 0
    for scale in (int(s) for s in a.scales.split(',')):
        path = os.path.join(a.weights or rt4ksr.CACHE_DIR, 'RT4KSR_x%d.pth' % scale)
        if not os.path.exists(path):
            raise SystemExit('no weights at %s\n'
                             'run scripts/rt4ksr/fetch_weights.py first' % path)
        blob = torch.load(path, map_location='cpu')
        sd = rt4ksr._strip(blob['state_dict'] if 'state_dict' in blob else blob)

        up = upstream.rt4ksr_rep(cfg(scale, True))
        up.load_state_dict(sd, strict=True)
        up.eval()

        mine = rt4ksr.RT4KSR(scale=scale, is_train=True)
        mine.load_state_dict(sd, strict=True)
        mine.eval()

        folded = rt4ksr.merge_rep(sd, scale)
        folded.eval()
        fast = rt4ksr.merge_rep(sd, scale, drop_dead_branch=True)
        fast.eval()

        x = torch.rand(1, 3, 96, 128)
        with torch.no_grad():
            y_up, y_mine = up(x), mine(x)
            y_fold, y_fast = folded(x), fast(x)

        d1, _ = rel(y_mine, y_up)
        d2, _ = rel(y_fold, y_mine)
        d3, _ = rel(y_fast, y_fold)
        n = sum(p.numel() for p in folded.parameters())

        ok1, ok2, ok3 = d1 == 0.0, d2 <= a.tol, d3 == 0.0
        print('x%d  output %s   params %s' % (scale, tuple(y_up.shape), format(n, ',')))
        print('  training class vs upstream         %s  max|d|=%.3e' % ('OK  ' if ok1 else 'FAIL', d1))
        print('  folded vs training class           %s  max|d|=%.3e' % ('OK  ' if ok2 else 'FAIL', d2))
        print('  dead branch dropped vs folded      %s  max|d|=%.3e' % ('OK  ' if ok3 else 'FAIL', d3))
        if not (ok1 and ok2 and ok3):
            bad += 1

    print('all claims hold' if not bad else '%d scale(s) FAILED' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
