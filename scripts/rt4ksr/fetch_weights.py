#!/usr/bin/env python3
"""Fetch the RT4KSR x2 and x3 weights.

Upstream keeps these in the git repo (1.7 MB and 1.9 MB), but this project does
not vendor binaries, so they are downloaded here instead. Installs to
~/.cache/macvideoupscaler/RT4KSR_x{2,3}.pth, which is where rt4ksr.load() looks.

Checksums are pinned to the bytes upstream ships, so a silently changed or
truncated download fails loudly instead of producing a differently-behaving model.

  python3 scripts/rt4ksr/fetch_weights.py [--dest DIR] [--force]
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

BASE = 'https://github.com/eduardzamfir/RT4KSR/raw/main/code/checkpoints'
FILES = {
    2: ('rt4ksr_x2.pth', 1699164,
        '9ec7500092848a480e058a9c703e4b2254bc0cc97928c076bba272177752124f'),
    3: ('rt4ksr_x3.pth', 1855452,
        '13229f15a7f3897096ab0ee92b7c3d3c64fd59b4e558fea3248147cec225e65e'),
}
DEFAULT_DEST = os.path.expanduser('~/.cache/macvideoupscaler')


def fetch(scale, dest, force):
    name, size, sha = FILES[scale]
    url = f'{BASE}/{name}'
    target = os.path.join(dest, 'RT4KSR_x%d.pth' % scale)
    if os.path.exists(target) and not force:
        if os.path.getsize(target) == size:
            print(f'already present: {target} ({size} bytes)')
            return 0
        print(f'{target} has the wrong size, re-downloading', file=sys.stderr)

    tmp = target + '.part'
    print(f'downloading {url}\n         -> {target}')
    try:
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        print(f'download failed: {e}', file=sys.stderr)
        return 1

    got = os.path.getsize(tmp)
    digest = hashlib.sha256(open(tmp, 'rb').read()).hexdigest()
    if got != size or digest != sha:
        os.unlink(tmp)
        print(f'{name}: got {got} bytes / {digest}\n'
              f'  expected {size} bytes / {sha}', file=sys.stderr)
        return 1
    os.rename(tmp, target)
    print(f'wrote {target} ({got} bytes)')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=DEFAULT_DEST)
    ap.add_argument('--scale', type=int, choices=[2, 3], action='append',
                    help='default: both')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    os.makedirs(a.dest, exist_ok=True)
    rc = 0
    for scale in (a.scale or [2, 3]):
        rc |= fetch(scale, a.dest, a.force)
    return rc


if __name__ == '__main__':
    sys.exit(main())
