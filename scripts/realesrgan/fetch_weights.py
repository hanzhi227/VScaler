#!/usr/bin/env python3
"""Fetch the official Real-ESRGAN x4 weights.

2.4 MB (anime video) and 4.7 MB (general), from the official v0.2.5.0 release.
Installs to ~/.cache/macvideoupscaler/, where srvgg.load() looks.

Pinned by size and sha256 so a changed or truncated download fails loudly.

  python3 scripts/realesrgan/fetch_weights.py [--dest DIR] [--force]
                                              [--only animevideov3|general-x4v3]
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

BASE = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'
DEFAULT_DEST = os.path.expanduser('~/.cache/macvideoupscaler')

# name -> (file, bytes, sha256). Both load into SRVGGNetCompact(strict=True) with
# the shape each row implies: 18 convs / 621,424 params, and 34 convs / 1,213,296.
WEIGHTS = {
    'animevideov3': ('realesr-animevideov3.pth', 2504012,
                     'b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d'),
    'general-x4v3': ('realesr-general-x4v3.pth', 4885111,
                     '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292'),
}


def fetch(name, dest, force):
    fname, nbytes, digest = WEIGHTS[name]
    target = os.path.join(dest, fname)
    if os.path.exists(target) and not force:
        size = os.path.getsize(target)
        if size == nbytes:
            print('already present: %s (%d bytes)' % (target, size))
            return 0
        print('%s has the wrong size, re-downloading' % target, file=sys.stderr)

    url = BASE + fname
    tmp = target + '.part'
    print('downloading %s\n         -> %s' % (url, target))
    try:
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        print('download failed: %s' % e, file=sys.stderr)
        return 1

    got = os.path.getsize(tmp)
    seen = hashlib.sha256(open(tmp, 'rb').read()).hexdigest()
    if got != nbytes or seen != digest:
        os.unlink(tmp)
        print('%s: got %d bytes / %s\n  expected %d bytes / %s'
              % (fname, got, seen, nbytes, digest), file=sys.stderr)
        return 1
    os.rename(tmp, target)
    print('wrote %s (%d bytes)' % (target, got))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=DEFAULT_DEST)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', choices=sorted(WEIGHTS), default=None,
                    help='fetch one checkpoint instead of both')
    a = ap.parse_args()

    os.makedirs(a.dest, exist_ok=True)
    names = [a.only] if a.only else sorted(WEIGHTS)
    rc = 0
    for name in names:
        rc |= fetch(name, a.dest, a.force)
    return rc


if __name__ == '__main__':
    sys.exit(main())
