#!/usr/bin/env python3
"""Read a safetensors file's architecture without downloading the weights.

A safetensors file begins with a little-endian u64 header length and then that
many bytes of JSON, listing every tensor with its dtype, shape and byte range.
The tensor *data* is everything after, which is the 99.9% that this does not
need: an HTTP range request for the first few megabytes returns the whole
architecture of a 6.8 GB checkpoint.

That is how `seedvr2.py`'s dimensions were established -- not from upstream's
`configs_3b/main.yaml`, which is misleading about both the MLP width and which
blocks are dual-stream. See `README.md`.

  # a remote checkpoint, header only
  python3 scripts/seedvr2/inspect_weights.py \\
      https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_3b_fp16.safetensors

  # every tensor, not just the summary
  python3 scripts/seedvr2/inspect_weights.py <url-or-path> --all

  # one block, to see the shape of a block
  python3 scripts/seedvr2/inspect_weights.py <url-or-path> --block 5
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from collections import defaultdict

DEFAULT_RANGE = 4 << 20   # 4 MiB: the 7B header is ~100 KB, so this is generous


def fetch_prefix(source, nbytes):
    """The first `nbytes` of a remote URL or local file."""
    if re.match(r"^https?://", source):
        out = subprocess.run(
            ["curl", "-sSL", "--max-time", "120", "-r", "0-%d" % (nbytes - 1), source],
            capture_output=True)
        if out.returncode != 0:
            raise SystemExit("curl failed: %s" % out.stderr.decode()[:300])
        return out.stdout
    with open(source, "rb") as fh:
        return fh.read(nbytes)


def read_header(source, nbytes=DEFAULT_RANGE):
    blob = fetch_prefix(source, nbytes)
    if len(blob) < 8:
        raise SystemExit("short read: %d bytes" % len(blob))
    n = struct.unpack("<Q", blob[:8])[0]
    if n > len(blob) - 8:
        raise SystemExit("header is %d bytes but only %d were read; raise --range"
                         % (n, len(blob) - 8))
    header = json.loads(blob[8:8 + n].decode())
    meta = header.pop("__metadata__", None)
    return header, meta


def numel(entry):
    n = 1
    for d in entry["shape"]:
        n *= d
    return n


def streams_of(block_tensors):
    """Which streams a block has: `txt`+`vid` (dual), or `all` (vid-only)."""
    found = set()
    for name, _ in block_tensors:
        parts = name.split(".")
        if len(parts) > 1 and parts[0] in ("ada", "attn", "mlp") and parts[1] in ("txt", "vid", "all"):
            found.add(parts[1])
        elif len(parts) > 2 and parts[0] == "attn" and parts[1] in ("norm_q", "norm_k"):
            found.add(parts[2])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="URL or local .safetensors path")
    ap.add_argument("--all", action="store_true", help="list every tensor, not a summary")
    ap.add_argument("--block", default=None, help="list one block's tensors, e.g. --block 5")
    ap.add_argument("--range", type=int, default=DEFAULT_RANGE, help="bytes to fetch")
    a = ap.parse_args()

    header, meta = read_header(a.source, a.range)
    total = sum(numel(v) for v in header.values())

    if meta:
        print("metadata: %s" % meta)
    dtypes = defaultdict(int)
    for v in header.values():
        dtypes[v["dtype"]] += 1
    print("%s" % os.path.basename(a.source.rstrip("/")))
    print("  %d tensors, %s" % (len(header), ", ".join("%s x%d" % (k, n)
                                                       for k, n in sorted(dtypes.items()))))
    print("  %.4f B parameters (%.2f GB at fp16)" % (total / 1e9, total * 2 / 1e9))
    print()

    if a.all:
        for k in sorted(header):
            v = header[k]
            print("  %-60s %-6s %s" % (k, v["dtype"], v["shape"]))
        return

    if a.block is not None:
        prefix = "blocks.%s." % a.block
        rows = [(k, v) for k, v in header.items() if k.startswith(prefix)]
        if not rows:
            raise SystemExit("no tensors for block %s" % a.block)
        for k, v in sorted(rows):
            print("  %-50s %-6s %s" % (k, v["dtype"], v["shape"]))
        return

    # Group by block, and by what kind of block it is.
    blocks = defaultdict(list)
    other = []
    for k, v in header.items():
        m = re.match(r"blocks\.(\d+)\.(.*)$", k)
        if m:
            blocks[int(m.group(1))].append((m.group(2), v))
        else:
            other.append((k, v))

    if blocks:
        kinds = defaultdict(list)
        for idx, tensors in blocks.items():
            kinds[tuple(sorted(streams_of(tensors)))].append(idx)
        print("  %d blocks" % len(blocks))
        for kind, idx in sorted(kinds.items()):
            params = sum(numel(v) for _, v in blocks[idx[0]])
            label = ", ".join(kind) if kind else "(unrecognised)"
            span = ("%d" % idx[0] if len(idx) == 1
                    else "%d..%d" % (idx[0], idx[-1]))
            print("    %-14s %2d blocks (%-9s) %8.3f M params each = %8.3f M"
                  % (label, len(idx), span, params / 1e6, params * len(idx) / 1e6))
        print()

    print("  non-block tensors:")
    for k, v in sorted(other):
        print("    %-54s %-6s %-24s %9.3f M" % (k, v["dtype"], v["shape"], numel(v) / 1e6))


if __name__ == "__main__":
    main()
