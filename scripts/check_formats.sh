#!/usr/bin/env bash
# Plays every file in the format matrix through the embedded player and reports,
# per file, whether it decoded, how many frames arrived, and whether its audio
# was fed to the renderer. This is the check behind the format-support table.
#
#   scripts/check_formats.sh [matrix-dir] [seconds-per-file]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${1:-/tmp/format-matrix}"
SECS="${2:-5}"
PLAYER="$ROOT/engine/MacVideoUpscaler/.build/debug/playercheck"
[ -x "$PLAYER" ] || PLAYER="$ROOT/engine/MacVideoUpscaler/.build/release/playercheck"
[ -x "$PLAYER" ] || { echo "build playercheck first (swift build)" >&2; exit 1; }
[ -d "$DIR" ] || { echo "no matrix dir $DIR — run scripts/make_format_matrix.sh" >&2; exit 1; }

printf '%-22s %-10s %6s %6s %8s  %s\n' FILE STATE FRAMES OUTPUT AUDIO-S FPS-NOTE
for f in "$DIR"/*; do
    name="$(basename "$f")"
    out="$("$PLAYER" --file "$f" --seconds "$SECS" --mode bypass --mute --quiet 2>/dev/null | tail -1)"
    printf '%-22s %s\n' "$name" "$(printf '%s' "$out" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("NO SUMMARY"); raise SystemExit
msg = d.get("message", "")
# A file with no video stream is meant to fail, and saying so plainly is the
# behaviour under test — report it as such instead of as a broken row.
if d.get("final_state") == "failed" and "no video" in msg:
    print("%-10s %6d %6s %8.2f  expected: %s" % ("failed", 0, "-", 0.0, msg[:52]))
    raise SystemExit
print("%-10s %6d %6s %8.2f  %.2fx RT%s" % (
    d.get("final_state","?"), d.get("frames",0), d.get("output","-"),
    d.get("audio_frames",0)/48000.0, d.get("realtime_ratio",0),
    ("  " + msg[:60]) if msg else ""))
')"
done
