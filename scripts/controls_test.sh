#!/bin/bash
# Verifies the app's playback controls headlessly, through exactly the code
# paths the GUI drives (PipelineController.pause/resume/seek/setKind):
#   pause freezes processing, resume continues, seek starts a NEW session at
#   the requested offset (A/V still in sync), and enhancement mode switches hot.
set -u
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$P/engine/MacVideoUpscaler/.build/debug/macvideoupscale"
FILE="${1:-$P/test-media/segments/naruto_1080p_aligned.mkv}"
SEEK_T="${2:-42}"   # seek target; must land in a moving section for A/V checks
PORT=8111
LOG=/tmp/controls_stdout.log
S1=/tmp/controls_session1.ts
S2=/tmp/controls_session2.ts

pkill -f "macvideoupscale --file" 2>/dev/null; sleep 1
: > "$LOG"; rm -f "$S1" "$S2"

"$ENGINE" --file "$FILE" --mode cnn --start 0 --port $PORT \
    --script "4:pause,8:resume,12:seek:$SEEK_T,18:mode:bypass,24:mode:cnn" \
    > "$LOG" 2>/tmp/controls_stderr.log &
CLI_PID=$!
START=$(date +%s)
sleep 3

curl -s --max-time 20 "http://127.0.0.1:$PORT/stream.ts?session=1" -o "$S1" 2>/dev/null &

# wait for the seek to land (session id becomes 2), then capture that session
for i in $(seq 1 60); do
  SESS=$(curl -s --max-time 2 "http://127.0.0.1:$PORT/status.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sessionId"])' 2>/dev/null)
  [ "${SESS:-1}" -ge 2 ] && break
  sleep 0.5
done
echo "[test] session after seek: ${SESS:-unknown} (waited $(( $(date +%s) - START ))s)"
curl -s --max-time 20 "http://127.0.0.1:$PORT/stream.ts?session=${SESS:-2}" -o "$S2" 2>/dev/null &

# let the mode-switch events fire, then stop
while [ $(( $(date +%s) - START )) -lt 30 ]; do sleep 1; done
kill $CLI_PID 2>/dev/null; wait $CLI_PID 2>/dev/null

echo "===== control events ====="
grep '"event"' "$LOG" | python3 -c '
import sys, json
for line in sys.stdin:
    try: d = json.loads(line)
    except Exception: continue
    print("  sched=%2ds wall=%4.1fs  %-6s %-7s state=%-10s session=%d frames=%-5d pos=%5.1fs mode=%s"
          % (d["at"], d["wall"], d["event"], str(d["value"] or ""), d["state"], d["session"], d["frames"], d["position"], d["mode"]))
'
echo
echo "===== processing state transitions ====="
python3 - "$LOG" <<'PY'
import json, sys
seq = []
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except Exception: continue
    if "event" in d: continue
    k = (d["state"], d["framesProcessed"])
    if not seq or seq[-1][0] != k[0]:
        seq.append(k)
for s, f in seq:
    print("  state=%-10s frames=%d" % (s, f))
PY
echo
echo "===== captured streams ====="
for f in "$S1" "$S2"; do
  if [ -s "$f" ]; then
    printf "  %-24s %9s bytes  " "$(basename "$f")" "$(stat -f%z "$f")"
    ffprobe -v error -show_entries stream=codec_type,width,height -show_entries format=duration -of csv=p=0 "$f" | sort -u | tr '\n' ' '
    echo
  else
    echo "  $(basename "$f"): empty"
  fi
done
echo
echo "===== post-seek A/V sync (session 2 vs reference at t=${SEEK_T}s) ====="
if [ -s "$S2" ]; then
  python3 "$P/scripts/check_av_sync.py" --reference "$FILE" --stream "$S2" --file-start "$SEEK_T" --dur 8 2>&1 | sed 's/^/  /' | head -12
fi
