#!/usr/bin/env bash
# Escalation round for one arm: limit-finding tasks through pi, run in
# PARALLEL with (or after) the standard battery while that arm's model is
# live. Usage: quant-trial-escalate.sh <arm-label>
set -uo pipefail
ARM=${1:?arm label}
REPO="$HOME/src/github.com/sparkstation"
OUT="$HOME/.sparkstation/quant-ab/night-$(date +%Y%m%d)"
LOG="$OUT/night.log"
RESULTS="$OUT/results.jsonl"
PI_TIMEOUT="${PI_TIMEOUT:-2400}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd "$REPO"

# agentic escalation: multi-file refactor that must keep end-to-end behavior
CLONE="$OUT/$ARM/RealtorZero-esc"
rm -rf "$CLONE"
git clone -q "$HOME/src/github.com/RealtorZero" "$CLONE"

log "=== ESCALATION [$ARM] start ==="
t0=$(date +%s)
( cd "$CLONE" && timeout "$PI_TIMEOUT" pi -p "You are working ONLY inside this directory (scratch clone; safe). Refactoring mission on a real codebase: extract ALL geometric math from src/realtorzero/tools/comps.py (the lat/lon bounding-box delta computation AND the flat-earth distance function) into a new module src/realtorzero/tools/geo_math.py with clean function interfaces, update comps.py to use it, preserving EXACT behavior. Then add a property-style test file that checks: distance is symmetric under point swap, zero at identical points, and the bounding box always contains every point within the radius (try several latitudes incl. 0 and 60). Run your tests to verify before finishing." > "$OUT/$ARM-esc-refactor-pi.txt" 2>&1 )
rc=$?; t1=$(date +%s)
v=$(python3 "$REPO/patches/pi-trial-verify.py" verify-a "$CLONE" 2>>"$LOG")
[ -z "$v" ] && v='{"pass": false, "error": "verify crashed"}'
gm="false"; [ -f "$CLONE/src/realtorzero/tools/geo_math.py" ] && gm="true"
echo "{\"arm\":\"$ARM\",\"mission\":\"esc-refactor\",\"wall_s\":$((t1-t0)),\"pi_rc\":$rc,\"verify\":{\"pass\":$( [ "$gm" = true ] && echo "$v" | python3 -c "import json,sys; print(str(json.load(sys.stdin)['pass']).lower())" || echo false),\"geo_module_created\":$gm,\"behavior_preserved\":$(echo "$v" | python3 -c "import json,sys; print(str(json.load(sys.stdin)['pass']).lower())" 2>/dev/null || echo false)}}" >> "$RESULTS"
log "    esc-refactor[$ARM]: geo_module=$gm behavior=$v wall=$((t1-t0))s"
git -C "$CLONE" diff > "$OUT/$ARM/esc-refactor.diff" 2>/dev/null

# hard escalation problems via solution.py missions
for tid in esc-interpreter esc-ot-converge esc-bounded-queue; do
  WS="$OUT/$ARM/prob-$tid"; mkdir -p "$WS"
  PROMPT=$(python3 - "$tid" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ab", "patches/qwen-quant-ab.py")
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
t = next(t for t in ab.TASKS if t["id"] == sys.argv[1])
p = t["prompt"].replace("Reply with ONLY a python code block containing transform and apply.", "").replace("Reply with ONLY a python code block.", "")
print(p + " Work ONLY in the current directory. Write your final solution as a single self-contained python file named solution.py (module-level definitions exactly as specified). Create and RUN your own tests here to verify before finishing — you are graded by hidden tests on solution.py only.")
PYEOF
)
  t0=$(date +%s)
  ( cd "$WS" && timeout "$PI_TIMEOUT" pi -p "$PROMPT" > "$OUT/$ARM-$tid-pi.txt" 2>&1 )
  rc=$?; t1=$(date +%s)
  v=$(python3 patches/qwen-quant-ab.py --check-solution "$tid" "$WS/solution.py" 2>>"$LOG")
  [ -z "$v" ] && v='{"pass": false, "error": "check crashed"}'
  echo "{\"arm\":\"$ARM\",\"mission\":\"$tid\",\"wall_s\":$((t1-t0)),\"pi_rc\":$rc,\"verify\":$v}" >> "$RESULTS"
  log "    check[$tid][$ARM]: $v wall=$((t1-t0))s"
done
log "=== ESCALATION [$ARM] complete ==="
