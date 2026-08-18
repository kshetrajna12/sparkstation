#!/usr/bin/env bash
# Re-run dsv4 missions invalidated by the 23:12 worker-rank death (NCCL
# breakage — infra failure, not model failure). Run AFTER the morning
# restore puts dsv4 back. Appends rows tagged rerun=true; grading takes the
# rerun row for these missions.
set -uo pipefail
REPO="$HOME/src/github.com/sparkstation"
OUT="$HOME/.sparkstation/quant-ab/night-$(date +%Y%m%d 2>/dev/null)"
[ -d "$OUT" ] || OUT="$HOME/.sparkstation/quant-ab/night-20260817"
LOG="$OUT/night.log"; RESULTS="$OUT/results.jsonl"
PI_TIMEOUT=1800
HARD_STOP_H=${HARD_STOP_H:-11}   # don't run past this local hour
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd "$REPO"

# sanity: default must be dsv4
got=$(curl -s --max-time 10 -H "Authorization: Bearer dummy-key" -H "Content-Type: application/json" \
  http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"default","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8,"chat_template_kwargs":{"thinking":false}}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('model',''))" 2>/dev/null)
echo "$got" | grep -qi deepseek || { log "RERUN ABORT: default is '$got', not dsv4"; exit 1; }

log "=== RERUN dsv4 invalidated missions start ==="
STD="wis-schedule regex-lite expr-eval edit-script lru-ttl real-fix-ports real-fix-comps"
ESC="esc-interpreter esc-ot-converge esc-bounded-queue"
for tid in $STD $ESC; do
  [ "$(date +%H)" -ge "$HARD_STOP_H" ] && { log "RERUN hard-stop reached"; break; }
  WS="$OUT/dsv4/prob-$tid"; rm -rf "$WS"; mkdir -p "$WS"
  PROMPT=$(python3 - "$tid" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ab", "patches/qwen-quant-ab.py")
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
t = next(t for t in ab.TASKS if t["id"] == sys.argv[1])
p = t["prompt"]
for cut in ("Reply with ONLY a python code block containing transform and apply.",
            "Reply with ONLY a python code block containing the fixed function.",
            "Reply with ONLY a python code block containing the corrected function (minimal change).",
            "Reply with ONLY a python code block."):
    p = p.replace(cut, "")
print(p + " Work ONLY in the current directory. Write your final solution as a single self-contained python file named solution.py (module-level definitions exactly as specified). Create and RUN your own tests here to verify before finishing — you are graded by hidden tests on solution.py only.")
PYEOF
)
  t0=$(date +%s)
  ( cd "$WS" && timeout "$PI_TIMEOUT" pi -p "$PROMPT" > "$OUT/dsv4-$tid-pi-rerun.txt" 2>&1 )
  rc=$?; t1=$(date +%s)
  v=$(python3 patches/qwen-quant-ab.py --check-solution "$tid" "$WS/solution.py" 2>>"$LOG")
  [ -z "$v" ] && v='{"pass": false, "error": "check crashed"}'
  echo "{\"arm\":\"dsv4\",\"mission\":\"prob-$tid\",\"wall_s\":$((t1-t0)),\"pi_rc\":$rc,\"verify\":$v,\"rerun\":true}" >> "$RESULTS"
  log "    RERUN check[$tid][dsv4]: $v wall=$((t1-t0))s"
done
log "=== RERUN dsv4 complete ==="
