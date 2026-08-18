#!/usr/bin/env bash
# Trial arm: qwen3.8-27b on the SGLang + DSpark recipe (hasso5703/dgx-spark-qwen38,
# 28-40 tok/s agentic on a single Spark). Runs the IDENTICAL 15-mission gauntlet
# through pi against the standalone server on worker1:30000.
#
# Preconditions: sparkstation stopped (GPU exclusive on both nodes), the recipe
# server RUNNING on worker1 (ssh worker1 'cd dgx-spark-qwen38 && ./run.sh'),
# pi config swapped by this script (reverted on exit).
set -uo pipefail
REPO="$HOME/src/github.com/sparkstation"
OUT="$HOME/.sparkstation/quant-ab/night-20260817"   # same trial dataset
LOG="$OUT/night.log"; RESULTS="$OUT/results.jsonl"
ARM=qwen-dspark
BASE="http://192.168.100.11:30000"
KEY=$(ssh 192.168.100.11 'cat ~/.config/qwen38/api-key')
PI_TIMEOUT="${PI_TIMEOUT:-3600}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd "$REPO"

# ── pi provider swap (revert on exit) ────────────────────────────────────
PI_MODELS="$HOME/.pi/agent/models.json"
cp "$PI_MODELS" "$PI_MODELS.pre-dspark-arm"
revert_pi() { cp "$PI_MODELS.pre-dspark-arm" "$PI_MODELS"; log "pi config reverted"; }
trap revert_pi EXIT
python3 - "$BASE" "$KEY" <<'PYEOF'
import json, sys
p = __import__("os").path.expanduser("~/.pi/agent/models.json")
c = json.load(open(p))
prov = c["providers"]["sparkstation"]
prov["baseUrl"] = sys.argv[1] + "/v1"
prov["apiKey"] = sys.argv[2]
for m in prov["models"]:
    m["id"] = "qwen3.8-27b"   # served name on the recipe server
json.dump(c, open(p, "w"), indent=2)
print("pi → dspark recipe server")
PYEOF

# sanity: server answers with the model
got=$(curl -s --max-time 20 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  "$BASE/v1/chat/completions" -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c "import json,sys; b=json.load(sys.stdin); print(b.get('model',''), b['choices'][0]['message'].get('content',''))" 2>/dev/null)
log "recipe server check: $got"
echo "$got" | grep -qi qwen || { log "ABORT: recipe server not answering"; exit 1; }

log "=== ARM: $ARM (SGLang + DSpark public draft) ==="
armdir="$OUT/$ARM"; mkdir -p "$armdir"

pi_mission() {
  local mid=$1 wd=$2 prompt=$3 t0 t1 rc
  log "--- [$ARM] pi mission: $mid"
  t0=$(date +%s)
  ( cd "$wd" && timeout "$PI_TIMEOUT" pi -p "$prompt" > "$OUT/$ARM-$mid-pi.txt" 2>&1 )
  rc=$?; t1=$(date +%s)
  log "    [$ARM] $mid: pi rc=$rc wall=$((t1-t0))s"
  echo "$rc $((t1-t0))"
}

# ── agentic missions (fresh clone) ───────────────────────────────────────
clone="$armdir/RealtorZero"
rm -rf "$clone"; git clone -q "$HOME/src/github.com/RealtorZero" "$clone"
python3 "$REPO/patches/pi-trial-verify.py" plant-a "$clone" >> "$LOG"

read -r rc wall <<< "$(pi_mission agentic-bugfix "$clone" \
  "You are working ONLY inside this directory (a scratch clone — safe to modify, never leave it). Bug report from production: find_comps() wrongly EXCLUDES comps whose bed count is unknown (beds is NULL, which is every Maricopa county record) whenever min_beds or max_beds is set. Per the design, unknown bed counts must PASS bed filters. Find the bug and fix it minimally.")"
v=$(python3 "$REPO/patches/pi-trial-verify.py" verify-a "$clone" 2>>"$LOG")
echo "{\"arm\":\"$ARM\",\"mission\":\"agentic-bugfix\",\"wall_s\":$wall,\"pi_rc\":$rc,\"verify\":$v}" >> "$RESULTS"
log "    verify-a: $v"

read -r rc wall <<< "$(pi_mission agentic-feature "$clone" \
  "Still ONLY inside this directory. Add a function median_ppsf(comps) to src/realtorzero/tools/comps.py: takes a list of dicts with sale_price and sqft keys, returns the median price-per-sqft as a float, ignoring comps whose sqft is missing, None, or zero. Also add a pytest test file covering it (normal case + ignore rules). Run your test to confirm it passes before finishing.")"
v=$(python3 "$REPO/patches/pi-trial-verify.py" verify-b "$clone" 2>>"$LOG")
echo "{\"arm\":\"$ARM\",\"mission\":\"agentic-feature\",\"wall_s\":$wall,\"pi_rc\":$rc,\"verify\":$v}" >> "$RESULTS"
log "    verify-b: $v"

read -r rc wall <<< "$(pi_mission agentic-research "$clone" \
  "Still ONLY inside this directory. Use your web search capability to find this week's average 30-year fixed US mortgage rate (Freddie Mac PMMS is the canonical source). Create MARKET_NOTES.md in the repo root containing: the rate as a percentage, the survey date, and the source URL.")"
v=$(python3 "$REPO/patches/pi-trial-verify.py" verify-c "$clone" 2>>"$LOG")
echo "{\"arm\":\"$ARM\",\"mission\":\"agentic-research\",\"verify\":$v,\"wall_s\":$wall,\"pi_rc\":$rc}" >> "$RESULTS"
log "    verify-c: $v"
git -C "$clone" diff > "$armdir/agentic.diff" 2>/dev/null

# ── hard + escalation problems (solution.py missions) ────────────────────
for tid in real-hard-valuation wis-schedule regex-lite expr-eval edit-script lru-ttl real-fix-ports real-fix-comps esc-interpreter esc-ot-converge esc-bounded-queue; do
  ws="$armdir/prob-$tid"; rm -rf "$ws"; mkdir -p "$ws"
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
  read -r rc wall <<< "$(pi_mission "prob-$tid" "$ws" "$PROMPT")"
  v=$(python3 patches/qwen-quant-ab.py --check-solution "$tid" "$ws/solution.py" 2>>"$LOG")
  [ -z "$v" ] && v='{"pass": false, "error": "check crashed"}'
  echo "{\"arm\":\"$ARM\",\"mission\":\"prob-$tid\",\"wall_s\":$wall,\"pi_rc\":$rc,\"verify\":$v}" >> "$RESULTS"
  log "    check[$tid][$ARM]: $v"
done
log "=== ARM $ARM complete ==="
