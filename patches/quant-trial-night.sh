#!/usr/bin/env bash
# Overnight three-arm daily-driver trial — ENTIRELY through pi (user's real
# env: skills, extensions, search, thinking=xhigh). Arms: dsv4 (coding
# profile) -> qwen NVFP4 -> qwen FP8 (both image-indexing profile, worker1).
# Per arm: 3 agentic missions in a fresh RealtorZero CLONE (real repo never
# touched) + 8 hard-problem missions where pi writes solution.py files that
# are verified in a bwrap sandbox. Coding profile is restored before morning
# NO MATTER WHAT (trap + deadline guard).
set -uo pipefail

REPO="$HOME/src/github.com/sparkstation"
OUT="$HOME/.sparkstation/quant-ab/night-$(date +%Y%m%d)"
LOG="$OUT/night.log"
RESULTS="$OUT/results.jsonl"
DEADLINE_H=7   # after 07:00 local: stop starting new work, restore, report
PI_TIMEOUT="${PI_TIMEOUT:-1800}"   # qwen arms need ~2x dsv4 (half the tok/s)
mkdir -p "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
rec() { echo "$1" >> "$RESULTS"; }
past_deadline() { [ "$(date +%H)" -ge "$DEADLINE_H" ] && [ "$(date +%H)" -lt 20 ]; }

RESTORED=0
restore() {
  [ "$RESTORED" = 1 ] && return
  RESTORED=1
  log "=== RESTORE: reverting models.yaml + coding profile back up ==="
  git -C "$REPO" checkout -- models.yaml 2>/dev/null
  sparkstation stop >> "$LOG" 2>&1
  sleep 5
  sparkstation start -d >> "$LOG" 2>&1
  for i in $(seq 1 80); do
    if curl -sf --max-time 3 -H "Authorization: Bearer dummy-key" \
         http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q dsv4-flash; then
      log "coding profile restored — dsv4 serving"; return
    fi
    sleep 15
  done
  log "WARN: dsv4 not confirmed after restore — check in the morning"
}
trap restore EXIT

wait_default_serves() {  # $1 = substring the default model id must contain
  for i in $(seq 1 90); do
    got=$(curl -s --max-time 5 -H "Authorization: Bearer dummy-key" \
      -H "Content-Type: application/json" http://127.0.0.1:8000/v1/chat/completions \
      -d '{"model":"default","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8,"chat_template_kwargs":{"thinking":false}}' \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('model',''))" 2>/dev/null)
    if echo "$got" | grep -qi "$1"; then log "default now serves: $got"; return 0; fi
    sleep 20
  done
  log "ERROR: default never became $1"; return 1
}

pi_mission() {  # $1=arm $2=mission-id $3=workdir $4=prompt
  local arm=$1 mid=$2 wd=$3 prompt=$4 t0 t1 rc
  log "--- [$arm] pi mission: $mid"
  t0=$(date +%s)
  ( cd "$wd" && timeout "$PI_TIMEOUT" pi -p "$prompt" > "$OUT/$arm-$mid-pi.txt" 2>&1 )
  rc=$?
  t1=$(date +%s)
  log "    [$arm] $mid: pi rc=$rc wall=$((t1-t0))s"
  echo "$rc $((t1-t0))"
}

run_arm() {  # $1 = arm label
  local arm=$1
  log "=== ARM: $arm ==="
  local armdir="$OUT/$arm"; mkdir -p "$armdir"

  # -- agentic missions in a fresh RealtorZero clone --
  local clone="$armdir/RealtorZero"
  rm -rf "$clone"
  git clone -q "$HOME/src/github.com/RealtorZero" "$clone"
  python3 "$REPO/patches/pi-trial-verify.py" plant-a "$clone" >> "$LOG"

  read -r rc wall <<< "$(pi_mission "$arm" agentic-bugfix "$clone" \
    "You are working ONLY inside this directory (a scratch clone — safe to modify, never leave it). Bug report from production: find_comps() wrongly EXCLUDES comps whose bed count is unknown (beds is NULL, which is every Maricopa county record) whenever min_beds or max_beds is set. Per the design, unknown bed counts must PASS bed filters. Find the bug and fix it minimally.")"
  v=$(python3 "$REPO/patches/pi-trial-verify.py" verify-a "$clone" 2>>"$LOG")
  rec "{\"arm\":\"$arm\",\"mission\":\"agentic-bugfix\",\"wall_s\":$wall,\"pi_rc\":$rc,\"verify\":$v}"
  log "    verify-a: $v"

  read -r rc wall <<< "$(pi_mission "$arm" agentic-feature "$clone" \
    "Still ONLY inside this directory. Add a function median_ppsf(comps) to src/realtorzero/tools/comps.py: takes a list of dicts with sale_price and sqft keys, returns the median price-per-sqft as a float, ignoring comps whose sqft is missing, None, or zero. Also add a pytest test file covering it (normal case + ignore rules). Run your test to confirm it passes before finishing.")"
  v=$(python3 "$REPO/patches/pi-trial-verify.py" verify-b "$clone" 2>>"$LOG")
  rec "{\"arm\":\"$arm\",\"mission\":\"agentic-feature\",\"wall_s\":$wall,\"pi_rc\":$rc,\"verify\":$v}"
  log "    verify-b: $v"

  read -r rc wall <<< "$(pi_mission "$arm" agentic-research "$clone" \
    "Still ONLY inside this directory. Use your web search capability to find this week's average 30-year fixed US mortgage rate (Freddie Mac PMMS is the canonical source). Create MARKET_NOTES.md in the repo root containing: the rate as a percentage, the survey date, and the source URL.")"
  v=$(python3 "$REPO/patches/pi-trial-verify.py" verify-c "$clone" 2>>"$LOG")
  rec "{\"arm\":\"$arm\",\"mission\":\"agentic-research\",\"verify\":$v,\"wall_s\":$wall,\"pi_rc\":$rc}"
  log "    verify-c: $v"

  git -C "$clone" diff > "$armdir/agentic.diff" 2>/dev/null

  # -- hard problems: pi writes solution.py, we verify in the bwrap sandbox --
  local tasks="real-hard-valuation wis-schedule regex-lite expr-eval edit-script lru-ttl real-fix-ports real-fix-comps"
  for tid in $tasks; do
    if past_deadline; then log "DEADLINE inside arm $arm — skipping remaining tasks"; break; fi
    local ws="$armdir/prob-$tid"; mkdir -p "$ws"
    local prompt
    prompt=$(python3 - "$tid" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ab", "patches/qwen-quant-ab.py")
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
t = next(t for t in ab.TASKS if t["id"] == sys.argv[1])
p = t["prompt"].replace("Reply with ONLY a python code block.", "").replace("Answer with ONLY a python code block defining", "Write a file defining")
print(p + " Work ONLY in the current directory. Write your final solution as a single self-contained python file named solution.py (module-level definitions exactly as specified). You may create and run scratch/test files here to verify your solution before finishing — you are graded only on solution.py passing hidden tests.")
PYEOF
)
    read -r rc wall <<< "$(pi_mission "$arm" "prob-$tid" "$ws" "$prompt")"
    v=$(cd "$REPO" && python3 patches/qwen-quant-ab.py --check-solution "$tid" "$ws/solution.py" 2>>"$LOG")
    [ -z "$v" ] && v='{"pass": false, "error": "check crashed"}'
    rec "{\"arm\":\"$arm\",\"mission\":\"prob-$tid\",\"wall_s\":$wall,\"pi_rc\":$rc,\"verify\":$v}"
    log "    check[$tid]: $v"
  done
  log "=== ARM $arm complete ==="
}

swap_worker_to_fp8() {
  log "=== swapping worker1 qwen: NVFP4 -> official FP8 ==="
  python3 - <<'PYEOF'
import re
p = "models.yaml"
s = open(p).read()
anchor = "      host: worker1\n      memory_gb: 90"
assert anchor in s
s = s.replace(anchor, "      name: \"Qwen/Qwen3.8-27B-FP8\"\n      quantization: \"none\"\n" + anchor, 1)
open(p, "w").write(s)
PYEOF
  sparkstation models stop qwen3.8-27b >> "$LOG" 2>&1
  sleep 10
  sparkstation models start qwen3.8-27b --profile image-indexing >> "$LOG" 2>&1 || true  # profile flag REQUIRED: without it the alias resolves through default_profile (coding) and boots NVFP4 — cost us the fp8 arm on night 1
}

cd "$REPO"
log "======== NIGHT TRIAL START ========"
log "output: $OUT"

if [ "${SKIP_TO_NVFP4:-0}" != "1" ]; then
  # ARM 1: dsv4 — coding profile is already live
  wait_default_serves "deepseek" || exit 1
  run_arm dsv4
fi

# ARM 2: qwen NVFP4 — switch profiles (skipped if already switched)
if ! past_deadline; then
  if [ "${SKIP_TO_NVFP4:-0}" != "1" ]; then
    log "=== switching to image-indexing profile (qwen NVFP4) ==="
    sparkstation stop >> "$LOG" 2>&1
    sleep 5
    sparkstation start -d --profile image-indexing >> "$LOG" 2>&1
  fi
  wait_default_serves "Qwen3.8-27B-NVFP4" && run_arm nvfp4
fi

# ARM 3: qwen FP8 — swap just the worker model
if ! past_deadline; then
  swap_worker_to_fp8
  wait_default_serves "Qwen3.8-27B-FP8" && run_arm fp8
fi

restore

# morning report
python3 - "$RESULTS" "$OUT/MORNING-REPORT.md" <<'PYEOF'
import json, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
arms = defaultdict(list)
for r in rows: arms[r["arm"]].append(r)
out = ["# Overnight daily-driver trial — three arms through pi\n"]
out.append("| mission | " + " | ".join(arms) + " |")
out.append("|---|" + "---|" * len(arms))
missions = sorted({r["mission"] for r in rows}, key=lambda m: (not m.startswith("agentic"), m))
for m in missions:
    cells = []
    for a in arms:
        match = [r for r in arms[a] if r["mission"] == m]
        if not match: cells.append("—")
        else:
            r = match[0]
            cells.append(("✅" if r["verify"].get("pass") else "❌") + f" {r['wall_s']}s")
    out.append(f"| {m} | " + " | ".join(cells) + " |")
out.append("")
for a in arms:
    n = len(arms[a]); p = sum(1 for r in arms[a] if r["verify"].get("pass"))
    tot = sum(r["wall_s"] for r in arms[a])
    out.append(f"**{a}**: {p}/{n} missions passed, total wall {tot//60}min")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("\n".join(out))
PYEOF
log "======== NIGHT TRIAL DONE — report: $OUT/MORNING-REPORT.md ========"
