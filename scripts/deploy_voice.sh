#!/usr/bin/env bash
# Deploy voicecascade/ from this checkout to the voice host, restart the
# managed stack, wait for readiness, optionally run the latency bench.
#   scripts/deploy_voice.sh [--bench N]
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${VOICE_HOST:-kshetrajna@192.168.101.12}
scp -q voicecascade/*.py scripts/cascade_ws_client.py scripts/latency_bench.sh "$HOST":cascade-bot/tmp_deploy/ 2>/dev/null || {
  ssh "$HOST" 'mkdir -p ~/cascade-bot/tmp_deploy'; scp -q voicecascade/*.py scripts/cascade_ws_client.py scripts/latency_bench.sh "$HOST":cascade-bot/tmp_deploy/; }
ssh "$HOST" 'cd ~/cascade-bot && mv tmp_deploy/cascade_ws_client.py tmp_deploy/latency_bench.sh scripts/ && mv tmp_deploy/*.py voicecascade/ && chmod +x scripts/latency_bench.sh && echo "files deployed"'
sparkstation models stop voicecascade >/dev/null 2>&1 || true
sparkstation models start voicecascade -p voice | tail -1
# readiness: the managed start already polls :7860, but STT preload logs last
for _ in $(seq 40); do ssh "$HOST" 'grep -aq "KyutaiSTT ready" /tmp/cascade-bot.log' 2>/dev/null && break; sleep 3; done
if [ "${1:-}" = "--bench" ]; then ssh "$HOST" "~/cascade-bot/scripts/latency_bench.sh ${2:-3}"; fi
