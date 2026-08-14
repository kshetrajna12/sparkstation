#!/usr/bin/env bash
# Submit our Qwen3.8-27B DGX Spark benchmarks to localmaxxing.
# Reads your API key from $LOCALMAXXING_KEY or prompts for it (never echoed, never logged).
set -euo pipefail
JSON="${1:?path to submissions json}"

if [ -z "${LOCALMAXXING_KEY:-}" ]; then
  read -rs -p "localmaxxing API key (bhk_...): " LOCALMAXXING_KEY; echo
fi

count=$(python3 -c "import json; print(len(json.load(open('$JSON'))['submissions']))")
for i in $(seq 0 $((count-1))); do
  payload=$(python3 -c "import json; print(json.dumps(json.load(open('$JSON'))['submissions'][$i]))")
  echo "Submitting entry $((i+1))/$count..."
  curl -sf -X POST "https://www.localmaxxing.com/api/speed-tests" \
    -H "Authorization: Bearer $LOCALMAXXING_KEY" \
    -H "Content-Type: application/json" \
    -d "$payload" && echo " ✓ accepted" || echo " ✗ FAILED (check key / fields)"
  [ "$i" -lt $((count-1)) ] && { echo "waiting 65s (rate limit)..."; sleep 65; }
done
echo "Done — entries appear after moderation at localmaxxing.com/en/models/Qwen/Qwen3.8-27B"
