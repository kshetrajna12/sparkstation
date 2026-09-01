#!/usr/bin/env bash
# Voice latency bench (run ON worker2): N turns each of fast / think / tool.
# Prints speech-end -> first-audio per turn and the median per lane.
N=${1:-3}
PY=$HOME/cascade-bot/.venv/bin/python; C=$HOME/cascade-bot/scripts/cascade_ws_client.py
SYS="You are Sparky, a friendly home robot. You have no internet access and no knowledge of live events or the user data yourself. For news, weather, live information or anything about the user, you MUST call the openclaw_agent_consult tool. Keep spoken answers to one or two short sentences."
median() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "n/a"} else if(NR%2){print a[(NR+1)/2]} else {print (a[NR/2]+a[NR/2+1])/2} }'; }
declare -A LAT
for lane in fast think tool; do
  vals=""
  for i in $(seq "$N"); do
    case $lane in
      fast)  out=$($PY $C 127.0.0.1 22 /tmp/eggs16k.wav 2>&1);;
      think) out=$($PY $C 127.0.0.1 26 /tmp/think16k.wav 2>&1);;
      tool)  out=$(CONFIGURE='{"brain":"default"}' SYS_PROMPT="$SYS" $PY $C 127.0.0.1 26 /tmp/news16k.wav /tmp/tools_openclaw.json 2>&1);;
    esac
    lat=$(echo "$out" | grep -oE "LATENCY[^:]*: [0-9.]+" | grep -oE "[0-9.]+$")
    tc=$(echo "$out" | grep -c "TOOL CALL:")
    said=$(echo "$out" | grep -m1 "BOT SAID" | cut -c13-70)
    echo "$lane #$i  latency=${lat:-none}s  toolcalls=$tc  said='$said'"
    [ -n "$lat" ] && vals="$vals$lat"$'\n'
    sleep 2
  done
  LAT[$lane]=$(printf "%s" "$vals" | median)
done
echo "MEDIANS  fast=${LAT[fast]}s  think=${LAT[think]}s  tool=${LAT[tool]}s"
