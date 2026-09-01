#!/usr/bin/env bash
# Blue-green LiteLLM manager — ZERO-DOWNTIME config reloads.
#
# LiteLLM only reads litellm.yaml at startup (1.79+ removed /config/reload).
# The old litellm-watch.sh restarted litellm IN PLACE on every config change,
# so the public API dropped for ~2s per model coming up during a profile
# bring-up — clients saw "All connection attempts failed" 502s (see the
# 2026-08-21 deep-profile churn).
#
# This manager instead runs litellm on one of two ports (blue/green). On a
# config change it brings the NEW litellm up on the IDLE port, health-checks
# it, then ATOMICALLY flips a pointer file that the :8000 proxy follows — new
# requests go to the new litellm, in-flight requests drain against the old one,
# and only then is the old litellm retired. The public port never sees a gap.
#
# Two robustness features on top:
#   - DEBOUNCE: a profile bring-up rewrites the config once per model as each
#     comes up. We wait for the config to be STABLE for DEBOUNCE seconds before
#     swapping, so N model transitions become ONE swap, not N.
#   - Liveness: if the active litellm dies, relaunch it in place.
#
# Usage: litellm-bluegreen.sh <blue-port> <green-port> <pointer-file>
# Env:   LITELLM_PYTHON, LITELLM_DRAIN_GRACE, LITELLM_READY_TIMEOUT, LITELLM_DEBOUNCE
set -u
CONFIG="gateway/litellm.yaml"
PORT_BLUE="${1:?usage: litellm-bluegreen.sh <blue-port> <green-port> <pointer-file>}"
PORT_GREEN="${2:?}"
POINTER="${3:?}"
PY="${LITELLM_PYTHON:-python3}"
GRACE="${LITELLM_DRAIN_GRACE:-300}"          # let old drain this long before kill (covers long generations)
READY_TIMEOUT="${LITELLM_READY_TIMEOUT:-90}" # max secs to wait for a new litellm to become ready
DEBOUNCE="${LITELLM_DEBOUNCE:-6}"            # config must be stable this long before we swap

cfg_sum() { md5sum "$CONFIG" 2>/dev/null | cut -d' ' -f1; }

start_litellm() { # <port> -> prints pid
  "$PY" -m litellm.proxy.proxy_cli --config "$CONFIG" --host 127.0.0.1 --port "$1" \
    >>"gateway/.litellm-$1.log" 2>&1 &
  echo $!
}

wait_ready() { # <port> <pid> -> 0 if it becomes ready, 1 if it dies or times out
  for _i in $(seq 1 "$READY_TIMEOUT"); do
    if curl -fsS "http://127.0.0.1:$1/health/readiness" >/dev/null 2>&1; then return 0; fi
    kill -0 "$2" 2>/dev/null || return 1   # died during boot
    sleep 1
  done
  return 1
}

# Graceful stop with SIGKILL escalation (uvicorn waits forever on open streams).
stop_litellm() { # <pid>
  local pid="$1"; [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null
  for _i in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$pid" 2>/dev/null
}

write_pointer() { printf '%s' "$1" > "$POINTER.tmp" && mv "$POINTER.tmp" "$POINTER"; }

# ── Initial boot ────────────────────────────────────────────────────────────
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&-; return 0; } || return 1; }
# Start on the first FREE slot (a draining litellm from a killed manager may
# still hold blue), and publish the pointer only once it answers readiness —
# the proxy must never be pointed at a port nothing listens on.
boot_litellm() { # sets active_port/active_pid, returns 0 when ready
  local try
  for try in 1 2; do
    for p in "$PORT_BLUE" "$PORT_GREEN"; do
      port_busy "$p" && { echo "[bluegreen] port $p busy; trying the other slot"; continue; }
      active_port="$p"; active_pid="$(start_litellm "$p")"
      if wait_ready "$p" "$active_pid"; then write_pointer "$p"; return 0; fi
      echo "[bluegreen] litellm on $p failed readiness; retrying"; stop_litellm "$active_pid"
    done
    sleep 3
  done
  return 1
}
if ! boot_litellm; then
  echo "[bluegreen] FATAL: could not bring up litellm on $PORT_BLUE or $PORT_GREEN"; exit 1
fi
active_sum="$(cfg_sum)"
echo "[bluegreen] litellm up on $active_port pid=$active_pid (config md5=${active_sum:-none})"

drain_and_kill() { # <old_pid> — background: let in-flight drain, then stop
  ( sleep "$GRACE"; stop_litellm "$1"; echo "[bluegreen] retired old litellm pid=$1" ) &
}

on_term() { stop_litellm "$active_pid"; exit 0; }
trap on_term TERM INT

pending_sum=""; pending_since=0
while true; do
  sleep 2

  # Liveness: relaunch the active litellm in place if it died.
  if ! kill -0 "$active_pid" 2>/dev/null; then
    echo "[bluegreen] active litellm ($active_port) died; relaunching"
    if ! boot_litellm; then echo "[bluegreen] relaunch failed on both slots; will retry"; sleep 5; continue; fi
    active_sum="$(cfg_sum)"; pending_sum=""
    continue
  fi

  cur="$(cfg_sum)"
  [ "$cur" = "$active_sum" ] && { pending_sum=""; continue; }   # no change vs live config

  # Config differs — debounce until it's been stable for DEBOUNCE seconds.
  now=$(date +%s)
  if [ "$cur" != "$pending_sum" ]; then
    pending_sum="$cur"; pending_since="$now"; continue          # changed again; reset timer
  fi
  [ $(( now - pending_since )) -lt "$DEBOUNCE" ] && continue     # not stable yet

  # ── Blue-green swap ─────────────────────────────────────────────────────
  if [ "$active_port" = "$PORT_BLUE" ]; then idle_port="$PORT_GREEN"; else idle_port="$PORT_BLUE"; fi
  echo "[bluegreen] config stable; bringing up new litellm on idle $idle_port"
  new_pid="$(start_litellm "$idle_port")"
  if wait_ready "$idle_port" "$new_pid"; then
    write_pointer "$idle_port"                                   # ATOMIC flip — proxy follows this
    echo "[bluegreen] flipped -> $idle_port; draining old $active_port (pid=$active_pid) for ${GRACE}s"
    drain_and_kill "$active_pid"
    active_port="$idle_port"; active_pid="$new_pid"; active_sum="$cur"; pending_sum=""
  else
    echo "[bluegreen] new litellm on $idle_port failed readiness in ${READY_TIMEOUT}s; keeping $active_port"
    stop_litellm "$new_pid"
    active_sum="$cur"; pending_sum=""   # don't hot-loop on a bad config; wait for the next change
  fi
done
