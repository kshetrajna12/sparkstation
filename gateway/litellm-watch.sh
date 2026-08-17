#!/usr/bin/env bash
# Runs LiteLLM under a config watcher.
#
# Why: LiteLLM only reads gateway/litellm.yaml at startup (1.79+ removed
# /config/reload, and /model/new needs a DB we don't run). The supervisor's
# gateway_sync rewrites the yaml as models come up/down — this wrapper
# restarts LiteLLM whenever the file's content actually changes, so models
# appear in the gateway incrementally during background autoload without
# anyone blocking on "all models loaded". Also restarts LiteLLM if it dies.
#
# Usage: litellm-watch.sh <port>   (cwd must be the project root)
# Env:   LITELLM_PYTHON — python interpreter to use (default: python3)
set -u
CONFIG="gateway/litellm.yaml"
PORT="${1:?usage: litellm-watch.sh <port>}"
PY="${LITELLM_PYTHON:-python3}"

child=""
on_term() {
  if [ -n "$child" ]; then
    kill "$child" 2>/dev/null
    wait "$child" 2>/dev/null
  fi
  exit 0
}
trap on_term TERM INT

while true; do
  sum=$(md5sum "$CONFIG" 2>/dev/null | cut -d' ' -f1)
  "$PY" -m litellm.proxy.proxy_cli \
    --config "$CONFIG" --host 127.0.0.1 --port "$PORT" &
  child=$!
  echo "[litellm-watch] started LiteLLM pid=$child (config md5=${sum:-none})"

  while kill -0 "$child" 2>/dev/null; do
    sleep 5
    new=$(md5sum "$CONFIG" 2>/dev/null | cut -d' ' -f1)
    if [ "$new" != "$sum" ]; then
      echo "[litellm-watch] config changed; restarting LiteLLM"
      kill "$child" 2>/dev/null
      # Escalate: uvicorn's graceful shutdown waits INDEFINITELY for open
      # connections — a client holding a streaming response wedged the
      # gateway for good on 2026-08-16. Give it 10s, then SIGKILL.
      for _i in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$child" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$child" 2>/dev/null; then
        echo "[litellm-watch] graceful shutdown stuck; SIGKILL"
        kill -9 "$child" 2>/dev/null
      fi
      wait "$child" 2>/dev/null
      break
    fi
  done
  wait "$child" 2>/dev/null
  child=""
  sleep 1
done
