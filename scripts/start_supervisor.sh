#!/bin/bash
# Start Sparkstation Supervisor

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate virtual environment if using uv
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Ensure data directory exists
mkdir -p data

# Start supervisor
echo "Starting Sparkstation Supervisor on ${HOST:-127.0.0.1}:${PORT:-9001}..."

uv run uvicorn supervisor.main:app \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-9001}" \
    --log-level "${LOG_LEVEL:-info}" \
    "$@"
