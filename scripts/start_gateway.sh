#!/bin/bash
# Start LiteLLM Gateway

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Load environment variables (filter out comments and empty lines)
if [ -f .env ]; then
    set -a
    source <(grep -v '^#' .env | grep -v '^$' | sed 's/#.*//')
    set +a
fi

# Activate virtual environment if using uv
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Start LiteLLM
echo "Starting LiteLLM Gateway on ${LITELLM_HOST:-127.0.0.1}:8000..."

uv run litellm \
    --config gateway/litellm.yaml \
    --host "${LITELLM_HOST:-127.0.0.1}" \
    --port 8000 \
    --detailed_debug \
    "$@"
