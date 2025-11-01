#!/bin/bash
#
# Verify Sparkstation backend installations (Docker-based)
#
# This script checks that SGLang Docker image is properly installed
# and can access CUDA GPUs.
#
# Usage:
#   ./scripts/verify_backends.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Sparkstation Backend Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Source .env file to get configuration
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    # Export variables from .env
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo -e "${GREEN}✓ Loaded configuration from .env${NC}"
else
    echo -e "${RED}✗ .env file not found!${NC}"
    echo "Run ./scripts/setup_backends.sh first"
    exit 1
fi

echo ""

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    PLATFORM="linux/arm64"
elif [ "$ARCH" = "x86_64" ]; then
    PLATFORM="linux/amd64"
else
    echo -e "${RED}✗ Unsupported architecture: $ARCH${NC}"
    exit 1
fi

# Check Docker mode
if [ "${USE_DOCKER:-true}" = "true" ]; then
    echo -e "${YELLOW}Checking Docker-based backend...${NC}"
    echo ""

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker not found!${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker available${NC}"

    # Check SGLang image
    SGLANG_IMAGE="${SGLANG_DOCKER_IMAGE:-lmsysorg/sglang:latest}"
    echo -e "${YELLOW}Checking SGLang Docker image: $SGLANG_IMAGE${NC}"

    if ! docker image inspect "$SGLANG_IMAGE" &> /dev/null; then
        echo -e "${RED}✗ SGLang image not found locally!${NC}"
        echo "  Run: docker pull --platform $PLATFORM $SGLANG_IMAGE"
        exit 1
    fi
    echo -e "${GREEN}✓ SGLang image found${NC}"

    # Test PyTorch CUDA
    echo -e "${YELLOW}Testing PyTorch CUDA in SGLang container...${NC}"
    CUDA_OUTPUT=$(docker run --platform "$PLATFORM" --gpus all --rm "$SGLANG_IMAGE" \
        python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1)

    echo "$CUDA_OUTPUT"

    if echo "$CUDA_OUTPUT" | grep -q "CUDA available: True"; then
        echo -e "${GREEN}✓ CUDA works in SGLang Docker container${NC}"
    else
        echo -e "${RED}✗ CUDA not available in SGLang Docker container${NC}"
        exit 1
    fi

    # Test SGLang import
    echo ""
    echo -e "${YELLOW}Testing SGLang import...${NC}"
    SGLANG_VERSION=$(docker run --platform "$PLATFORM" --rm "$SGLANG_IMAGE" \
        python -c "import sglang; print(sglang.__version__)" 2>&1 | tail -1)

    if [ -n "$SGLANG_VERSION" ]; then
        echo -e "${GREEN}✓ SGLang $SGLANG_VERSION${NC}"
    else
        echo -e "${RED}✗ Failed to import sglang${NC}"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}✓ SGLang Docker backend verified${NC}"

else
    # Subprocess mode (conda/micromamba)
    echo -e "${YELLOW}Checking subprocess-based backend (conda/micromamba)...${NC}"

    if [ -n "$SGLANG_PYTHON_PATH" ] && [ -f "$SGLANG_PYTHON_PATH" ]; then
        echo "  Python path: $SGLANG_PYTHON_PATH"

        # Check Python version
        PYTHON_VERSION=$("$SGLANG_PYTHON_PATH" --version 2>&1)
        echo "  $PYTHON_VERSION"

        # Check SGLang import
        if "$SGLANG_PYTHON_PATH" -c "import sglang" 2>/dev/null; then
            SGLANG_VERSION=$("$SGLANG_PYTHON_PATH" -c "import sglang; print(sglang.__version__)")
            echo -e "  ${GREEN}✓ SGLang $SGLANG_VERSION${NC}"
        else
            echo -e "  ${RED}✗ Failed to import sglang${NC}"
            exit 1
        fi

        # Check PyTorch CUDA
        if "$SGLANG_PYTHON_PATH" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            TORCH_VERSION=$("$SGLANG_PYTHON_PATH" -c "import torch; print(torch.__version__)")
            CUDA_VERSION=$("$SGLANG_PYTHON_PATH" -c "import torch; print(torch.version.cuda)")
            GPU_COUNT=$("$SGLANG_PYTHON_PATH" -c "import torch; print(torch.cuda.device_count())")
            GPU_NAME=$("$SGLANG_PYTHON_PATH" -c "import torch; print(torch.cuda.get_device_name(0))")
            echo -e "  ${GREEN}✓ PyTorch $TORCH_VERSION (CUDA $CUDA_VERSION)${NC}"
            echo -e "  ${GREEN}✓ GPUs: $GPU_COUNT x $GPU_NAME${NC}"
        else
            echo -e "  ${RED}✗ CUDA not available in PyTorch${NC}"
            exit 1
        fi

        echo -e "${GREEN}✓ SGLang subprocess backend verified${NC}"
    else
        echo -e "${YELLOW}! SGLANG_PYTHON_PATH not set or not found${NC}"
        echo "  Expected: SGLANG_PYTHON_PATH in .env"
        echo "  See DEPLOYMENT_PRODUCTION.md for conda/micromamba setup"
    fi
fi

echo ""
echo -e "${YELLOW}vLLM backend: Not installed yet (to be added later)${NC}"
echo ""

# Check Sparkstation
echo -e "${YELLOW}Checking Sparkstation installation...${NC}"
if command -v uv &> /dev/null; then
    echo -e "  ${GREEN}✓ uv found${NC}"

    # Check if venv exists
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "  ${GREEN}✓ Virtual environment exists${NC}"

        # Check FastAPI import
        if "$PROJECT_ROOT/.venv/bin/python" -c "import fastapi" 2>/dev/null; then
            echo -e "  ${GREEN}✓ FastAPI installed${NC}"
        else
            echo -e "  ${YELLOW}! FastAPI not found, run: uv sync${NC}"
        fi
    else
        echo -e "  ${YELLOW}! Virtual environment not found, run: uv sync${NC}"
    fi
else
    echo -e "  ${RED}✗ uv not found${NC}"
    echo "  Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Verification Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ "${USE_DOCKER:-true}" = "true" ]; then
    echo "SGLang Docker backend is properly installed and CUDA is available."
else
    echo "SGLang subprocess backend is properly installed and CUDA is available."
fi

echo ""
echo -e "${BLUE}You can now:${NC}"
echo ""
echo "1. Start the Sparkstation supervisor:"
echo "   ./scripts/start_supervisor.sh"
echo ""
echo "2. Start a model via Sparkstation API"
echo "   # See README.md for examples"
echo ""
echo -e "${YELLOW}Note: vLLM support will be added in a future update${NC}"
echo ""
