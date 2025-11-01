#!/bin/bash
#
# Master setup script for Sparkstation backends (Docker-based)
#
# This script pulls Docker images for SGLang and vLLM backends
# and configures Sparkstation to use them.
#
# Usage:
#   ./scripts/setup_backends.sh
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
echo -e "${BLUE}Sparkstation Backend Setup (Docker)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check for docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found!${NC}"
    echo ""
    echo "Docker is required for Sparkstation backends."
    echo "Install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Check for nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found!${NC}"
    echo "NVIDIA GPU driver is required."
    exit 1
fi
echo -e "${GREEN}✓ NVIDIA driver found${NC}"

# Check for NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA Container Toolkit not working!${NC}"
    echo ""
    echo "Install NVIDIA Container Toolkit:"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    exit 1
fi
echo -e "${GREEN}✓ NVIDIA Container Toolkit working${NC}"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo -e "${GREEN}✓ Detected ARM64/aarch64 architecture (DGX Spark)${NC}"
    PLATFORM="linux/arm64"
elif [ "$ARCH" = "x86_64" ]; then
    echo -e "${GREEN}✓ Detected x86_64 architecture${NC}"
    PLATFORM="linux/amd64"
else
    echo -e "${RED}✗ Unsupported architecture: $ARCH${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pulling Backend Docker Images${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Pull vLLM image
echo -e "${YELLOW}Pulling vLLM Docker image (this may take a few minutes)...${NC}"
docker pull --platform "$PLATFORM" nvcr.io/nvidia/vllm:25.10-py3
echo -e "${GREEN}✓ vLLM image pulled${NC}"

echo ""

# Verify vLLM image works with CUDA
echo -e "${YELLOW}Verifying vLLM image with CUDA...${NC}"
CUDA_TEST=$(docker run --platform "$PLATFORM" --gpus all --rm nvcr.io/nvidia/vllm:25.10-py3 \
    python -c "import torch; print('CUDA:', torch.cuda.is_available())" 2>&1 | grep "CUDA: True" || echo "")

if [ -z "$CUDA_TEST" ]; then
    echo -e "${RED}✗ CUDA not available in vLLM Docker image!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CUDA works in vLLM Docker image${NC}"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Configuring Sparkstation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Update .env file
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env from .env.example..."
    cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
fi

# Function to update or add env var
update_env_var() {
    local key=$1
    local value=$2
    local file=$3

    if grep -q "^${key}=" "$file"; then
        # Update existing line
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^${key}=.*|${key}=${value}|" "$file"
        else
            sed -i "s|^${key}=.*|${key}=${value}|" "$file"
        fi
    elif grep -q "^# ${key}=" "$file"; then
        # Uncomment and update
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^# ${key}=.*|${key}=${value}|" "$file"
        else
            sed -i "s|^# ${key}=.*|${key}=${value}|" "$file"
        fi
    else
        # Add new line
        echo "${key}=${value}" >> "$file"
    fi
}

update_env_var "USE_DOCKER" "true" "$ENV_FILE"
update_env_var "VLLM_DOCKER_IMAGE" "nvcr.io/nvidia/vllm:25.10-py3" "$ENV_FILE"

echo -e "${GREEN}✓ Configuration updated${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Backend Docker image installed:"
echo "  - vLLM: nvcr.io/nvidia/vllm:25.10-py3 ($PLATFORM)"
echo ""
echo "Configuration file updated:"
echo "  - $ENV_FILE"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo "1. Verify backends are working:"
echo "   ./scripts/verify_backends.sh"
echo ""
echo "2. Start Sparkstation supervisor:"
echo "   ./scripts/start_supervisor.sh"
echo ""
echo "3. Start LiteLLM gateway:"
echo "   ./scripts/start_gateway.sh"
echo ""
echo "4. Start a model (see README.md for examples)"
echo ""
