#!/bin/bash
#
# Setup vLLM backend environment for Sparkstation
#
# This script creates a separate micromamba/conda environment for vLLM
# with all required dependencies including CUDA support.
#
# Usage:
#   ./scripts/setup_vllm_env.sh [install_path]
#
# Default install_path: /opt/backends/vllm
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default installation path
INSTALL_PATH="${1:-/opt/backends/vllm}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}vLLM Backend Environment Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Installation path: $INSTALL_PATH"
echo ""

# Check for micromamba or conda
if command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
    echo -e "${GREEN}✓ Found micromamba${NC}"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo -e "${YELLOW}! Using conda (micromamba recommended for faster installs)${NC}"
else
    echo -e "${RED}✗ Neither micromamba nor conda found!${NC}"
    echo ""
    echo "Install micromamba (recommended):"
    echo "  curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba"
    echo "  sudo mv bin/micromamba /usr/local/bin/"
    echo ""
    echo "Or install conda/miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check CUDA version
echo ""
echo -e "${YELLOW}Checking CUDA version...${NC}"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    echo -e "${GREEN}✓ CUDA Version: $CUDA_VERSION${NC}"

    # Determine PyTorch CUDA wheel version
    if [[ "$CUDA_VERSION" == 13.* ]]; then
        # CUDA 13.x (Grace Blackwell) - use cu124 wheels
        TORCH_CUDA="cu124"
        echo -e "${GREEN}✓ CUDA 13.x detected (using cu124 wheels)${NC}"
    elif [[ "$CUDA_VERSION" == 12.4* ]] || [[ "$CUDA_VERSION" == 12.5* ]]; then
        TORCH_CUDA="cu124"
    elif [[ "$CUDA_VERSION" == 12.6* ]]; then
        TORCH_CUDA="cu126"
    elif [[ "$CUDA_VERSION" == 12.1* ]] || [[ "$CUDA_VERSION" == 12.2* ]] || [[ "$CUDA_VERSION" == 12.3* ]]; then
        TORCH_CUDA="cu121"
    else
        echo -e "${YELLOW}! Unsupported CUDA version. Defaulting to cu124${NC}"
        TORCH_CUDA="cu124"
    fi
    echo -e "${GREEN}✓ Using PyTorch wheel: $TORCH_CUDA${NC}"
else
    echo -e "${RED}✗ nvidia-smi not found. Is NVIDIA driver installed?${NC}"
    exit 1
fi

# Create installation directory
echo ""
echo -e "${YELLOW}Creating installation directory...${NC}"
if [[ "$INSTALL_PATH" == /opt/* ]]; then
    # Need sudo for /opt
    sudo mkdir -p "$INSTALL_PATH"
    sudo chown $USER:$USER "$INSTALL_PATH"
else
    mkdir -p "$INSTALL_PATH"
fi
echo -e "${GREEN}✓ Directory created: $INSTALL_PATH${NC}"

# Create environment
echo ""
echo -e "${YELLOW}Creating conda environment (this may take a few minutes)...${NC}"
if [ "$CONDA_CMD" = "micromamba" ]; then
    micromamba create -y -p "$INSTALL_PATH" -c conda-forge python=3.11
else
    conda create -y -p "$INSTALL_PATH" python=3.11
fi
echo -e "${GREEN}✓ Python 3.11 environment created${NC}"

# Install PyTorch
echo ""
echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
if [ "$CONDA_CMD" = "micromamba" ]; then
    micromamba run -p "$INSTALL_PATH" pip install "torch==2.4.*" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
else
    conda run -p "$INSTALL_PATH" pip install "torch==2.4.*" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
fi
echo -e "${GREEN}✓ PyTorch 2.4 installed${NC}"

# Install vLLM
echo ""
echo -e "${YELLOW}Installing vLLM...${NC}"
if [ "$CONDA_CMD" = "micromamba" ]; then
    micromamba run -p "$INSTALL_PATH" pip install "vllm==0.6.*"
else
    conda run -p "$INSTALL_PATH" pip install "vllm==0.6.*"
fi
echo -e "${GREEN}✓ vLLM installed${NC}"

# Verify installation
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"
if [ "$CONDA_CMD" = "micromamba" ]; then
    VLLM_VERSION=$(micromamba run -p "$INSTALL_PATH" python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    TORCH_VERSION=$(micromamba run -p "$INSTALL_PATH" python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    CUDA_AVAILABLE=$(micromamba run -p "$INSTALL_PATH" python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
else
    VLLM_VERSION=$(conda run -p "$INSTALL_PATH" python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    TORCH_VERSION=$(conda run -p "$INSTALL_PATH" python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    CUDA_AVAILABLE=$(conda run -p "$INSTALL_PATH" python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Environment details:"
echo "  - Location: $INSTALL_PATH"
echo "  - Python interpreter: $INSTALL_PATH/bin/python"
echo "  - vLLM version: $VLLM_VERSION"
echo "  - PyTorch version: $TORCH_VERSION"
echo "  - CUDA available: $CUDA_AVAILABLE"
echo ""

if [ "$CUDA_AVAILABLE" = "False" ]; then
    echo -e "${RED}⚠ WARNING: CUDA not available in PyTorch!${NC}"
    echo "This likely means CUDA version mismatch. Try reinstalling PyTorch with correct CUDA version."
    exit 1
fi

echo -e "${GREEN}✓ vLLM backend ready to use${NC}"
echo ""
echo "To use this environment in Sparkstation, add to .env:"
echo "  VLLM_PYTHON_PATH=$INSTALL_PATH/bin/python"
echo ""
echo "To test the installation:"
echo "  $INSTALL_PATH/bin/python -m vllm.entrypoints.openai.api_server --help"
echo ""
