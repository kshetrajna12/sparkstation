#!/bin/bash
#
# Setup SGLang backend environment for Sparkstation
#
# This script creates a separate micromamba/conda environment for SGLang
# with all required dependencies including CUDA support.
#
# Usage:
#   ./scripts/setup_sglang_env.sh [install_path]
#
# Default install_path: /opt/backends/sglang
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default installation path
INSTALL_PATH="${1:-/opt/backends/sglang}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SGLang Backend Environment Setup${NC}"
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

# Activate function
activate_env() {
    if [ "$CONDA_CMD" = "micromamba" ]; then
        eval "$(micromamba shell hook --shell bash)"
        micromamba activate "$INSTALL_PATH"
    else
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$INSTALL_PATH"
    fi
}

# Install SGLang with Blackwell ARM64 support
echo ""
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo -e "${YELLOW}Installing SGLang with official Blackwell ARM64 support...${NC}"
    echo -e "${YELLOW}This will install PyTorch and all CUDA dependencies automatically${NC}"

    if [ "$CONDA_CMD" = "micromamba" ]; then
        # Use official blackwell-aarch64 extra for DGX Spark / Grace Blackwell
        micromamba run -p "$INSTALL_PATH" pip install "sglang[blackwell-aarch64]>=0.5.0"
    else
        conda run -p "$INSTALL_PATH" pip install "sglang[blackwell-aarch64]>=0.5.0"
    fi
    echo -e "${GREEN}✓ SGLang with Blackwell ARM64 support installed${NC}"
else
    echo -e "${YELLOW}Installing SGLang for x86_64...${NC}"

    # x86_64 - install PyTorch first, then SGLang
    if [ "$CONDA_CMD" = "micromamba" ]; then
        micromamba run -p "$INSTALL_PATH" pip install "torch==2.4.*" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
        micromamba run -p "$INSTALL_PATH" pip install "sglang[all]>=0.5.0"
    else
        conda run -p "$INSTALL_PATH" pip install "torch==2.4.*" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
        conda run -p "$INSTALL_PATH" pip install "sglang[all]>=0.5.0"
    fi
    echo -e "${GREEN}✓ SGLang installed${NC}"
fi

# Verify installation
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"
if [ "$CONDA_CMD" = "micromamba" ]; then
    SGLANG_VERSION=$(micromamba run -p "$INSTALL_PATH" python -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
    TORCH_VERSION=$(micromamba run -p "$INSTALL_PATH" python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    CUDA_AVAILABLE=$(micromamba run -p "$INSTALL_PATH" python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
else
    SGLANG_VERSION=$(conda run -p "$INSTALL_PATH" python -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
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
echo "  - SGLang version: $SGLANG_VERSION"
echo "  - PyTorch version: $TORCH_VERSION"
echo "  - CUDA available: $CUDA_AVAILABLE"
echo ""

if [ "$CUDA_AVAILABLE" = "False" ]; then
    echo -e "${RED}⚠ WARNING: CUDA not available in PyTorch!${NC}"
    echo "This likely means CUDA version mismatch. Try reinstalling PyTorch with correct CUDA version."
    exit 1
fi

echo -e "${GREEN}✓ SGLang backend ready to use${NC}"
echo ""
echo "To use this environment in Sparkstation, add to .env:"
echo "  SGLANG_PYTHON_PATH=$INSTALL_PATH/bin/python"
echo ""
echo "To test the installation:"
echo "  $INSTALL_PATH/bin/python -m sglang.launch_server --help"
echo ""
