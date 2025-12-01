#!/bin/bash
################################################################################
# CUDA Detection Script
# 
# This script helps you find where CUDA is installed on your system.
# Run this to determine the correct CUDA_HOME path.
################################################################################

echo "========================================"
echo "CUDA Detection Script"
echo "========================================"
echo ""

# Check if nvidia-smi works
echo "1. Checking nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --version
    echo "✓ nvidia-smi found"
else
    echo "✗ nvidia-smi not found"
fi
echo ""

# Check for nvcc
echo "2. Checking for nvcc compiler..."
if command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
    echo "✓ nvcc found at: $NVCC_PATH"
    nvcc --version
    CUDA_HOME_FROM_NVCC=$(dirname $(dirname $NVCC_PATH))
    echo "→ Suggested CUDA_HOME: $CUDA_HOME_FROM_NVCC"
else
    echo "✗ nvcc not in PATH"
fi
echo ""

# Check common CUDA locations
echo "3. Checking common CUDA installation directories..."
for dir in /usr/local/cuda /usr/local/cuda-12.0 /usr/local/cuda-12.1 /usr/local/cuda-12.2 /usr/local/cuda-11.8 /usr/local/cuda-11.7 /opt/cuda; do
    if [ -d "$dir" ]; then
        echo "✓ Found: $dir"
        if [ -f "$dir/bin/nvcc" ]; then
            echo "  → Has nvcc: YES"
            $dir/bin/nvcc --version | grep "release"
        else
            echo "  → Has nvcc: NO"
        fi
    fi
done
echo ""

# Check PyTorch CUDA
echo "4. Checking PyTorch CUDA configuration..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version (PyTorch): {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "✗ PyTorch not available or error"
echo ""

# Search for CUDA libraries
echo "5. Searching for CUDA libraries..."
LIBCUDART=$(ldconfig -p 2>/dev/null | grep libcudart.so | head -1)
if [ -n "$LIBCUDART" ]; then
    echo "✓ Found libcudart:"
    echo "  $LIBCUDART"
    LIB_PATH=$(echo $LIBCUDART | awk '{print $NF}')
    if [ -L "$LIB_PATH" ]; then
        REAL_PATH=$(readlink -f "$LIB_PATH")
        CUDA_LIB_DIR=$(dirname "$REAL_PATH")
        CUDA_HOME_FROM_LIB=$(dirname "$CUDA_LIB_DIR")
        echo "→ Suggested CUDA_HOME: $CUDA_HOME_FROM_LIB"
    fi
else
    echo "✗ libcudart not found in ldconfig"
fi
echo ""

# Check environment variables
echo "6. Checking existing environment variables..."
if [ -n "$CUDA_HOME" ]; then
    echo "CUDA_HOME is set: $CUDA_HOME"
else
    echo "CUDA_HOME is not set"
fi

if [ -n "$CUDA_PATH" ]; then
    echo "CUDA_PATH is set: $CUDA_PATH"
else
    echo "CUDA_PATH is not set"
fi
echo ""

# Check for module system
echo "7. Checking for module system (HPC clusters)..."
if command -v module &> /dev/null; then
    echo "✓ Module system detected"
    echo "Available CUDA modules:"
    module avail cuda 2>&1 | grep -i cuda || echo "  No CUDA modules found"
    echo ""
    echo "To load CUDA module, try:"
    echo "  module load cuda"
    echo "  module load cuda/12.0"
else
    echo "✗ No module system detected"
fi
echo ""

# Final recommendation
echo "========================================"
echo "RECOMMENDATION"
echo "========================================"
echo ""
echo "Based on the above, try setting CUDA_HOME to one of these:"
echo ""

# Collect all found paths
declare -a FOUND_PATHS=()

if command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
    CUDA_HOME_FROM_NVCC=$(dirname $(dirname $NVCC_PATH))
    FOUND_PATHS+=("$CUDA_HOME_FROM_NVCC")
fi

for dir in /usr/local/cuda /usr/local/cuda-12.0 /usr/local/cuda-12.1 /usr/local/cuda-11.8; do
    if [ -d "$dir" ] && [ -f "$dir/bin/nvcc" ]; then
        FOUND_PATHS+=("$dir")
    fi
done

if [ ${#FOUND_PATHS[@]} -eq 0 ]; then
    echo "❌ No valid CUDA installation found!"
    echo ""
    echo "Possible issues:"
    echo "  1. CUDA is not installed"
    echo "  2. CUDA is in a non-standard location"
    echo "  3. You need to load a module: module load cuda"
    echo ""
    echo "If PyTorch CUDA works, you might be able to skip DeepSpeed"
    echo "and just use FSDP instead (similar communication patterns):"
    echo "  torchrun --nproc_per_node=4 profile_llm_parallel_pt_fsdp.py"
else
    echo "Try one of these commands:"
    echo ""
    for path in "${FOUND_PATHS[@]}"; do
        echo "  export CUDA_HOME=$path"
    done
    echo ""
    echo "To make it permanent, add to ~/.bashrc:"
    echo "  echo 'export CUDA_HOME=${FOUND_PATHS[0]}' >> ~/.bashrc"
    echo "  source ~/.bashrc"
fi
echo ""
echo "========================================"

