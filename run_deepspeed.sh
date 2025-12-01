#!/bin/bash
################################################################################
# DeepSpeed Launch Script
# 
# This script sets up the environment and launches DeepSpeed ZeRO-3 profiling.
# It handles CUDA_HOME configuration which is required by DeepSpeed.
#
# Usage: ./run_deepspeed.sh
################################################################################

set -e  # Exit on error

echo "=================================="
echo "DeepSpeed ZeRO-3 Launch Script"
echo "=================================="

# ============================================================================
# CUDA Configuration
# ============================================================================

# Try to find CUDA automatically
if [ -z "$CUDA_HOME" ]; then
    echo "CUDA_HOME not set, attempting to find CUDA..."
    
    # Try common locations
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Found CUDA at: $CUDA_HOME"
    elif [ -d "/usr/local/cuda-12.0" ]; then
        export CUDA_HOME=/usr/local/cuda-12.0
        echo "Found CUDA at: $CUDA_HOME"
    elif [ -d "/usr/local/cuda-11.8" ]; then
        export CUDA_HOME=/usr/local/cuda-11.8
        echo "Found CUDA at: $CUDA_HOME"
    elif [ -d "/opt/cuda" ]; then
        export CUDA_HOME=/opt/cuda
        echo "Found CUDA at: $CUDA_HOME"
    else
        # Try to find nvcc
        NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
        if [ -n "$NVCC_PATH" ]; then
            export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
            echo "Found CUDA via nvcc at: $CUDA_HOME"
        else
            echo "ERROR: Could not find CUDA installation!"
            echo "Please set CUDA_HOME manually:"
            echo "  export CUDA_HOME=/path/to/cuda"
            echo "  ./run_deepspeed.sh"
            exit 1
        fi
    fi
else
    echo "Using existing CUDA_HOME: $CUDA_HOME"
fi

# Set CUDA paths
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA
if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
    echo "Please check your CUDA installation"
    exit 1
fi

echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version:"
$CUDA_HOME/bin/nvcc --version | grep "release"

# ============================================================================
# Conda Environment
# ============================================================================

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/anaconda3/etc/profile.d/conda.sh
else
    echo "WARNING: Could not find conda.sh, assuming conda is already initialized"
fi

echo ""
echo "Activating conda environment: translate"
conda activate translate

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# ============================================================================
# DeepSpeed Configuration
# ============================================================================

NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
echo ""
echo "Launching DeepSpeed with $NUM_GPUS GPUs"
echo "Script: profile_llm_parallel_ds.py"
echo "=================================="
echo ""

# ============================================================================
# Launch DeepSpeed
# ============================================================================

deepspeed --num_gpus=$NUM_GPUS profile_llm_parallel_ds.py

echo ""
echo "=================================="
echo "DeepSpeed profiling complete!"
echo "=================================="

