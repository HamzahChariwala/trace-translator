#!/bin/bash

# TraceTranslator Environment Setup Script
# This script creates and configures the conda environment for model loading

echo "================================================"
echo "TraceTranslator Environment Setup"
echo "================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo ""
echo "Step 1: Creating conda environment 'translate'..."
echo "------------------------------------------------"

# Remove existing environment if it exists
if conda env list | grep -q "^translate "; then
    echo "Environment 'translate' already exists. Removing it..."
    conda env remove -n translate -y
fi

# Create environment from yml file
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "✓ Environment created successfully"
else
    echo "✗ Failed to create environment"
    exit 1
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate translate"
echo ""
echo "To test the model loader, run:"
echo "  python model_loader.py"
echo ""
echo "To deactivate when done:"
echo "  conda deactivate"
echo ""

