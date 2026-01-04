#!/bin/bash

# RL-LNS Environment Installation Script

set -e  # Exit immediately if a command exits with a non-zero status.

ENV_NAME="rl-lns"
YAML_FILE="environment.yaml"

echo "============================================================"
echo "   RL-LNS Environment Setup Script"
echo "============================================================"

# 1. Check for Conda
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# 2. Create Conda Environment
echo ""
echo "[1/4] Creating Conda environment '$ENV_NAME' from $YAML_FILE..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and reinstall? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME
        conda env create -f $YAML_FILE
    else
        echo "Skipping environment creation."
    fi
else
    conda env create -f $YAML_FILE
fi

# 3. Install Gurobi
echo ""
echo "[2/4] Installing Gurobi (Required for MILP solving)..."
# We need to run this in the new environment. 
# Since 'conda activate' doesn't work well in scripts without init, we use 'conda run'.
conda run -n $ENV_NAME conda install -c gurobi gurobi -y

# 4. Install Flash Attention (Optional)
echo ""
echo "[3/4] Attempting to install Flash Attention 2 (Optional, requires CUDA)..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing flash-attn..."
    conda run -n $ENV_NAME pip install flash-attn --no-build-isolation || echo "Warning: Flash Attention installation failed. You can try manually later."
else
    echo "No NVIDIA GPU detected or nvidia-smi not found. Skipping Flash Attention."
fi

# 5. Model Download Helper
echo ""
echo "[4/4] Model Setup Instructions"
echo "You need the Qwen2.5-7B-Instruct model."
echo "Option A: Auto-download (Recommended)"
echo "  Just run the training script, and it will download to ~/.cache/huggingface/"
echo ""
echo "Option B: Manual download to local folder"
echo "  Run the following command after activating the environment:"
echo "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct --local-dir-use-symlinks False"

echo ""
echo "============================================================"
echo "   Setup Complete!"
echo "============================================================"
echo "To start working, run:"
echo ""
echo "    conda activate $ENV_NAME"
echo "    grbgetkey YOUR_LICENSE_KEY  # If you haven't activated Gurobi yet"
echo ""
