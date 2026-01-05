#!/bin/bash
#SBATCH -J gnn_sft
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.log

# ============================================================
#   Main Method: GNN Embedding + Physics-Informed SFT
#   Purpose: Train with GNN structure embedding and physics loss
# ============================================================

set -e

# Create logs directory
mkdir -p logs

# Load modules
module load miniconda/24.9.2
module load cuda/12.1

# Activate conda environment
source activate rl-lns

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_ENDPOINT=https://hf-mirror.com

# Load RL-LNS service configuration (wandb, gurobi, etc.)
if [ -f "$HOME/.rl_lns_config" ]; then
    export $(cat "$HOME/.rl_lns_config" | xargs)
fi

# Print environment info
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "============================================================"

# Run training (use main.py subcommand)
python src/main.py train-sft --config configs/training.yaml --mode gnn

echo "============================================================"
echo "End Time: $(date)"
echo "============================================================"
