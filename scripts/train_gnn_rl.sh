#!/bin/bash
#SBATCH -J gnn_rl
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
#   Main Method: GRPO Reinforcement Learning
#   Purpose: Fine-tune with GRPO RL using solver feedback
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

# Use all available CPUs for parallel reward calculation (Gurobi solving)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print environment info
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Start Time: $(date)"
echo "============================================================"

# Run training (use main.py subcommand)
# If you have a pretrained SFT model, pass it with --pretrained <path>
python src/main.py train-grpo --config configs/training.yaml

echo "============================================================"
echo "End Time: $(date)"
echo "============================================================"
