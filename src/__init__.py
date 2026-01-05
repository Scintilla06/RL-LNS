"""
Deep-Structure RL-LNS Solver for Hard MILPs

A neural-guided Large Neighborhood Search solver that uses:
- GNN encoder to represent MILP as bipartite graphs
- Qwen2.5-7B backbone with LoRA fine-tuning
- Physics-informed SFT + GRPO training
- Heuristic evolution with EOH

Modules:
- data: Data preprocessing and dataset classes
- model: GNN tokenizer, text tokenizer, prediction heads, NeuroSolver
- training: Physics-informed SFT and GRPO training
- evolution: EOH heuristic evolution
- problems: MILP problem definitions
- llm: LLM API interfaces
- utils: Configuration and utilities
"""

__version__ = "0.1.0"
__author__ = "RL-LNS Team"

# Core model
from .model.neuro_solver import NeuroSolver
from .model.gnn_tokenizer import GNNTokenizer, BipartiteGNN
from .model.text_tokenizer import TextTokenizerWrapper
from .model.heads import PredictionHead, MultiTaskHead, SolutionOutput

# Training
from .training.physics_loss import PhysicsInformedLoss, FeasibilityChecker
from .training.sft_trainer import SFTTrainer
from .training.grpo_loop import GRPOTrainer

# Data
from .datalib.preprocess import MILPPreprocessor
from .datalib.dataset import MILPGraphDataset, MILPTextDataset

# Evolution
from .evolution.eoh import EOH, Paras

__all__ = [
    # Model
    'NeuroSolver',
    'GNNTokenizer',
    'BipartiteGNN',
    'TextTokenizerWrapper',
    'PredictionHead',
    'MultiTaskHead',
    'SolutionOutput',
    # Training
    'PhysicsInformedLoss',
    'FeasibilityChecker',
    'SFTTrainer',
    'GRPOTrainer',
    # Data
    'MILPPreprocessor',
    'MILPGraphDataset',
    'MILPTextDataset',
    # Evolution
    'EOH',
    'Paras',
]
