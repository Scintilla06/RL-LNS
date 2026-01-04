# Training modules
"""
Training modules for Physics-Informed SFT and GRPO.
"""

from .physics_loss import (
    TaskLoss,
    ConstraintLoss,
    IntegralityLoss,
    PhysicsInformedLoss,
    FeasibilityChecker,
    AccuracyMetrics,
)
from .sft_trainer import SFTTrainer
from .grpo_loop import (
    RewardCalculator,
    GroupSampler,
    GRPOTrainer,
)

__all__ = [
    # Loss functions
    'TaskLoss',
    'ConstraintLoss',
    'IntegralityLoss',
    'PhysicsInformedLoss',
    'FeasibilityChecker',
    'AccuracyMetrics',
    # SFT
    'SFTTrainer',
    # GRPO
    'RewardCalculator',
    'GroupSampler',
    'GRPOTrainer',
]
