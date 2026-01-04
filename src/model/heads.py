"""
Prediction heads for MILP solution prediction.

Attached to LLM's last hidden state to predict:
- Primal solution (0/1 for binary variables)
- Uncertainty (variance/confidence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PredictionHead(nn.Module):
    """
    Prediction head for binary variable solutions.
    
    Takes variable hidden states and outputs probability of x_i = 1.
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        intermediate_dim: int = 512,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        """
        Args:
            hidden_dim: Input hidden dimension (LLM hidden size).
            intermediate_dim: Intermediate MLP dimension.
            dropout: Dropout rate.
            num_layers: Number of MLP layers.
        """
        super().__init__()
        
        layers = []
        
        # First layer
        layers.extend([
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        # Final layer - output single logit
        layers.append(nn.Linear(intermediate_dim if num_layers > 1 else hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict solution probabilities.
        
        Args:
            hidden_states: Variable hidden states of shape (batch, n_vars, hidden_dim)
                          or (n_vars, hidden_dim).
        
        Returns:
            Predicted probabilities of shape (batch, n_vars) or (n_vars,).
        """
        # MLP forward
        logits = self.mlp(hidden_states)  # (..., 1)
        logits = logits.squeeze(-1)  # (...,)
        
        # Sigmoid for probability
        probs = torch.sigmoid(logits)
        
        return probs


class UncertaintyHead(nn.Module):
    """
    Uncertainty head for prediction confidence estimation.
    
    Outputs variance/uncertainty for each variable prediction.
    Higher uncertainty indicates less confident predictions.
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        intermediate_dim: int = 256,
        dropout: float = 0.1,
        min_variance: float = 1e-6,
    ):
        """
        Args:
            hidden_dim: Input hidden dimension.
            intermediate_dim: Intermediate MLP dimension.
            dropout: Dropout rate.
            min_variance: Minimum variance to ensure numerical stability.
        """
        super().__init__()
        
        self.min_variance = min_variance
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict uncertainty (variance).
        
        Args:
            hidden_states: Variable hidden states of shape (batch, n_vars, hidden_dim)
                          or (n_vars, hidden_dim).
        
        Returns:
            Predicted variance of shape (batch, n_vars) or (n_vars,).
        """
        # MLP forward
        log_var = self.mlp(hidden_states)  # (..., 1)
        log_var = log_var.squeeze(-1)  # (...,)
        
        # Softplus to ensure positive variance, plus minimum
        variance = F.softplus(log_var) + self.min_variance
        
        return variance


class DualHead(nn.Module):
    """
    Dual head for reduced cost prediction.
    
    Predicts reduced costs (shadow prices) for variables.
    Useful for warm-starting dual solvers.
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        intermediate_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Input hidden dimension.
            intermediate_dim: Intermediate MLP dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict reduced costs.
        
        Args:
            hidden_states: Variable hidden states of shape (batch, n_vars, hidden_dim)
                          or (n_vars, hidden_dim).
        
        Returns:
            Predicted reduced costs of shape (batch, n_vars) or (n_vars,).
        """
        # MLP forward - no activation, can be positive or negative
        reduced_costs = self.mlp(hidden_states)  # (..., 1)
        reduced_costs = reduced_costs.squeeze(-1)  # (...,)
        
        return reduced_costs


class MultiTaskHead(nn.Module):
    """
    Combined multi-task head for all predictions.
    
    Combines:
    - Primal prediction (solution)
    - Uncertainty estimation
    - Dual prediction (optional)
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        intermediate_dim: int = 512,
        dropout: float = 0.1,
        include_dual: bool = False,
    ):
        """
        Args:
            hidden_dim: Input hidden dimension.
            intermediate_dim: Intermediate MLP dimension.
            dropout: Dropout rate.
            include_dual: Whether to include dual head.
        """
        super().__init__()
        
        self.include_dual = include_dual
        
        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.primal_head = nn.Linear(intermediate_dim, 1)
        self.uncertainty_head = nn.Linear(intermediate_dim, 1)
        
        if include_dual:
            self.dual_head = nn.Linear(intermediate_dim, 1)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict all outputs.
        
        Args:
            hidden_states: Variable hidden states.
        
        Returns:
            Tuple of (primal_probs, uncertainty, dual_costs).
            dual_costs is None if include_dual=False.
        """
        # Shared encoding
        shared = self.shared(hidden_states)
        
        # Primal prediction
        primal_logits = self.primal_head(shared).squeeze(-1)
        primal_probs = torch.sigmoid(primal_logits)
        
        # Uncertainty prediction
        log_var = self.uncertainty_head(shared).squeeze(-1)
        uncertainty = F.softplus(log_var) + 1e-6
        
        # Dual prediction
        dual_costs = None
        if self.include_dual:
            dual_costs = self.dual_head(shared).squeeze(-1)
        
        return primal_probs, uncertainty, dual_costs


class SolutionOutput:
    """
    Container for model outputs.
    """
    
    def __init__(
        self,
        primal: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        dual: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            primal: Predicted solution probabilities (n_vars,).
            uncertainty: Prediction uncertainty (n_vars,).
            dual: Predicted reduced costs (n_vars,).
            hidden_states: Raw hidden states (n_vars, hidden_dim).
        """
        self.primal = primal
        self.uncertainty = uncertainty
        self.dual = dual
        self.hidden_states = hidden_states
    
    def to_discrete(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Convert probabilities to discrete solution.
        
        Args:
            threshold: Probability threshold for x_i = 1.
        
        Returns:
            Discrete solution (n_vars,) with values in {0, 1}.
        """
        return (self.primal > threshold).float()
    
    def sample(self, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample discrete solution from probabilities.
        
        Args:
            temperature: Sampling temperature.
        
        Returns:
            Sampled solution (n_vars,) with values in {0, 1}.
        """
        if temperature != 1.0:
            # Apply temperature to logits
            logits = torch.logit(self.primal.clamp(1e-7, 1 - 1e-7))
            probs = torch.sigmoid(logits / temperature)
        else:
            probs = self.primal
        
        return torch.bernoulli(probs)
    
    def most_uncertain(self, k: int) -> torch.Tensor:
        """
        Get indices of k most uncertain variables.
        
        Useful for LNS neighborhood selection.
        
        Args:
            k: Number of variables to select.
        
        Returns:
            Indices of k most uncertain variables.
        """
        if self.uncertainty is None:
            # Use distance from 0.5 as uncertainty proxy
            uncertainty = 0.5 - torch.abs(self.primal - 0.5)
        else:
            uncertainty = self.uncertainty
        
        _, indices = torch.topk(uncertainty, k)
        return indices
