"""
Prediction heads for MILP solution prediction.

Attached to LLM's last hidden state to predict:
- Primal solution (0/1 for binary, real values for integer/continuous)
- Uncertainty (variance/confidence)

Supports mixed variable types with hybrid activation mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# Variable type constants
VAR_BINARY = 0
VAR_CONTINUOUS = 1
VAR_INTEGER = 2


class PredictionHead(nn.Module):
    """
    Prediction head for mixed-variable MILP solutions.
    
    Uses hybrid activation mechanism:
    - Binary variables: Sigmoid activation (probability of x_i = 1)
    - Integer/Continuous variables: Softplus or Identity activation
    
    Output is raw logits; activation is applied based on variable types.
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
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        var_types: Optional[torch.Tensor] = None,
        var_lb: Optional[torch.Tensor] = None,
        var_ub: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict solution values with hybrid activation.
        
        Args:
            hidden_states: Variable hidden states of shape (batch, n_vars, hidden_dim)
                          or (n_vars, hidden_dim).
            var_types: Variable types (n_vars,) with values in {0, 1, 2}.
                      0=binary, 1=continuous, 2=integer.
                      If None, assumes all binary.
            var_lb: Lower bounds for scaling continuous/integer outputs.
            var_ub: Upper bounds for scaling continuous/integer outputs.
        
        Returns:
            Predicted values of shape (batch, n_vars) or (n_vars,).
            For binary: probabilities in (0, 1)
            For integer/continuous: scaled values respecting bounds
        """
        # MLP forward
        logits = self.mlp(hidden_states)  # (..., 1)
        logits = logits.squeeze(-1)  # (...,)
        
        # If no var_types provided, return logits (backward compatible)
        if var_types is None:
            return logits
        
        # Apply hybrid activation based on variable types
        output = torch.zeros_like(logits)
        
        # Binary variables: Sigmoid
        binary_mask = var_types == VAR_BINARY
        if binary_mask.any():
            output[binary_mask] = torch.sigmoid(logits[binary_mask])
        
        # Continuous variables: scaled to bounds using sigmoid
        continuous_mask = var_types == VAR_CONTINUOUS
        if continuous_mask.any():
            # Use sigmoid to map to [lb, ub] range
            sig_out = torch.sigmoid(logits[continuous_mask])
            if var_lb is not None and var_ub is not None:
                lb = var_lb[continuous_mask]
                ub = var_ub[continuous_mask]
                output[continuous_mask] = lb + sig_out * (ub - lb)
            else:
                # Default to [0, inf) using softplus
                output[continuous_mask] = F.softplus(logits[continuous_mask])
        
        # Integer variables: similar to continuous but will be rounded later
        integer_mask = var_types == VAR_INTEGER
        if integer_mask.any():
            # Use sigmoid to map to [lb, ub] range
            sig_out = torch.sigmoid(logits[integer_mask])
            if var_lb is not None and var_ub is not None:
                lb = var_lb[integer_mask]
                ub = var_ub[integer_mask]
                output[integer_mask] = lb + sig_out * (ub - lb)
            else:
                # Default to non-negative using softplus
                output[integer_mask] = F.softplus(logits[integer_mask])
        
        return output
    
    def forward_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Return raw logits without activation (for loss computation).
        
        Args:
            hidden_states: Variable hidden states.
        
        Returns:
            Raw logits of shape (batch, n_vars) or (n_vars,).
        """
        logits = self.mlp(hidden_states)
        return logits.squeeze(-1)


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
    Container for model outputs with support for mixed variable types.
    """
    
    def __init__(
        self,
        primal: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        dual: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        var_types: Optional[torch.Tensor] = None,
        var_lb: Optional[torch.Tensor] = None,
        var_ub: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            primal: Predicted solution values (n_vars,).
                    For binary: probabilities in (0, 1)
                    For integer/continuous: scaled values
            uncertainty: Prediction uncertainty (n_vars,).
            dual: Predicted reduced costs (n_vars,).
            hidden_states: Raw hidden states (n_vars, hidden_dim).
            var_types: Variable types (n_vars,) with values {0: binary, 1: continuous, 2: integer}.
            var_lb: Lower bounds (n_vars,).
            var_ub: Upper bounds (n_vars,).
        """
        self.primal = primal
        self.uncertainty = uncertainty
        self.dual = dual
        self.hidden_states = hidden_states
        self.var_types = var_types
        self.var_lb = var_lb
        self.var_ub = var_ub
    
    def to_discrete(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Convert predictions to discrete solution respecting variable types.
        
        Args:
            threshold: Probability threshold for binary variables (x_i = 1 if p > threshold).
        
        Returns:
            Discrete solution (n_vars,).
            Binary: {0, 1}
            Integer: rounded to nearest integer
            Continuous: unchanged
        """
        if self.var_types is None:
            # Backward compatible: assume all binary
            return (self.primal > threshold).float()
        
        solution = self.primal.clone()
        
        # Binary: threshold
        binary_mask = self.var_types == VAR_BINARY
        if binary_mask.any():
            solution[binary_mask] = (self.primal[binary_mask] > threshold).float()
        
        # Integer: round to nearest integer, respecting bounds
        integer_mask = self.var_types == VAR_INTEGER
        if integer_mask.any():
            rounded = torch.round(self.primal[integer_mask])
            if self.var_lb is not None and self.var_ub is not None:
                rounded = torch.clamp(rounded, 
                                      self.var_lb[integer_mask], 
                                      self.var_ub[integer_mask])
            solution[integer_mask] = rounded
        
        # Continuous: keep as is, but clamp to bounds
        continuous_mask = self.var_types == VAR_CONTINUOUS
        if continuous_mask.any() and self.var_lb is not None and self.var_ub is not None:
            solution[continuous_mask] = torch.clamp(
                self.primal[continuous_mask],
                self.var_lb[continuous_mask],
                self.var_ub[continuous_mask]
            )
        
        return solution
    
    def sample(self, temperature: float = 1.0, sigma: float = 0.1) -> torch.Tensor:
        """
        Sample solution from predictions using type-appropriate distributions.
        
        Args:
            temperature: Sampling temperature for binary variables.
            sigma: Standard deviation for Gaussian sampling of integer/continuous variables.
        
        Returns:
            Sampled solution (n_vars,).
            Binary: Bernoulli sample
            Integer: Gaussian sample rounded to integer
            Continuous: Gaussian sample
        """
        if self.var_types is None:
            # Backward compatible: assume all binary
            if temperature != 1.0:
                logits = torch.logit(self.primal.clamp(1e-7, 1 - 1e-7))
                probs = torch.sigmoid(logits / temperature)
            else:
                probs = self.primal
            return torch.bernoulli(probs)
        
        sample = torch.zeros_like(self.primal)
        
        # Binary: Bernoulli sampling
        binary_mask = self.var_types == VAR_BINARY
        if binary_mask.any():
            probs = self.primal[binary_mask]
            if temperature != 1.0:
                logits = torch.logit(probs.clamp(1e-7, 1 - 1e-7))
                probs = torch.sigmoid(logits / temperature)
            sample[binary_mask] = torch.bernoulli(probs)
        
        # Integer: Gaussian sample + round
        integer_mask = self.var_types == VAR_INTEGER
        if integer_mask.any():
            mean = self.primal[integer_mask]
            # Use uncertainty as variance if available, otherwise use sigma
            if self.uncertainty is not None:
                std = torch.sqrt(self.uncertainty[integer_mask])
            else:
                std = torch.full_like(mean, sigma)
            noise = torch.randn_like(mean) * std
            sampled = mean + noise
            # Round and clamp to bounds
            rounded = torch.round(sampled)
            if self.var_lb is not None and self.var_ub is not None:
                rounded = torch.clamp(rounded, 
                                      self.var_lb[integer_mask], 
                                      self.var_ub[integer_mask])
            sample[integer_mask] = rounded
        
        # Continuous: Gaussian sample
        continuous_mask = self.var_types == VAR_CONTINUOUS
        if continuous_mask.any():
            mean = self.primal[continuous_mask]
            if self.uncertainty is not None:
                std = torch.sqrt(self.uncertainty[continuous_mask])
            else:
                std = torch.full_like(mean, sigma)
            noise = torch.randn_like(mean) * std
            sampled = mean + noise
            # Clamp to bounds
            if self.var_lb is not None and self.var_ub is not None:
                sampled = torch.clamp(sampled,
                                      self.var_lb[continuous_mask],
                                      self.var_ub[continuous_mask])
            sample[continuous_mask] = sampled
        
        return sample
    
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
            # Use type-specific uncertainty proxies
            if self.var_types is None:
                # Binary: distance from 0.5
                uncertainty = 0.5 - torch.abs(self.primal - 0.5)
            else:
                uncertainty = torch.zeros_like(self.primal)
                
                # Binary: confidence = |p - 0.5| * 2, uncertainty = 1 - confidence
                binary_mask = self.var_types == VAR_BINARY
                if binary_mask.any():
                    conf = torch.abs(self.primal[binary_mask] - 0.5) * 2
                    uncertainty[binary_mask] = 1 - conf
                
                # Integer: integrality gap |x - round(x)|
                integer_mask = self.var_types == VAR_INTEGER
                if integer_mask.any():
                    int_gap = torch.abs(self.primal[integer_mask] - 
                                       torch.round(self.primal[integer_mask]))
                    uncertainty[integer_mask] = int_gap
                
                # Continuous: typically not selected for LNS, low uncertainty
                continuous_mask = self.var_types == VAR_CONTINUOUS
                if continuous_mask.any():
                    uncertainty[continuous_mask] = 0.0
        else:
            uncertainty = self.uncertainty
        
        _, indices = torch.topk(uncertainty, min(k, len(uncertainty)))
        return indices
    
    def get_type_specific_features(self) -> dict:
        """
        Get type-specific features for LNS score function.
        
        Returns:
            Dict with features per variable type:
            - binary: neural_conf (classification confidence)
            - integer: neural_variance, integrality_gap
            - continuous: neural_rc (reduced cost)
        """
        features = {
            'neural_conf': None,  # For binary
            'neural_variance': None,  # For integer
            'integrality_gap': None,  # For integer
            'neural_rc': self.dual,  # For continuous (and all)
        }
        
        if self.var_types is None:
            # All binary
            features['neural_conf'] = torch.abs(self.primal - 0.5) * 2
            return features
        
        n_vars = self.primal.size(0)
        
        # Binary confidence
        features['neural_conf'] = torch.zeros(n_vars, device=self.primal.device)
        binary_mask = self.var_types == VAR_BINARY
        if binary_mask.any():
            features['neural_conf'][binary_mask] = torch.abs(self.primal[binary_mask] - 0.5) * 2
        
        # Integer features
        features['neural_variance'] = self.uncertainty if self.uncertainty is not None else torch.zeros(n_vars, device=self.primal.device)
        features['integrality_gap'] = torch.zeros(n_vars, device=self.primal.device)
        integer_mask = self.var_types == VAR_INTEGER
        if integer_mask.any():
            features['integrality_gap'][integer_mask] = torch.abs(
                self.primal[integer_mask] - torch.round(self.primal[integer_mask])
            )
        
        return features
