"""
Prediction heads for MILP solution prediction.

Attached to LLM's last hidden state to predict:
- Primal solution (0/1 for binary, real values for integer/continuous)
- Uncertainty (variance/confidence)

Supports mixed variable types with unified normalization:
- All predictions output sigmoid [0, 1] representing normalized position in [lb, ub]
- Use denormalize() to convert back to original value space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..utils.constants import VAR_BINARY, VAR_CONTINUOUS, VAR_INTEGER


# ============================================================================
# Normalization utilities
# ============================================================================

def normalize(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """
    Normalize values from [lb, ub] to [0, 1].
    
    Args:
        x: Values in original space.
        lb: Lower bounds.
        ub: Upper bounds.
    
    Returns:
        Normalized values in [0, 1].
    """
    range_val = (ub - lb).clamp(min=1e-8)
    return (x - lb) / range_val


def denormalize(x_norm: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """
    Denormalize values from [0, 1] to [lb, ub].
    
    Args:
        x_norm: Normalized values in [0, 1].
        lb: Lower bounds.
        ub: Upper bounds.
    
    Returns:
        Values in original space [lb, ub].
    """
    return x_norm * (ub - lb) + lb


class PredictionHead(nn.Module):
    """
    Prediction head for mixed-variable MILP solutions.
    
    Uses unified normalization: all variables output sigmoid in [0, 1],
    representing the normalized position within [lb, ub].
    
    For Binary variables (lb=0, ub=1), this is equivalent to probability.
    For Integer/Continuous, use denormalize() to get original values.
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
        Predict normalized solution values in [0, 1].
        
        All variable types output sigmoid([0, 1]) representing:
        - Binary: probability of x_i = 1 (since lb=0, ub=1)
        - Integer/Continuous: normalized position in [lb, ub]
        
        Args:
            hidden_states: Variable hidden states of shape (batch, n_vars, hidden_dim)
                          or (n_vars, hidden_dim).
            var_types: Variable types (n_vars,). Not used in forward but kept for API.
            var_lb: Lower bounds. Not used in forward (normalization in loss/output).
            var_ub: Upper bounds. Not used in forward (normalization in loss/output).
        
        Returns:
            Normalized predictions in [0, 1] of shape (batch, n_vars) or (n_vars,).
        """
        # MLP forward
        logits = self.mlp(hidden_states)  # (..., 1)
        logits = logits.squeeze(-1)  # (...,)
        
        # Unified sigmoid activation: all outputs in [0, 1]
        # For Binary (lb=0, ub=1): this IS the probability
        # For others: this is the normalized position in [lb, ub]
        output = torch.sigmoid(logits)
        
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
    
    The `primal` field contains NORMALIZED values in [0, 1].
    Use `get_denormalized()` to get values in original space [lb, ub].
    Use `get_solution()` to get final solution (with Integer rounding).
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
            primal: NORMALIZED predicted values in [0, 1] (n_vars,).
                    Represents position within [lb, ub] range.
            uncertainty: Prediction uncertainty (n_vars,).
            dual: Predicted reduced costs (n_vars,).
            hidden_states: Raw hidden states (n_vars, hidden_dim).
            var_types: Variable types (n_vars,) with values {0: binary, 1: continuous, 2: integer}.
            var_lb: Lower bounds (n_vars,).
            var_ub: Upper bounds (n_vars,).
        """
        self.primal = primal  # Normalized [0, 1]
        self.uncertainty = uncertainty
        self.dual = dual
        self.hidden_states = hidden_states
        self.var_types = var_types
        self.var_lb = var_lb
        self.var_ub = var_ub
    
    def get_denormalized(self) -> torch.Tensor:
        """
        Get denormalized predictions in original value space [lb, ub].
        
        Returns:
            Denormalized values (n_vars,).
            Binary: same as primal (lb=0, ub=1)
            Integer/Continuous: lb + primal * (ub - lb)
        """
        if self.var_lb is None or self.var_ub is None:
            # No bounds info: return primal as-is
            return self.primal
        
        return denormalize(self.primal, self.var_lb, self.var_ub)
    
    def get_solution(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get final solution with proper discretization.
        
        This is the method to call when you need a usable MILP solution.
        
        Args:
            threshold: Probability threshold for binary variables.
        
        Returns:
            Final solution (n_vars,).
            Binary: {0, 1} based on threshold
            Integer: rounded to nearest integer within bounds
            Continuous: denormalized continuous value
        """
        # First denormalize to original space
        denorm = self.get_denormalized()
        
        if self.var_types is None:
            # Assume all binary
            return (denorm > threshold).float()
        
        solution = denorm.clone()
        
        # Binary: threshold (denorm is in [0, 1] since lb=0, ub=1)
        binary_mask = self.var_types == VAR_BINARY
        if binary_mask.any():
            solution[binary_mask] = (denorm[binary_mask] > threshold).float()
        
        # Integer: round to nearest integer, clamp to bounds
        integer_mask = self.var_types == VAR_INTEGER
        if integer_mask.any():
            rounded = torch.round(denorm[integer_mask])
            if self.var_lb is not None and self.var_ub is not None:
                rounded = torch.clamp(rounded, 
                                      self.var_lb[integer_mask], 
                                      self.var_ub[integer_mask])
            solution[integer_mask] = rounded
        
        # Continuous: already denormalized, just clamp to bounds
        continuous_mask = self.var_types == VAR_CONTINUOUS
        if continuous_mask.any() and self.var_lb is not None and self.var_ub is not None:
            solution[continuous_mask] = torch.clamp(
                denorm[continuous_mask],
                self.var_lb[continuous_mask],
                self.var_ub[continuous_mask]
            )
        
        return solution
    
    def to_discrete(self, threshold: float = 0.5) -> torch.Tensor:
        """
        DEPRECATED: Use get_solution() instead.
        
        Convert predictions to discrete solution respecting variable types.
        """
        return self.get_solution(threshold)
    
    def sample(self, temperature: float = 1.0, sigma: float = 0.1) -> torch.Tensor:
        """
        Sample solution from predictions using type-appropriate distributions.
        
        Sampling happens in NORMALIZED [0, 1] space, then denormalized.
        
        Args:
            temperature: Sampling temperature for binary variables.
            sigma: Standard deviation for Gaussian sampling in normalized space.
        
        Returns:
            Sampled solution in ORIGINAL value space (n_vars,).
            Binary: {0, 1} from Bernoulli
            Integer: rounded after denormalization
            Continuous: denormalized Gaussian sample
        """
        if self.var_types is None:
            # Backward compatible: assume all binary (already in [0,1])
            probs = self.primal
            if temperature != 1.0:
                logits = torch.logit(probs.clamp(1e-7, 1 - 1e-7))
                probs = torch.sigmoid(logits / temperature)
            return torch.bernoulli(probs)
        
        # Sample in normalized [0, 1] space
        sample_norm = torch.zeros_like(self.primal)
        
        # Binary: Bernoulli sampling (already in [0,1] since lb=0, ub=1)
        binary_mask = self.var_types == VAR_BINARY
        if binary_mask.any():
            probs = self.primal[binary_mask]
            if temperature != 1.0:
                logits = torch.logit(probs.clamp(1e-7, 1 - 1e-7))
                probs = torch.sigmoid(logits / temperature)
            sample_norm[binary_mask] = torch.bernoulli(probs)
        
        # Integer/Continuous: Gaussian sampling in normalized space
        non_binary_mask = (self.var_types == VAR_INTEGER) | (self.var_types == VAR_CONTINUOUS)
        if non_binary_mask.any():
            mean = self.primal[non_binary_mask]
            # Use uncertainty if available, otherwise use sigma
            if self.uncertainty is not None:
                std = torch.sqrt(self.uncertainty[non_binary_mask]).clamp(min=1e-6)
            else:
                std = torch.full_like(mean, sigma)
            noise = torch.randn_like(mean) * std
            sampled = (mean + noise).clamp(0, 1)  # Stay in [0, 1]
            sample_norm[non_binary_mask] = sampled
        
        # Denormalize to original space
        if self.var_lb is not None and self.var_ub is not None:
            sample_denorm = denormalize(sample_norm, self.var_lb, self.var_ub)
        else:
            sample_denorm = sample_norm
        
        # Round integers after denormalization
        integer_mask = self.var_types == VAR_INTEGER
        if integer_mask.any():
            sample_denorm[integer_mask] = torch.round(sample_denorm[integer_mask])
            if self.var_lb is not None and self.var_ub is not None:
                sample_denorm[integer_mask] = torch.clamp(
                    sample_denorm[integer_mask],
                    self.var_lb[integer_mask],
                    self.var_ub[integer_mask]
                )
        
        return sample_denorm
    
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
            # Use type-specific uncertainty proxies based on normalized values
            if self.var_types is None:
                # Binary: distance from 0.5 in normalized space
                uncertainty = 0.5 - torch.abs(self.primal - 0.5)
            else:
                uncertainty = torch.zeros_like(self.primal)
                
                # Binary: confidence = |p - 0.5| * 2, uncertainty = 1 - confidence
                binary_mask = self.var_types == VAR_BINARY
                if binary_mask.any():
                    conf = torch.abs(self.primal[binary_mask] - 0.5) * 2
                    uncertainty[binary_mask] = 1 - conf
                
                # Integer: integrality gap computed on DENORMALIZED values
                integer_mask = self.var_types == VAR_INTEGER
                if integer_mask.any():
                    if self.var_lb is not None and self.var_ub is not None:
                        # Denormalize to get actual values, then compute gap
                        denorm_int = denormalize(
                            self.primal[integer_mask],
                            self.var_lb[integer_mask],
                            self.var_ub[integer_mask]
                        )
                        int_gap = torch.abs(denorm_int - torch.round(denorm_int))
                        # Normalize gap by range for comparability
                        range_val = (self.var_ub[integer_mask] - self.var_lb[integer_mask]).clamp(min=1)
                        uncertainty[integer_mask] = int_gap / range_val
                    else:
                        # Fallback: use normalized value distance from 0.5
                        uncertainty[integer_mask] = 0.5 - torch.abs(self.primal[integer_mask] - 0.5)
                
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
            - neural_conf: Classification confidence for binary (in [0, 1])
            - neural_variance: Uncertainty for integer
            - integrality_gap: |denorm_x - round(denorm_x)| for integer
            - neural_rc: Reduced cost for all (if dual head enabled)
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
        device = self.primal.device
        
        # Binary confidence: |p - 0.5| * 2 (ranges from 0 to 1)
        features['neural_conf'] = torch.zeros(n_vars, device=device)
        binary_mask = self.var_types == VAR_BINARY
        if binary_mask.any():
            features['neural_conf'][binary_mask] = torch.abs(self.primal[binary_mask] - 0.5) * 2
        
        # Integer features
        features['neural_variance'] = self.uncertainty if self.uncertainty is not None else torch.zeros(n_vars, device=device)
        features['integrality_gap'] = torch.zeros(n_vars, device=device)
        integer_mask = self.var_types == VAR_INTEGER
        if integer_mask.any():
            if self.var_lb is not None and self.var_ub is not None:
                # Compute gap on denormalized values
                denorm_int = denormalize(
                    self.primal[integer_mask],
                    self.var_lb[integer_mask],
                    self.var_ub[integer_mask]
                )
                features['integrality_gap'][integer_mask] = torch.abs(
                    denorm_int - torch.round(denorm_int)
                )
            else:
                # Fallback: use normalized integrality proxy
                features['integrality_gap'][integer_mask] = torch.abs(
                    self.primal[integer_mask] - torch.round(self.primal[integer_mask])
                )
        
        return features
