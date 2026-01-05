"""
Physics-informed loss functions for MILP solution prediction.

Implements:
- Task Loss: BCE for binary variables, Huber for integer/continuous
- Constraint Loss: Penalizes constraint violations (uses denormalized values)
- Integrality Loss: Pushes toward integer values (uses denormalized values)

IMPORTANT: All predictions (pred) are expected to be NORMALIZED in [0, 1].
- For constraint and integrality loss, we denormalize before computation.
- For task loss, we normalize the target to [0, 1] for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from ..utils.constants import VAR_BINARY, VAR_CONTINUOUS, VAR_INTEGER
from ..model.heads import normalize, denormalize


class TaskLoss(nn.Module):
    """
    Task loss for mixed-variable MILP prediction.
    
    Expects predictions in NORMALIZED [0, 1] space.
    
    Uses stratified loss based on variable type:
    - Binary: BCE (pred is already sigmoid, target is 0/1)
    - Integer/Continuous: Huber Loss on normalized values
    """
    
    def __init__(self, label_smoothing: float = 0.0, huber_delta: float = 0.1):
        """
        Args:
            label_smoothing: Label smoothing factor for binary variables.
            huber_delta: Delta parameter for Huber loss (in normalized [0,1] space).
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.huber_delta = huber_delta
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        var_types: torch.Tensor,
        var_lb: Optional[torch.Tensor] = None,
        var_ub: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute stratified task loss.
        
        Args:
            pred: NORMALIZED predictions in [0, 1] (n_vars,).
            target: Target values in ORIGINAL space (n_vars,).
            var_types: Variable types (n_vars,) with values {0, 1, 2}.
            var_lb: Lower bounds (n_vars,). Required for normalizing target.
            var_ub: Upper bounds (n_vars,). Required for normalizing target.
            weight: Optional per-variable weights.
        
        Returns:
            Scalar loss value.
        """
        # Ensure 1D tensors
        if pred.dim() > 1:
            pred = pred.squeeze(0)
        if target.dim() > 1:
            target = target.squeeze(0)
        
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        count = 0
        
        # Binary variables: BCE loss
        # pred is already sigmoid output in [0, 1], target is 0/1
        binary_mask = var_types == VAR_BINARY
        if binary_mask.any():
            binary_pred = pred[binary_mask]
            binary_target = target[binary_mask].float()
            
            if self.label_smoothing > 0:
                binary_target = binary_target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            # Use BCE (not with_logits) since pred is already sigmoid
            binary_weight = weight[binary_mask] if weight is not None else None
            binary_loss = F.binary_cross_entropy(
                binary_pred.clamp(1e-7, 1 - 1e-7), 
                binary_target, 
                weight=binary_weight, 
                reduction='mean'
            )
            total_loss = total_loss + binary_loss * binary_mask.sum()
            count += binary_mask.sum()
        
        # Integer/Continuous variables: Huber loss on NORMALIZED values
        non_binary_mask = (var_types == VAR_INTEGER) | (var_types == VAR_CONTINUOUS)
        if non_binary_mask.any():
            non_binary_pred = pred[non_binary_mask]
            non_binary_target = target[non_binary_mask]
            
            # Normalize target to [0, 1] for comparison
            if var_lb is not None and var_ub is not None:
                target_norm = normalize(
                    non_binary_target,
                    var_lb[non_binary_mask],
                    var_ub[non_binary_mask]
                )
            else:
                # Fallback: assume target is already normalized
                target_norm = non_binary_target
            
            # Huber loss in normalized space
            non_binary_loss = F.smooth_l1_loss(
                non_binary_pred, target_norm, beta=self.huber_delta, reduction='mean'
            )
            total_loss = total_loss + non_binary_loss * non_binary_mask.sum()
            count += non_binary_mask.sum()
        
        if count > 0:
            return total_loss / count
        return total_loss


class ConstraintLoss(nn.Module):
    """
    Constraint violation loss.
    
    IMPORTANT: Expects predictions in NORMALIZED [0, 1] space.
    Denormalizes to original value space before computing Ax - b.
    
    Penalizes predictions that violate MILP constraints:
    L_constr = sum_i max(0, sum_j a_ij * x_j - b_i)  for <= constraints
    L_constr = sum_i max(0, b_i - sum_j a_ij * x_j)  for >= constraints
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Reduction method ("mean", "sum", "none").
        """
        super().__init__()
        self.reduction = reduction
    
    def _denormalize_pred(
        self,
        pred: torch.Tensor,
        var_lb: Optional[torch.Tensor],
        var_ub: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Denormalize predictions from [0, 1] to original value space.
        """
        if var_lb is None or var_ub is None:
            return pred
        return denormalize(pred, var_lb, var_ub)
    
    def forward(
        self,
        pred: torch.Tensor,
        constr_matrix: List[Tuple[int, int, float]],
        constr_rhs: torch.Tensor,
        constr_sense: torch.Tensor,
        n_vars: int,
        n_constrs: int,
        var_types: Optional[torch.Tensor] = None,
        var_lb: Optional[torch.Tensor] = None,
        var_ub: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute constraint violation loss for mixed-variable MILP.
        
        Args:
            pred: NORMALIZED predictions in [0, 1] (n_vars,).
            constr_matrix: List of (constr_idx, var_idx, coeff) tuples.
            constr_rhs: Right-hand side values (n_constrs,).
            constr_sense: Constraint sense (n_constrs,): 1=<=, 2=>=, 3===.
            n_vars: Number of variables.
            n_constrs: Number of constraints.
            var_types: Variable types (n_vars,).
            var_lb: Lower bounds (n_vars,). Required for denormalization.
            var_ub: Upper bounds (n_vars,). Required for denormalization.
        
        Returns:
            Constraint violation loss.
        """
        device = pred.device
        
        # Denormalize predictions to original value space for constraint checking
        values = self._denormalize_pred(pred, var_lb, var_ub)
        
        # Compute Ax for each constraint
        ax = torch.zeros(n_constrs, device=device)
        
        for constr_idx, var_idx, coeff in constr_matrix:
            ax[constr_idx] += coeff * values[var_idx]
        
        # Compute violations based on constraint sense
        violations = torch.zeros(n_constrs, device=device)
        
        # <= constraints: violation = max(0, Ax - b)
        le_mask = constr_sense == 1
        violations[le_mask] = F.relu(ax[le_mask] - constr_rhs[le_mask])
        
        # >= constraints: violation = max(0, b - Ax)
        ge_mask = constr_sense == 2
        violations[ge_mask] = F.relu(constr_rhs[ge_mask] - ax[ge_mask])
        
        # == constraints: violation = |Ax - b|
        eq_mask = constr_sense == 3
        violations[eq_mask] = torch.abs(ax[eq_mask] - constr_rhs[eq_mask])
        
        # Reduce
        if self.reduction == "mean":
            return violations.mean()
        elif self.reduction == "sum":
            return violations.sum()
        else:
            return violations
    
    def forward_batch(
        self,
        pred: torch.Tensor,
        A: torch.Tensor,
        b: torch.Tensor,
        sense: torch.Tensor,
        var_types: Optional[torch.Tensor] = None,
        var_lb: Optional[torch.Tensor] = None,
        var_ub: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch version using dense constraint matrix.
        
        Args:
            pred: NORMALIZED predictions in [0, 1] (batch, n_vars) or (n_vars,).
            A: Constraint matrix (n_constrs, n_vars).
            b: RHS values (n_constrs,).
            sense: Constraint sense (n_constrs,).
            var_types: Variable types (n_vars,).
            var_lb: Lower bounds (n_vars,). Required for denormalization.
            var_ub: Upper bounds (n_vars,). Required for denormalization.
        
        Returns:
            Constraint violation loss.
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        
        # Denormalize predictions to original value space
        values = self._denormalize_pred(pred.squeeze(0), var_lb, var_ub).unsqueeze(0)
        
        # Compute Ax: (batch, n_constrs)
        if A.is_sparse:
            # A is (n_constrs, n_vars) sparse
            # values is (batch, n_vars) dense
            # We want values @ A.T -> (batch, n_constrs)
            # Use (A @ values.T).T
            
            # NOTE: CUDA sparse mm does NOT support FP16 (Half).
            # When using mixed-precision training (autocast), we must
            # temporarily convert to float32 for the sparse operation.
            
            original_dtype = values.dtype
            
            # Disable autocast to prevent implicit casting back to half
            with torch.amp.autocast('cuda', enabled=False):
                # Force both A and values to float32 for sparse mm
                A_float = A.float() if A.dtype != torch.float32 else A
                values_float = values.float() if values.dtype != torch.float32 else values
                
                ax_t = torch.sparse.mm(A_float, values_float.t())
                ax = ax_t.t()
        else:
            ax = torch.matmul(values, A.T)
        
        # Ensure ax is float32 for loss calculation stability
        if ax.dtype != torch.float32:
            ax = ax.float()
            
        # Expand b for batch: (1, n_constrs)
        b = b.unsqueeze(0)
        
        # Ensure b is float32
        if b.dtype != torch.float32:
            b = b.float()
        
        # Compute violations
        violations = torch.zeros_like(ax)
        
        le_mask = sense == 1
        violations[:, le_mask] = F.relu(ax[:, le_mask] - b[:, le_mask])
        
        ge_mask = sense == 2
        violations[:, ge_mask] = F.relu(b[:, ge_mask] - ax[:, ge_mask])
        
        eq_mask = sense == 3
        violations[:, eq_mask] = torch.abs(ax[:, eq_mask] - b[:, eq_mask])
        
        if self.reduction == "mean":
            return violations.mean()
        elif self.reduction == "sum":
            return violations.sum()
        else:
            return violations


class IntegralityLoss(nn.Module):
    """
    Integrality loss to push predictions toward integer values.
    
    IMPORTANT: Expects predictions in NORMALIZED [0, 1] space.
    Denormalizes to original value space before computing cosine loss.
    
    Uses periodic cosine loss:
    L_int = sum_j (1 - cos(2 * pi * x_j))
    
    This loss is 0 when x_j is an integer and maximum when x_j = 0.5.
    
    For mixed-variable MILP:
    - Binary variables: Always applied (they must be 0 or 1)
    - Integer variables: Applied to push toward integers
    - Continuous variables: NEVER applied (would force continuous to integers)
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Reduction method ("mean", "sum", "none").
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor,
        var_types: torch.Tensor,
        var_lb: Optional[torch.Tensor] = None,
        var_ub: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute selective integrality loss.
        
        Args:
            pred: NORMALIZED predictions in [0, 1] (n_vars,).
            var_types: Variable types (n_vars,) with values {0, 1, 2}.
            var_lb: Lower bounds (n_vars,). Required for denormalization.
            var_ub: Upper bounds (n_vars,). Required for denormalization.
        
        Returns:
            Integrality loss (only for binary and integer variables).
        """
        # Ensure 1D
        if pred.dim() > 1:
            pred = pred.squeeze(0)
        
        # Create mask for variables that need integrality
        # Binary (0) and Integer (2), NOT Continuous (1)
        binary_mask = (var_types == VAR_BINARY)
        integer_mask = (var_types == VAR_INTEGER)
        integrality_mask = binary_mask | integer_mask
        
        if not integrality_mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Compute loss directly on masked variables
        loss_sum = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        count = 0
        
        # Binary variables: pred is already probability in [0, 1]
        # Cosine loss: 1 - cos(2*pi*p) is 0 at p=0 or p=1, max at p=0.5
        if binary_mask.any():
            binary_probs = pred[binary_mask]
            binary_loss = 1 - torch.cos(2 * torch.pi * binary_probs)
            loss_sum = loss_sum + binary_loss.sum()
            count += binary_mask.sum()
        
        # Integer variables: denormalize first, then compute loss
        if integer_mask.any():
            # Denormalize to get actual integer values
            if var_lb is not None and var_ub is not None:
                integer_vals = denormalize(
                    pred[integer_mask],
                    var_lb[integer_mask],
                    var_ub[integer_mask]
                )
            else:
                integer_vals = pred[integer_mask]
            
            # Cosine loss on actual values
            integer_loss = 1 - torch.cos(2 * torch.pi * integer_vals)
            loss_sum = loss_sum + integer_loss.sum()
            count += integer_mask.sum()
        
        if self.reduction == "mean":
            return loss_sum / count if count > 0 else loss_sum
        elif self.reduction == "sum":
            return loss_sum
        else:
            # Return full-size tensor with zeros for continuous
            full_loss = torch.zeros_like(pred)
            if binary_mask.any():
                full_loss[binary_mask] = 1 - torch.cos(2 * torch.pi * pred[binary_mask])
            if integer_mask.any():
                if var_lb is not None and var_ub is not None:
                    int_vals = denormalize(pred[integer_mask], var_lb[integer_mask], var_ub[integer_mask])
                else:
                    int_vals = pred[integer_mask]
                full_loss[integer_mask] = 1 - torch.cos(2 * torch.pi * int_vals)
            return full_loss


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for mixed-variable MILP solution prediction.
    
    IMPORTANT: Expects predictions in NORMALIZED [0, 1] space.
    
    L = L_task + lambda_1 * L_constraint + lambda_2 * L_integrality
    
    Supports:
    - Stratified task loss (BCE for binary, Huber for integer/continuous)
    - Selective integrality loss (only for binary and integer variables)
    - Unified constraint penalty (for all variable types)
    """
    
    def __init__(
        self,
        lambda_constraint: float = 0.1,
        lambda_integrality: float = 0.01,
        label_smoothing: float = 0.0,
        huber_delta: float = 0.1,
    ):
        """
        Args:
            lambda_constraint: Weight for constraint loss.
            lambda_integrality: Weight for integrality loss.
            label_smoothing: Label smoothing for binary task loss.
            huber_delta: Delta parameter for Huber loss (in normalized [0,1] space).
        """
        super().__init__()
        
        self.lambda_constraint = lambda_constraint
        self.lambda_integrality = lambda_integrality
        
        self.task_loss = TaskLoss(label_smoothing=label_smoothing, huber_delta=huber_delta)
        self.constraint_loss = ConstraintLoss()
        self.integrality_loss = IntegralityLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        var_types: torch.Tensor,
        var_lb: Optional[torch.Tensor] = None,
        var_ub: Optional[torch.Tensor] = None,
        constr_matrix: Optional[List[Tuple[int, int, float]]] = None,
        constr_rhs: Optional[torch.Tensor] = None,
        constr_sense: Optional[torch.Tensor] = None,
        n_vars: Optional[int] = None,
        n_constrs: Optional[int] = None,
        A: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        sense: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined physics-informed loss for mixed-variable MILP.
        
        Args:
            pred: NORMALIZED predictions in [0, 1] (n_vars,).
            target: Target values in ORIGINAL space (n_vars,).
            var_types: Variable types (n_vars,) with values {0, 1, 2}.
            var_lb: Lower bounds (n_vars,). Required for normalization.
            var_ub: Upper bounds (n_vars,). Required for normalization.
            constr_matrix: Sparse constraint matrix (list format).
            constr_rhs: RHS values.
            constr_sense: Constraint sense.
            n_vars: Number of variables.
            n_constrs: Number of constraints.
            A: Dense constraint matrix.
            b: RHS values (dense format).
            sense: Constraint sense (dense format).
        
        Returns:
            Dict with 'total', 'task', 'constraint', 'integrality' losses.
        """
        # Task loss (stratified based on var_types, normalizes target internally)
        l_task = self.task_loss(pred, target, var_types=var_types, var_lb=var_lb, var_ub=var_ub)
        
        # Constraint loss (denormalizes pred to original space)
        if A is not None and b is not None and sense is not None:
            # Dense format
            l_constr = self.constraint_loss.forward_batch(
                pred, A, b, sense, 
                var_types=var_types, var_lb=var_lb, var_ub=var_ub
            )
        elif constr_matrix is not None and constr_rhs is not None and constr_sense is not None:
            # Sparse format
            l_constr = self.constraint_loss(
                pred, constr_matrix, constr_rhs, constr_sense, n_vars, n_constrs, 
                var_types=var_types, var_lb=var_lb, var_ub=var_ub
            )
        else:
            l_constr = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Integrality loss (denormalizes pred for integer variables)
        l_int = self.integrality_loss(pred, var_types=var_types, var_lb=var_lb, var_ub=var_ub)
        
        # Combined loss
        total = l_task + self.lambda_constraint * l_constr + self.lambda_integrality * l_int
        
        return {
            'total': total,
            'task': l_task,
            'constraint': l_constr,
            'integrality': l_int,
        }


class FeasibilityChecker:
    """
    Check feasibility of solutions and compute violation metrics.
    """
    
    @staticmethod
    def check_feasibility(
        solution: torch.Tensor,
        constr_matrix: List[Tuple[int, int, float]],
        constr_rhs: torch.Tensor,
        constr_sense: torch.Tensor,
        n_vars: int,
        n_constrs: int,
        tol: float = 1e-6,
    ) -> Tuple[bool, int, float]:
        """
        Check if solution is feasible.
        
        Args:
            solution: Solution to check (n_vars,).
            constr_matrix: Constraint matrix in sparse format.
            constr_rhs: RHS values.
            constr_sense: Constraint sense.
            n_vars: Number of variables.
            n_constrs: Number of constraints.
            tol: Tolerance for constraint satisfaction.
        
        Returns:
            Tuple of (is_feasible, num_violations, total_violation).
        """
        device = solution.device
        
        # Compute Ax
        ax = torch.zeros(n_constrs, device=device)
        for constr_idx, var_idx, coeff in constr_matrix:
            ax[constr_idx] += coeff * solution[var_idx]
        
        # Check violations
        violations = torch.zeros(n_constrs, device=device)
        
        le_mask = constr_sense == 1
        violations[le_mask] = F.relu(ax[le_mask] - constr_rhs[le_mask] - tol)
        
        ge_mask = constr_sense == 2
        violations[ge_mask] = F.relu(constr_rhs[ge_mask] - ax[ge_mask] - tol)
        
        eq_mask = constr_sense == 3
        violations[eq_mask] = F.relu(torch.abs(ax[eq_mask] - constr_rhs[eq_mask]) - tol)
        
        num_violations = (violations > 0).sum().item()
        total_violation = violations.sum().item()
        is_feasible = num_violations == 0
        
        return is_feasible, num_violations, total_violation
    
    @staticmethod
    def compute_objective(
        solution: torch.Tensor,
        obj_coeffs: torch.Tensor,
        obj_sense: int = 1,
    ) -> float:
        """
        Compute objective value.
        
        Args:
            solution: Solution (n_vars,).
            obj_coeffs: Objective coefficients (n_vars,).
            obj_sense: 1 for minimize, -1 for maximize.
        
        Returns:
            Objective value (negated if maximization for consistent "lower is better").
        """
        obj_val = (solution * obj_coeffs).sum().item()
        return obj_val * obj_sense  # Positive for minimize, negative for maximize


class AccuracyMetrics:
    """
    Compute accuracy metrics for solution prediction.
    """
    
    @staticmethod
    def compute_accuracy(
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute prediction accuracy metrics.
        
        Args:
            pred: Predicted probabilities (n_vars,).
            target: Target labels (n_vars,).
            threshold: Threshold for discretization.
        
        Returns:
            Dict with 'accuracy', 'precision', 'recall', 'f1'.
        """
        pred_binary = (pred > threshold).float()
        
        correct = (pred_binary == target).float()
        accuracy = correct.mean().item()
        
        # For binary classification metrics
        tp = ((pred_binary == 1) & (target == 1)).sum().float()
        fp = ((pred_binary == 1) & (target == 0)).sum().float()
        fn = ((pred_binary == 0) & (target == 1)).sum().float()
        
        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-8))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    @staticmethod
    def compute_solution_quality(
        pred_solution: torch.Tensor,
        opt_solution: torch.Tensor,
        obj_coeffs: torch.Tensor,
        obj_sense: int = 1,
    ) -> Dict[str, float]:
        """
        Compute solution quality metrics.
        
        Args:
            pred_solution: Predicted solution (n_vars,).
            opt_solution: Optimal solution (n_vars,).
            obj_coeffs: Objective coefficients.
            obj_sense: Objective sense.
        
        Returns:
            Dict with 'pred_obj', 'opt_obj', 'gap', 'relative_gap'.
        """
        pred_obj = (pred_solution * obj_coeffs).sum().item()
        opt_obj = (opt_solution * obj_coeffs).sum().item()
        
        gap = abs(pred_obj - opt_obj)
        
        if abs(opt_obj) > 1e-8:
            relative_gap = gap / abs(opt_obj)
        else:
            relative_gap = gap
        
        return {
            'pred_obj': pred_obj,
            'opt_obj': opt_obj,
            'gap': gap,
            'relative_gap': relative_gap,
        }
