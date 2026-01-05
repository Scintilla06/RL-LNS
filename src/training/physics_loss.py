"""
Physics-informed loss functions for MILP solution prediction.

Implements:
- Task Loss: BCE for binary variables, Huber for integer/continuous
- Constraint Loss: Penalizes constraint violations
- Integrality Loss: Pushes predictions toward integer values (only for integer vars)

Supports mixed variable types (Binary, Integer, Continuous).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


# Variable type constants
VAR_BINARY = 0
VAR_CONTINUOUS = 1
VAR_INTEGER = 2


class TaskLoss(nn.Module):
    """
    Task loss for mixed-variable MILP prediction.
    
    Uses stratified loss based on variable type:
    - Binary: Binary Cross Entropy
    - Integer/Continuous: Huber Loss (SmoothL1)
    """
    
    def __init__(self, label_smoothing: float = 0.0, huber_delta: float = 1.0):
        """
        Args:
            label_smoothing: Label smoothing factor for binary variables.
            huber_delta: Delta parameter for Huber loss.
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.huber_delta = huber_delta
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        var_types: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute stratified task loss.
        
        Args:
            pred: Predicted values (n_vars,) or (batch, n_vars).
                  For binary: logits (before sigmoid)
                  For integer/continuous: predicted values
            target: Target labels (n_vars,) or (batch, n_vars).
            var_types: Variable types (n_vars,) with values {0, 1, 2}.
                      If None, assumes all binary.
            weight: Optional per-variable weights.
        
        Returns:
            Scalar loss value.
        """
        if var_types is None:
            # Backward compatible: assume all binary
            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight, reduction='none')
            return loss.mean()
        
        # Align pred and target shapes
        # Handle case where one is (batch, n_vars) and other is (n_vars,)
        if pred.shape != target.shape:
            if pred.dim() == 1 and target.dim() == 2 and target.shape[0] == 1:
                target = target.squeeze(0)
            elif pred.dim() == 2 and pred.shape[0] == 1 and target.dim() == 1:
                pred = pred.squeeze(0)
        
        total_loss = torch.tensor(0.0, device=pred.device)
        count = 0
        
        # Ensure shapes match for masking
        # If pred/target are (batch, n_vars) but var_types is (n_vars,),
        # we need to expand var_types or flatten pred/target
        
        if pred.dim() > 1 and var_types.dim() == 1:
            # Case: pred is (batch, n_vars), var_types is (n_vars,)
            # Expand var_types to (batch, n_vars)
            var_types = var_types.unsqueeze(0).expand_as(pred)
        
        # Binary variables: BCE loss
        binary_mask = var_types == VAR_BINARY
        if binary_mask.any():
            binary_pred = pred[binary_mask]
            binary_target = target[binary_mask]
            
            if self.label_smoothing > 0:
                binary_target = binary_target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
            binary_weight = weight[binary_mask] if weight is not None else None
            binary_loss = F.binary_cross_entropy_with_logits(
                binary_pred, binary_target, weight=binary_weight, reduction='mean'
            )
            total_loss = total_loss + binary_loss * binary_mask.sum()
            count += binary_mask.sum()
        
        # Integer variables: Huber loss
        integer_mask = var_types == VAR_INTEGER
        if integer_mask.any():
            integer_pred = pred[integer_mask]
            integer_target = target[integer_mask]
            
            integer_loss = F.smooth_l1_loss(
                integer_pred, integer_target, beta=self.huber_delta, reduction='mean'
            )
            total_loss = total_loss + integer_loss * integer_mask.sum()
            count += integer_mask.sum()
        
        # Continuous variables: Huber loss
        continuous_mask = var_types == VAR_CONTINUOUS
        if continuous_mask.any():
            continuous_pred = pred[continuous_mask]
            continuous_target = target[continuous_mask]
            
            continuous_loss = F.smooth_l1_loss(
                continuous_pred, continuous_target, beta=self.huber_delta, reduction='mean'
            )
            total_loss = total_loss + continuous_loss * continuous_mask.sum()
            count += continuous_mask.sum()
        
        if count > 0:
            return total_loss / count
        return total_loss


class ConstraintLoss(nn.Module):
    """
    Constraint violation loss.
    
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
    
    def _get_solution_values(
        self,
        pred: torch.Tensor,
        var_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get solution values from predictions based on variable types.
        
        For constraint checking, we need actual values:
        - Binary: sigmoid(pred) gives probability in (0, 1)
        - Integer/Continuous: pred gives direct value
        """
        if var_types is None:
            # Backward compatible: assume all binary
            return torch.sigmoid(pred)
        
        values = pred.clone()
        
        # Binary: apply sigmoid
        binary_mask = var_types == VAR_BINARY
        if binary_mask.any():
            values[binary_mask] = torch.sigmoid(pred[binary_mask])
        
        # Integer and Continuous: use raw values (already in value space)
        # No transformation needed
        
        return values
    
    def forward(
        self,
        pred: torch.Tensor,
        constr_matrix: List[Tuple[int, int, float]],
        constr_rhs: torch.Tensor,
        constr_sense: torch.Tensor,
        n_vars: int,
        n_constrs: int,
        var_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute constraint violation loss for mixed-variable MILP.
        
        Args:
            pred: Predicted solution (n_vars,).
            constr_matrix: List of (constr_idx, var_idx, coeff) tuples.
            constr_rhs: Right-hand side values (n_constrs,).
            constr_sense: Constraint sense (n_constrs,): 1=<=, 2=>=, 3===.
            n_vars: Number of variables.
            n_constrs: Number of constraints.
            var_types: Variable types (n_vars,). If None, assumes all binary.
        
        Returns:
            Constraint violation loss.
        """
        device = pred.device
        
        # Get solution values (handles different variable types)
        values = self._get_solution_values(pred, var_types)
        
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
    ) -> torch.Tensor:
        """
        Batch version using dense constraint matrix.
        
        Args:
            pred: Predicted solution (batch, n_vars) or (n_vars,).
            A: Constraint matrix (n_constrs, n_vars).
            b: RHS values (n_constrs,).
            sense: Constraint sense (n_constrs,).
            var_types: Variable types (n_vars,). If None, assumes all binary.
        
        Returns:
            Constraint violation loss.
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        
        # Get solution values (handles different variable types)
        if var_types is not None:
            values = self._get_solution_values(pred.squeeze(0), var_types).unsqueeze(0)
        else:
            values = torch.sigmoid(pred)
        
        # Compute Ax: (batch, n_constrs)
        if A.is_sparse:
            # A is (n_constrs, n_vars) sparse
            # values is (batch, n_vars) dense
            # We want values @ A.T -> (batch, n_constrs)
            # Use (A @ values.T).T
            
            # NOTE: CUDA sparse mm does NOT support FP16 (Half).
            # When using mixed-precision training (autocast), we must
            # temporarily convert to float32 for the sparse operation.
            
            # DEBUG: Print dtypes to diagnose the issue
            print(f"[DEBUG ConstraintLoss.forward_batch] Sparse MM Check:")
            print(f"  A: dtype={A.dtype}, shape={A.shape}, is_sparse={A.is_sparse}")
            print(f"  values: dtype={values.dtype}, shape={values.shape}")
            
            original_dtype = values.dtype
            
            # Disable autocast to prevent implicit casting back to half
            with torch.amp.autocast('cuda', enabled=False):
                # Force both A and values to float32 for sparse mm
                
                # Ensure A is float32
                A_float = A.float() if A.dtype != torch.float32 else A
                
                # Ensure values is float32
                values_float = values.float() if values.dtype != torch.float32 else values
                
                print(f"[DEBUG ConstraintLoss.forward_batch] Executing sparse.mm with float32:")
                print(f"  A_float: dtype={A_float.dtype}")
                print(f"  values_float: dtype={values_float.dtype}")
                
                ax_t = torch.sparse.mm(A_float, values_float.t())
                ax = ax_t.t()
            
            # Note: We keep ax in float32 for loss calculation stability
            # (No need to convert back to half)
        else:
            ax = torch.matmul(values, A.T)
        
        # Ensure ax is float32 for loss calculation stability and to match b
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
    
    Uses periodic cosine loss:
    L_int = sum_j (1 - cos(2 * pi * x_j))
    
    This loss is 0 when x_j is an integer and maximum when x_j = 0.5.
    
    IMPORTANT: For mixed-variable MILP:
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
        var_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute selective integrality loss.
        
        Args:
            pred: Predicted values (n_vars,) or (batch, n_vars).
                  For binary: should be logits (will apply sigmoid)
                  For integer: raw values
            var_types: Variable types (n_vars,) with values {0, 1, 2}.
                      If None, assumes all binary (backward compatible).
        
        Returns:
            Integrality loss (only for binary and integer variables).
        """
        if var_types is None:
            # Backward compatible: assume all binary
            probs = torch.sigmoid(pred)
            loss = 1 - torch.cos(2 * torch.pi * probs)
            
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss
        
        # Create mask for variables that need integrality
        # Binary (0) and Integer (2), NOT Continuous (1)
        integrality_mask = (var_types == VAR_BINARY) | (var_types == VAR_INTEGER)
        
        if not integrality_mask.any():
            return torch.tensor(0.0, device=pred.device)
        
        # Compute loss only for masked variables
        masked_pred = pred[integrality_mask]
        
        # For binary variables, apply sigmoid first
        binary_in_mask = var_types[integrality_mask] == VAR_BINARY
        
        loss_values = torch.zeros_like(masked_pred)
        
        # Binary: use sigmoid output
        if binary_in_mask.any():
            binary_probs = torch.sigmoid(masked_pred[binary_in_mask])
            loss_values[binary_in_mask] = 1 - torch.cos(2 * torch.pi * binary_probs)
        
        # Integer: use raw values
        integer_in_mask = var_types[integrality_mask] == VAR_INTEGER
        if integer_in_mask.any():
            integer_vals = masked_pred[integer_in_mask]
            loss_values[integer_in_mask] = 1 - torch.cos(2 * torch.pi * integer_vals)
        
        if self.reduction == "mean":
            return loss_values.mean()
        elif self.reduction == "sum":
            return loss_values.sum()
        else:
            # Return full-size tensor with zeros for continuous
            full_loss = torch.zeros_like(pred)
            full_loss[integrality_mask] = loss_values
            return full_loss


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for mixed-variable MILP solution prediction.
    
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
        huber_delta: float = 1.0,
    ):
        """
        Args:
            lambda_constraint: Weight for constraint loss.
            lambda_integrality: Weight for integrality loss.
            label_smoothing: Label smoothing for binary task loss.
            huber_delta: Delta parameter for Huber loss (integer/continuous).
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
        var_types: Optional[torch.Tensor] = None,
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
        
        Supports both sparse (constr_matrix) and dense (A, b) constraint formats.
        
        Args:
            pred: Predicted values (logits for binary, values for others).
            target: Target labels.
            var_types: Variable types (n_vars,) with values {0, 1, 2}.
                      If None, assumes all binary.
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
        # Task loss (stratified based on var_types)
        l_task = self.task_loss(pred, target, var_types=var_types)
        
        # Constraint loss (unified for all variable types)
        if A is not None and b is not None and sense is not None:
            # Dense format
            l_constr = self.constraint_loss.forward_batch(pred, A, b, sense, var_types=var_types)
        elif constr_matrix is not None and constr_rhs is not None and constr_sense is not None:
            # Sparse format
            l_constr = self.constraint_loss(
                pred, constr_matrix, constr_rhs, constr_sense, n_vars, n_constrs, var_types=var_types
            )
        else:
            l_constr = torch.tensor(0.0, device=pred.device)
        
        # Integrality loss (selective: only for binary and integer variables)
        l_int = self.integrality_loss(pred, var_types=var_types)
        
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
