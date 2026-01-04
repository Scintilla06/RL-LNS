"""
Physics-informed loss functions for MILP solution prediction.

Implements:
- Task Loss: BCE for binary variables
- Constraint Loss: Penalizes constraint violations
- Integrality Loss: Pushes predictions toward integer values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class TaskLoss(nn.Module):
    """
    Task loss for binary variable prediction.
    
    Uses Binary Cross Entropy against optimal solution labels.
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        """
        Args:
            label_smoothing: Label smoothing factor (0 = no smoothing).
        """
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute BCE loss.
        
        Args:
            pred: Predicted probabilities (n_vars,) or (batch, n_vars).
            target: Target labels (n_vars,) or (batch, n_vars).
            weight: Optional per-variable weights.
        
        Returns:
            Scalar loss value.
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE
        # loss = F.binary_cross_entropy(pred, target, weight=weight, reduction='none')
        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight, reduction='none')
        
        return loss.mean()


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
    
    def forward(
        self,
        pred: torch.Tensor,
        constr_matrix: List[Tuple[int, int, float]],
        constr_rhs: torch.Tensor,
        constr_sense: torch.Tensor,
        n_vars: int,
        n_constrs: int,
    ) -> torch.Tensor:
        """
        Compute constraint violation loss.
        
        Args:
            pred: Predicted solution (n_vars,).
            constr_matrix: List of (constr_idx, var_idx, coeff) tuples.
            constr_rhs: Right-hand side values (n_constrs,).
            constr_sense: Constraint sense (n_constrs,): 1=<=, 2=>=, 3===.
            n_vars: Number of variables.
            n_constrs: Number of constraints.
        
        Returns:
            Constraint violation loss.
        """
        device = pred.device
        
        # Compute Ax for each constraint
        ax = torch.zeros(n_constrs, device=device)
        
        # Use sigmoid to get probabilities for constraint calculation
        probs = torch.sigmoid(pred)
        
        for constr_idx, var_idx, coeff in constr_matrix:
            ax[constr_idx] += coeff * probs[var_idx]
        
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
    ) -> torch.Tensor:
        """
        Batch version using dense constraint matrix.
        
        Args:
            pred: Predicted solution (batch, n_vars) or (n_vars,).
            A: Constraint matrix (n_constrs, n_vars).
            b: RHS values (n_constrs,).
            sense: Constraint sense (n_constrs,).
        
        Returns:
            Constraint violation loss.
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            
        # Use sigmoid to get probabilities for constraint calculation
        probs = torch.sigmoid(pred)
        
        # Compute Ax: (batch, n_constrs)
        ax = torch.matmul(probs, A.T)
        
        # Expand b for batch: (1, n_constrs)
        b = b.unsqueeze(0)
        
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
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Reduction method ("mean", "sum", "none").
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute integrality loss.
        
        Args:
            pred: Predicted values (n_vars,) or (batch, n_vars).
        
        Returns:
            Integrality loss.
        """
        # Use sigmoid to get probabilities
        probs = torch.sigmoid(pred)
        
        # Periodic cosine loss
        loss = 1 - torch.cos(2 * torch.pi * probs)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for MILP solution prediction.
    
    L = L_task + lambda_1 * L_constraint + lambda_2 * L_integrality
    """
    
    def __init__(
        self,
        lambda_constraint: float = 0.1,
        lambda_integrality: float = 0.01,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            lambda_constraint: Weight for constraint loss.
            lambda_integrality: Weight for integrality loss.
            label_smoothing: Label smoothing for task loss.
        """
        super().__init__()
        
        self.lambda_constraint = lambda_constraint
        self.lambda_integrality = lambda_integrality
        
        self.task_loss = TaskLoss(label_smoothing=label_smoothing)
        self.constraint_loss = ConstraintLoss()
        self.integrality_loss = IntegralityLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
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
        Compute combined physics-informed loss.
        
        Supports both sparse (constr_matrix) and dense (A, b) constraint formats.
        
        Args:
            pred: Predicted probabilities.
            target: Target labels.
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
        # Task loss
        l_task = self.task_loss(pred, target)
        
        # Constraint loss
        if A is not None and b is not None and sense is not None:
            # Dense format
            l_constr = self.constraint_loss.forward_batch(pred, A, b, sense)
        elif constr_matrix is not None and constr_rhs is not None and constr_sense is not None:
            # Sparse format
            l_constr = self.constraint_loss(
                pred, constr_matrix, constr_rhs, constr_sense, n_vars, n_constrs
            )
        else:
            l_constr = torch.tensor(0.0, device=pred.device)
        
        # Integrality loss
        l_int = self.integrality_loss(pred)
        
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
