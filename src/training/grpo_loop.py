"""
GRPO (Group Relative Policy Optimization) Training for MILP Solution Improvement.

Implements:
- Group sampling with G samples per instance
- Reward computation based on solution quality
- GRPO loss with relative advantages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Optional, Dict, Any, List, Tuple, Callable
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from ..model.neuro_solver import NeuroSolver
from ..model.heads import SolutionOutput
from .physics_loss import FeasibilityChecker


class RewardCalculator:
    """
    Calculate rewards for MILP solutions.
    
    Rewards are based on:
    - Objective value improvement
    - Feasibility (constraint satisfaction)
    - Solution quality vs baseline
    """
    
    def __init__(
        self,
        baseline_type: str = "mean",
        infeasibility_penalty: float = 10.0,
        obj_scale: float = 1.0,
    ):
        """
        Args:
            baseline_type: Type of baseline ("mean", "min", "ref").
            infeasibility_penalty: Penalty for infeasible solutions.
            obj_scale: Scale factor for objective values.
        """
        self.baseline_type = baseline_type
        self.infeasibility_penalty = infeasibility_penalty
        self.obj_scale = obj_scale
    
    def compute_rewards(
        self,
        solutions: torch.Tensor,
        obj_coeffs: torch.Tensor,
        obj_sense: int,
        constr_matrix: List[Tuple[int, int, float]],
        constr_rhs: torch.Tensor,
        constr_sense: torch.Tensor,
        n_vars: int,
        n_constrs: int,
        reference_obj: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute rewards for a group of solutions.
        
        Args:
            solutions: Group of solutions (G, n_vars).
            obj_coeffs: Objective coefficients (n_vars,).
            obj_sense: 1 for minimize, -1 for maximize.
            constr_matrix: Constraint matrix.
            constr_rhs: RHS values.
            constr_sense: Constraint sense.
            n_vars: Number of variables.
            n_constrs: Number of constraints.
            reference_obj: Reference objective (e.g., optimal).
        
        Returns:
            Tuple of (rewards, info_dict).
        """
        G = solutions.size(0)
        device = solutions.device
        
        rewards = torch.zeros(G, device=device)
        feasible_mask = torch.zeros(G, dtype=torch.bool, device=device)
        obj_values = torch.zeros(G, device=device)
        
        for i in range(G):
            sol = solutions[i]
            
            # Compute objective
            obj_val = (sol * obj_coeffs).sum()
            obj_values[i] = obj_val
            
            # Check feasibility
            is_feasible, num_violations, total_violation = FeasibilityChecker.check_feasibility(
                sol, constr_matrix, constr_rhs, constr_sense, n_vars, n_constrs
            )
            feasible_mask[i] = is_feasible
            
            # Compute reward
            if is_feasible:
                # Objective-based reward (lower is better for minimization)
                if obj_sense == 1:  # minimize
                    rewards[i] = -obj_val * self.obj_scale
                else:  # maximize
                    rewards[i] = obj_val * self.obj_scale
            else:
                # Penalty for infeasibility
                rewards[i] = -self.infeasibility_penalty - total_violation
        
        # Compute relative advantages
        if self.baseline_type == "mean":
            baseline = rewards.mean()
        elif self.baseline_type == "min":
            baseline = rewards.min()
        elif self.baseline_type == "ref" and reference_obj is not None:
            if obj_sense == 1:
                baseline = -reference_obj * self.obj_scale
            else:
                baseline = reference_obj * self.obj_scale
        else:
            baseline = 0.0
        
        advantages = rewards - baseline
        
        info = {
            'rewards': rewards,
            'advantages': advantages,
            'baseline': baseline,
            'obj_values': obj_values,
            'feasible_mask': feasible_mask,
            'num_feasible': feasible_mask.sum().item(),
            'feasibility_rate': feasible_mask.float().mean().item(),
        }
        
        return advantages, info


class GroupSampler:
    """
    Sample groups of solutions from the model.
    """
    
    def __init__(
        self,
        group_size: int = 16,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Args:
            group_size: Number of samples per instance (G).
            temperature: Sampling temperature.
            top_k: If set, sample from top-k predictions.
        """
        self.group_size = group_size
        self.temperature = temperature
        self.top_k = top_k
    
    def sample(
        self,
        model: NeuroSolver,
        data: Any,
        mode: str = "gnn",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample G solutions from the model.
        
        Args:
            model: NeuroSolver model.
            data: Input data.
            mode: Model mode ("gnn" or "text").
        
        Returns:
            Tuple of (samples, log_probs, probs).
            - samples: (G, n_vars) binary solutions
            - log_probs: (G, n_vars) log probabilities
            - probs: (n_vars,) predicted probabilities
        """
        # Get model predictions
        output = model(data=data, mode=mode)
        probs = output.primal  # (n_vars,)
        
        # Apply temperature
        if self.temperature != 1.0:
            # Convert to logits, scale, convert back
            logits = torch.log(probs / (1 - probs + 1e-8))
            logits = logits / self.temperature
            probs_scaled = torch.sigmoid(logits)
        else:
            probs_scaled = probs
        
        # Sample G solutions
        n_vars = probs.size(0)
        device = probs.device
        
        samples = torch.zeros(self.group_size, n_vars, device=device)
        log_probs = torch.zeros(self.group_size, n_vars, device=device)
        
        for g in range(self.group_size):
            # Bernoulli sampling
            sample = torch.bernoulli(probs_scaled)
            samples[g] = sample
            
            # Compute log probability
            lp = sample * torch.log(probs + 1e-8) + (1 - sample) * torch.log(1 - probs + 1e-8)
            log_probs[g] = lp
        
        return samples, log_probs, probs


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    
    Features:
    - Sample G solutions per instance
    - Compute rewards based on solution quality
    - GRPO loss with relative advantages
    - KL divergence regularization
    """
    
    def __init__(
        self,
        model: NeuroSolver,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        group_size: int = 16,
        batch_size: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 1,
        max_grad_norm: float = 1.0,
        kl_coef: float = 0.01,
        entropy_coef: float = 0.01,
        temperature: float = 1.0,
        baseline_type: str = "mean",
        infeasibility_penalty: float = 10.0,
        output_dir: str = "./outputs_grpo",
        logging_steps: int = 10,
        save_steps: int = 500,
        fp16: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_mode: str = "online",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Pretrained NeuroSolver model.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            group_size: Number of samples per instance (G).
            batch_size: Batch size (typically 1 for GRPO).
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            num_epochs: Number of epochs.
            max_grad_norm: Max gradient norm.
            kl_coef: KL divergence coefficient.
            entropy_coef: Entropy bonus coefficient.
            temperature: Sampling temperature.
            baseline_type: Type of baseline for advantages.
            infeasibility_penalty: Penalty for infeasible solutions.
            output_dir: Output directory.
            logging_steps: Steps between logging.
            save_steps: Steps between saving.
            fp16: Use mixed precision.
            wandb_project: WandB project.
            wandb_run_name: WandB run name.
            device: Device for training.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.group_size = group_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.temperature = temperature
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.fp16 = fp16
        
        self.device = device or model.device
        
        # Initialize components
        self.sampler = GroupSampler(
            group_size=group_size,
            temperature=temperature,
        )
        
        self.reward_calculator = RewardCalculator(
            baseline_type=baseline_type,
            infeasibility_penalty=infeasibility_penalty,
        )
        
        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # Store reference model (frozen) for KL divergence
        self.ref_model = None  # Will be set during training
        
        # WandB
        self.use_wandb = HAS_WANDB and wandb_project is not None
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                mode=wandb_mode,
                config={
                    'group_size': group_size,
                    'learning_rate': learning_rate,
                    'kl_coef': kl_coef,
                    'entropy_coef': entropy_coef,
                    'temperature': temperature,
                }
            )
        
        # Training state
        self.global_step = 0
    
    def _extract_constraints(self, data: "HeteroData") -> Dict[str, Any]:
        """Extract constraint information from graph data."""
        edge_index = data['var', 'participates', 'constr'].edge_index
        edge_attr = data['var', 'participates', 'constr'].edge_attr
        
        constr_matrix = []
        for i in range(edge_index.size(1)):
            var_idx = edge_index[0, i].item()
            constr_idx = edge_index[1, i].item()
            coeff = edge_attr[i, 0].item()
            constr_matrix.append((constr_idx, var_idx, coeff))
        
        constr_features = data['constr'].x
        constr_rhs = constr_features[:, 0]
        constr_sense = constr_features[:, 1:4].argmax(dim=1) + 1
        
        # Extract objective
        var_features = data['var'].x
        obj_coeffs = var_features[:, 0]  # First feature is obj coefficient
        
        return {
            'constr_matrix': constr_matrix,
            'constr_rhs': constr_rhs,
            'constr_sense': constr_sense,
            'obj_coeffs': obj_coeffs,
            'obj_sense': 1,  # Assume minimization
            'n_vars': data['var'].x.size(0),
            'n_constrs': data['constr'].x.size(0),
        }
    
    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        probs: torch.Tensor,
        ref_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss.
        
        L_GRPO = -E[A * log_prob] + kl_coef * KL(pi || ref_pi) - entropy_coef * H(pi)
        
        Args:
            log_probs: Log probabilities of samples (G, n_vars).
            advantages: Relative advantages (G,).
            probs: Current probabilities (n_vars,).
            ref_probs: Reference probabilities for KL (n_vars,).
        
        Returns:
            Dict with 'total', 'policy', 'kl', 'entropy' losses.
        """
        G, n_vars = log_probs.size()
        
        # Policy gradient loss: -E[A * sum(log_prob)]
        # Expand advantages: (G,) -> (G, 1)
        advantages_expanded = advantages.unsqueeze(1)
        
        # Weighted log probs: (G, n_vars)
        weighted_log_probs = advantages_expanded * log_probs
        
        # Policy loss: negative expected weighted log prob
        policy_loss = -weighted_log_probs.mean()
        
        # KL divergence loss
        if ref_probs is not None and self.kl_coef > 0:
            # KL(pi || ref_pi) = sum[pi * log(pi/ref_pi)]
            kl_div = probs * (torch.log(probs + 1e-8) - torch.log(ref_probs + 1e-8))
            kl_div += (1 - probs) * (torch.log(1 - probs + 1e-8) - torch.log(1 - ref_probs + 1e-8))
            kl_loss = kl_div.mean()
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device)
        
        # Entropy bonus
        # H(pi) = -sum[pi * log(pi) + (1-pi) * log(1-pi)]
        entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
        entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy
        
        # Total loss
        total_loss = policy_loss + self.kl_coef * kl_loss + self.entropy_coef * entropy_loss
        
        return {
            'total': total_loss,
            'policy': policy_loss,
            'kl': kl_loss,
            'entropy': entropy.mean(),
        }
    
    def train_step(self, data: Any) -> Dict[str, float]:
        """
        Single GRPO training step.
        
        Args:
            data: Input data (single instance).
        
        Returns:
            Dict of metrics.
        """
        data = data.to(self.device)
        
        # Extract constraints and objectives
        constr_info = self._extract_constraints(data)
        
        # Sample G solutions
        self.model.eval()  # Use eval mode for sampling
        with torch.no_grad():
            samples, log_probs, probs = self.sampler.sample(
                self.model, data, mode="gnn"
            )
            
            # Get reference probabilities
            if self.ref_model is not None:
                ref_output = self.ref_model(data=data, mode="gnn")
                ref_probs = ref_output.primal.detach()
            else:
                ref_probs = probs.detach().clone()
        
        # Compute rewards
        advantages, reward_info = self.reward_calculator.compute_rewards(
            solutions=samples,
            obj_coeffs=constr_info['obj_coeffs'],
            obj_sense=constr_info['obj_sense'],
            constr_matrix=constr_info['constr_matrix'],
            constr_rhs=constr_info['constr_rhs'],
            constr_sense=constr_info['constr_sense'],
            n_vars=constr_info['n_vars'],
            n_constrs=constr_info['n_constrs'],
        )
        
        # Training mode for loss computation
        self.model.train()
        
        # Recompute log probs with gradients
        output = self.model(data=data, mode="gnn")
        current_probs = output.primal
        
        # Compute log probs for sampled solutions
        log_probs_grad = samples * torch.log(current_probs + 1e-8) + \
                         (1 - samples) * torch.log(1 - current_probs + 1e-8)
        
        # Compute GRPO loss
        if self.fp16:
            with torch.cuda.amp.autocast():
                losses = self.compute_grpo_loss(
                    log_probs=log_probs_grad,
                    advantages=advantages,
                    probs=current_probs,
                    ref_probs=ref_probs,
                )
        else:
            losses = self.compute_grpo_loss(
                log_probs=log_probs_grad,
                advantages=advantages,
                probs=current_probs,
                ref_probs=ref_probs,
            )
        
        # Backward pass
        if self.fp16:
            self.scaler.scale(losses['total']).backward()
        else:
            losses['total'].backward()
        
        metrics = {k: v.item() for k, v in losses.items()}
        metrics.update({
            'feasibility_rate': reward_info['feasibility_rate'],
            'mean_reward': reward_info['rewards'].mean().item(),
            'mean_obj': reward_info['obj_values'].mean().item(),
        })
        
        return metrics
    
    def train(self):
        """
        Main GRPO training loop.
        """
        # Create data loader (batch size = 1 for GRPO)
        if HAS_PYG:
            from torch_geometric.loader import DataLoader as PyGDataLoader
            train_loader = PyGDataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=0,
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=True,
                collate_fn=lambda x: x[0],
            )
        
        print(f"Starting GRPO training:")
        print(f"  Num samples: {len(self.train_dataset)}")
        print(f"  Num epochs: {self.num_epochs}")
        print(f"  Group size: {self.group_size}")
        print(f"  Learning rate: {self.learning_rate}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            epoch_metrics = {
                'total': 0, 'policy': 0, 'kl': 0, 'entropy': 0,
                'feasibility_rate': 0, 'mean_reward': 0, 'mean_obj': 0,
            }
            num_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for step, data in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(data)
                
                # Accumulate metrics
                for k, v in metrics.items():
                    epoch_metrics[k] += v
                num_steps += 1
                
                # Gradient clipping and optimizer step
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_metrics = {k: v / num_steps for k, v in epoch_metrics.items()}
                    
                    progress_bar.set_postfix({
                        'loss': f"{avg_metrics['total']:.4f}",
                        'feas': f"{avg_metrics['feasibility_rate']:.2f}",
                    })
                    
                    if self.use_wandb:
                        wandb.log({
                            'grpo/loss': avg_metrics['total'],
                            'grpo/policy_loss': avg_metrics['policy'],
                            'grpo/kl_loss': avg_metrics['kl'],
                            'grpo/entropy': avg_metrics['entropy'],
                            'grpo/feasibility_rate': avg_metrics['feasibility_rate'],
                            'grpo/mean_reward': avg_metrics['mean_reward'],
                            'grpo/step': self.global_step,
                        })
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
            
            # End of epoch
            avg_metrics = {k: v / num_steps for k, v in epoch_metrics.items()}
            print(f"Epoch {epoch + 1} - Loss: {avg_metrics['total']:.4f}, "
                  f"Feasibility: {avg_metrics['feasibility_rate']:.2%}")
            
            self.save_checkpoint(f'epoch_{epoch + 1}')
        
        print("\nGRPO training complete!")
        self.save_checkpoint('final')
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(checkpoint_dir))
        
        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_dir / 'trainer_state.pt')
        
        print(f"Checkpoint saved to {checkpoint_dir}")
