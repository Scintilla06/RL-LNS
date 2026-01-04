"""
SFT Trainer for Physics-Informed Supervised Fine-Tuning.

Trains the NeuroSolver model on MILP instances with:
- Task loss (BCE against optimal solutions)
- Constraint loss (penalize violations)
- Integrality loss (push toward integer values)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict, Any, List, Callable
from tqdm import tqdm
import json
from pathlib import Path

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
from .physics_loss import PhysicsInformedLoss, FeasibilityChecker, AccuracyMetrics


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for NeuroSolver.
    
    Features:
    - Physics-informed loss (task + constraint + integrality)
    - Gradient accumulation for effective batch size
    - Mixed precision training
    - Learning rate scheduling with warmup
    - WandB logging
    """
    
    def __init__(
        self,
        model: NeuroSolver,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_ratio: float = 0.03,
        max_grad_norm: float = 1.0,
        lambda_constraint: float = 0.1,
        lambda_integrality: float = 0.01,
        output_dir: str = "./outputs",
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        fp16: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_mode: str = "online",
        max_samples_per_epoch: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: NeuroSolver model to train.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            batch_size: Batch size per step.
            gradient_accumulation_steps: Steps to accumulate gradients.
            learning_rate: Peak learning rate.
            weight_decay: Weight decay for AdamW.
            num_epochs: Number of training epochs.
            warmup_ratio: Ratio of warmup steps.
            max_grad_norm: Maximum gradient norm for clipping.
            lambda_constraint: Weight for constraint loss.
            lambda_integrality: Weight for integrality loss.
            output_dir: Output directory for checkpoints.
            logging_steps: Steps between logging.
            eval_steps: Steps between evaluation.
            save_steps: Steps between saving checkpoints.
            fp16: Whether to use mixed precision.
            wandb_project: WandB project name.
            wandb_run_name: WandB run name.
            wandb_mode: WandB mode (online/offline).
            max_samples_per_epoch: Maximum number of samples to use per epoch.
            device: Device for training.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.effective_batch_size = batch_size * gradient_accumulation_steps
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.max_samples_per_epoch = max_samples_per_epoch
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.fp16 = fp16
        
        self.device = device or model.device
        
        # Initialize loss function
        self.loss_fn = PhysicsInformedLoss(
            lambda_constraint=lambda_constraint,
            lambda_integrality=lambda_integrality,
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler (created in train())
        self.scheduler = None
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # WandB
        self.use_wandb = HAS_WANDB and wandb_project is not None
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                mode=wandb_mode,
                config={
                    'batch_size': batch_size,
                    'effective_batch_size': self.effective_batch_size,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'lambda_constraint': lambda_constraint,
                    'lambda_integrality': lambda_integrality,
                }
            )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters with/without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = AdamW([
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.learning_rate)
        
        return optimizer
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=self.learning_rate * 0.1,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
    
    def _prepare_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Prepare batch for training.
        
        Handles both graph and text modes.
        """
        if HAS_PYG and isinstance(batch, HeteroData):
            # Graph mode
            return {
                'data': batch.to(self.device),
                'target': batch['var'].y.to(self.device) if hasattr(batch['var'], 'y') else None,
                'mode': 'gnn',
            }
        elif isinstance(batch, dict):
            # Could be text mode or processed graph
            return batch
        elif isinstance(batch, list):
            # List of samples - process first one
            return self._prepare_batch(batch[0])
        else:
            raise ValueError(f"Unknown batch type: {type(batch)}")
    
    def _extract_constraints(self, data: "HeteroData") -> Dict[str, Any]:
        """Extract constraint information from graph data."""
        # Build constraint matrix from edges
        edge_index = data['var', 'participates', 'constr'].edge_index
        edge_attr = data['var', 'participates', 'constr'].edge_attr
        
        constr_matrix = []
        for i in range(edge_index.size(1)):
            var_idx = edge_index[0, i].item()
            constr_idx = edge_index[1, i].item()
            coeff = edge_attr[i, 0].item()
            constr_matrix.append((constr_idx, var_idx, coeff))
        
        # Get constraint RHS and sense from constraint node features
        # Features: [rhs, sense_onehot(3)]
        constr_features = data['constr'].x
        constr_rhs = constr_features[:, 0]
        constr_sense = constr_features[:, 1:4].argmax(dim=1) + 1  # 1, 2, or 3
        
        return {
            'constr_matrix': constr_matrix,
            'constr_rhs': constr_rhs,
            'constr_sense': constr_sense,
            'n_vars': data['var'].x.size(0),
            'n_constrs': data['constr'].x.size(0),
        }
    
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Training batch.
        
        Returns:
            Dict of loss values.
        """
        prepared = self._prepare_batch(batch)
        
        # Forward pass
        if self.fp16:
            with torch.cuda.amp.autocast():
                output = self.model(data=prepared.get('data'), mode=prepared.get('mode', 'gnn'))
        else:
            output = self.model(data=prepared.get('data'), mode=prepared.get('mode', 'gnn'))
        
        # Get target
        target = prepared.get('target')
        if target is None:
            raise ValueError("No target labels in batch")
        
        # Extract constraints for physics loss
        if 'data' in prepared and prepared['mode'] == 'gnn':
            constr_info = self._extract_constraints(prepared['data'])
        else:
            constr_info = {}
        
        # Compute loss
        if self.fp16:
            with torch.cuda.amp.autocast():
                losses = self.loss_fn(
                    pred=output.primal,
                    target=target,
                    **constr_info,
                )
        else:
            losses = self.loss_fn(
                pred=output.primal,
                target=target,
                **constr_info,
            )
        
        # Scale loss for gradient accumulation
        loss = losses['total'] / self.gradient_accumulation_steps
        
        # Backward pass
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train(self):
        """
        Main training loop.
        """
        # Create data loader
        train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        
        # Calculate total steps
        num_training_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        
        # Create scheduler
        self._create_scheduler(num_training_steps)
        
        print(f"Starting training:")
        print(f"  Num samples: {len(self.train_dataset)}")
        print(f"  Num epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.effective_batch_size}")
        print(f"  Total steps: {num_training_steps}")
        print(f"  Trainable params: {self.model.get_trainable_params():,}")
        
        self.model.train()
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            epoch_losses = {'total': 0, 'task': 0, 'constraint': 0, 'integrality': 0}
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Check max samples
                if self.max_samples_per_epoch is not None and step * self.batch_size >= self.max_samples_per_epoch:
                    break
                
                # Training step
                losses = self.train_step(batch)
                
                # Accumulate losses
                for k, v in losses.items():
                    epoch_losses[k] += v
                num_batches += 1
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
                        
                        progress_bar.set_postfix({
                            'loss': f"{avg_losses['total']:.4f}",
                            'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                        })
                        
                        if self.use_wandb:
                            wandb.log({
                                'train/loss': avg_losses['total'],
                                'train/task_loss': avg_losses['task'],
                                'train/constraint_loss': avg_losses['constraint'],
                                'train/integrality_loss': avg_losses['integrality'],
                                'train/learning_rate': self.scheduler.get_last_lr()[0],
                                'train/step': self.global_step,
                            })
                    
                    # Evaluation
                    if self.val_dataset is not None and self.global_step % self.eval_steps == 0:
                        val_metrics = self.evaluate()
                        
                        if val_metrics['loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['loss']
                            self.save_checkpoint('best')
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f'step_{self.global_step}')
            
            # End of epoch
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            print(f"Epoch {epoch + 1} - Loss: {avg_losses['total']:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}')
        
        print("\nTraining complete!")
        self.save_checkpoint('final')
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Dict of evaluation metrics.
        """
        if self.val_dataset is None:
            return {}
        
        self.model.eval()
        val_loader = self._create_dataloader(self.val_dataset, shuffle=False)
        
        total_losses = {'total': 0, 'task': 0, 'constraint': 0, 'integrality': 0}
        total_accuracy = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                prepared = self._prepare_batch(batch)
                
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        output = self.model(data=prepared.get('data'), mode=prepared.get('mode', 'gnn'))
                else:
                    output = self.model(data=prepared.get('data'), mode=prepared.get('mode', 'gnn'))
                
                target = prepared.get('target')
                
                if target is None:
                    continue
                
                # Extract constraints
                if 'data' in prepared and prepared['mode'] == 'gnn':
                    constr_info = self._extract_constraints(prepared['data'])
                else:
                    constr_info = {}
                
                # Compute loss
                losses = self.loss_fn(
                    pred=output.primal,
                    target=target,
                    **constr_info,
                )
                
                for k, v in losses.items():
                    total_losses[k] += v.item()
                
                # Compute accuracy
                metrics = AccuracyMetrics.compute_accuracy(output.primal, target)
                total_accuracy += metrics['accuracy']
                num_samples += 1
        
        # Average metrics
        avg_metrics = {
            'loss': total_losses['total'] / num_samples,
            'task_loss': total_losses['task'] / num_samples,
            'constraint_loss': total_losses['constraint'] / num_samples,
            'integrality_loss': total_losses['integrality'] / num_samples,
            'accuracy': total_accuracy / num_samples,
        }
        
        print(f"Validation - Loss: {avg_metrics['loss']:.4f}, Accuracy: {avg_metrics['accuracy']:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_metrics['loss'],
                'val/accuracy': avg_metrics['accuracy'],
                'val/step': self.global_step,
            })
        
        return avg_metrics
    
    def _create_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        """Create DataLoader for dataset."""
        if HAS_PYG:
            from torch_geometric.loader import DataLoader as PyGDataLoader
            return PyGDataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0,  # Single-threaded for simplicity
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=lambda x: x,
            )
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_dir))
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_dir / 'trainer_state.pt')
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(path)
        
        # Load model
        self.model = NeuroSolver.from_pretrained(str(checkpoint_dir))
        
        # Load training state
        state = torch.load(checkpoint_dir / 'trainer_state.pt')
        self.global_step = state['global_step']
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if state['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.best_val_loss = state['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_dir}")
