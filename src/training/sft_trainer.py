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
        
        # Store base lambda for scheduling
        self.base_lambda_integrality = lambda_integrality
        
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
        self.scaler = torch.amp.GradScaler('cuda') if fp16 else None
        
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
            # Graph mode - extract constraint info from dense matrices if available
            result = {
                'data': batch.to(self.device),
                'target': batch['var'].y.to(self.device) if hasattr(batch['var'], 'y') else None,
                'mode': 'gnn',
            }
            # Use dense constraint matrices if available (new format)
            if hasattr(batch, 'A') and batch.A is not None:
                result['A'] = batch.A.to(self.device)
                result['b'] = batch.b.to(self.device)
                result['sense'] = batch.sense.to(self.device)
            return result
        elif isinstance(batch, dict):
            # Text mode - constraint info already in dict
            result = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.to(self.device)
                else:
                    result[k] = v
            return result
        elif isinstance(batch, list):
            # List of samples - process first one
            return self._prepare_batch(batch[0])
        else:
            raise ValueError(f"Unknown batch type: {type(batch)}")
    
    def _extract_var_bounds(self, prepared: Dict[str, Any], data: Any = None) -> Dict[str, torch.Tensor]:
        """
        Extract variable bounds (var_lb, var_ub) from prepared batch or graph data.
        
        Args:
            prepared: Prepared batch dictionary.
            data: Graph data object (for GNN mode).
        
        Returns:
            Dict with 'var_lb' and 'var_ub' if available.
        """
        result = {}
        
        # Try extracting from prepared dict first
        for key in ['var_lb', 'var_ub']:
            if key in prepared and prepared[key] is not None:
                val = prepared[key]
                if isinstance(val, list):
                    val = val[0]
                elif isinstance(val, torch.Tensor) and val.dim() > 1:
                    val = val[0]
                if isinstance(val, torch.Tensor):
                    result[key] = val.to(self.device)
        
        # Try extracting from graph data
        if data is not None:
            for key in ['var_lb', 'var_ub']:
                if key not in result and hasattr(data, key) and getattr(data, key) is not None:
                    result[key] = getattr(data, key).to(self.device)
        
        return result
    
    def _extract_constraints(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract constraint information from prepared batch.
        
        Supports both:
        - Dense format (A, b, sense) from new preprocessed data
        - COO components (A_row, A_col, A_val, A_shape) for text mode
        - Sparse format (from graph edges) for backward compatibility
        """
        # COO components format (from text mode dataset)
        if 'A_row' in prepared and prepared['A_row'] is not None:
            # Handle batched COO components - reconstruct sparse tensor for first sample
            A_row = prepared['A_row']
            A_col = prepared['A_col']
            A_val = prepared['A_val']
            A_shape = prepared['A_shape']
            
            # If batched (list or has batch dim), take first sample
            if isinstance(A_row, list):
                A_row, A_col, A_val, A_shape = A_row[0], A_col[0], A_val[0], A_shape[0]
            elif isinstance(A_row, torch.Tensor) and A_row.dim() > 1:
                A_row, A_col, A_val, A_shape = A_row[0], A_col[0], A_val[0], A_shape[0]
            
            # Ensure 1D tensors for indices
            A_row = A_row.flatten().long()
            A_col = A_col.flatten().long()
            A_val = A_val.flatten()
            
            # Convert A_shape to tuple (handles tensor, list, or tuple)
            if isinstance(A_shape, torch.Tensor):
                if A_shape.dim() > 1:
                    A_shape = A_shape[0]
                shape_tuple = tuple(A_shape.flatten().tolist())
            elif isinstance(A_shape, (list, tuple)):
                # Handle nested list (e.g. [[h, w]])
                if len(A_shape) > 0 and isinstance(A_shape[0], (list, tuple, torch.Tensor)):
                    A_shape = A_shape[0]
                
                if isinstance(A_shape, torch.Tensor):
                     shape_tuple = tuple(A_shape.flatten().tolist())
                else:
                     shape_tuple = tuple(int(x) for x in A_shape)
            else:
                shape_tuple = (int(A_shape[0]), int(A_shape[1]))
            
            # Build sparse tensor: indices should be (2, nnz)
            indices = torch.stack([A_row, A_col], dim=0)
            A = torch.sparse_coo_tensor(
                indices, A_val, shape_tuple
            ).to(self.device)
            
            # Extract var_types if available
            var_types = None
            if 'var_types' in prepared and prepared['var_types'] is not None:
                vt = prepared['var_types']
                if isinstance(vt, list):
                    var_types = vt[0].to(self.device)
                elif isinstance(vt, torch.Tensor) and vt.dim() > 1:
                    var_types = vt[0].to(self.device)
                else:
                    var_types = vt.to(self.device)
            
            result = {
                'A': A,
                'b': prepared['b'][0] if isinstance(prepared['b'], list) or (isinstance(prepared['b'], torch.Tensor) and prepared['b'].dim() > 1) else prepared['b'],
                'sense': prepared['sense'][0] if isinstance(prepared['sense'], list) or (isinstance(prepared['sense'], torch.Tensor) and prepared['sense'].dim() > 1) else prepared['sense'],
            }
            if var_types is not None:
                result['var_types'] = var_types
            # Add variable bounds
            result.update(self._extract_var_bounds(prepared))
            return result
        
        # Prefer dense format if available
        if 'A' in prepared and prepared['A'] is not None:
            result = {
                'A': prepared['A'],
                'b': prepared['b'],
                'sense': prepared['sense'],
            }
            # Also check for var_types in prepared
            if 'var_types' in prepared and prepared['var_types'] is not None:
                result['var_types'] = prepared['var_types'].to(self.device) if isinstance(prepared['var_types'], torch.Tensor) else prepared['var_types']
            # Add variable bounds
            result.update(self._extract_var_bounds(prepared))
            return result
        
        # Fall back to extracting from graph structure
        if 'data' in prepared and prepared.get('mode') == 'gnn':
            data = prepared['data']
            
            # New format: COO components stored separately
            # Note: PyG DataLoader batches these, so we need to extract per-sample
            if hasattr(data, 'A_row'):
                # For batched data, A_row/A_col/A_val are concatenated
                # A_shape becomes batched tensor [batch_size, 2]
                # We only support batch_size=1 for constraint info
                A_row = data.A_row.flatten().long()
                A_col = data.A_col.flatten().long()
                A_val = data.A_val.flatten().float()
                
                # Handle A_shape: could be (n_constrs, n_vars) tuple, 
                # [n_constrs, n_vars] tensor, or batched [batch, 2] tensor
                A_shape = data.A_shape
                if isinstance(A_shape, torch.Tensor):
                    if A_shape.dim() == 2:  # Batched: [batch, 2]
                        A_shape = A_shape[0]  # Take first sample
                    shape_tuple = (int(A_shape[0].item()), int(A_shape[1].item()))
                elif isinstance(A_shape, (list, tuple)):
                    # Could be nested list from batching: [[n_constrs, n_vars], ...]
                    first = A_shape[0]
                    if isinstance(first, (list, tuple, torch.Tensor)):
                        # Nested - take first sample
                        if isinstance(first, torch.Tensor):
                            shape_tuple = (int(first[0].item()), int(first[1].item()))
                        else:
                            shape_tuple = (int(first[0]), int(first[1]))
                    else:
                        # Simple tuple (n_constrs, n_vars)
                        shape_tuple = (int(A_shape[0]), int(A_shape[1]))
                else:
                    raise ValueError(f"Unknown A_shape type: {type(A_shape)}")
                
                indices = torch.stack([A_row, A_col], dim=0)
                A = torch.sparse_coo_tensor(
                    indices, A_val, shape_tuple
                ).to(self.device)
                
                result = {
                    'A': A,
                    'b': data.b.to(self.device),
                    'sense': data.sense.long().to(self.device),
                }
                # Extract var_types from graph data
                if hasattr(data, 'var_types') and data.var_types is not None:
                    result['var_types'] = data.var_types.long().to(self.device)
                # Add variable bounds
                result.update(self._extract_var_bounds(prepared, data))
                return result
            
            # Legacy format: sparse tensor directly
            if hasattr(data, 'A') and data.A is not None:
                result = {
                    'A': data.A.to(self.device),
                    'b': data.b.to(self.device),
                    'sense': data.sense.long().to(self.device),
                }
                if hasattr(data, 'var_types') and data.var_types is not None:
                    result['var_types'] = data.var_types.long().to(self.device)
                # Add variable bounds
                result.update(self._extract_var_bounds(prepared, data))
                return result
            
            # Legacy: Build from edges
            edge_index = data['var', 'participates', 'constr'].edge_index
            edge_attr = data['var', 'participates', 'constr'].edge_attr
            
            # Optimize: Build sparse A tensor directly from graph structure
            # edge_index is [var_idx, constr_idx]
            # We want A[constr_idx, var_idx] = coeff
            
            indices = torch.stack([edge_index[1], edge_index[0]])
            values = edge_attr.squeeze(-1)
            
            n_vars = data['var'].x.size(0)
            n_constrs = data['constr'].x.size(0)
            
            A = torch.sparse_coo_tensor(indices, values, (n_constrs, n_vars)).to(self.device)
            
            constr_features = data['constr'].x
            b = constr_features[:, 0]
            sense = constr_features[:, 1:4].argmax(dim=1) + 1
            
            result = {
                'A': A,
                'b': b,
                'sense': sense,
            }
            # Try to get var_types from graph data
            if hasattr(data, 'var_types') and data.var_types is not None:
                result['var_types'] = data.var_types.long().to(self.device)
            # Add variable bounds
            result.update(self._extract_var_bounds(prepared, data))
            return result
        
        # No constraint info available
        return {}
    
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Training batch.
        
        Returns:
            Dict of loss values.
        """
        prepared = self._prepare_batch(batch)
        mode = prepared.get('mode', 'gnn')
        # Handle batched mode (list of strings from collate)
        if isinstance(mode, list):
            mode = mode[0]
        
        # Handle batched text (list of strings from collate) - take first sample
        text = prepared.get('text')
        if isinstance(text, list):
            text = text[0]
        
        # Forward pass based on mode
        if self.fp16:
            with torch.amp.autocast('cuda'):
                if mode == 'text':
                    output = self.model(text=text, mode='text')
                else:
                    output = self.model(data=prepared.get('data'), mode=mode)
        else:
            if mode == 'text':
                output = self.model(text=text, mode='text')
            else:
                output = self.model(data=prepared.get('data'), mode=mode)
        
        # Get target
        target = prepared.get('target')
        if target is None:
            raise ValueError("No target labels in batch")
        
        # Extract constraints for physics loss (unified for all modes)
        constr_info = self._extract_constraints(prepared)
        
        # Compute loss
        if self.fp16:
            with torch.amp.autocast('cuda'):
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
        
        # Determine batches per epoch
        total_batches = len(train_loader)
        if self.max_samples_per_epoch is not None:
            max_batches = (self.max_samples_per_epoch + self.batch_size - 1) // self.batch_size
            batches_per_epoch = min(total_batches, max_batches)
        else:
            batches_per_epoch = total_batches
        
        # Calculate total steps
        num_training_steps = batches_per_epoch * self.num_epochs // self.gradient_accumulation_steps
        
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
        import itertools
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            epoch_losses = {'total': 0, 'task': 0, 'constraint': 0, 'integrality': 0}
            num_batches = 0
            
            # Running losses for logging (reset every logging_steps)
            logging_losses = {'total': 0, 'task': 0, 'constraint': 0, 'integrality': 0}
            logging_batches = 0
            
            # Use islice to limit iterator if needed
            if self.max_samples_per_epoch is not None:
                epoch_iterator = itertools.islice(train_loader, batches_per_epoch)
            else:
                epoch_iterator = train_loader
            
            progress_bar = tqdm(epoch_iterator, total=batches_per_epoch, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Update integrality weight schedule
                # Warmup for first 20% of steps
                if num_training_steps > 0:
                    progress = self.global_step / num_training_steps
                    if progress < 0.2:
                        self.loss_fn.lambda_integrality = 0.0
                    else:
                        # Linear ramp up from 0 to base_lambda
                        # (progress - 0.2) goes from 0 to 0.8
                        # factor goes from 0 to 1
                        factor = min(1.0, (progress - 0.2) / 0.8)
                        self.loss_fn.lambda_integrality = self.base_lambda_integrality * factor

                # Training step
                losses = self.train_step(batch)
                
                # Accumulate losses
                for k, v in losses.items():
                    epoch_losses[k] += v
                    logging_losses[k] += v
                num_batches += 1
                logging_batches += 1
                
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
                        # Calculate average since last log
                        avg_losses = {k: v / logging_batches for k, v in logging_losses.items()}
                        
                        progress_bar.set_postfix({
                            'loss': f"{avg_losses['total']:.4f}",
                            'task': f"{avg_losses['task']:.4f}",
                            'cons': f"{avg_losses['constraint']:.4f}",
                            'int': f"{avg_losses['integrality']:.4f}",
                            'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                        })
                        
                        if self.use_wandb:
                            wandb.log({
                                'train/loss': avg_losses['total'],
                                'train/task_loss': avg_losses['task'],
                                'train/constraint_loss': avg_losses['constraint'],
                                'train/integrality_loss': avg_losses['integrality'],
                                'train/learning_rate': self.scheduler.get_last_lr()[0],
                                'train/lambda_integrality': self.loss_fn.lambda_integrality,
                                'train/step': self.global_step,
                            })
                        
                        # Reset logging counters
                        logging_losses = {'total': 0, 'task': 0, 'constraint': 0, 'integrality': 0}
                        logging_batches = 0
                    
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
                # Use the same method as in train_step
                constr_info = self._extract_constraints(prepared)
                
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
        
        print(f"Validation - Loss: {avg_metrics['loss']:.4f}, Task: {avg_metrics['task_loss']:.4f}, Cons: {avg_metrics['constraint_loss']:.4f}, Int: {avg_metrics['integrality_loss']:.4f}, Accuracy: {avg_metrics['accuracy']:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_metrics['loss'],
                'val/accuracy': avg_metrics['accuracy'],
                'val/step': self.global_step,
            })
        
        return avg_metrics
    
    def _collate_fn(self, batch):
        """Custom collate function to handle sparse tensors."""
        from torch.utils.data.dataloader import default_collate
        
        if len(batch) == 0:
            return batch
            
        elem = batch[0]
        
        # Handle dictionary (recurse)
        if isinstance(elem, dict):
            return {key: self._collate_fn([d[key] for d in batch]) for key in elem}
            
        # Handle sparse tensors
        if isinstance(elem, torch.Tensor) and elem.is_sparse:
            # If batch size is 1, return the tensor directly (no batch dimension added)
            if len(batch) == 1:
                return batch[0]
            # If batch size > 1, return list (downstream needs to handle this)
            return batch
            
        # Fallback to default_collate for everything else
        return default_collate(batch)
    
    def _create_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        """Create DataLoader for dataset."""
        # Check if dataset returns Data/HeteroData objects (Graph mode)
        is_graph_data = False
        if HAS_PYG and hasattr(dataset, '__getitem__') and len(dataset) > 0:
            try:
                item = dataset[0]
                if isinstance(item, HeteroData) or (hasattr(item, 'edge_index') and hasattr(item, 'x')):
                    is_graph_data = True
            except:
                pass

        if HAS_PYG and is_graph_data:
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
                collate_fn=self._collate_fn,
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
