"""
NeuroSolver: Unified model wrapper for MILP solution prediction.

Supports two input modes:
- GNN mode: Graph input → GNN Tokenizer → Qwen → Prediction
- Text mode: Text input → Chunked Tokenizer → Qwen → Prediction

Key features:
- Disables RoPE positional encoding (uses RWPE for graph structure)
- Supports attention mask configuration (optional graph-aware mask)
- Multi-task output heads (primal, uncertainty, dual)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from .gnn_tokenizer import GNNTokenizer
from .text_tokenizer import TextTokenizerWrapper, ChunkedTextEncoder
from .heads import PredictionHead, UncertaintyHead, MultiTaskHead, SolutionOutput


class NeuroSolver(nn.Module):
    """
    Neural solver for MILP problems.
    
    Architecture:
    1. Input Encoder (GNN or Text)
    2. Qwen2.5-7B backbone (with RoPE disabled)
    3. Multi-task prediction heads
    
    The model processes MILP instances and predicts:
    - Binary variable values (0/1 probabilities)
    - Prediction uncertainty (for LNS guidance)
    """
    
    def __init__(
        self,
        backbone: str = "Qwen/Qwen2.5-7B-Instruct",
        mode: str = "gnn",
        load_in_4bit: bool = True,
        use_flash_attention: bool = True,
        disable_rope: bool = True,
        use_graph_attn_mask: bool = False,
        gnn_hidden_dim: int = 256,
        gnn_num_layers: int = 2,
        gnn_encoder_type: str = "GINEConv",
        chunk_size: int = 8192,
        chunk_stride: int = 4096,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_target_modules: Optional[list] = None,
        include_uncertainty: bool = True,
        include_dual: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            backbone: HuggingFace model name.
            mode: Input mode ("gnn" or "text").
            load_in_4bit: Whether to load model in 4-bit quantization.
            use_flash_attention: Whether to use FlashAttention-2.
            disable_rope: Whether to disable RoPE positional encoding.
            use_graph_attn_mask: Whether to use graph-structure attention mask.
            gnn_hidden_dim: GNN hidden dimension.
            gnn_num_layers: Number of GNN layers.
            gnn_encoder_type: Type of GNN encoder ("GINEConv" or "GATv2").
            chunk_size: Chunk size for text mode.
            chunk_stride: Chunk stride for text mode.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_target_modules: LoRA target modules.
            include_uncertainty: Whether to predict uncertainty.
            include_dual: Whether to predict dual values.
            device: Device for model.
        """
        super().__init__()
        
        self.mode = mode
        self.disable_rope = disable_rope
        self.use_graph_attn_mask = use_graph_attn_mask
        self.include_uncertainty = include_uncertainty
        self.include_dual = include_dual
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load backbone model
        self.qwen, self.tokenizer = self._load_backbone(
            backbone=backbone,
            load_in_4bit=load_in_4bit,
            use_flash_attention=use_flash_attention,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
        )
        
        # Get hidden size from model config
        self.hidden_size = self.qwen.config.hidden_size  # 3584 for Qwen2.5-7B
        
        # Initialize input encoders based on mode
        if mode == "gnn":
            if not HAS_PYG:
                raise ImportError("torch_geometric required for GNN mode")
            self.gnn_tokenizer = GNNTokenizer(
                gnn_hidden_dim=gnn_hidden_dim,
                llm_hidden_dim=self.hidden_size,
                gnn_num_layers=gnn_num_layers,
                gnn_encoder_type=gnn_encoder_type,
            ).to(self.device)
        
        if mode == "text" or mode == "both":
            self.text_tokenizer = TextTokenizerWrapper(
                tokenizer=self.tokenizer,
                chunk_size=chunk_size,
                stride=chunk_stride,
            )
        
        # Prediction heads
        self.pred_head = PredictionHead(
            hidden_dim=self.hidden_size,
            intermediate_dim=512,
        ).to(self.device)
        
        if include_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                hidden_dim=self.hidden_size,
            ).to(self.device)
        
        if include_dual:
            from .heads import DualHead
            self.dual_head = DualHead(
                hidden_dim=self.hidden_size,
            ).to(self.device)
    
    def _load_backbone(
        self,
        backbone: str,
        load_in_4bit: bool,
        use_flash_attention: bool,
        lora_r: int,
        lora_alpha: int,
        lora_target_modules: Optional[list],
    ) -> Tuple[nn.Module, Any]:
        """
        Load Qwen backbone with optional LoRA.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        model = AutoModelForCausalLM.from_pretrained(backbone, **model_kwargs)
        
        # Apply LoRA
        target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()
        
        print(f"Loaded {backbone} with transformers (4-bit: {load_in_4bit})")
        
        return model, tokenizer
    
    def _get_position_ids(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Get position IDs for the model.
        
        If disable_rope=True, returns zeros to effectively disable positional encoding.
        """
        if self.disable_rope:
            # All zeros to disable RoPE
            return torch.zeros(1, seq_len, dtype=torch.long, device=device)
        else:
            # Standard sequential position IDs
            return torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    
    def _build_graph_attention_mask(
        self, 
        data: "HeteroData",
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        """
        Build attention mask based on graph structure.
        
        Only allows attention between connected nodes.
        Currently not implemented - returns None for full attention.
        """
        if not self.use_graph_attn_mask:
            return None
        
        # TODO: Implement graph-aware attention mask
        # For now, use full attention
        return None
    
    def forward_gnn(
        self, 
        data: "HeteroData",
    ) -> SolutionOutput:
        """
        Forward pass for GNN mode.
        
        Args:
            data: HeteroData graph with 'var' and 'constr' nodes.
        
        Returns:
            SolutionOutput with predictions.
        """
        # Move data to device
        data = data.to(self.device)
        n_vars = data['var'].x.size(0)
        
        # GNN encoding → inputs_embeds
        inputs_embeds = self.gnn_tokenizer(data)  # (1, seq_len, hidden_size)
        seq_len = inputs_embeds.size(1)
        
        # Get position IDs (zeros if RoPE disabled)
        position_ids = self._get_position_ids(seq_len, self.device)
        
        # Build attention mask
        attention_mask = self._build_graph_attention_mask(data, seq_len)
        
        # Forward through Qwen
        outputs = self.qwen.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_size)
        
        # Extract variable hidden states (first n_vars positions)
        var_hidden = hidden_states[:, :n_vars, :]  # (1, n_vars, hidden_size)
        
        # Prediction heads
        primal = self.pred_head(var_hidden).squeeze(0)  # (n_vars,)
        
        uncertainty = None
        if self.include_uncertainty:
            uncertainty = self.uncertainty_head(var_hidden).squeeze(0)
        
        dual = None
        if self.include_dual:
            dual = self.dual_head(var_hidden).squeeze(0)
        
        return SolutionOutput(
            primal=primal,
            uncertainty=uncertainty,
            dual=dual,
            hidden_states=var_hidden.squeeze(0),
        )
    
    def forward_text(
        self, 
        text: str,
    ) -> SolutionOutput:
        """
        Forward pass for text mode.
        
        Args:
            text: MILP problem in text format.
        
        Returns:
            SolutionOutput with predictions.
        """
        # Use text tokenizer to get variable hidden states
        var_hidden, n_vars = self.text_tokenizer(
            text, self.qwen.model, self.device
        )  # (n_vars, hidden_size)
        
        # Add batch dimension
        var_hidden = var_hidden.unsqueeze(0)  # (1, n_vars, hidden_size)
        
        # Prediction heads
        primal = self.pred_head(var_hidden).squeeze(0)
        
        uncertainty = None
        if self.include_uncertainty:
            uncertainty = self.uncertainty_head(var_hidden).squeeze(0)
        
        dual = None
        if self.include_dual:
            dual = self.dual_head(var_hidden).squeeze(0)
        
        return SolutionOutput(
            primal=primal,
            uncertainty=uncertainty,
            dual=dual,
            hidden_states=var_hidden.squeeze(0),
        )
    
    def forward(
        self,
        data: Optional["HeteroData"] = None,
        text: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> SolutionOutput:
        """
        Unified forward pass.
        
        Args:
            data: HeteroData graph (for GNN mode).
            text: Text input (for text mode).
            mode: Override default mode.
        
        Returns:
            SolutionOutput with predictions.
        """
        mode = mode or self.mode
        
        if mode == "gnn":
            if data is None:
                raise ValueError("data required for GNN mode")
            return self.forward_gnn(data)
        
        elif mode == "text":
            if text is None:
                raise ValueError("text required for text mode")
            return self.forward_text(text)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def predict(
        self,
        data: Optional["HeteroData"] = None,
        text: Optional[str] = None,
        threshold: float = 0.5,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict discrete solution.
        
        Args:
            data: HeteroData graph.
            text: Text input.
            threshold: Probability threshold for discretization.
            mode: Input mode.
        
        Returns:
            Tuple of (solution, uncertainty).
        """
        with torch.no_grad():
            output = self.forward(data=data, text=text, mode=mode)
        
        solution = output.to_discrete(threshold)
        return solution, output.uncertainty
    
    def sample_solutions(
        self,
        data: Optional["HeteroData"] = None,
        text: Optional[str] = None,
        n_samples: int = 16,
        temperature: float = 1.0,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Sample multiple solutions (for GRPO).
        
        Args:
            data: HeteroData graph.
            text: Text input.
            n_samples: Number of solutions to sample.
            temperature: Sampling temperature.
            mode: Input mode.
        
        Returns:
            Sampled solutions of shape (n_samples, n_vars).
        """
        with torch.no_grad():
            output = self.forward(data=data, text=text, mode=mode)
        
        samples = []
        for _ in range(n_samples):
            sample = output.sample(temperature=temperature)
            samples.append(sample)
        
        return torch.stack(samples, dim=0)
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA weights
        self.qwen.save_pretrained(path)
        
        # Save tokenizers and heads
        torch.save({
            'gnn_tokenizer': self.gnn_tokenizer.state_dict() if hasattr(self, 'gnn_tokenizer') else None,
            'pred_head': self.pred_head.state_dict(),
            'uncertainty_head': self.uncertainty_head.state_dict() if self.include_uncertainty else None,
            'dual_head': self.dual_head.state_dict() if self.include_dual else None,
            'config': {
                'mode': self.mode,
                'disable_rope': self.disable_rope,
                'use_graph_attn_mask': self.use_graph_attn_mask,
                'include_uncertainty': self.include_uncertainty,
                'include_dual': self.include_dual,
            }
        }, os.path.join(path, 'neuro_solver.pt'))
        
        print(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "NeuroSolver":
        """Load model from checkpoint."""
        import os
        
        # Load config
        checkpoint = torch.load(os.path.join(path, 'neuro_solver.pt'))
        config = checkpoint['config']
        
        # Update with any overrides
        config.update(kwargs)
        
        # Create model
        model = cls(**config)
        
        # Load weights
        if checkpoint['gnn_tokenizer'] is not None:
            model.gnn_tokenizer.load_state_dict(checkpoint['gnn_tokenizer'])
        
        model.pred_head.load_state_dict(checkpoint['pred_head'])
        
        if checkpoint['uncertainty_head'] is not None:
            model.uncertainty_head.load_state_dict(checkpoint['uncertainty_head'])
        
        if checkpoint['dual_head'] is not None:
            model.dual_head.load_state_dict(checkpoint['dual_head'])
        
        print(f"Model loaded from {path}")
        return model
