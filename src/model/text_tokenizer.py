"""
Text Tokenizer for MILP problems.

Key design: Appends [VAR_0], [VAR_1], ..., [VAR_n] tokens at the END
of the text sequence. Due to causal attention in decoder-only LLMs,
these tokens can attend to ALL preceding content (the full MILP description).

This makes text mode FAIR with GNN mode, where variable embeddings are also
placed at the end of the sequence after constraint embeddings.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Any
import re


# Special token format for variable prediction positions
VAR_PRED_TOKEN_FORMAT = "[VAR_{}]"


def count_variables_in_text(text: str) -> int:
    """Count the number of unique variables in MILP text."""
    pattern = re.compile(r'x\[(\d+)\]')
    var_indices = set(int(m.group(1)) for m in pattern.finditer(text))
    return max(var_indices) + 1 if var_indices else 0


def append_var_tokens(text: str, n_vars: int) -> str:
    """
    Append variable prediction tokens at the end of text.
    
    This is CRITICAL for causal attention: by placing [VAR_i] tokens
    at the END of the sequence, they can attend to ALL preceding tokens
    (the entire MILP description), enabling fair comparison with GNN mode.
    
    Args:
        text: Original MILP text.
        n_vars: Number of variables.
    
    Returns:
        Text with appended variable tokens.
    """
    var_tokens = " ".join(VAR_PRED_TOKEN_FORMAT.format(i) for i in range(n_vars))
    return text + "\n" + var_tokens


class TextTokenizerWrapper(nn.Module):
    """
    Wrapper that provides same interface as GNNTokenizer for text input.
    
    Key design: Appends [VAR_0], [VAR_1], ..., [VAR_n] tokens at the END
    of the text sequence. Due to causal attention, these tokens can
    attend to ALL preceding content (the full MILP description).
    
    This makes text mode FAIR with GNN mode, where variables are also
    placed at the end of the sequence.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 32768,
        chunk_size: int = 8192,
        stride: int = 4096,
        max_vars: int = 1000,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length for direct processing.
            chunk_size: Chunk size for long sequences (reserved for future use).
            stride: Stride for chunked processing (reserved for future use).
            max_vars: Maximum number of variables to support.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_vars = max_vars
        self.chunk_size = chunk_size
        self.stride = stride
        
        # Add special tokens for variable positions if not already added
        self._ensure_var_tokens_added()
    
    def _ensure_var_tokens_added(self):
        """Add [VAR_i] special tokens to tokenizer if not present."""
        existing_tokens = set(self.tokenizer.get_vocab().keys())
        new_tokens = []
        
        for i in range(self.max_vars):
            token = VAR_PRED_TOKEN_FORMAT.format(i)
            if token not in existing_tokens:
                new_tokens.append(token)
        
        if new_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
    
    def get_var_token_ids(self, n_vars: int) -> List[int]:
        """Get token IDs for [VAR_0] to [VAR_{n_vars-1}]."""
        return [
            self.tokenizer.convert_tokens_to_ids(VAR_PRED_TOKEN_FORMAT.format(i))
            for i in range(n_vars)
        ]
    
    def forward(
        self,
        text: str,
        model: nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        """
        Process text input through LLM.
        
        Pipeline:
        1. Count variables in text to determine n_vars
        2. Append [VAR_0], ..., [VAR_{n-1}] tokens at end
        3. Run through LLM
        4. Extract hidden states at the [VAR_i] positions (last n_vars tokens)
        
        Args:
            text: Input MILP text.
            model: LLM model.
            device: Device.
        
        Returns:
            Tuple of (var_hiddens, n_vars).
        """
        # Count variables in text
        n_vars = count_variables_in_text(text)
        
        if n_vars == 0:
            # No variables found - return empty
            hidden_dim = model.config.hidden_size
            return torch.zeros(0, hidden_dim, device=device), 0
        
        if n_vars > self.max_vars:
            raise ValueError(f"Number of variables ({n_vars}) exceeds max_vars ({self.max_vars})")
        
        # Append [VAR_i] tokens at the end
        text_with_vars = append_var_tokens(text, n_vars)
        
        # Tokenize
        encoding = self.tokenizer(
            text_with_vars,
            return_tensors='pt',
            truncation=False,
        )
        input_ids = encoding['input_ids'].to(device)
        seq_len = input_ids.size(1)
        
        if seq_len <= self.max_length:
            # Direct processing
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Get last layer hidden state
            hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            
            # Extract hidden states for the LAST n_vars tokens (the [VAR_i] tokens)
            # These tokens can attend to ALL preceding content due to causal attention
            var_hiddens = hidden[0, -n_vars:, :]  # (n_vars, hidden_dim)
            
            return var_hiddens, n_vars
        
        else:
            # For long sequences, truncate with warning
            # The [VAR_i] tokens need to be preserved at the end
            import warnings
            warnings.warn(
                f"Sequence length ({seq_len}) exceeds max_length ({self.max_length}). "
                "Truncating from the beginning to preserve [VAR_i] tokens at the end."
            )
            
            # Keep the last max_length tokens (which includes [VAR_i] tokens)
            input_ids = input_ids[:, -self.max_length:]
            
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            hidden = outputs.hidden_states[-1]
            
            # Extract last n_vars tokens
            var_hiddens = hidden[0, -n_vars:, :]
            
            return var_hiddens, n_vars


# For backward compatibility - ChunkedTextEncoder is now just an alias
# The actual chunking is handled internally by TextTokenizerWrapper
ChunkedTextEncoder = TextTokenizerWrapper
