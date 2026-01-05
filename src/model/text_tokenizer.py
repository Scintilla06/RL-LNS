"""
Chunked Text Tokenizer for handling long MILP text sequences.

Splits long sequences (>64K tokens) into overlapping chunks,
processes each chunk independently, then aggregates hidden states
for variable nodes.

Uses special <VAR> anchor token for precise variable position mapping.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import re


# Special token for variable anchoring
VAR_ANCHOR_TOKEN = "<VAR>"


class VariablePositionMapper:
    """
    Maps variable indices to their token positions in text.
    
    Uses <VAR> anchor tokens for precise position mapping:
    - Preprocess text to add <VAR> after each variable: x[i] -> x[i]<VAR>
    - During mapping, find <VAR> token positions directly
    
    This avoids issues with BPE tokenization splitting variable names.
    """
    
    def __init__(self, tokenizer: Any):
        """
        Args:
            tokenizer: HuggingFace tokenizer.
        """
        self.tokenizer = tokenizer
        self.var_token_id = None
        
        # Try to get <VAR> token ID (may not exist if not added yet)
        self._init_var_token()
    
    def _init_var_token(self):
        """Initialize <VAR> token ID if it exists in tokenizer."""
        if VAR_ANCHOR_TOKEN in self.tokenizer.get_vocab():
            self.var_token_id = self.tokenizer.convert_tokens_to_ids(VAR_ANCHOR_TOKEN)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Add <VAR> anchor token after each variable occurrence.
        
        Transforms: x[123] -> x[123]<VAR>
        
        Args:
            text: Original MILP text.
        
        Returns:
            Text with <VAR> anchors added.
        """
        # Replace x[i] with x[i]<VAR>
        return re.sub(r'(x\[\d+\])', r'\1' + VAR_ANCHOR_TOKEN, text)
    
    def map_variables_fast(
        self,
        input_ids: torch.Tensor,
        text: str,
    ) -> Dict[int, List[int]]:
        """
        Fast variable mapping using <VAR> anchor tokens.
        
        Args:
            input_ids: Tokenized input ids.
            text: Preprocessed text (with <VAR> anchors).
        
        Returns:
            Dict mapping var_idx to list of token positions.
        """
        if self.var_token_id is None:
            # Fallback to regex-based mapping
            return self.map_variables(text, input_ids)
        
        var_positions = {}
        
        # Find all <VAR> token positions
        var_token_positions = (input_ids == self.var_token_id).nonzero(as_tuple=True)[0].tolist()
        
        # Extract variable indices from text (order matches <VAR> positions)
        pattern = re.compile(r'x\[(\d+)\]' + re.escape(VAR_ANCHOR_TOKEN))
        matches = list(pattern.finditer(text))
        
        if len(matches) != len(var_token_positions):
            # Mismatch - fall back to regex mapping
            return self.map_variables(text, input_ids)
        
        for match, token_pos in zip(matches, var_token_positions):
            var_idx = int(match.group(1))
            if var_idx not in var_positions:
                var_positions[var_idx] = []
            var_positions[var_idx].append(token_pos)
        
        return var_positions
    
    def map_variables(
        self, 
        text: str, 
        input_ids: torch.Tensor,
        offset_mapping: Optional[torch.Tensor] = None,
    ) -> Dict[int, List[int]]:
        """
        Map variable indices to token positions.
        
        Args:
            text: Original text string.
            input_ids: Tokenized input ids.
            offset_mapping: Pre-computed offset mapping (optional, avoids re-tokenization).
        
        Returns:
            Dict mapping var_idx to list of token positions.
        """
        # Only tokenize if offset_mapping not provided
        if offset_mapping is None:
            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                return_tensors='pt',
                truncation=False,
            )
            offset_mapping = encoding.get('offset_mapping', None)
            if offset_mapping is not None:
                offset_mapping = offset_mapping[0]
        
        var_positions = {}
        
        # Pre-convert offset_mapping to list for faster access
        offset_list = None
        if offset_mapping is not None:
            offset_list = offset_mapping.tolist()
        
        # Find all x[i] patterns using compiled regex
        pattern = re.compile(r'x\[(\d+)\]')
        matches = list(pattern.finditer(text))
        
        current_token_idx = 0
        num_tokens = len(offset_list) if offset_list else 0
        
        for match in matches:
            var_idx = int(match.group(1))
            char_start = match.start()
            char_end = match.end()
            
            if offset_list is not None:
                token_positions = []
                
                # 1. Fast forward to the first token that might overlap
                # We maintain current_token_idx across matches since matches are sequential
                while current_token_idx < num_tokens:
                    # Handle potential None or invalid offsets
                    offsets = offset_list[current_token_idx]
                    if not offsets or len(offsets) < 2:
                        current_token_idx += 1
                        continue
                        
                    tok_start, tok_end = offsets
                    if tok_start is None or tok_end is None:
                        current_token_idx += 1
                        continue
                        
                    # If token ends before match starts, we can skip it
                    if tok_end <= char_start:
                        current_token_idx += 1
                    else:
                        # This token ends after match starts, so it might overlap
                        break
                
                # 2. Collect all overlapping tokens starting from current_token_idx
                # Use a temp index to scan forward without losing our place
                temp_idx = current_token_idx
                while temp_idx < num_tokens:
                    offsets = offset_list[temp_idx]
                    if not offsets or len(offsets) < 2:
                        temp_idx += 1
                        continue
                        
                    tok_start, tok_end = offsets
                    if tok_start is None or tok_end is None:
                        temp_idx += 1
                        continue
                    
                    # If token starts after match ends, no more overlaps possible
                    if tok_start >= char_end:
                        break
                    
                    # If we are here, tok_end > char_start (from step 1) 
                    # AND tok_start < char_end (from check above)
                    # So it overlaps.
                    token_positions.append(temp_idx)
                    temp_idx += 1
            else:
                # Fallback: estimate based on character position
                token_positions = [char_start // 4]
            
            if var_idx not in var_positions:
                var_positions[var_idx] = []
            var_positions[var_idx].extend(token_positions)
        
        return var_positions
    
    def _char_to_token_positions(
        self,
        offset_mapping: torch.Tensor,
        char_start: int,
        char_end: int,
    ) -> List[int]:
        """Convert character span to token positions (tensor version)."""
        positions = []
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping.tolist()):
            if tok_start is None or tok_end is None:
                continue
            # Check if token overlaps with character span
            if tok_end > char_start and tok_start < char_end:
                positions.append(tok_idx)
        return positions



class ChunkedTextEncoder(nn.Module):
    """
    Encoder for long text sequences using chunking strategy.
    
    For sequences longer than max_length:
    1. Split into overlapping chunks
    2. Process each chunk independently with LLM
    3. Aggregate hidden states for each variable
    """
    
    def __init__(
        self,
        tokenizer: Any,
        chunk_size: int = 8192,
        stride: int = 4096,
        aggregation: str = "mean",
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer.
            chunk_size: Maximum tokens per chunk.
            stride: Stride between chunks (overlap = chunk_size - stride).
            aggregation: How to aggregate hidden states ("mean", "max", "last").
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride
        self.aggregation = aggregation
        
        self.position_mapper = VariablePositionMapper(tokenizer)
    
    def create_chunks(
        self, 
        input_ids: torch.Tensor,
    ) -> List[Tuple[int, int, torch.Tensor]]:
        """
        Split input_ids into overlapping chunks.
        
        Args:
            input_ids: Full token sequence (seq_len,).
        
        Returns:
            List of (start_pos, end_pos, chunk_ids) tuples.
        """
        seq_len = input_ids.size(0)
        
        if seq_len <= self.chunk_size:
            return [(0, seq_len, input_ids)]
        
        chunks = []
        start = 0
        
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunk_ids = input_ids[start:end]
            chunks.append((start, end, chunk_ids))
            
            if end >= seq_len:
                break
            
            start += self.stride
        
        return chunks
    
    def map_vars_to_chunks(
        self,
        var_positions: Dict[int, List[int]],
        chunks: List[Tuple[int, int, torch.Tensor]],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Map each variable to the chunks that contain it.
        
        Args:
            var_positions: Dict mapping var_idx to global token positions.
            chunks: List of (start, end, chunk_ids) tuples.
        
        Returns:
            Dict mapping var_idx to list of (chunk_idx, local_position) tuples.
        """
        var_chunk_map = {}
        
        for var_idx, positions in var_positions.items():
            var_chunk_map[var_idx] = []
            
            for pos in positions:
                for chunk_idx, (start, end, _) in enumerate(chunks):
                    if start <= pos < end:
                        local_pos = pos - start
                        var_chunk_map[var_idx].append((chunk_idx, local_pos))
        
        return var_chunk_map
    
    def aggregate_hidden_states(
        self,
        chunk_hiddens: List[torch.Tensor],
        var_chunk_map: Dict[int, List[Tuple[int, int]]],
        n_vars: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        """
        Aggregate hidden states for each variable across chunks.
        
        Args:
            chunk_hiddens: List of hidden states for each chunk, 
                           each of shape (1, chunk_len, hidden_dim).
            var_chunk_map: Mapping from var_idx to (chunk_idx, local_pos).
            n_vars: Number of variables.
            hidden_dim: Hidden dimension.
        
        Returns:
            Aggregated hidden states of shape (n_vars, hidden_dim).
        """
        device = chunk_hiddens[0].device
        var_hiddens = torch.zeros(n_vars, hidden_dim, device=device)
        var_counts = torch.zeros(n_vars, device=device)
        
        for var_idx in range(n_vars):
            if var_idx not in var_chunk_map or not var_chunk_map[var_idx]:
                # Variable not found in text - use zero embedding
                continue
            
            chunk_positions = var_chunk_map[var_idx]
            hidden_list = []
            
            for chunk_idx, local_pos in chunk_positions:
                if local_pos < chunk_hiddens[chunk_idx].size(1):
                    h = chunk_hiddens[chunk_idx][0, local_pos, :]
                    hidden_list.append(h)
            
            if hidden_list:
                stacked = torch.stack(hidden_list, dim=0)
                
                if self.aggregation == "mean":
                    var_hiddens[var_idx] = stacked.mean(dim=0)
                elif self.aggregation == "max":
                    var_hiddens[var_idx] = stacked.max(dim=0)[0]
                elif self.aggregation == "last":
                    var_hiddens[var_idx] = stacked[-1]
                
                var_counts[var_idx] = len(hidden_list)
        
        return var_hiddens
    
    def encode(
        self,
        text: str,
        use_var_anchor: bool = True,
    ) -> Dict[str, Any]:
        """
        Tokenize and prepare text for chunked processing.
        
        Args:
            text: Input text string.
            use_var_anchor: Whether to use <VAR> anchor tokens for precise mapping.
                           Requires tokenizer to have <VAR> as a special token.
        
        Returns:
            Dict with 'chunks', 'var_positions', 'var_chunk_map', 'n_vars'.
        """
        # Optionally preprocess text to add <VAR> anchor tokens
        processed_text = text
        if use_var_anchor:
            processed_text = VariablePositionMapper.preprocess_text(text)
        
        # Tokenize full text with offset_mapping
        encoding = self.tokenizer(
            processed_text,
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=False,
        )
        input_ids = encoding['input_ids'].squeeze(0)
        offset_mapping = encoding.get('offset_mapping')
        if offset_mapping is not None:
            offset_mapping = offset_mapping[0]  # Remove batch dim
        
        # Get variable positions
        if use_var_anchor:
            # Fast mapping using <VAR> anchor tokens
            var_positions = self.position_mapper.map_variables_fast(
                processed_text, input_ids, offset_mapping=offset_mapping
            )
        else:
            # Fallback to offset-based mapping (pass offset_mapping to avoid re-tokenization)
            var_positions = self.position_mapper.map_variables(
                text, input_ids, offset_mapping=offset_mapping
            )
        
        n_vars = max(var_positions.keys()) + 1 if var_positions else 0
        
        # Create chunks
        chunks = self.create_chunks(input_ids)
        
        # Map variables to chunks
        var_chunk_map = self.map_vars_to_chunks(var_positions, chunks)
        
        return {
            'input_ids': input_ids,
            'chunks': chunks,
            'var_positions': var_positions,
            'var_chunk_map': var_chunk_map,
            'n_vars': n_vars,
            'processed_text': processed_text,  # Include for reference
        }
    
    def forward(
        self,
        text: str,
        model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Full forward pass: encode text, process chunks, aggregate.
        
        Args:
            text: Input text string.
            model: LLM model with forward method.
            device: Device for computation.
        
        Returns:
            Variable hidden states of shape (n_vars, hidden_dim).
        """
        # Encode text
        encoded = self.encode(text)
        chunks = encoded['chunks']
        var_chunk_map = encoded['var_chunk_map']
        n_vars = encoded['n_vars']
        
        if n_vars == 0:
            raise ValueError("No variables found in text")
        
        # Process each chunk
        chunk_hiddens = []
        for start, end, chunk_ids in chunks:
            chunk_ids = chunk_ids.unsqueeze(0).to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=chunk_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Get last layer hidden state from hidden_states tuple
            hidden = outputs.hidden_states[-1]  # (1, chunk_len, hidden_dim)
            chunk_hiddens.append(hidden)
        
        # Get hidden dimension from first chunk
        hidden_dim = chunk_hiddens[0].size(-1)
        
        # Aggregate for variables
        var_hiddens = self.aggregate_hidden_states(
            chunk_hiddens, var_chunk_map, n_vars, hidden_dim
        )
        
        return var_hiddens


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
            chunk_size: Chunk size for long sequences.
            stride: Stride for chunked processing.
            max_vars: Maximum number of variables to support.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_vars = max_vars
        
        # Add special tokens for variable positions if not already added
        self._ensure_var_tokens_added()
        
        self.chunked_encoder = ChunkedTextEncoder(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            stride=stride,
        )
    
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
            # For long sequences, use chunked processing
            # The [VAR_i] tokens are in the last chunk
            # We need to ensure the last chunk processes them with full context
            
            # For now, fall back to direct processing with truncation warning
            # TODO: Implement proper chunked processing for very long sequences
            import warnings
            warnings.warn(
                f"Sequence length ({seq_len}) exceeds max_length ({self.max_length}). "
                "Truncating to max_length. Consider using chunked processing."
            )
            
            input_ids = input_ids[:, :self.max_length]
            
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            hidden = outputs.hidden_states[-1]
            
            # If truncated, we may have lost some [VAR_i] tokens
            # Use whatever we have at the end
            actual_n_vars = min(n_vars, self.max_length - (seq_len - n_vars))
            if actual_n_vars < n_vars:
                warnings.warn(f"Only {actual_n_vars}/{n_vars} variable tokens fit in truncated sequence")
            
            var_hiddens = hidden[0, -actual_n_vars:, :]
            
            # Pad if needed
            if actual_n_vars < n_vars:
                hidden_dim = hidden.size(-1)
                padding = torch.zeros(n_vars - actual_n_vars, hidden_dim, device=device)
                var_hiddens = torch.cat([padding, var_hiddens], dim=0)
            
            return var_hiddens, n_vars
