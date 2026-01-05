"""
Chunked Text Tokenizer for handling long MILP text sequences.

Splits long sequences (>64K tokens) into overlapping chunks,
processes each chunk independently, then aggregates hidden states
for variable nodes.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import re


class VariablePositionMapper:
    """
    Maps variable indices to their token positions in text.
    
    Tracks where each variable (x[i]) appears in the tokenized text,
    enabling aggregation of hidden states for prediction.
    """
    
    def __init__(self, tokenizer: Any):
        """
        Args:
            tokenizer: HuggingFace tokenizer.
        """
        self.tokenizer = tokenizer
    
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
        
        # Find all x[i] patterns
        for match in re.finditer(r'x\[(\d+)\]', text):
            var_idx = int(match.group(1))
            char_start = match.start()
            char_end = match.end()
            
            if offset_mapping is not None:
                # Use offset mapping for precise token positions
                token_positions = self._char_to_token_positions(
                    offset_mapping, char_start, char_end
                )
            else:
                # Fallback: estimate based on character position
                # Rough heuristic: ~4 characters per token on average
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
        """Convert character span to token positions."""
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
    ) -> Dict[str, Any]:
        """
        Tokenize and prepare text for chunked processing.
        
        Args:
            text: Input text string.
        
        Returns:
            Dict with 'chunks', 'var_positions', 'var_chunk_map', 'n_vars'.
        """
        # Tokenize full text with offset_mapping
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=False,
        )
        input_ids = encoding['input_ids'].squeeze(0)
        offset_mapping = encoding.get('offset_mapping')
        if offset_mapping is not None:
            offset_mapping = offset_mapping[0]  # Remove batch dim
        
        # Get variable positions (pass offset_mapping to avoid re-tokenization)
        var_positions = self.position_mapper.map_variables(text, input_ids, offset_mapping=offset_mapping)
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
            
            with torch.cuda.amp.autocast():
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


class TextTokenizerWrapper(nn.Module):
    """
    Wrapper that provides same interface as GNNTokenizer for text input.
    
    Handles both short sequences (direct processing) and long sequences (chunked).
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 32768,
        chunk_size: int = 8192,
        stride: int = 4096,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length for direct processing.
            chunk_size: Chunk size for long sequences.
            stride: Stride for chunked processing.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.chunked_encoder = ChunkedTextEncoder(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            stride=stride,
        )
    
    def forward(
        self,
        text: str,
        model: nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        """
        Process text input through LLM.
        
        Args:
            text: Input text.
            model: LLM model.
            device: Device.
        
        Returns:
            Tuple of (var_hiddens, n_vars).
        """
        # Tokenize once with offset_mapping to avoid re-tokenization later
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=False,
        )
        seq_len = encoding['input_ids'].size(1)
        offset_mapping = encoding.get('offset_mapping')
        if offset_mapping is not None:
            offset_mapping = offset_mapping[0]  # Remove batch dim
        
        if seq_len <= self.max_length:
            # Direct processing
            input_ids = encoding['input_ids'].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Get last layer hidden state from hidden_states tuple
            hidden = outputs.hidden_states[-1]
            
            # Get variable positions and aggregate (pass offset_mapping to avoid re-tokenization)
            var_positions = self.chunked_encoder.position_mapper.map_variables(
                text, encoding['input_ids'].squeeze(0), offset_mapping=offset_mapping
            )
            n_vars = max(var_positions.keys()) + 1 if var_positions else 0
            
            # Simple aggregation for short sequences
            var_hiddens = torch.zeros(n_vars, hidden.size(-1), device=device)
            for var_idx, positions in var_positions.items():
                if positions:
                    h = hidden[0, positions, :].mean(dim=0)
                    var_hiddens[var_idx] = h
            
            return var_hiddens, n_vars
        
        else:
            # Chunked processing
            var_hiddens = self.chunked_encoder(text, model, device)
            n_vars = var_hiddens.size(0)
            return var_hiddens, n_vars
