"""
PyTorch Dataset classes for MILP data.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json

try:
    from torch_geometric.data import HeteroData, Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class MILPGraphDataset(Dataset):
    """
    Dataset for MILP instances in graph format.
    
    Each sample is a HeteroData graph with:
    - var nodes: (n_vars, 4) features [obj_coeff, lb, ub, x_lp]
    - constr nodes: (n_constrs, 4) features [rhs, sense_onehot]
    - edges: var <-> constr with edge_attr [a_ij]
    - labels: var.y = optimal solution
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
    ):
        """
        Args:
            data_path: Path to .pt file containing list of HeteroData.
            transform: Optional transform to apply.
            pre_transform: Optional pre-transform.
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.pre_transform = pre_transform
        
        # Load data
        self.data_list = torch.load(self.data_path)
        
        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(d) for d in self.data_list]
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> HeteroData:
        data = self.data_list[idx]
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    @staticmethod
    def collate_fn(batch: List[HeteroData]) -> Dict[str, Any]:
        """
        Custom collate function for batching graphs.
        
        Since graphs have different sizes, we don't batch them into tensors.
        Instead, return a list for processing one at a time, or use PyG Batch.
        """
        if HAS_PYG:
            return Batch.from_data_list(batch)
        else:
            return batch


class MILPTextDataset(Dataset):
    """
    Dataset for MILP instances in text format with preprocessed constraint info.
    
    Loads from preprocessed .pt files that contain:
    - text: MILP problem text
    - target: optimal solution
    - A, b, sense: constraint matrix information
    - var_types: variable types
    
    This ensures text mode can use the same PhysicsInformedLoss as GNN mode.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 65536,
        chunk_size: int = 8192,
        stride: int = 4096,
    ):
        """
        Args:
            data_path: Path to preprocessed .pt file (train_text.pt or val_text.pt).
            tokenizer: HuggingFace tokenizer (optional, for on-the-fly tokenization).
            max_length: Maximum total sequence length.
            chunk_size: Size of each chunk for long sequences.
            stride: Stride between chunks (overlap = chunk_size - stride).
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.stride = stride
        
        # Load preprocessed data
        self.data_list = torch.load(self.data_path)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data_list[idx]
        
        result = {
            'text': sample.text,
            'n_vars': sample.n_vars,
            'n_constrs': sample.n_constrs,
            'target': torch.tensor(sample.target, dtype=torch.float32),
            'A': torch.tensor(sample.A, dtype=torch.float32),
            'b': torch.tensor(sample.b, dtype=torch.float32),
            'sense': torch.tensor(sample.sense, dtype=torch.long),
            'var_types': torch.tensor(sample.var_types, dtype=torch.long),
            'mode': 'text',
        }
        
        if sample.lp_relaxation is not None:
            result['lp_relaxation'] = torch.tensor(sample.lp_relaxation, dtype=torch.float32)
        
        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                sample.text,
                truncation=False,
                return_tensors='pt',
            )
            result['input_ids'] = encoding['input_ids'].squeeze(0)
            result['attention_mask'] = encoding['attention_mask'].squeeze(0)
        
        return result


class MILPTextDatasetLegacy(Dataset):
    """
    Dataset for MILP instances in text format (for baseline comparison).
    
    Handles long sequences by storing variable position mappings.
    """
    
    def __init__(
        self,
        json_path: str,
        tokenizer: Any,
        max_length: int = 65536,
        chunk_size: int = 8192,
        stride: int = 4096,
    ):
        """
        Args:
            json_path: Path to JSON dataset.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum total sequence length.
            chunk_size: Size of each chunk for long sequences.
            stride: Stride between chunks (overlap = chunk_size - stride).
        """
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.stride = stride
        
        # Load data
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        # Get text input
        text = sample['input']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=False,  # Don't truncate - we handle long sequences
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Parse solution for labels
        solution = self._parse_solution(sample.get('output', ''))
        
        # Extract variable positions in token space
        var_positions = self._extract_var_positions(text, input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'solution': solution,
            'var_positions': var_positions,
            'n_vars': len(solution),
            'text': text,
        }
    
    def _parse_solution(self, output_str: str) -> torch.Tensor:
        """Parse solution string to tensor."""
        import re
        pattern = r'x\[(\d+)\]\s*=\s*(\d+)'
        matches = re.findall(pattern, output_str)
        
        if not matches:
            return torch.tensor([])
        
        max_idx = max(int(m[0]) for m in matches)
        solution = torch.zeros(max_idx + 1)
        
        for var_idx, value in matches:
            solution[int(var_idx)] = float(value)
        
        return solution
    
    def _extract_var_positions(
        self, 
        text: str, 
        input_ids: torch.Tensor
    ) -> Dict[int, List[int]]:
        """
        Extract token positions for each variable.
        
        Returns dict mapping var_idx to list of token positions
        where that variable appears.
        """
        import re
        
        # Decode tokens to find variable positions
        # This is approximate - exact mapping requires tokenizer specifics
        var_positions = {}
        
        # Find all x[i] patterns in text
        for match in re.finditer(r'x\[(\d+)\]', text):
            var_idx = int(match.group(1))
            char_start = match.start()
            char_end = match.end()
            
            # Approximate token position (rough estimate)
            # More accurate would require offset_mapping from tokenizer
            approx_token_pos = char_start // 4  # Rough estimate
            
            if var_idx not in var_positions:
                var_positions[var_idx] = []
            var_positions[var_idx].append(approx_token_pos)
        
        return var_positions
    
    def get_chunks(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Split long sequence into overlapping chunks.
        
        Args:
            input_ids: Full token sequence.
        
        Returns:
            List of chunk tensors.
        """
        seq_len = input_ids.size(0)
        
        if seq_len <= self.chunk_size:
            return [input_ids]
        
        chunks = []
        start = 0
        
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunks.append(input_ids[start:end])
            
            if end >= seq_len:
                break
            
            start += self.stride
        
        return chunks


class CombinedDataset(Dataset):
    """
    Dataset that provides both graph and text representations.
    Useful for ablation studies comparing modalities.
    """
    
    def __init__(
        self,
        graph_dataset: MILPGraphDataset,
        text_dataset: MILPTextDataset,
    ):
        """
        Args:
            graph_dataset: Graph format dataset.
            text_dataset: Text format dataset.
        """
        assert len(graph_dataset) == len(text_dataset), \
            "Datasets must have same length"
        
        self.graph_dataset = graph_dataset
        self.text_dataset = text_dataset
    
    def __len__(self) -> int:
        return len(self.graph_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'graph': self.graph_dataset[idx],
            'text': self.text_dataset[idx],
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    mode: str = 'graph',
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with appropriate collate function.
    
    Args:
        dataset: Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of workers.
        mode: 'graph' or 'text'.
    
    Returns:
        DataLoader instance.
    """
    if mode == 'graph' and HAS_PYG:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        return PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: x,  # Don't collate - process one at a time
        )
