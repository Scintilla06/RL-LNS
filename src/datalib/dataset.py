"""

            stride: Stride between chunks (overlap = chunk_size - stride).
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.stride = stride
        
        # Load text data (minimal: just text and instance_id)
        self.text_data = torch.load(self.data_path)
        
        # Load graph data for constraint info
        if graph_data_path is None:
            # Infer graph path: train_text.pt -> train.pt
            graph_data_path = str(self.data_path).replace('_text.pt', '.pt')
        self.graph_data = torch.load(graph_data_path)
    
    def __len__(self) -> int:
        return len(self.text_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text_sample = self.text_data[idx]
        graph_sample = self.graph_data[idx]
        
        # Get text from text data
        text = text_sample['text'] if isinstance(text_sample, dict) else text_sample.text
        
        # Extract sparse A components from graph data's COO format
        # Store as separate tensors to allow proper batching (sparse tensors can't be batched)
        if hasattr(graph_sample, 'A_row'):
            # New format: COO components
            A_row = graph_sample.A_row.long()
            A_col = graph_sample.A_col.long()
            A_val = graph_sample.A_val
            # A_shape may be tuple or tensor
            if isinstance(graph_sample.A_shape, tuple):
                A_shape = torch.tensor(graph_sample.A_shape, dtype=torch.int64)
            else:
                A_shape = graph_sample.A_shape.long()
        elif hasattr(graph_sample, 'A'):
            # Legacy format: sparse tensor directly - extract COO components
            A_sparse = graph_sample.A.coalesce()
            A_row = A_sparse.indices()[0]
            A_col = A_sparse.indices()[1]
            A_val = A_sparse.values()
            A_shape = torch.tensor(A_sparse.shape, dtype=torch.int64)
        else:
            raise ValueError("Graph data missing constraint matrix")
        
        result = {
            'text': text,
            'n_vars': graph_sample.n_vars,
            'n_constrs': graph_sample.n_constrs,
            'target': graph_sample['var'].y if hasattr(graph_sample['var'], 'y') else torch.zeros(graph_sample.n_vars),
            'A_row': A_row,
            'A_col': A_col,
            'A_val': A_val,
            'A_shape': A_shape,
            'b': graph_sample.b,
            'sense': graph_sample.sense.long(),
            'var_types': graph_sample.var_types.long(),
            'mode': 'text',
        }
        
        # LP relaxation is in var features (4th column)
        if hasattr(graph_sample['var'], 'x') and graph_sample['var'].x.shape[1] >= 4:
            result['lp_relaxation'] = graph_sample['var'].x[:, 3]
        
        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=False,
                return_tensors='pt',
            )
            result['input_ids'] = encoding['input_ids'].squeeze(0)
            result['attention_mask'] = encoding['attention_mask'].squeeze(0)
        
        return result



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
