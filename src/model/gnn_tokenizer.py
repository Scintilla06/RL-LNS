"""
GNN-based Structural Tokenizer for MILP problems.

Maps MILP bipartite graph to Qwen's embedding space.
Each graph node becomes a token in the sequence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

try:
    from torch_geometric.nn import GINEConv, GATv2Conv, HeteroConv, Linear
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_dense_adj, add_self_loops
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class FourierFeatureMapper(nn.Module):
    """
    Maps scalar features to high-dimensional Fourier features.
    
    x -> [sin(2^0 * pi * x), cos(2^0 * pi * x), sin(2^1 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    
    This helps the network learn high-frequency functions of the input.
    """
    
    def __init__(self, input_dim: int, num_frequencies: int = 8, include_input: bool = True):
        """
        Args:
            input_dim: Number of input features.
            num_frequencies: Number of frequency bands (L).
            include_input: Whether to include original input in output.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Precompute frequency scales: 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer('freqs', freqs)
        
        # Output dimension
        self.output_dim = input_dim * num_frequencies * 2
        if include_input:
            self.output_dim += input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., input_dim).
        
        Returns:
            Fourier features of shape (..., output_dim).
        """
        # x: (..., input_dim)
        # Expand for broadcasting: (..., input_dim, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Compute scaled inputs: (..., input_dim, num_frequencies)
        scaled = x_expanded * self.freqs * math.pi
        
        # Apply sin and cos: (..., input_dim, num_frequencies, 2)
        sin_features = torch.sin(scaled)
        cos_features = torch.cos(scaled)
        
        # Interleave sin and cos: (..., input_dim, num_frequencies * 2)
        fourier = torch.stack([sin_features, cos_features], dim=-1)
        fourier = fourier.reshape(*x.shape[:-1], -1)
        
        if self.include_input:
            return torch.cat([x, fourier], dim=-1)
        return fourier


class RWPEEncoder(nn.Module):
    """
    Random Walk Positional Encoding for graph nodes.
    
    Computes the diagonal of the k-step random walk matrix: diag(A^k D^{-k})
    where A is the adjacency matrix and D is the degree matrix.
    """
    
    def __init__(self, walk_length: int = 16, hidden_dim: int = 64):
        """
        Args:
            walk_length: Number of random walk steps (k).
            hidden_dim: Output dimension after MLP projection.
        """
        super().__init__()
        self.walk_length = walk_length
        
        # MLP to project RWPE features
        self.mlp = nn.Sequential(
            nn.Linear(walk_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute RWPE for all nodes.
        
        Args:
            edge_index: Edge index tensor (2, num_edges).
            num_nodes: Number of nodes.
            edge_weight: Optional edge weights.
        
        Returns:
            RWPE features of shape (num_nodes, hidden_dim).
        """
        device = edge_index.device
        
        # Build adjacency matrix
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)
        
        # Add self-loops for random walk
        edge_index_loop, edge_weight_loop = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        
        # Compute degree
        row = edge_index_loop[0]
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, row, edge_weight_loop)
        deg_inv = 1.0 / deg.clamp(min=1e-8)
        
        # Normalize adjacency for random walk: D^{-1} A
        norm_edge_weight = edge_weight_loop * deg_inv[row]
        
        # Compute random walk probabilities
        # P = D^{-1} A, we want diag(P^k) for k = 1, ..., walk_length
        rwpe = torch.zeros(num_nodes, self.walk_length, device=device)
        
        # Initialize: P^0 = I
        prob = torch.ones(num_nodes, device=device)
        
        for k in range(self.walk_length):
            # One step of random walk
            # prob_new[j] = sum_i prob[i] * P[i,j]
            new_prob = torch.zeros(num_nodes, device=device)
            new_prob.scatter_add_(
                0, 
                edge_index_loop[1], 
                prob[edge_index_loop[0]] * norm_edge_weight
            )
            prob = new_prob
            rwpe[:, k] = prob
        
        return self.mlp(rwpe)


class DegreeEncoder(nn.Module):
    """
    Encode node degree as learnable embeddings.
    """
    
    def __init__(self, max_degree: int = 100, hidden_dim: int = 64):
        """
        Args:
            max_degree: Maximum degree to embed (higher degrees are clamped).
            hidden_dim: Embedding dimension.
        """
        super().__init__()
        self.max_degree = max_degree
        self.embedding = nn.Embedding(max_degree + 1, hidden_dim)
    
    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute degree embeddings.
        
        Args:
            edge_index: Edge index tensor (2, num_edges).
            num_nodes: Number of nodes.
        
        Returns:
            Degree embeddings of shape (num_nodes, hidden_dim).
        """
        device = edge_index.device
        
        # Compute degree
        deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=device))
        
        # Clamp to max_degree
        deg = deg.clamp(max=self.max_degree)
        
        return self.embedding(deg)


class BipartiteGNNLayer(nn.Module):
    """
    Single layer of bipartite GNN for MILP graph.
    
    Messages pass between variable and constraint nodes.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        edge_dim: int = 1,
        encoder_type: str = "GINEConv",
    ):
        """
        Args:
            hidden_dim: Hidden dimension.
            edge_dim: Edge feature dimension.
            encoder_type: Type of GNN layer ("GINEConv" or "GATv2").
        """
        super().__init__()
        self.encoder_type = encoder_type
        
        if not HAS_PYG:
            raise ImportError("torch_geometric required for BipartiteGNN")
        
        if encoder_type == "GINEConv":
            # GINEConv: Graph Isomorphism Network with Edge features
            nn_var = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn_constr = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv_var_to_constr = GINEConv(nn_var, edge_dim=edge_dim)
            self.conv_constr_to_var = GINEConv(nn_constr, edge_dim=edge_dim)
        
        elif encoder_type == "GATv2":
            # GATv2: Graph Attention Network v2
            self.conv_var_to_constr = GATv2Conv(
                hidden_dim, hidden_dim, edge_dim=edge_dim, heads=4, concat=False
            )
            self.conv_constr_to_var = GATv2Conv(
                hidden_dim, hidden_dim, edge_dim=edge_dim, heads=4, concat=False
            )
        
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Layer norms
        self.norm_var = nn.LayerNorm(hidden_dim)
        self.norm_constr = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x_var: torch.Tensor,
        x_constr: torch.Tensor,
        edge_index_v2c: torch.Tensor,
        edge_index_c2v: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_var: Variable node features (n_vars, hidden_dim).
            x_constr: Constraint node features (n_constrs, hidden_dim).
            edge_index_v2c: Var->Constr edges (2, n_edges).
            edge_index_c2v: Constr->Var edges (2, n_edges).
            edge_attr: Edge attributes (n_edges, edge_dim).
        
        Returns:
            Updated (x_var, x_constr).
        """
        # Message passing: var -> constr
        x_constr_new = self.conv_var_to_constr(
            (x_var, x_constr), edge_index_v2c, edge_attr
        )
        x_constr = self.norm_constr(x_constr + x_constr_new)
        
        # Message passing: constr -> var
        x_var_new = self.conv_constr_to_var(
            (x_constr, x_var), edge_index_c2v, edge_attr
        )
        x_var = self.norm_var(x_var + x_var_new)
        
        return x_var, x_constr


class BipartiteGNN(nn.Module):
    """
    Full bipartite GNN encoder for MILP graphs.
    
    Encodes variable and constraint nodes, then concatenates them
    in fixed order [variables, constraints] to form input sequence for LLM.
    """
    
    def __init__(
        self,
        var_input_dim: int = 4,        # [obj_coeff, lb, ub, x_lp]
        constr_input_dim: int = 4,     # [rhs, sense_onehot(3)]
        edge_dim: int = 1,             # [a_ij]
        hidden_dim: int = 256,
        num_layers: int = 2,
        encoder_type: str = "GINEConv",
        num_fourier_freqs: int = 8,
        rwpe_walk_length: int = 16,
        use_rwpe: bool = True,
        use_degree: bool = True,
    ):
        """
        Args:
            var_input_dim: Variable node feature dimension.
            constr_input_dim: Constraint node feature dimension.
            edge_dim: Edge feature dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of GNN layers.
            encoder_type: Type of GNN layer.
            num_fourier_freqs: Number of Fourier frequencies.
            rwpe_walk_length: Random walk length for RWPE.
            use_rwpe: Whether to use RWPE.
            use_degree: Whether to use degree encoding.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_rwpe = use_rwpe
        self.use_degree = use_degree
        
        # Feature encoders
        self.var_fourier = FourierFeatureMapper(var_input_dim, num_fourier_freqs)
        self.constr_fourier = FourierFeatureMapper(constr_input_dim, num_fourier_freqs)
        self.edge_fourier = FourierFeatureMapper(edge_dim, num_fourier_freqs // 2)
        
        var_feat_dim = self.var_fourier.output_dim
        constr_feat_dim = self.constr_fourier.output_dim
        edge_feat_dim = self.edge_fourier.output_dim
        
        # Positional encodings
        if use_rwpe:
            self.rwpe = RWPEEncoder(rwpe_walk_length, hidden_dim // 4)
            var_feat_dim += hidden_dim // 4
            constr_feat_dim += hidden_dim // 4
        
        if use_degree:
            self.degree_enc = DegreeEncoder(100, hidden_dim // 4)
            var_feat_dim += hidden_dim // 4
            constr_feat_dim += hidden_dim // 4
        
        # Input projections
        self.var_input_proj = nn.Linear(var_feat_dim, hidden_dim)
        self.constr_input_proj = nn.Linear(constr_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim // 4)
        
        # GNN layers
        self.layers = nn.ModuleList([
            BipartiteGNNLayer(hidden_dim, hidden_dim // 4, encoder_type)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, data: "HeteroData") -> torch.Tensor:
        """
        Encode MILP graph to node embeddings.
        
        Args:
            data: HeteroData with 'var' and 'constr' node types.
        
        Returns:
            Node embeddings of shape (n_vars + n_constrs, hidden_dim).
            Ordered as [all variables, all constraints].
        """
        # Get features
        x_var = data['var'].x  # (n_vars, var_input_dim)
        x_constr = data['constr'].x  # (n_constrs, constr_input_dim)
        
        edge_index_v2c = data['var', 'participates', 'constr'].edge_index
        edge_attr = data['var', 'participates', 'constr'].edge_attr
        edge_index_c2v = data['constr', 'contains', 'var'].edge_index
        
        n_vars = x_var.size(0)
        n_constrs = x_constr.size(0)
        n_nodes = n_vars + n_constrs
        
        # Apply Fourier feature mapping
        x_var = self.var_fourier(x_var)
        x_constr = self.constr_fourier(x_constr)
        edge_attr = self.edge_fourier(edge_attr)
        
        # Build combined graph for positional encodings
        # Offset constraint indices by n_vars
        combined_edge_index = torch.cat([
            edge_index_v2c + torch.tensor([[0], [n_vars]], device=edge_index_v2c.device),
            edge_index_c2v + torch.tensor([[n_vars], [0]], device=edge_index_c2v.device),
        ], dim=1)
        
        # Add positional encodings
        if self.use_rwpe:
            rwpe = self.rwpe(combined_edge_index, n_nodes)
            x_var = torch.cat([x_var, rwpe[:n_vars]], dim=-1)
            x_constr = torch.cat([x_constr, rwpe[n_vars:]], dim=-1)
        
        if self.use_degree:
            deg_enc = self.degree_enc(combined_edge_index, n_nodes)
            x_var = torch.cat([x_var, deg_enc[:n_vars]], dim=-1)
            x_constr = torch.cat([x_constr, deg_enc[n_vars:]], dim=-1)
        
        # Project to hidden dim
        x_var = self.var_input_proj(x_var)
        x_constr = self.constr_input_proj(x_constr)
        edge_attr = self.edge_proj(edge_attr)
        
        # Apply GNN layers
        for layer in self.layers:
            x_var, x_constr = layer(
                x_var, x_constr,
                edge_index_v2c, edge_index_c2v,
                edge_attr
            )
        
        # Concatenate in fixed order: [variables, constraints]
        x = torch.cat([x_var, x_constr], dim=0)  # (n_vars + n_constrs, hidden_dim)
        x = self.output_norm(x)
        
        return x


class EmbeddingProjector(nn.Module):
    """
    Projects GNN embeddings to LLM hidden dimension.
    
    GNN output (256) -> Qwen hidden size (3584)
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 3584,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: GNN output dimension.
            output_dim: LLM hidden dimension.
            num_layers: Number of MLP layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        layers = []
        hidden_dim = (input_dim + output_dim) // 2
        
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        # Final layer
        layers.append(nn.Linear(hidden_dim if num_layers > 1 else input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings.
        
        Args:
            x: GNN output of shape (seq_len, gnn_dim).
        
        Returns:
            Projected embeddings of shape (seq_len, llm_dim).
        """
        x = self.mlp(x)
        x = self.output_norm(x)
        return x


class GNNTokenizer(nn.Module):
    """
    Complete GNN-based tokenizer that converts MILP graphs to LLM input embeddings.
    
    Pipeline:
    1. BipartiteGNN encodes graph structure -> (N, gnn_hidden_dim)
    2. EmbeddingProjector maps to LLM space -> (N, llm_hidden_dim)
    3. Output is inputs_embeds for the LLM, no position_ids needed
    """
    
    def __init__(
        self,
        gnn_hidden_dim: int = 256,
        llm_hidden_dim: int = 3584,
        gnn_num_layers: int = 2,
        gnn_encoder_type: str = "GINEConv",
        **kwargs,
    ):
        """
        Args:
            gnn_hidden_dim: GNN hidden dimension.
            llm_hidden_dim: LLM hidden dimension (Qwen2.5-7B = 3584).
            gnn_num_layers: Number of GNN layers.
            gnn_encoder_type: Type of GNN encoder.
            **kwargs: Additional arguments for BipartiteGNN.
        """
        super().__init__()
        
        self.gnn = BipartiteGNN(
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            encoder_type=gnn_encoder_type,
            **kwargs,
        )
        
        self.projector = EmbeddingProjector(
            input_dim=gnn_hidden_dim,
            output_dim=llm_hidden_dim,
        )
        
        self.gnn_hidden_dim = gnn_hidden_dim
        self.llm_hidden_dim = llm_hidden_dim
    
    def forward(self, data: "HeteroData") -> torch.Tensor:
        """
        Convert graph to LLM input embeddings.
        
        Args:
            data: HeteroData graph.
        
        Returns:
            inputs_embeds of shape (1, n_vars + n_constrs, llm_hidden_dim).
        """
        # GNN encoding
        node_embeds = self.gnn(data)  # (N, gnn_hidden_dim)
        
        # Project to LLM space
        llm_embeds = self.projector(node_embeds)  # (N, llm_hidden_dim)
        
        # Add batch dimension
        return llm_embeds.unsqueeze(0)  # (1, N, llm_hidden_dim)
    
    def get_n_vars(self, data: "HeteroData") -> int:
        """Get number of variables from graph."""
        return data['var'].x.size(0)
