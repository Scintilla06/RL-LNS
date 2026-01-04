"""
Data preprocessing for MILP problems.
Converts JSON/LP files to PyG graph format with LP relaxation solutions.
"""

import json
import re
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

try:
    from torch_geometric.data import HeteroData, Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not installed. Graph features will be limited.")

try:
    from gurobipy import GRB, read, Model
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("Warning: gurobipy not installed. LP relaxation will not be available.")


@dataclass
class MILPInstance:
    """Represents a single MILP instance."""
    n_vars: int                          # Number of variables
    n_constrs: int                       # Number of constraints
    obj_coeffs: np.ndarray               # Objective coefficients (n_vars,)
    obj_sense: int                       # 1 for minimize, -1 for maximize
    
    # Constraint matrix in sparse format
    constr_matrix: List[Tuple[int, int, float]]  # List of (constr_idx, var_idx, coeff)
    constr_rhs: np.ndarray               # Right-hand side (n_constrs,)
    constr_sense: np.ndarray             # Constraint sense: 1=<=, 2=>=, 3===
    
    # Variable bounds and types
    var_lb: np.ndarray                   # Lower bounds (n_vars,)
    var_ub: np.ndarray                   # Upper bounds (n_vars,)
    var_types: np.ndarray                # 0=binary, 1=continuous, 2=integer
    
    # Solutions
    optimal_solution: Optional[np.ndarray] = None  # Optimal solution (n_vars,)
    lp_relaxation: Optional[np.ndarray] = None     # LP relaxation solution (n_vars,)
    
    # Metadata
    instance_id: Optional[str] = None


class LPFormatParser:
    """Parse LP format strings to extract MILP structure."""
    
    @staticmethod
    def parse(lp_string: str) -> MILPInstance:
        """
        Parse LP format string to MILPInstance.
        
        Args:
            lp_string: MILP in LP format.
        
        Returns:
            MILPInstance object.
        """
        lines = lp_string.strip().split('\n')
        
        # State machine
        section = None
        obj_sense = 1  # Default minimize
        obj_terms = []
        constraints = []
        current_constr = []
        bounds = {}
        binaries = set()
        generals = set()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('\\'):
                continue
            
            line_lower = line.lower()
            
            # Section detection
            if line_lower.startswith('minimize'):
                section = 'objective'
                obj_sense = 1
                continue
            elif line_lower.startswith('maximize'):
                section = 'objective'
                obj_sense = -1
                continue
            elif line_lower.startswith('subject to') or line_lower == 'st':
                section = 'constraints'
                continue
            elif line_lower.startswith('bounds'):
                section = 'bounds'
                continue
            elif line_lower.startswith('binary') or line_lower.startswith('binaries'):
                section = 'binary'
                continue
            elif line_lower.startswith('general') or line_lower.startswith('generals'):
                section = 'general'
                continue
            elif line_lower.startswith('end'):
                break
            
            # Parse content based on section
            if section == 'objective':
                obj_terms.extend(LPFormatParser._parse_linear_expr(line))
            elif section == 'constraints':
                if any(op in line for op in ['<=', '>=', '=']):
                    if current_constr:
                        constraints.append(current_constr)
                    current_constr = [line]
                else:
                    current_constr.append(line)
            elif section == 'bounds':
                LPFormatParser._parse_bounds(line, bounds)
            elif section == 'binary':
                binaries.update(re.findall(r'x\[(\d+)\]', line))
            elif section == 'general':
                generals.update(re.findall(r'x\[(\d+)\]', line))
        
        if current_constr:
            constraints.append(current_constr)
        
        # Build instance
        return LPFormatParser._build_instance(
            obj_terms, obj_sense, constraints, bounds, binaries, generals
        )
    
    @staticmethod
    def _parse_linear_expr(expr: str) -> List[Tuple[int, float]]:
        """Parse linear expression to list of (var_idx, coeff)."""
        terms = []
        # Match patterns like: 0.5 x[0], -1.2 x[1], + x[2]
        pattern = r'([+-]?\s*\d*\.?\d*)\s*x\[(\d+)\]'
        for match in re.finditer(pattern, expr):
            coeff_str = match.group(1).replace(' ', '')
            if coeff_str in ['', '+']:
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)
            var_idx = int(match.group(2))
            terms.append((var_idx, coeff))
        return terms
    
    @staticmethod
    def _parse_bounds(line: str, bounds: Dict):
        """Parse bounds line."""
        # Simple bounds: x[0] <= 1, x[0] >= 0, 0 <= x[0] <= 1
        pass  # For binary/01 problems, bounds are typically 0-1
    
    @staticmethod
    def _build_instance(
        obj_terms: List[Tuple[int, float]],
        obj_sense: int,
        constraints: List[List[str]],
        bounds: Dict,
        binaries: set,
        generals: set
    ) -> MILPInstance:
        """Build MILPInstance from parsed components."""
        # Determine number of variables
        all_vars = set()
        for var_idx, _ in obj_terms:
            all_vars.add(var_idx)
        
        constr_matrix = []
        constr_rhs = []
        constr_sense_list = []
        
        for constr_idx, constr_lines in enumerate(constraints):
            constr_str = ' '.join(constr_lines)
            
            # Determine constraint sense and split
            if '<=' in constr_str:
                sense = 1
                lhs, rhs = constr_str.split('<=')
            elif '>=' in constr_str:
                sense = 2
                lhs, rhs = constr_str.split('>=')
            elif '=' in constr_str:
                sense = 3
                lhs, rhs = constr_str.split('=')
            else:
                continue
            
            # Parse LHS terms
            terms = LPFormatParser._parse_linear_expr(lhs)
            for var_idx, coeff in terms:
                all_vars.add(var_idx)
                constr_matrix.append((constr_idx, var_idx, coeff))
            
            # Parse RHS
            try:
                rhs_val = float(rhs.strip())
            except:
                rhs_val = 0.0
            
            constr_rhs.append(rhs_val)
            constr_sense_list.append(sense)
        
        n_vars = max(all_vars) + 1 if all_vars else 0
        n_constrs = len(constr_rhs)
        
        # Build arrays
        obj_coeffs = np.zeros(n_vars)
        for var_idx, coeff in obj_terms:
            obj_coeffs[var_idx] = coeff
        
        var_lb = np.zeros(n_vars)
        var_ub = np.ones(n_vars)  # Default for binary
        var_types = np.zeros(n_vars, dtype=np.int32)  # Default binary
        
        for var_str in generals:
            var_idx = int(var_str)
            if var_idx < n_vars:
                var_types[var_idx] = 2  # Integer
        
        return MILPInstance(
            n_vars=n_vars,
            n_constrs=n_constrs,
            obj_coeffs=obj_coeffs,
            obj_sense=obj_sense,
            constr_matrix=constr_matrix,
            constr_rhs=np.array(constr_rhs),
            constr_sense=np.array(constr_sense_list),
            var_lb=var_lb,
            var_ub=var_ub,
            var_types=var_types,
        )


class SolutionParser:
    """Parse solution strings."""
    
    @staticmethod
    def parse(solution_str: str, n_vars: int) -> np.ndarray:
        """
        Parse solution string to numpy array.
        
        Args:
            solution_str: Solution in format "x[0] = 1\nx[1] = 0\n..."
            n_vars: Number of variables.
        
        Returns:
            Solution array (n_vars,).
        """
        solution = np.zeros(n_vars)
        pattern = r'x\[(\d+)\]\s*=\s*(\d+)'
        for match in re.finditer(pattern, solution_str):
            var_idx = int(match.group(1))
            value = int(match.group(2))
            if var_idx < n_vars:
                solution[var_idx] = value
        return solution


class LPRelaxationSolver:
    """Solve LP relaxation using Gurobi."""
    
    @staticmethod
    def solve(instance: MILPInstance) -> np.ndarray:
        """
        Solve LP relaxation of MILP instance.
        
        Args:
            instance: MILPInstance object.
        
        Returns:
            LP relaxation solution (n_vars,).
        """
        if not HAS_GUROBI:
            # Return midpoint as fallback
            return np.full(instance.n_vars, 0.5)
        
        try:
            model = Model("LP_Relaxation")
            model.setParam('OutputFlag', 0)
            
            # Add continuous variables (relaxation)
            x = []
            for i in range(instance.n_vars):
                x.append(model.addVar(
                    lb=instance.var_lb[i],
                    ub=instance.var_ub[i],
                    vtype=GRB.CONTINUOUS  # Relaxed to continuous
                ))
            
            model.update()
            
            # Set objective
            obj = sum(instance.obj_coeffs[i] * x[i] for i in range(instance.n_vars))
            if instance.obj_sense == -1:
                model.setObjective(obj, GRB.MAXIMIZE)
            else:
                model.setObjective(obj, GRB.MINIMIZE)
            
            # Build constraint matrix lookup
            constr_terms = {}
            for constr_idx, var_idx, coeff in instance.constr_matrix:
                if constr_idx not in constr_terms:
                    constr_terms[constr_idx] = []
                constr_terms[constr_idx].append((var_idx, coeff))
            
            # Add constraints
            for i in range(instance.n_constrs):
                if i not in constr_terms:
                    continue
                lhs = sum(coeff * x[var_idx] for var_idx, coeff in constr_terms[i])
                rhs = instance.constr_rhs[i]
                sense = instance.constr_sense[i]
                
                if sense == 1:  # <=
                    model.addConstr(lhs <= rhs)
                elif sense == 2:  # >=
                    model.addConstr(lhs >= rhs)
                else:  # ==
                    model.addConstr(lhs == rhs)
            
            # Solve
            model.optimize()
            
            if model.Status == GRB.OPTIMAL:
                return np.array([x[i].X for i in range(instance.n_vars)])
            else:
                return np.full(instance.n_vars, 0.5)
        
        except Exception as e:
            print(f"LP relaxation failed: {e}")
            return np.full(instance.n_vars, 0.5)


class GraphBuilder:
    """Build PyG graph from MILP instance."""
    
    @staticmethod
    def build(instance: MILPInstance) -> "HeteroData":
        """
        Build heterogeneous graph from MILP instance.
        
        Graph structure:
        - Variable nodes: n_vars nodes with features [obj_coeff, lb, ub, x_lp]
        - Constraint nodes: n_constrs nodes with features [rhs, sense]
        - Edges: var -> constr with edge feature [a_ij]
        
        Node order: [all variable nodes, all constraint nodes]
        
        Args:
            instance: MILPInstance object.
        
        Returns:
            HeteroData graph.
        """
        if not HAS_PYG:
            raise ImportError("torch_geometric required for graph building")
        
        # Variable node features: [obj_coeff, lb, ub, x_lp]
        x_lp = instance.lp_relaxation if instance.lp_relaxation is not None else np.full(instance.n_vars, 0.5)
        var_features = np.stack([
            instance.obj_coeffs,
            instance.var_lb,
            instance.var_ub,
            x_lp,
        ], axis=1)  # (n_vars, 4)
        
        # Constraint node features: [rhs, sense_onehot(3)]
        sense_onehot = np.zeros((instance.n_constrs, 3))
        for i, s in enumerate(instance.constr_sense):
            sense_onehot[i, s - 1] = 1.0
        constr_features = np.concatenate([
            instance.constr_rhs.reshape(-1, 1),
            sense_onehot,
        ], axis=1)  # (n_constrs, 4)
        
        # Build edge index and edge attributes
        edge_index_var_to_constr = []
        edge_attr = []
        
        for constr_idx, var_idx, coeff in instance.constr_matrix:
            edge_index_var_to_constr.append([var_idx, constr_idx])
            edge_attr.append([coeff])
        
        edge_index = np.array(edge_index_var_to_constr).T if edge_index_var_to_constr else np.zeros((2, 0))
        edge_attr = np.array(edge_attr) if edge_attr else np.zeros((0, 1))
        
        # Create HeteroData
        data = HeteroData()
        
        data['var'].x = torch.tensor(var_features, dtype=torch.float32)
        data['var'].n_nodes = instance.n_vars
        
        data['constr'].x = torch.tensor(constr_features, dtype=torch.float32)
        data['constr'].n_nodes = instance.n_constrs
        
        data['var', 'participates', 'constr'].edge_index = torch.tensor(edge_index, dtype=torch.long)
        data['var', 'participates', 'constr'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        # Reverse edges for message passing
        data['constr', 'contains', 'var'].edge_index = torch.tensor(
            np.array([edge_index[1], edge_index[0]]), dtype=torch.long
        )
        data['constr', 'contains', 'var'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        # Store labels if available
        if instance.optimal_solution is not None:
            data['var'].y = torch.tensor(instance.optimal_solution, dtype=torch.float32)
        
        # Store metadata
        data.n_vars = instance.n_vars
        data.n_constrs = instance.n_constrs
        data.obj_sense = instance.obj_sense
        
        # Store dense constraint matrix for loss computation
        # Build dense matrix A from sparse representation
        A = np.zeros((instance.n_constrs, instance.n_vars))
        for constr_idx, var_idx, coeff in instance.constr_matrix:
            A[constr_idx, var_idx] = coeff
        
        data.A = torch.tensor(A, dtype=torch.float32)  # (n_constrs, n_vars)
        data.b = torch.tensor(instance.constr_rhs, dtype=torch.float32)  # (n_constrs,)
        data.sense = torch.tensor(instance.constr_sense, dtype=torch.long)  # (n_constrs,)
        
        # Store variable types for future general MILP support
        data.var_types = torch.tensor(instance.var_types, dtype=torch.long)  # (n_vars,)
        
        return data


class TextFormatter:
    """Format MILP instance as text with variable position tracking."""
    
    @staticmethod
    def format(instance: MILPInstance) -> Tuple[str, Dict[int, Tuple[int, int]]]:
        """
        Format MILP instance as text string.
        
        Args:
            instance: MILPInstance object.
        
        Returns:
            Tuple of (text, var_positions) where var_positions maps
            var_idx to (start_char, end_char) in the text.
        """
        lines = []
        var_positions = {}
        current_pos = 0
        
        # Objective
        obj_type = "Minimize" if instance.obj_sense == 1 else "Maximize"
        lines.append(f"{obj_type}")
        current_pos += len(lines[-1]) + 1
        
        obj_terms = []
        for i in range(instance.n_vars):
            if instance.obj_coeffs[i] != 0:
                start = current_pos + len(' '.join(obj_terms)) + (3 if obj_terms else 2)
                term = f"{instance.obj_coeffs[i]:.6f} x[{i}]"
                var_positions[i] = (start, start + len(term))
                obj_terms.append(term)
        
        lines.append("  " + " + ".join(obj_terms))
        current_pos += len(lines[-1]) + 1
        
        # Constraints
        lines.append("Subject To")
        current_pos += len(lines[-1]) + 1
        
        # Group constraint terms
        constr_terms = {}
        for constr_idx, var_idx, coeff in instance.constr_matrix:
            if constr_idx not in constr_terms:
                constr_terms[constr_idx] = []
            constr_terms[constr_idx].append((var_idx, coeff))
        
        for i in range(instance.n_constrs):
            terms = constr_terms.get(i, [])
            sense_str = {1: "<=", 2: ">=", 3: "="}[instance.constr_sense[i]]
            
            term_strs = []
            for var_idx, coeff in terms:
                term_strs.append(f"{coeff:.6f} x[{var_idx}]")
            
            constr_str = f"  c{i}: " + " + ".join(term_strs) + f" {sense_str} {instance.constr_rhs[i]:.6f}"
            lines.append(constr_str)
            current_pos += len(lines[-1]) + 1
        
        text = "\n".join(lines)
        return text, var_positions


@dataclass
class TextDataSample:
    """Preprocessed text data sample with constraint information."""
    text: str                            # MILP in text format
    n_vars: int                          # Number of variables
    n_constrs: int                       # Number of constraints
    target: np.ndarray                   # Optimal solution (n_vars,)
    A: np.ndarray                        # Constraint matrix (n_constrs, n_vars)
    b: np.ndarray                        # RHS (n_constrs,)
    sense: np.ndarray                    # Constraint sense (n_constrs,)
    var_types: np.ndarray                # Variable types (n_vars,)
    lp_relaxation: Optional[np.ndarray] = None  # LP relaxation (n_vars,)
    instance_id: Optional[int] = None


class MILPPreprocessor:
    """Main preprocessor class for MILP data."""
    
    def __init__(self, config: Optional[Any] = None, compute_lp_relaxation: bool = True):
        self.config = config
        self.compute_lp_relaxation = compute_lp_relaxation
    
    def _process_instance(self, sample: Dict, idx: int) -> Tuple[MILPInstance, "HeteroData", TextDataSample]:
        """
        Process a single sample into all formats.
        
        Returns:
            Tuple of (MILPInstance, HeteroData graph, TextDataSample)
        """
        # Parse LP format
        instance = LPFormatParser.parse(sample['input'])
        instance.instance_id = str(idx)
        
        # Parse optimal solution
        if 'output' in sample:
            instance.optimal_solution = SolutionParser.parse(
                sample['output'], instance.n_vars
            )
        
        # Compute LP relaxation
        if self.compute_lp_relaxation:
            instance.lp_relaxation = LPRelaxationSolver.solve(instance)
        
        # Build graph data
        graph = GraphBuilder.build(instance)
        graph.instance_id = idx
        
        # Build text data with constraint information
        text_data = self._build_text_data(instance, sample['input'], idx)
        
        return instance, graph, text_data
    
    def _build_text_data(self, instance: MILPInstance, raw_text: str, idx: int) -> TextDataSample:
        """Build text data sample with all constraint information."""
        # Build dense constraint matrix
        A = np.zeros((instance.n_constrs, instance.n_vars))
        for constr_idx, var_idx, coeff in instance.constr_matrix:
            A[constr_idx, var_idx] = coeff
        
        return TextDataSample(
            text=raw_text,
            n_vars=instance.n_vars,
            n_constrs=instance.n_constrs,
            target=instance.optimal_solution if instance.optimal_solution is not None else np.zeros(instance.n_vars),
            A=A,
            b=instance.constr_rhs,
            sense=instance.constr_sense,
            var_types=instance.var_types,
            lp_relaxation=instance.lp_relaxation,
            instance_id=idx,
        )
    
    def process_json_dataset(
        self,
        json_path: str,
        output_dir: str,
        compute_lp_relaxation: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Process JSON dataset to both graph and text formats.
        
        Args:
            json_path: Path to JSON file.
            output_dir: Output directory for processed data.
            compute_lp_relaxation: Whether to compute LP relaxation.
            max_samples: Maximum number of samples to process.
        """
        self.compute_lp_relaxation = compute_lp_relaxation
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading JSON from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"Processing {len(data)} samples...")
        graph_data = []
        text_data = []
        
        for idx, sample in enumerate(tqdm(data)):
            try:
                instance, graph, text_sample = self._process_instance(sample, idx)
                graph_data.append(graph)
                text_data.append(text_sample)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        # Save processed data
        print(f"Saving {len(graph_data)} processed samples...")
        
        # Split train/val (10% validation)
        val_size = int(len(graph_data) * 0.1)
        
        # Graph data
        train_graph = graph_data[val_size:]
        val_graph = graph_data[:val_size]
        torch.save(train_graph, output_path / "train.pt")
        torch.save(val_graph, output_path / "val.pt")
        
        # Text data (with constraints)
        train_text = text_data[val_size:]
        val_text = text_data[:val_size]
        torch.save(train_text, output_path / "train_text.pt")
        torch.save(val_text, output_path / "val_text.pt")
        
        print(f"Saved to {output_path}")
        print(f"  Graph - Train: {len(train_graph)}, Val: {len(val_graph)}")
        print(f"  Text  - Train: {len(train_text)}, Val: {len(val_text)}")
    
    def process_single(self, lp_string: str, solution_string: Optional[str] = None) -> "HeteroData":
        """
        Process a single MILP instance.
        
        Args:
            lp_string: MILP in LP format.
            solution_string: Optional solution string.
        
        Returns:
            HeteroData graph.
        """
        instance = LPFormatParser.parse(lp_string)
        
        if solution_string:
            instance.optimal_solution = SolutionParser.parse(solution_string, instance.n_vars)
        
        instance.lp_relaxation = LPRelaxationSolver.solve(instance)
        
        return GraphBuilder.build(instance)


if __name__ == "__main__":
    # Test preprocessing
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/train_dataset_huge.json")
    parser.add_argument("--output", type=str, default="./data/processed")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_lp", action="store_true", help="Skip LP relaxation")
    args = parser.parse_args()
    
    preprocessor = MILPPreprocessor()
    preprocessor.process_json_dataset(
        args.input,
        args.output,
        compute_lp_relaxation=not args.no_lp,
        max_samples=args.max_samples,
    )
