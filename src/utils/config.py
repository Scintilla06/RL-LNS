"""
Configuration classes for Deep-Structure RL-LNS Solver.
Refactored from LLM-LNS.py Paras and GetPrompts classes.
"""

import os
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    mode: str = "gnn"  # "gnn" or "text"
    backbone: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    use_flash_attention: bool = True
    disable_rope: bool = True  # Disable positional encoding for graph input
    use_graph_attn_mask: bool = False  # Optional graph-structure attention mask
    
    # GNN Tokenizer
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 2
    gnn_encoder_type: str = "GINEConv"  # "GINEConv" or "GATv2"
    
    # Text Tokenizer (for long sequences)
    chunk_size: int = 8192
    chunk_stride: int = 4096  # Overlap
    
    # Projection
    qwen_hidden_size: int = 3584  # Qwen2.5-7B hidden size
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Training configuration."""
    # General
    seed: int = 42
    output_dir: str = "./outputs"
    
    # SFT
    sft_epochs: int = 3
    sft_batch_size: int = 1
    sft_gradient_accumulation: int = 16
    sft_learning_rate: float = 2e-4
    sft_warmup_ratio: float = 0.03
    sft_max_grad_norm: float = 1.0
    
    # Physics-Informed Loss weights
    lambda_constraint: float = 0.1
    lambda_integrality: float = 0.01
    
    # GRPO
    grpo_group_size: int = 16
    grpo_epochs: int = 1
    grpo_learning_rate: float = 1e-5
    grpo_kl_coef: float = 0.1
    
    # Reward
    infeasibility_penalty: float = 10.0


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "./data"
    processed_dir: str = "./data/processed"
    train_file: str = "train_dataset_huge.json"
    val_split: float = 0.1
    max_vars: int = 1000
    max_constrs: int = 1000
    
    # LP file path for raw instances
    lp_dir: Optional[str] = None


@dataclass
class EvolutionConfig:
    """Evolution (LLM-LNS) configuration."""
    method: str = "eoh"  # 'eoh', 'ael', 'ls', 'sa'
    problem: str = "milp_construct"
    
    # Population
    pop_size: int = 5
    n_pop: int = 5
    
    # Operators
    operators: List[str] = field(default_factory=lambda: ["e1", "e2", "m1", "m2"])
    operator_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    m: int = 2  # Number of parents for crossover
    
    # Selection and Management
    selection: str = "prob_rank"
    management: str = "pop_greedy"
    
    # LLM API
    llm_api_endpoint: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    
    # Evaluation
    eva_timeout: int = 1500  # 5 * 300
    prompt_eva_timeout: int = 30
    
    # Debug
    debug_mode: bool = False
    n_proc: int = -1  # -1 for auto


@dataclass 
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    
    # Experiment
    exp_name: str = "default"
    wandb_project: str = "deep-structure-lns"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Auto-detect number of processes
        if self.evolution.n_proc == -1:
            self.evolution.n_proc = multiprocessing.cpu_count()
        
        # Create output directories
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.data.processed_dir, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            evolution=EvolutionConfig(**config_dict.get("evolution", {})),
            exp_name=config_dict.get("exp_name", "default"),
            wandb_project=config_dict.get("wandb_project", "deep-structure-lns"),
            wandb_entity=config_dict.get("wandb_entity"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)


class PromptTemplates:
    """
    Prompt templates for LNS neighborhood selection.
    Refactored from GetPrompts class.
    """
    
    def __init__(self):
        self.task = (
            "Given an initial feasible solution and a current solution to a "
            "Mixed-Integer Linear Programming (MILP) problem, with a complete "
            "description of the constraints and objective function. "
            "We want to improve the current solution using Large Neighborhood Search (LNS). "
            "The task can be solved step-by-step by starting from the current solution "
            "and iteratively selecting a subset of decision variables to relax and re-optimize. "
            "In each step, most decision variables are fixed to their values in the current solution, "
            "and only a small subset is allowed to change. "
            "You need to score all the decision variables based on the information I give you, "
            "and I will choose the decision variables with high scores as neighborhood selection. "
            "To avoid getting stuck in local optima, the choice of the subset can incorporate "
            "a degree of randomness. "
            "You can also consider the correlation between decision variables, for example, "
            "assigning similar scores to variables involved in the same constraint, "
            "which often exhibit high correlation. This will help me select decision variables "
            "from the same constraint. "
            "Of course, I also welcome other interesting strategies that you might suggest."
        )
        
        self.func_name = "select_neighborhood"
        
        self.func_inputs = [
            "n", "m", "k", "site", "value", "constraint",
            "initial_solution", "current_solution", "objective_coefficient"
        ]
        
        self.func_outputs = ["neighbor_score"]
        
        self.inout_info = (
            "'n': Number of decision variables in the problem instance. 'n' is an integer number. "
            "'m': Number of constraints in the problem instance. 'm' is an integer number. "
            "'k': k[i] indicates the number of decision variables involved in the ith constraint. "
            "'k' is a Numpy array with length m. "
            "'site': site[i][j] indicates which decision variable is involved in the jth position "
            "of the ith constraint. 'site' is a list of Numpy arrays. The length of the list is m. "
            "'value': value[i][j] indicates the coefficient of the jth decision variable in the "
            "ith constraint. 'value' is a list of Numpy arrays. The length of the list is m. "
            "'constraint': constraint[i] indicates the right-hand side value of the ith constraint. "
            "'constraint' is a Numpy array with length m. "
            "'initial_solution': initial_solution[i] indicates the initial value of the i-th "
            "decision variable. initial_solution is a Numpy array with length n. "
            "'current_solution': current_solution[i] indicates the current value of the i-th "
            "decision variable. current_solution is a Numpy array with length n. "
            "'objective_coefficient': objective_coefficient[i] indicates the objective function "
            "coefficient corresponding to the i-th decision variable. objective_coefficient is "
            "a Numpy array with length n. "
            "'initial_solution', 'current_solution', and 'objective_coefficient' are numpy arrays "
            "with length n. The i-th element of the arrays corresponds to the i-th decision variable. "
            "This corresponds to the Set Cover MILP problem, where all decision variables are binary "
            "(0-1 variables), and all constraints are in the form of LHS >= RHS. "
            "'neighbor_score' is also a numpy array that you need to create manually. "
            "The i-th element of the arrays corresponds to the i-th decision variable."
        )
        
        self.other_info = (
            "All are Numpy arrays. I don't give you 'neighbor_score' so that you need to create "
            "it manually. The length of the 'neighbor_score' array is also 'n'."
        )


def get_prompts() -> PromptTemplates:
    """Get prompt templates instance."""
    return PromptTemplates()
