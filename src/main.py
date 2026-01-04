#!/usr/bin/env python3
"""
Deep-Structure RL-LNS Solver - Main Entry Point

This script provides a unified interface for:
1. Data preprocessing (JSON -> PyG graphs)
2. Model training (SFT + GRPO)
3. Heuristic evolution (EOH)
4. Inference and evaluation

Usage:
    python main.py preprocess --config configs/data.yaml
    python main.py train-sft --config configs/training.yaml
    python main.py train-grpo --config configs/training.yaml
    python main.py evolve --config configs/evolution.yaml
    python main.py infer --model outputs/sft/best --input problem.lp
"""

import argparse
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def preprocess(args):
    """Preprocess raw JSON data to PyG graph format."""
    print("=" * 50)
    print("Data Preprocessing")
    print("=" * 50)
    
    config = load_config(args.config)
    
    from src.datalib.preprocess import MILPPreprocessor
    
    # Initialize preprocessor
    preprocessor = MILPPreprocessor(
        compute_lp_relaxation=config.get('preprocessing', {}).get('compute_lp_relaxation', True)
    )
    
    # Load raw data
    raw_config = config.get('raw', {})
    train_json = raw_config.get('train_json', 'data/train_dataset_huge.json')
    
    print(f"Loading data from {train_json}...")
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Split train/val if needed
    val_split = raw_config.get('val_split_ratio', 0.1)
    if val_split > 0:
        import random
        random.seed(42)
        random.shuffle(data)
        split_idx = int(len(data) * (1 - val_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        print(f"Split: {len(train_data)} train, {len(val_data)} val")
    else:
        train_data = data
        val_data = []
    
    # Process and save
    processed_config = config.get('processed', {})
    train_dir = processed_config.get('train_dir', 'data/processed/train')
    val_dir = processed_config.get('val_dir', 'data/processed/val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Processing training data to {train_dir}...")
    for i, sample in enumerate(train_data):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(train_data)}")
        try:
            graph = preprocessor.process_sample(sample)
            import torch
            torch.save(graph, f"{train_dir}/sample_{i:06d}.pt")
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
    
    if val_data:
        print(f"Processing validation data to {val_dir}...")
        for i, sample in enumerate(val_data):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(val_data)}")
            try:
                graph = preprocessor.process_sample(sample)
                import torch
                torch.save(graph, f"{val_dir}/sample_{i:06d}.pt")
            except Exception as e:
                print(f"  Error processing sample {i}: {e}")
    
    print("Preprocessing complete!")


def train_sft(args):
    """Train model with Physics-Informed SFT."""
    print("=" * 50)
    print("Physics-Informed SFT Training")
    print("=" * 50)
    
    # Load configs
    training_config = load_config(args.config)
    model_config = load_config(args.model_config) if args.model_config else load_config('configs/model.yaml')
    
    import torch
    from src.model.neuro_solver import NeuroSolver
    from src.datalib.dataset import MILPGraphDataset, MILPTextDataset, create_dataloader
    from src.training.sft_trainer import SFTTrainer
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get mode from args or config
    mode = args.mode if hasattr(args, 'mode') and args.mode else 'gnn'
    print(f"Training mode: {mode}")
    
    # Determine GNN layers based on mode
    if mode == 'mlp':
        # MLP mode: use 0 GNN layers (just projection)
        gnn_num_layers = 0
        actual_mode = 'gnn'  # Model still uses gnn forward path
    elif mode == 'text':
        gnn_num_layers = model_config['gnn']['num_layers']
        actual_mode = 'text'
    else:  # gnn
        gnn_num_layers = model_config['gnn']['num_layers']
        actual_mode = 'gnn'
    
    # Initialize model
    print("Initializing model...")
    model = NeuroSolver(
        backbone=model_config['model']['name'],
        mode=actual_mode,
        gnn_hidden_dim=model_config['gnn']['hidden_dim'],
        gnn_num_layers=gnn_num_layers,
        load_in_4bit=model_config['model']['load_in_4bit'],
        lora_r=model_config['model']['lora_r'],
        lora_alpha=model_config['model']['lora_alpha'],
        device=device,
    )
    
    # Load datasets based on mode
    sft_config = training_config['sft']
    
    if mode == 'text':
        # Text mode: use text dataset with constraint info
        train_data_path = sft_config.get('train_data_text', 'data/processed/train_text.pt')
        val_data_path = sft_config.get('val_data_text', 'data/processed/val_text.pt')
        
        print(f"Loading text training data from {train_data_path}...")
        train_dataset = MILPTextDataset(train_data_path, tokenizer=model.tokenizer)
        
        val_dataset = None
        if val_data_path:
            print(f"Loading text validation data from {val_data_path}...")
            val_dataset = MILPTextDataset(val_data_path, tokenizer=model.tokenizer)
    else:
        # GNN/MLP mode: use graph dataset
        print(f"Loading graph training data from {sft_config['train_data']}...")
        train_dataset = MILPGraphDataset(sft_config['train_data'])
        
        val_dataset = None
        if sft_config.get('val_data'):
            print(f"Loading graph validation data from {sft_config['val_data']}...")
            val_dataset = MILPGraphDataset(sft_config['val_data'])
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=sft_config['batch_size'],
        gradient_accumulation_steps=sft_config['gradient_accumulation_steps'],
        learning_rate=sft_config['learning_rate'],
        weight_decay=sft_config['weight_decay'],
        num_epochs=sft_config['num_epochs'],
        warmup_ratio=sft_config['warmup_ratio'],
        max_grad_norm=sft_config['max_grad_norm'],
        lambda_constraint=sft_config['lambda_constraint'],
        lambda_integrality=sft_config['lambda_integrality'],
        output_dir=sft_config['output_dir'],
        logging_steps=sft_config['logging_steps'],
        eval_steps=sft_config['eval_steps'],
        save_steps=sft_config['save_steps'],
        fp16=sft_config['fp16'],
        wandb_project=training_config.get('wandb', {}).get('project'),
        wandb_run_name=training_config.get('wandb', {}).get('run_name'),
        wandb_mode=training_config.get('wandb', {}).get('mode', 'online'),
        max_samples_per_epoch=sft_config.get('max_samples_per_epoch'),
        device=device,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    print("Training complete!")


def train_grpo(args):
    """Train model with GRPO."""
    print("=" * 50)
    print("GRPO Training")
    print("=" * 50)
    
    # Load configs
    training_config = load_config(args.config)
    
    import torch
    from src.model.neuro_solver import NeuroSolver
    from src.datalib.dataset import MILPGraphDataset
    from src.training.grpo_loop import GRPOTrainer
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained model
    grpo_config = training_config['grpo']
    pretrained_path = args.pretrained or training_config['sft']['output_dir'] + '/best'
    
    print(f"Loading pretrained model from {pretrained_path}...")
    model = NeuroSolver.from_pretrained(pretrained_path, device=device)
    
    # Load dataset
    sft_config = training_config['sft']
    print(f"Loading training data from {sft_config['train_data']}...")
    train_dataset = MILPGraphDataset(sft_config['train_data'])
    print(f"Train samples: {len(train_dataset)}")
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        train_dataset=train_dataset,
        group_size=grpo_config['group_size'],
        batch_size=grpo_config['batch_size'],
        learning_rate=grpo_config['learning_rate'],
        weight_decay=grpo_config['weight_decay'],
        num_epochs=grpo_config['num_epochs'],
        max_grad_norm=grpo_config['max_grad_norm'],
        kl_coef=grpo_config['kl_coef'],
        entropy_coef=grpo_config['entropy_coef'],
        temperature=grpo_config['temperature'],
        baseline_type=grpo_config['baseline_type'],
        infeasibility_penalty=grpo_config['infeasibility_penalty'],
        output_dir=grpo_config['output_dir'],
        logging_steps=grpo_config['logging_steps'],
        save_steps=grpo_config['save_steps'],
        fp16=grpo_config['fp16'],
        wandb_project=training_config.get('wandb', {}).get('project'),
        wandb_run_name=training_config.get('wandb', {}).get('run_name'),
        wandb_mode=training_config.get('wandb', {}).get('mode', 'online'),
        device=device,
    )
    
    # Train
    print("Starting GRPO training...")
    trainer.train()
    print("GRPO training complete!")


def evolve(args):
    """Run heuristic evolution with EOH."""
    print("=" * 50)
    print("Heuristic Evolution (EOH)")
    print("=" * 50)
    
    config = load_config(args.config)
    
    from src.evolution.eoh import EOH, Paras, create_folders
    from src.problems.milp import MILPProblem
    
    # Import selection and management methods
    from src.evolution.selection import prob_rank, equal, roulette_wheel, tournament
    from src.evolution.management import pop_greedy, ls_greedy, ls_sa
    
    # Create parameters
    paras = Paras()
    paras.set_paras(
        method=config.get('method', 'eoh'),
        problem=config.get('problem', 'milp_construct'),
        llm_api_endpoint=config['llm']['api_endpoint'],
        llm_api_key=config['llm']['api_key'],
        llm_model=config['llm']['model'],
        ec_pop_size=config['ec']['pop_size'],
        ec_n_pop=config['ec']['n_pop'],
        exp_n_proc=config['exp']['n_proc'],
        exp_debug_mode=config['exp']['debug_mode'],
        exp_output_path=config['exp']['output_path'],
    )
    
    # Create output folders
    create_folders(paras.exp_output_path)
    
    # Initialize problem
    problem = MILPProblem()
    
    # Get selection method
    selection_name = config.get('selection', 'prob_rank')
    selection_methods = {
        'prob_rank': prob_rank,
        'equal': equal,
        'roulette_wheel': roulette_wheel,
        'tournament': tournament,
    }
    select = selection_methods.get(selection_name, prob_rank)
    
    # Get management method
    management_name = config.get('management', 'pop_greedy')
    management_methods = {
        'pop_greedy': pop_greedy,
        'ls_greedy': ls_greedy,
        'ls_sa': ls_sa,
    }
    manage = management_methods.get(management_name, pop_greedy)
    
    # Run EOH
    eoh = EOH(paras, problem, select, manage)
    eoh.run()
    
    print("Evolution complete!")


def infer(args):
    """Run inference on MILP problem."""
    print("=" * 50)
    print("Inference")
    print("=" * 50)
    
    import torch
    from src.model.neuro_solver import NeuroSolver
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = NeuroSolver.from_pretrained(args.model, device=device)
    model.eval()
    
    # TODO: Implement inference pipeline
    print("Inference pipeline not yet fully implemented")
    print("Model loaded successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Deep-Structure RL-LNS Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('--config', type=str, default='configs/data.yaml',
                                   help='Data configuration file')
    
    # Train SFT command
    sft_parser = subparsers.add_parser('train-sft', help='Train with Physics-Informed SFT')
    sft_parser.add_argument('--config', type=str, default='configs/training.yaml',
                           help='Training configuration file')
    sft_parser.add_argument('--model-config', type=str, default=None,
                           help='Model configuration file')
    sft_parser.add_argument('--mode', type=str, default='gnn', choices=['gnn', 'mlp', 'text'],
                           help='Training mode: gnn (default), mlp (no GNN layers), text (text input)')
    
    # Train GRPO command
    grpo_parser = subparsers.add_parser('train-grpo', help='Train with GRPO')
    grpo_parser.add_argument('--config', type=str, default='configs/training.yaml',
                            help='Training configuration file')
    grpo_parser.add_argument('--pretrained', type=str, default=None,
                            help='Pretrained model path')
    
    # Evolve command
    evolve_parser = subparsers.add_parser('evolve', help='Run heuristic evolution')
    evolve_parser.add_argument('--config', type=str, default='configs/evolution.yaml',
                              help='Evolution configuration file')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', type=str, required=True,
                             help='Model checkpoint path')
    infer_parser.add_argument('--input', type=str, required=True,
                             help='Input MILP problem (LP file or JSON)')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        preprocess(args)
    elif args.command == 'train-sft':
        train_sft(args)
    elif args.command == 'train-grpo':
        train_grpo(args)
    elif args.command == 'evolve':
        evolve(args)
    elif args.command == 'infer':
        infer(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
