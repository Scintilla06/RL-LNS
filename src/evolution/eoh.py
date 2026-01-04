"""
EOH (Evolution of Heuristics) main algorithm.

Implements the main evolution loop with:
- Population initialization
- Prompt evolution
- Code evolution
- Population management
"""

import os
import json
import time
import random
import heapq
from pathlib import Path
from typing import List, Dict, Any, Optional

from .operators import InterfaceEC, InterfaceECPrompt


def create_folders(results_path: str):
    """Create result folders and subfolders."""
    folder_path = os.path.join(results_path, "results")
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    subfolders = ["history", "pops", "pops_best"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)


class EOH:
    """
    Evolution of Heuristics (EOH) algorithm.
    
    Main evolution loop that:
    1. Initializes population of prompts and algorithms
    2. Evolves both prompts and algorithms
    3. Manages population to maintain diversity and quality
    """
    
    def __init__(
        self,
        paras: Any,
        problem: Any,
        select: Any,
        manage: Any,
    ):
        """
        Args:
            paras: Parameter configuration object.
            problem: Problem evaluation interface.
            select: Parent selection method.
            manage: Population management method.
        """
        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model
        
        # Prompt evolution settings
        self.pop_size_cross = 2
        self.pop_size_variation = 2
        self.problem_type = "minimization"
        
        # Population settings
        self.pop_size = paras.ec_pop_size
        self.n_pop = paras.ec_n_pop
        
        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjusting to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m
        
        self.debug_mode = paras.exp_debug_mode
        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.prompt_timeout = paras.prompt_eva_timeout
        self.use_numba = paras.eva_numba_decorator
        
        print("- EOH parameters loaded -")
        random.seed(2024)
    
    def add2pop(self, population: List[Dict], offspring: List[Dict]):
        """Add offspring to population."""
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if self.debug_mode:
                        print("Duplicated result, retrying...")
            population.append(off)
    
    def add2pop_prompt(self, population: List[Dict], offspring: List[Dict]):
        """Add prompt offspring to population."""
        for off in offspring:
            for ind in population:
                if ind['prompt'] == off['prompt']:
                    if self.debug_mode:
                        print("Duplicated result, retrying...")
            population.append(off)
    
    def run(self):
        """Run EOH evolution loop."""
        print("- Evolution Start -")
        time_start = time.time()
        
        interface_prob = self.prob
        
        # Initialize prompt evolution interfaces
        interface_prompt_cross = InterfaceECPrompt(
            self.pop_size_cross, self.m,
            self.api_endpoint, self.api_key, self.llm_model,
            self.debug_mode, self.select, self.exp_n_proc,
            self.prompt_timeout, self.problem_type
        )
        
        interface_prompt_variation = InterfaceECPrompt(
            self.pop_size_variation, self.m,
            self.api_endpoint, self.api_key, self.llm_model,
            self.debug_mode, self.select, self.exp_n_proc,
            self.prompt_timeout, self.problem_type
        )
        
        # Initialize code evolution interface
        interface_ec = InterfaceEC(
            self.pop_size, self.m,
            self.api_endpoint, self.api_key, self.llm_model,
            self.debug_mode, interface_prob,
            select=self.select, n_p=self.exp_n_proc,
            timeout=self.timeout, use_numba=self.use_numba
        )
        
        # Initialize prompt populations
        print("Creating initial prompts...")
        cross_operators = interface_prompt_cross.population_generation("initial_cross")
        variation_operators = interface_prompt_variation.population_generation("initial_variation")
        
        print("Initial prompts created:")
        for prompt in cross_operators:
            print(f"  Cross Prompt: {prompt['prompt']}")
        for prompt in variation_operators:
            print(f"  Variation Prompt: {prompt['prompt']}")
        
        print("=" * 50)
        
        # Initialize algorithm population
        print("Creating initial population...")
        population = interface_ec.population_generation()
        population = self.manage.population_management(population, self.pop_size)
        
        print("Initial population:")
        for off in population:
            print(f"  Obj: {off['objective']}", end=" | ")
        print()
        
        # Save initial population
        filename = f"{self.output_path}/results/pops/population_generation_0.json"
        with open(filename, 'w') as f:
            json.dump(population, f, indent=5)
        
        print("=" * 50)
        
        # Evolution loop
        worst = []
        delay_turn = 3
        change_flag = 0
        last = -1
        max_k = 4
        
        for pop_idx in range(self.n_pop):
            # Prompt evolution trigger
            if change_flag:
                change_flag -= 1
                if change_flag == 0:
                    cross_operators = self.manage.population_management(
                        cross_operators, self.pop_size_cross
                    )
                    for prompt in cross_operators:
                        print(f"Cross Prompt: {prompt['prompt']}")
                    
                    variation_operators = self.manage.population_management(
                        variation_operators, self.pop_size_variation
                    )
                    for prompt in variation_operators:
                        print(f"Variation Prompt: {prompt['prompt']}")
            
            # Check for stagnation
            if len(worst) >= delay_turn and \
               worst[-1] == worst[-delay_turn] and \
               pop_idx - last > delay_turn:
                # Evolve prompts
                parents, offsprings = interface_prompt_cross.get_algorithm(
                    cross_operators, 'cross'
                )
                self.add2pop_prompt(cross_operators, offsprings)
                
                parents, offsprings = interface_prompt_cross.get_algorithm(
                    cross_operators, 'variation'
                )
                self.add2pop_prompt(cross_operators, offsprings)
                
                for prompt in cross_operators:
                    print(f"Cross Prompt: {prompt['prompt']}")
                    prompt["objective"] = 1e9
                    prompt["number"] = []
                
                parents, offsprings = interface_prompt_variation.get_algorithm(
                    variation_operators, 'cross'
                )
                self.add2pop_prompt(variation_operators, offsprings)
                
                parents, offsprings = interface_prompt_variation.get_algorithm(
                    variation_operators, 'variation'
                )
                self.add2pop_prompt(variation_operators, offsprings)
                
                for prompt in variation_operators:
                    print(f"Variation Prompt: {prompt['prompt']}")
                    prompt["objective"] = 1e9
                    prompt["number"] = []
                
                change_flag = 2
                last = pop_idx
            
            # Crossover operations
            for i, cross_op in enumerate(cross_operators):
                prompt = cross_op["prompt"]
                print(f" OP: cross, [{i + 1}/{len(cross_operators)}] ", end="| ")
                
                parents, offsprings = interface_ec.get_algorithm(
                    population, "cross", prompt
                )
                self.add2pop(population, offsprings)
                
                for off in offsprings:
                    print(f" Obj: {off['objective']}", end=" |")
                    if off['objective'] is None:
                        continue
                    
                    if len(cross_op["number"]) < max_k:
                        heapq.heappush(cross_op["number"], -off['objective'])
                    else:
                        if off['objective'] < -cross_op["number"][0]:
                            heapq.heapreplace(cross_op["number"], -off['objective'])
                    
                    cross_op["objective"] = -sum(cross_op["number"]) / len(cross_op["number"])
                
                # Population management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f" Cross {i + 1} obj: {cross_op['objective']}")
            
            # Variation operations
            for i, var_op in enumerate(variation_operators):
                prompt = var_op["prompt"]
                print(f" OP: variation, [{i + 1}/{len(variation_operators)}] ", end="| ")
                
                parents, offsprings = interface_ec.get_algorithm(
                    population, "variation", prompt
                )
                self.add2pop(population, offsprings)
                
                for off in offsprings:
                    print(f" Obj: {off['objective']}", end=" |")
                    if off['objective'] is None:
                        continue
                    
                    if len(var_op["number"]) < max_k:
                        heapq.heappush(var_op["number"], -off['objective'])
                    else:
                        if off['objective'] < -var_op["number"][0]:
                            heapq.heapreplace(var_op["number"], -off['objective'])
                    
                    var_op["objective"] = -sum(var_op["number"]) / len(var_op["number"])
                
                # Population management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f" Variation {i + 1} obj: {var_op['objective']}")
            
            # Save population
            filename = f"{self.output_path}/results/pops/population_generation_{pop_idx + 1}.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            
            # Save best
            filename = f"{self.output_path}/results/pops_best/population_generation_{pop_idx + 1}.json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)
            
            # Progress report
            elapsed = (time.time() - time_start) / 60
            print(f"--- {pop_idx + 1} of {self.n_pop} populations finished. Time: {elapsed:.1f}m")
            print("Pop Objs:", end=" ")
            for ind in population:
                print(f"{ind['objective']}", end=" ")
            worst.append(population[-1]['objective'])
            print()


class Paras:
    """Parameter configuration for EOH."""
    
    def __init__(self):
        # General settings
        self.method = 'eoh'
        self.problem = 'milp_construct'
        self.selection = None
        self.management = None
        
        # EC settings
        self.ec_pop_size = 5
        self.ec_n_pop = 5
        self.ec_operators = None
        self.ec_m = 2
        self.ec_operator_weights = None
        
        # LLM settings
        self.llm_api_endpoint = None
        self.llm_api_key = None
        self.llm_model = None
        
        # Experiment settings
        self.exp_debug_mode = False
        self.exp_output_path = "./outputs/"
        self.exp_n_proc = 1
        
        # Evaluation settings
        self.eva_timeout = 5 * 300
        self.prompt_eva_timeout = 30
        self.eva_numba_decorator = False
    
    def set_parallel(self):
        """Set number of processes."""
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set number of processes to {num_processes}")
    
    def set_ec(self):
        """Set EC parameters."""
        if self.management is None:
            if self.method in ['ael', 'eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'
        
        if self.selection is None:
            self.selection = 'prob_rank'
        
        if self.ec_operators is None:
            if self.method == 'eoh':
                self.ec_operators = ['e1', 'e2', 'm1', 'm2']
                if self.ec_operator_weights is None:
                    self.ec_operator_weights = [1, 1, 1, 1]
            elif self.method == 'ael':
                self.ec_operators = ['crossover', 'mutation']
                if self.ec_operator_weights is None:
                    self.ec_operator_weights = [1, 1]
            elif self.method in ['ls', 'sa']:
                self.ec_operators = ['m1']
                if self.ec_operator_weights is None:
                    self.ec_operator_weights = [1]
        
        if self.method in ['ls', 'sa'] and self.ec_pop_size > 1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> Single-point-based, set pop size to 1")
    
    def set_evaluation(self):
        """Set evaluation parameters."""
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            self.eva_numba_decorator = True
        elif self.problem == 'milp_construct':
            self.eva_timeout = 350 * 5
    
    def set_paras(self, **kwargs):
        """Set all parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.set_parallel()
        self.set_ec()
        self.set_evaluation()
