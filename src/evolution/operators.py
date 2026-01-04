"""
Evolutionary operators for heuristic evolution.

Includes:
- InterfaceEC: Evolutionary computation interface for code evolution
- InterfaceECPrompt: Evolutionary computation interface for prompt evolution
"""

import numpy as np
import re
import time
import warnings
import concurrent.futures
from typing import Optional, List, Dict, Any, Tuple
from joblib import Parallel, delayed

from ..llm.api import Evolution, EvolutionPrompt
from ..utils.code_utils import add_numba_decorator


class InterfaceEC:
    """
    Evolutionary computation interface for code evolution.
    
    Handles population generation, offspring creation, and fitness evaluation.
    """
    
    def __init__(
        self,
        pop_size: int,
        m: int,
        api_endpoint: str,
        api_key: str,
        llm_model: str,
        debug_mode: bool,
        interface_prob: Any,
        select: Any,
        n_p: int,
        timeout: float,
        use_numba: bool = False,
    ):
        """
        Args:
            pop_size: Population size.
            m: Number of parents for crossover.
            api_endpoint: LLM API endpoint.
            api_key: LLM API key.
            llm_model: LLM model identifier.
            debug_mode: Whether to print debug info.
            interface_prob: Problem evaluation interface.
            select: Parent selection method.
            n_p: Number of parallel processes.
            timeout: Evaluation timeout.
            use_numba: Whether to add numba decorators.
        """
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        
        self.evol = Evolution(
            api_endpoint, api_key, llm_model, debug_mode, prompts
        )
        
        self.m = m
        self.debug = debug_mode
        self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba
        
        if not self.debug:
            warnings.filterwarnings("ignore")
    
    def add2pop(self, population: List[Dict], offspring: Dict) -> bool:
        """Add offspring to population if not duplicate."""
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("Duplicated result, retrying...")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self, population: List[Dict], code: str) -> bool:
        """Check if code already exists in population."""
        for ind in population:
            if code == ind['code']:
                return True
        return False
    
    def _get_alg(
        self,
        pop: List[Dict],
        operator: str,
        prompt: str,
    ) -> Tuple[Optional[List[Dict]], Dict]:
        """
        Generate offspring using specified operator.
        
        Args:
            pop: Current population.
            operator: Evolution operator ("initial", "cross", "variation").
            prompt: Additional prompt text.
        
        Returns:
            Tuple of (parents, offspring).
        """
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        
        if operator == "initial":
            parents = None
            [offspring['code'], offspring['algorithm']] = self.evol.initial()
        elif operator == "cross":
            parents = self.select.parent_selection(pop, self.m)
            [offspring['code'], offspring['algorithm']] = self.evol.cross(parents, prompt)
        elif operator == "variation":
            parents = self.select.parent_selection(pop, 1)
            [offspring['code'], offspring['algorithm']] = self.evol.variation(parents[0], prompt)
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        return parents, offspring
    
    def get_offspring(
        self,
        pop: List[Dict],
        operator: str,
        prompt: str,
    ) -> Tuple[Optional[List[Dict]], Dict]:
        """
        Generate and evaluate offspring.
        
        Args:
            pop: Current population.
            operator: Evolution operator.
            prompt: Additional prompt text.
        
        Returns:
            Tuple of (parents, offspring with fitness).
        """
        try:
            p, offspring = self._get_alg(pop, operator, prompt)
            
            # Add numba decorator if enabled
            if self.use_numba:
                pattern = r"def\s+(\w+)\s*\(.*\):"
                match = re.search(pattern, offspring['code'])
                if match:
                    function_name = match.group(1)
                    code = add_numba_decorator(offspring['code'], function_name)
                else:
                    code = offspring['code']
            else:
                code = offspring['code']
            
            # Handle duplicate code
            n_retry = 0
            while self.check_duplicate(pop, offspring['code']):
                n_retry += 1
                if self.debug:
                    print("Duplicated code, retrying...")
                
                p, offspring = self._get_alg(pop, operator, prompt)
                
                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    if match:
                        function_name = match.group(1)
                        code = add_numba_decorator(offspring['code'], function_name)
                    else:
                        code = offspring['code']
                else:
                    code = offspring['code']
                
                if n_retry > 1:
                    break
            
            # Evaluate fitness
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code)
                fitness = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5) if fitness is not None else None
                future.cancel()
        
        except Exception as e:
            if self.debug:
                print(f"Error in get_offspring: {e}")
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None
        
        return p, offspring
    
    def get_algorithm(
        self,
        pop: List[Dict],
        operator: str,
        prompt: str,
    ) -> Tuple[List, List[Dict]]:
        """
        Generate multiple offspring in parallel.
        
        Args:
            pop: Current population.
            operator: Evolution operator.
            prompt: Additional prompt text.
        
        Returns:
            Tuple of (all_parents, all_offspring).
        """
        results = []
        try:
            results = Parallel(n_jobs=self.n_p, timeout=self.timeout + 15)(
                delayed(self.get_offspring)(pop, operator, prompt)
                for _ in range(self.pop_size)
            )
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel timeout.")
        
        time.sleep(2)
        
        out_p = []
        out_off = []
        
        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> Offspring: {off}")
        
        return out_p, out_off
    
    def population_generation(self) -> List[Dict]:
        """Generate initial population."""
        n_create = 2
        population = []
        
        for _ in range(n_create):
            _, pop = self.get_algorithm([], 'initial', [])
            population.extend(pop)
        
        return population
    
    def population_generation_seed(
        self,
        seeds: List[Dict],
        n_p: int,
    ) -> List[Dict]:
        """Generate population from seed algorithms."""
        population = []
        
        fitness = Parallel(n_jobs=n_p)(
            delayed(self.interface_eval.evaluate)(seed['code'])
            for seed in seeds
        )
        
        for i, seed in enumerate(seeds):
            try:
                seed_alg = {
                    'algorithm': seed['algorithm'],
                    'code': seed['code'],
                    'objective': None,
                    'other_inf': None
                }
                
                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)
            except Exception as e:
                print(f"Error in seed algorithm: {e}")
        
        print(f"Initialization finished! Got {len(seeds)} seed algorithms")
        return population


class InterfaceECPrompt:
    """
    Evolutionary computation interface for prompt evolution.
    
    Evolves prompts that guide code generation.
    """
    
    def __init__(
        self,
        pop_size: int,
        m: int,
        api_endpoint: str,
        api_key: str,
        llm_model: str,
        debug_mode: bool,
        select: Any,
        n_p: int,
        timeout: float,
        problem_type: str,
    ):
        """
        Args:
            pop_size: Population size.
            m: Number of parents for crossover.
            api_endpoint: LLM API endpoint.
            api_key: LLM API key.
            llm_model: LLM model identifier.
            debug_mode: Whether to print debug info.
            select: Parent selection method.
            n_p: Number of parallel processes.
            timeout: Evaluation timeout.
            problem_type: Problem type ("minimization" or "maximization").
        """
        self.pop_size = pop_size
        self.evol = EvolutionPrompt(
            api_endpoint, api_key, llm_model, debug_mode, problem_type
        )
        self.m = m
        self.debug = debug_mode
        self.select = select
        self.n_p = n_p
        self.timeout = timeout
        
        if not self.debug:
            warnings.filterwarnings("ignore")
    
    def add2pop(self, population: List[Dict], offspring: Dict) -> bool:
        """Add offspring to population if not duplicate."""
        for ind in population:
            if ind['prompt'] == offspring['prompt']:
                if self.debug:
                    print("Duplicated result, retrying...")
                return False
        population.append(offspring)
        return True
    
    def extract_first_quoted_string(self, text: str) -> str:
        """Extract first quoted string from text."""
        match = re.search(r'"(.*?)"', text)
        if match:
            text = match.group(1)
        prefix = "Prompt: "
        if text.startswith(prefix):
            return text[len(prefix):].strip()
        return text
    
    def _get_alg(
        self,
        pop: List[Dict],
        operator: str,
    ) -> Tuple[List[Dict], Dict, List[Dict]]:
        """Generate offspring prompt using specified operator."""
        offspring = {
            'prompt': None,
            'objective': None,
            'number': None
        }
        off_set = []
        
        if operator == "initial_cross":
            parents = []
            prompt_list = self.evol.initialize("cross")
            for prompt in prompt_list:
                off = {
                    'prompt': prompt,
                    'objective': 1e9,
                    'number': []
                }
                off_set.append(off)
        elif operator == "initial_variation":
            parents = []
            prompt_list = self.evol.initialize("variation")
            for prompt in prompt_list:
                off = {
                    'prompt': prompt,
                    'objective': 1e9,
                    'number': []
                }
                off_set.append(off)
        elif operator == "cross":
            parents = self.select.parent_selection(pop, self.m)
            prompt_now = self.evol.cross(parents)
            try:
                prompt_new = self.extract_first_quoted_string(prompt_now)
            except Exception as e:
                print(f"Prompt cross error: {e}")
                prompt_new = prompt_now
            offspring["prompt"] = prompt_new
            offspring["objective"] = 1e9
            offspring["number"] = []
        elif operator == "variation":
            parents = self.select.parent_selection(pop, 1)
            prompt_now = self.evol.variation(parents)
            try:
                prompt_new = self.extract_first_quoted_string(prompt_now)
            except Exception as e:
                print(f"Prompt variation error: {e}")
                prompt_new = prompt_now
            offspring["prompt"] = prompt_new
            offspring["objective"] = 1e9
            offspring["number"] = []
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        return parents, offspring, off_set
    
    def get_offspring(
        self,
        pop: List[Dict],
        operator: str,
    ) -> Tuple[Optional[List[Dict]], Dict, List[Dict]]:
        """Generate offspring prompt."""
        try:
            p, offspring, off_set = self._get_alg(pop, operator)
        except Exception as e:
            print(f"get_offspring error: {e}")
            offspring = {
                'prompt': None,
                'objective': None,
                'number': None
            }
            p = None
            off_set = None
        
        return p, offspring, off_set
    
    def get_algorithm(
        self,
        pop: List[Dict],
        operator: str,
    ) -> Tuple[List, List[Dict]]:
        """Generate multiple prompt offspring in parallel."""
        results = []
        try:
            if operator in ['cross', 'variation']:
                results = Parallel(n_jobs=self.n_p, timeout=self.timeout + 15)(
                    delayed(self.get_offspring)(pop, operator)
                    for _ in range(self.pop_size)
                )
            else:
                results = Parallel(n_jobs=self.n_p, timeout=self.timeout + 15)(
                    delayed(self.get_offspring)(pop, operator)
                    for _ in range(1)
                )
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel timeout.")
        
        time.sleep(2)
        
        out_p = []
        out_off = []
        
        for p, off, off_set in results:
            out_p.append(p)
            if operator in ['cross', 'variation']:
                out_off.append(off)
            else:
                out_off.extend(off_set)
            if self.debug:
                print(f">>> Offspring: {off}")
        
        return out_p, out_off
    
    def population_generation(self, initial_type: str) -> List[Dict]:
        """Generate initial prompt population."""
        n_create = 1
        population = []
        
        for _ in range(n_create):
            _, pop = self.get_algorithm([], initial_type)
            population.extend(pop)
        
        return population
