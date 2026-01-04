"""
LLM API interfaces for evolution.

Includes:
- InterfaceAPI: Low-level HTTP API client
- InterfaceLLM: High-level LLM interface
- Evolution: Code generation with LLM
- EvolutionPrompt: Prompt evolution with LLM
"""

import http.client
import json
import re
from typing import Optional, List, Dict, Any


class InterfaceAPI:
    """
    Low-level HTTP API client for LLM communication.
    """
    
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model_llm: str,
        debug_mode: bool = False,
        n_trial: int = 5,
    ):
        """
        Args:
            api_endpoint: API endpoint URL (e.g., api.openai.com).
            api_key: API authentication key.
            model_llm: Model identifier (e.g., gpt-4o-mini).
            debug_mode: Whether to print debug info.
            n_trial: Maximum retry attempts.
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_llm = model_llm
        self.debug_mode = debug_mode
        self.n_trial = n_trial
    
    def get_response(self, prompt_content: str) -> Optional[str]:
        """
        Send prompt to LLM and get response.
        
        Args:
            prompt_content: User prompt text.
        
        Returns:
            LLM response text, or None if failed.
        """
        payload = json.dumps({
            "model": self.model_llm,
            "messages": [
                {"role": "user", "content": prompt_content}
            ],
        })
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",
        }
        
        response = None
        n_trial = 0
        
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint)
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in API: {e}. Retrying...")
                continue
        
        return response


class InterfaceLLM:
    """
    High-level LLM interface with connection validation.
    """
    
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model_llm: str,
        debug_mode: bool = False,
    ):
        """
        Args:
            api_endpoint: API endpoint URL.
            api_key: API authentication key.
            model_llm: Model identifier.
            debug_mode: Whether to print debug info.
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_llm = model_llm
        self.debug_mode = debug_mode
        
        print("- Checking LLM API...")
        
        if not api_key or not api_endpoint:
            raise ValueError("API endpoint and key must be provided")
        
        # Create API interface
        self.interface_llm = InterfaceAPI(
            self.api_endpoint,
            self.api_key,
            self.model_llm,
            self.debug_mode,
        )
        
        # Test connection
        res = self.interface_llm.get_response("1+1=?")
        if res is None:
            raise ConnectionError("Failed to connect to LLM API")
        
        print("- LLM API connection verified")
    
    def get_response(self, prompt_content: str) -> Optional[str]:
        """Get LLM response for prompt."""
        return self.interface_llm.get_response(prompt_content)


class Evolution:
    """
    LLM-based code generation for heuristic evolution.
    
    Generates Python code for neighborhood selection heuristics
    based on task descriptions and evolutionary operators.
    """
    
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model_llm: str,
        debug_mode: bool,
        prompts: Any,
    ):
        """
        Args:
            api_endpoint: API endpoint URL.
            api_key: API authentication key.
            model_llm: Model identifier.
            debug_mode: Whether to print debug info.
            prompts: GetPrompts instance with task/function info.
        """
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        
        # Format inputs/outputs
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join(f"'{s}'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = f"'{self.prompt_func_inputs[0]}'"
        
        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join(f"'{s}'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = f"'{self.prompt_func_outputs[0]}'"
        
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_llm = model_llm
        self.debug_mode = debug_mode
        
        self.interface_llm = InterfaceLLM(
            self.api_endpoint, self.api_key, self.model_llm, self.debug_mode
        )
    
    def get_prompt_initial(self) -> str:
        """Generate prompt for initial algorithm creation."""
        return (
            f"{self.prompt_task}\n"
            f"First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, implement it in Python as a function named "
            f"{self.prompt_func_name}. This function should accept {len(self.prompt_func_inputs)} input(s): "
            f"{self.joined_inputs}. The function should return {len(self.prompt_func_outputs)} output(s): "
            f"{self.joined_outputs}. {self.prompt_inout_inf} "
            f"{self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )
    
    def get_prompt_cross(self, indivs: List[Dict], prompt: str) -> str:
        """Generate prompt for crossover-based creation."""
        prompt_indiv = ""
        for i, ind in enumerate(indivs):
            prompt_indiv += (
                f"No.{i+1} algorithm's thought, objective function value, and the corresponding code are:\n"
                f"{ind['algorithm']}\n{ind['objective']}\n{ind['code']}\n"
            )
        
        return (
            f"{self.prompt_task}\n"
            f"I have {len(indivs)} existing algorithm's thought, objective function value with their codes as follows:\n"
            f"{prompt_indiv}{prompt}\n"
            f"First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, implement it in Python as a function named "
            f"{self.prompt_func_name}. This function should accept {len(self.prompt_func_inputs)} input(s): "
            f"{self.joined_inputs}. The function should return {len(self.prompt_func_outputs)} output(s): "
            f"{self.joined_outputs}. {self.prompt_inout_inf} "
            f"{self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )
    
    def get_prompt_variation(self, indiv: Dict, prompt: str) -> str:
        """Generate prompt for variation-based creation."""
        return (
            f"{self.prompt_task}\n"
            f"I have one algorithm with its code as follows.\n"
            f"Algorithm description: {indiv['algorithm']}\n"
            f"Code:\n{indiv['code']}\n"
            f"{prompt}\n"
            f"First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, implement it in Python as a function named "
            f"{self.prompt_func_name}. This function should accept {len(self.prompt_func_inputs)} input(s): "
            f"{self.joined_inputs}. The function should return {len(self.prompt_func_outputs)} output(s): "
            f"{self.joined_outputs}. {self.prompt_inout_inf} "
            f"{self.prompt_other_inf}\n"
            f"Do not give additional explanations."
        )
    
    def _get_alg(self, prompt_content: str) -> List[str]:
        """
        Parse LLM response to extract algorithm and code.
        
        Returns:
            [code_string, algorithm_description]
        """
        response = self.interface_llm.get_response(prompt_content)
        
        if self.debug_mode:
            print(f"\n>>> LLM Response:\n{response}")
        
        # Extract algorithm description
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)
        
        # Extract code
        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)
        
        # Retry if extraction failed
        n_retry = 0
        while len(algorithm) == 0 or len(code) == 0:
            if self.debug_mode:
                print("Extraction failed, retrying...")
            
            response = self.interface_llm.get_response(prompt_content)
            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)
            
            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)
            
            n_retry += 1
            if n_retry > 3:
                break
        
        algorithm = algorithm[0] if algorithm else ""
        code = code[0] if code else ""
        
        # Complete the return statement
        code_all = code + " " + ", ".join(self.prompt_func_outputs)
        
        return [code_all, algorithm]
    
    def initial(self) -> List[str]:
        """Generate initial algorithm."""
        prompt_content = self.get_prompt_initial()
        if self.debug_mode:
            print(f"\n>>> Initial Prompt:\n{prompt_content}")
        return self._get_alg(prompt_content)
    
    def cross(self, parents: List[Dict], prompt: str) -> List[str]:
        """Generate algorithm via crossover."""
        prompt_content = self.get_prompt_cross(parents, prompt)
        if self.debug_mode:
            print(f"\n>>> Cross Prompt:\n{prompt_content}")
        return self._get_alg(prompt_content)
    
    def variation(self, parent: Dict, prompt: str) -> List[str]:
        """Generate algorithm via variation."""
        prompt_content = self.get_prompt_variation(parent, prompt)
        if self.debug_mode:
            print(f"\n>>> Variation Prompt:\n{prompt_content}")
        return self._get_alg(prompt_content)


class EvolutionPrompt:
    """
    LLM-based prompt evolution.
    
    Evolves the prompts themselves to improve algorithm generation.
    """
    
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model_llm: str,
        debug_mode: bool,
        problem_type: str = "minimization",
    ):
        """
        Args:
            api_endpoint: API endpoint URL.
            api_key: API authentication key.
            model_llm: Model identifier.
            debug_mode: Whether to print debug info.
            problem_type: "minimization" or "maximization".
        """
        self.prompt_task = (
            f"We are working on solving a {problem_type} problem. "
            "Our objective is to leverage the capabilities of the Language Model (LLM) "
            "to generate heuristic algorithms that can efficiently tackle this problem. "
            "We have already developed a set of initial prompts and observed the corresponding outputs. "
            "However, to improve the effectiveness of these algorithms, we need your assistance "
            "in carefully analyzing the existing prompts and their results. "
            f"Based on this analysis, we ask you to generate new prompts that will help us "
            f"achieve better outcomes in solving the {problem_type} problem."
        )
        
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_llm = model_llm
        self.debug_mode = debug_mode
        
        self.interface_llm = InterfaceLLM(
            self.api_endpoint, self.api_key, self.model_llm, self.debug_mode
        )
    
    def get_prompt_cross(self, prompts_indivs: List[Dict]) -> str:
        """Generate prompt for crossover."""
        prompt_indiv = ""
        for i, p in enumerate(prompts_indivs):
            prompt_indiv += (
                f"No.{i+1} prompt's tasks assigned to LLM, and objective function value are:\n"
                f"{p['prompt']}\n{p['objective']}\n"
            )
        
        return (
            f"{self.prompt_task}\n"
            f"I have {len(prompts_indivs)} existing prompt with objective function value as follows:\n"
            f"{prompt_indiv}"
            f"Please help me create a new prompt that has a totally different form from the given ones "
            f"but can be motivated from them.\n"
            f"Please describe your new prompt and main steps in one sentences.\n"
            f"Do not give additional explanations!!! Just one sentences.\n"
            f"Do not give additional explanations!!! Just one sentences."
        )
    
    def get_prompt_variation(self, prompts_indivs: List[Dict]) -> str:
        """Generate prompt for variation."""
        return (
            f"{self.prompt_task}\n"
            f"I have one prompt with its objective function value as follows.\n"
            f"prompt description: {prompts_indivs[0]['prompt']}\n"
            f"objective function value:\n{prompts_indivs[0]['objective']}\n"
            f"Please assist me in creating a new prompt that has a different form "
            f"but can be a modified version of the algorithm provided.\n"
            f"Please describe your new prompt and main steps in one sentences.\n"
            f"Do not give additional explanations!!! Just one sentences.\n"
            f"Do not give additional explanations!!! Just one sentences."
        )
    
    def initialize(self, prompt_type: str) -> List[str]:
        """Get initial prompts."""
        if prompt_type == 'cross':
            return [
                'Please help me create a new algorithm that has a totally different form from the given ones.',
                'Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.'
            ]
        else:
            return [
                'Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.',
                'Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.'
            ]
    
    def cross(self, parents: List[Dict]) -> str:
        """Generate new prompt via crossover."""
        prompt_content = self.get_prompt_cross(parents)
        return self.interface_llm.get_response(prompt_content)
    
    def variation(self, parents: List[Dict]) -> str:
        """Generate new prompt via variation."""
        prompt_content = self.get_prompt_variation(parents)
        return self.interface_llm.get_response(prompt_content)
