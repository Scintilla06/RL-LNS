"""
MILP Problem definitions and data loading.

Includes:
- GetData: Load MILP instances from LP files
- GetPrompts: Prompt templates for LNS neighborhood selection
- PROBLEMCONST: MILP problem wrapper with evaluation methods
"""

import os
import copy
import numpy as np
import json
import time
import types
import sys
import warnings
import concurrent.futures
from pathlib import Path
import traceback
from typing import Dict, List, Tuple, Any, Optional

try:
    from gurobipy import GRB, read, Model
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("Warning: Gurobi not available. MILP solving disabled.")


class GetPrompts:
    """
    Prompt templates for LNS neighborhood selection.
    
    Defines the task description, function signature, and input/output specifications
    for LLM-generated heuristics.
    """
    
    def __init__(self):
        # Task description
        self.prompt_task = (
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
        
        # Function name
        self.prompt_func_name = "select_neighborhood"
        
        # Function inputs
        self.prompt_func_inputs = [
            "n", "m", "k", "site", "value", "constraint",
            "initial_solution", "current_solution", "objective_coefficient"
        ]
        
        # Function outputs
        self.prompt_func_outputs = ["neighbor_score"]
        
        # Input/output descriptions
        self.prompt_inout_inf = (
            "'n': Number of decision variables in the problem instance. 'n' is a integer number. "
            "'m': Number of constraints in the problem instance. 'm' is a integer number. "
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
            "This corresponds to the Set Cover MILP problem, where all decision variables are "
            "binary (0-1 variables), and all constraints are in the form of LHS >= RHS. "
            "'neighbor_score' is also a numpy array that you need to create manually. "
            "The i-th element of the arrays corresponds to the i-th decision variable."
        )
        
        # Other info
        self.prompt_other_inf = (
            "All are Numpy arrays. I don't give you 'neighbor_score' so that you need to "
            "create it manually. The length of the 'neighbor_score' array is also 'n'."
        )
    
    def get_task(self) -> str:
        return self.prompt_task
    
    def get_func_name(self) -> str:
        return self.prompt_func_name
    
    def get_func_inputs(self) -> List[str]:
        return self.prompt_func_inputs
    
    def get_func_outputs(self) -> List[str]:
        return self.prompt_func_outputs
    
    def get_inout_inf(self) -> str:
        return self.prompt_inout_inf
    
    def get_other_inf(self) -> str:
        return self.prompt_other_inf


class GetData:
    """
    Load MILP instances from LP files.
    
    Parses Gurobi LP files and extracts:
    - Variable information (types, bounds, objectives)
    - Constraint information (coefficients, RHS, sense)
    """
    
    def generate_instances(self, lp_path: str) -> List[Tuple]:
        """
        Load MILP instances from LP files in the given directory.
        
        Args:
            lp_path: Path to directory containing .lp files.
        
        Returns:
            List of instance tuples containing all MILP data.
        """
        if not HAS_GUROBI:
            raise RuntimeError("Gurobi is required to load LP files")
        
        sample_files = [str(path) for path in Path(lp_path).glob("*.lp")]
        instance_data = []
        
        for f in sample_files:
            model = read(f)
            value_to_num = {}
            value_num = 0
            
            # Extract problem dimensions
            n = model.NumVars
            m = model.NumConstrs
            k = []
            site = []
            value = []
            constraint = []
            constraint_type = []
            
            # Parse constraints
            for cnstr in model.getConstrs():
                # Constraint sense: 1=<=, 2=>=, 3===
                if cnstr.Sense == '<':
                    constraint_type.append(1)
                elif cnstr.Sense == '>':
                    constraint_type.append(2)
                else:
                    constraint_type.append(3)
                
                constraint.append(cnstr.RHS)
                
                # Parse constraint row
                now_site = []
                now_value = []
                row = model.getRow(cnstr)
                k.append(row.size())
                
                for i in range(row.size()):
                    var_name = row.getVar(i).VarName
                    if var_name not in value_to_num:
                        value_to_num[var_name] = value_num
                        value_num += 1
                    now_site.append(value_to_num[var_name])
                    now_value.append(row.getCoeff(i))
                
                site.append(now_site)
                value.append(now_value)
            
            # Parse variable info
            coefficient = {}
            lower_bound = {}
            upper_bound = {}
            value_type = {}
            
            for val in model.getVars():
                if val.VarName not in value_to_num:
                    value_to_num[val.VarName] = value_num
                    value_num += 1
                idx = value_to_num[val.VarName]
                coefficient[idx] = val.Obj
                lower_bound[idx] = val.LB
                upper_bound[idx] = val.UB
                value_type[idx] = val.Vtype
            
            # Get objective sense (1=minimize, -1=maximize)
            obj_type = model.ModelSense
            
            # Solve to get initial solution
            model.setObjective(0, GRB.MAXIMIZE)
            model.optimize()
            new_sol = {}
            for val in model.getVars():
                if val.VarName not in value_to_num:
                    value_to_num[val.VarName] = value_num
                    value_num += 1
                new_sol[value_to_num[val.VarName]] = val.x
            
            # Convert to numpy arrays
            new_site = []
            new_value = []
            new_constraint = np.zeros(m)
            new_constraint_type = np.zeros(m, dtype=int)
            
            for i in range(m):
                new_site.append(np.array(site[i], dtype=int))
                new_value.append(np.array(value[i]))
                new_constraint[i] = constraint[i]
                new_constraint_type[i] = constraint_type[i]
            
            new_coefficient = np.zeros(n)
            new_lower_bound = np.zeros(n)
            new_upper_bound = np.zeros(n)
            new_value_type = np.zeros(n, dtype=int)
            new_new_sol = np.zeros(n)
            
            for i in range(n):
                new_coefficient[i] = coefficient.get(i, 0)
                new_lower_bound[i] = lower_bound.get(i, 0)
                new_upper_bound[i] = upper_bound.get(i, 1)
                vtype = value_type.get(i, 'B')
                if vtype == 'B':
                    new_value_type[i] = 0
                elif vtype == 'C':
                    new_value_type[i] = 1
                else:
                    new_value_type[i] = 2
                new_new_sol[i] = new_sol.get(i, 0)
            
            instance_data.append((
                n, m, k, new_site, new_value, new_constraint,
                new_constraint_type, new_coefficient, obj_type,
                new_lower_bound, new_upper_bound, new_value_type, new_new_sol
            ))
        
        return instance_data


class MILPProblem:
    """
    MILP problem wrapper with LNS evaluation methods.
    
    Provides:
    - Gurobi solver interface
    - Heuristic evaluation with LNS
    - Code execution framework
    """
    
    def __init__(
        self,
        lp_path: str = "./SC_easy_instance/LP",
        time_limit: float = 100.0,
        n_parallel: int = 5,
        epsilon: float = 1e-3,
    ):
        """
        Args:
            lp_path: Path to LP files directory.
            time_limit: Time limit per instance.
            n_parallel: Number of parallel evaluations.
            epsilon: Convergence tolerance.
        """
        self.path = lp_path
        self.set_time = time_limit
        self.n_p = n_parallel
        self.epsilon = epsilon
        
        self.prompts = GetPrompts()
        
        # Load instances
        getData = GetData()
        self.instance_data = getData.generate_instances(self.path)
        
        print(f"Loaded {len(self.instance_data)} MILP instances")
    
    def solve_subproblem(
        self,
        n: int,
        m: int,
        k: List[int],
        site: List[np.ndarray],
        value: List[np.ndarray],
        constraint: np.ndarray,
        constraint_type: np.ndarray,
        coefficient: np.ndarray,
        time_limit: float,
        obj_type: int,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        value_type: np.ndarray,
        now_sol: np.ndarray,
        now_col: np.ndarray,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Solve MILP subproblem using Gurobi.
        
        Args:
            n: Number of variables.
            m: Number of constraints.
            k: Number of variables per constraint.
            site: Variable indices per constraint.
            value: Coefficients per constraint.
            constraint: RHS values.
            constraint_type: Constraint types (1=<=, 2=>=, 3===).
            coefficient: Objective coefficients.
            time_limit: Time limit.
            obj_type: Objective sense.
            lower_bound: Variable lower bounds.
            upper_bound: Variable upper bounds.
            value_type: Variable types (0=binary, 1=continuous, 2=integer).
            now_sol: Current solution.
            now_col: Variables to optimize (1) vs fix (0).
        
        Returns:
            Tuple of (new_solution, objective, success_flag).
        """
        if not HAS_GUROBI:
            return None, -1, -1
        
        begin_time = time.time()
        model = Model("Gurobi")
        
        # Map unfixed variables
        site_to_new = {}
        new_to_site = {}
        new_num = 0
        x = []
        
        for i in range(n):
            if now_col[i] == 1:
                site_to_new[i] = new_num
                new_to_site[new_num] = i
                new_num += 1
                
                if value_type[i] == 0:
                    x.append(model.addVar(lb=lower_bound[i], ub=upper_bound[i], vtype=GRB.BINARY))
                elif value_type[i] == 1:
                    x.append(model.addVar(lb=lower_bound[i], ub=upper_bound[i], vtype=GRB.CONTINUOUS))
                else:
                    x.append(model.addVar(lb=lower_bound[i], ub=upper_bound[i], vtype=GRB.INTEGER))
        
        # Set objective
        coeff = 0
        for i in range(n):
            if now_col[i] == 1:
                coeff += x[site_to_new[i]] * coefficient[i]
            else:
                coeff += now_sol[i] * coefficient[i]
        
        if obj_type == -1:
            model.setObjective(coeff, GRB.MAXIMIZE)
        else:
            model.setObjective(coeff, GRB.MINIMIZE)
        
        # Add constraints
        for i in range(m):
            constr = 0
            has_var = False
            
            for j in range(k[i]):
                var_idx = site[i][j]
                if now_col[var_idx] == 1:
                    constr += x[site_to_new[var_idx]] * value[i][j]
                    has_var = True
                else:
                    constr += now_sol[var_idx] * value[i][j]
            
            if has_var:
                if constraint_type[i] == 1:
                    model.addConstr(constr <= constraint[i])
                elif constraint_type[i] == 2:
                    model.addConstr(constr >= constraint[i])
                else:
                    model.addConstr(constr == constraint[i])
        
        # Solve
        model.setParam('OutputFlag', 0)
        remaining_time = time_limit - (time.time() - begin_time)
        if remaining_time <= 0:
            return None, -1, -1
        model.setParam('TimeLimit', remaining_time)
        model.optimize()
        
        try:
            new_sol = np.zeros(n)
            for i in range(n):
                if now_col[i] == 0:
                    new_sol[i] = now_sol[i]
                else:
                    if value_type[i] == 1:  # Continuous
                        new_sol[i] = x[site_to_new[i]].X
                    else:  # Binary or Integer
                        new_sol[i] = int(x[site_to_new[i]].X)
            
            return new_sol, model.ObjVal, 1
        except:
            return None, -1, -1
    
    def compute_objective(
        self,
        n: int,
        coefficient: np.ndarray,
        solution: np.ndarray,
    ) -> float:
        """Compute objective value."""
        return np.dot(coefficient, solution)
    
    def evaluate_heuristic_single(
        self,
        instance_data: Tuple,
        heuristic_module: Any,
    ) -> float:
        """
        Evaluate heuristic on a single instance.
        
        Args:
            instance_data: MILP instance tuple.
            heuristic_module: Module with select_neighborhood function.
        
        Returns:
            Final objective value.
        """
        n, m, k, site, value, constraint, constraint_type, coefficient, \
            obj_type, lower_bound, upper_bound, value_type, initial_sol = instance_data
        
        parts = 10
        begin_time = time.time()
        turn_ans = [self.compute_objective(n, coefficient, initial_sol)]
        now_sol = initial_sol.copy()
        
        try:
            while time.time() - begin_time <= self.set_time:
                # Get neighborhood scores from heuristic
                neighbor_score = heuristic_module.select_neighborhood(
                    n, m,
                    copy.deepcopy(k),
                    copy.deepcopy(site),
                    copy.deepcopy(value),
                    copy.deepcopy(constraint),
                    copy.deepcopy(initial_sol),
                    copy.deepcopy(now_sol),
                    copy.deepcopy(coefficient)
                )
                
                # Select variables with highest scores
                indices = np.argsort(neighbor_score)[::-1]
                color = np.zeros(n)
                for i in range(n // parts):
                    color[indices[i]] = 1
                
                if self.set_time - (time.time() - begin_time) <= 0:
                    break
                
                # Solve subproblem
                new_sol, now_val, now_flag = self.solve_subproblem(
                    n, m, k, site, value, constraint, constraint_type, coefficient,
                    min(self.set_time - (time.time() - begin_time), self.set_time / 5),
                    obj_type, lower_bound, upper_bound, value_type, now_sol, color
                )
                
                if now_flag == -1:
                    continue
                
                now_sol = new_sol
                turn_ans.append(now_val)
                
                # Adaptive neighborhood size
                if len(turn_ans) > 3 and \
                   abs(turn_ans[-1] - turn_ans[-3]) <= self.epsilon * abs(turn_ans[-1]) and \
                   parts >= 3:
                    parts -= 1
            
            return turn_ans[-1]
            
        except Exception as e:
            print(f"MILP Error: {e}")
            traceback.print_exc()
            return 1e9
    
    def evaluate_heuristic(self, heuristic_module: Any) -> float:
        """
        Evaluate heuristic on all instances.
        
        Args:
            heuristic_module: Module with select_neighborhood function.
        
        Returns:
            Average objective value across instances.
        """
        results = []
        
        try:
            for instance_data in self.instance_data:
                result = self._run_with_timeout(
                    150,
                    self.evaluate_heuristic_single,
                    instance_data,
                    heuristic_module
                )
                
                if result is not None:
                    results.append(result)
                else:
                    results.append(1e9)
                    
        except Exception as e:
            print(f"Parallel MILP Error: {e}")
            traceback.print_exc()
            results = [1e9]
        
        return sum(results) / len(results)
    
    def _run_with_timeout(
        self,
        time_limit: float,
        func,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """Run function with timeout."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=time_limit)
            except concurrent.futures.TimeoutError:
                print(f"Function {func.__name__} timed out after {time_limit} seconds.")
                return None
    
    def evaluate(self, code_string: str) -> Optional[float]:
        """
        Evaluate a code string as a heuristic.
        
        Args:
            code_string: Python code defining select_neighborhood function.
        
        Returns:
            Average objective value, or None if execution failed.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Create module from code
                heuristic_module = types.ModuleType("heuristic_module")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module
                
                # Evaluate heuristic
                fitness = self.evaluate_heuristic(heuristic_module)
                return fitness
                
        except Exception as e:
            print(f"Greedy MILP Error: {e}")
            return None
