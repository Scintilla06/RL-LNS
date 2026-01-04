"""
Code manipulation utilities.
Refactored from LLM-LNS.py helper functions.
"""

import ast
from typing import Sequence, Optional


def add_import_package_statement(
    program: str, 
    package_name: str, 
    as_name: Optional[str] = None, 
    *, 
    check_imported: bool = True
) -> str:
    """
    Add 'import package_name as as_name' to the program code.
    
    Args:
        program: The Python code string.
        package_name: The package to import.
        as_name: Optional alias for the import.
        check_imported: If True, skip adding if already imported.
    
    Returns:
        Modified program string with import added.
    """
    tree = ast.parse(program)
    
    if check_imported:
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == package_name:
                        return program
            elif isinstance(node, ast.ImportFrom):
                if node.module == package_name:
                    return program
    
    import_node = ast.Import(names=[ast.alias(name=package_name, asname=as_name)])
    tree.body.insert(0, import_node)
    program = ast.unparse(tree)
    return program


def add_numba_decorator(
    program: str,
    function_name: str | Sequence[str],
) -> str:
    """
    Add @numba.jit(nopython=True) decorator to specified function(s).
    
    Args:
        program: The Python code string.
        function_name: Name(s) of function(s) to decorate.
    
    Returns:
        Modified program string with numba decorator added.
    """
    if isinstance(function_name, str):
        function_name = [function_name]
    
    for fname in function_name:
        program = _add_numba_decorator_single(program, fname)
    
    return program


def _add_numba_decorator_single(program: str, function_name: str) -> str:
    """Add numba decorator to a single function."""
    tree = ast.parse(program)
    
    # Check if numba is already imported
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numba":
                    numba_imported = True
                    break
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numba":
                numba_imported = True
                break
    
    # Add numba import if needed
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name="numba", asname=None)])
        tree.body.insert(0, import_node)
    
    # Find and decorate the function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Create decorator: numba.jit(nopython=True)
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="numba", ctx=ast.Load()),
                    attr="jit",
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[ast.keyword(arg="nopython", value=ast.Constant(value=True))]
            )
            node.decorator_list.insert(0, decorator)
            break
    
    return ast.unparse(tree)


def add_np_random_seed(program: str, seed: int = 2024) -> str:
    """
    Add np.random.seed(seed) after numpy import.
    
    Args:
        program: The Python code string.
        seed: Random seed value.
    
    Returns:
        Modified program string with random seed added.
    """
    tree = ast.parse(program)
    
    # Check if numpy is imported and find its alias
    numpy_alias = None
    numpy_import_idx = -1
    
    for idx, node in enumerate(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    numpy_alias = alias.asname if alias.asname else "numpy"
                    numpy_import_idx = idx
                    break
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numpy":
                numpy_import_idx = idx
                # For 'from numpy import *', use numpy directly
                numpy_alias = "numpy"
                break
    
    # If numpy not imported, add it
    if numpy_alias is None:
        numpy_alias = "np"
        import_node = ast.Import(names=[ast.alias(name="numpy", asname="np")])
        tree.body.insert(0, import_node)
        numpy_import_idx = 0
    
    # Create seed statement: np.random.seed(seed)
    seed_stmt = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id=numpy_alias, ctx=ast.Load()),
                    attr="random",
                    ctx=ast.Load()
                ),
                attr="seed",
                ctx=ast.Load()
            ),
            args=[ast.Constant(value=seed)],
            keywords=[]
        )
    )
    
    # Insert after numpy import
    tree.body.insert(numpy_import_idx + 1, seed_stmt)
    
    return ast.unparse(tree)


def extract_code_from_response(response: str, func_name: str = "select_neighborhood") -> tuple[str, str]:
    """
    Extract algorithm description and code from LLM response.
    
    Args:
        response: LLM response string.
        func_name: Expected function name.
    
    Returns:
        Tuple of (code, algorithm_description).
    """
    import re
    
    # Extract algorithm description from braces
    algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
    if len(algorithm) == 0:
        if 'python' in response:
            algorithm = [response.split('python')[0]]
        elif 'import' in response:
            algorithm = [response.split('import')[0]]
        else:
            algorithm = [""]
    
    # Extract code
    code = re.findall(r"import.*return", response, re.DOTALL)
    if len(code) == 0:
        code = re.findall(r"def.*return", response, re.DOTALL)
    
    if len(algorithm) == 0 or len(code) == 0:
        return "", ""
    
    algorithm_str = algorithm[0] if algorithm else ""
    code_str = code[0] if code else ""
    
    return code_str, algorithm_str


def validate_python_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Python code syntax.
    
    Args:
        code: Python code string.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
