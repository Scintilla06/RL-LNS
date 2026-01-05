"""
Shared constants for MILP variable types and default bounds.

Variable Type Encoding:
- VAR_BINARY (0): Binary variables {0, 1}
- VAR_CONTINUOUS (1): Continuous variables in [lb, ub]
- VAR_INTEGER (2): Integer variables in [lb, ub] ∩ ℤ
"""

# Variable type constants
VAR_BINARY = 0
VAR_CONTINUOUS = 1
VAR_INTEGER = 2

# Default bounds for unbounded variables
DEFAULT_LB = -1e4
DEFAULT_UB = 1e4

# Binary variable bounds (always [0, 1])
BINARY_LB = 0.0
BINARY_UB = 1.0
