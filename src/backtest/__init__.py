"""
Backtesting Module
==================
Realistic backtesting with transaction costs and capacity analysis.
"""

from .costs import CostModel, estimate_capacity, apply_costs_to_returns

__all__ = [
    "CostModel",
    "estimate_capacity", 
    "apply_costs_to_returns",
]

