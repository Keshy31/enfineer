"""
Statistical Analysis Module
===========================
Rigorous statistical tests for validating trading signals.
"""

from .statistical_tests import (
    bootstrap_sharpe_ci,
    bootstrap_sharpe_difference_test,
    compute_regime_transition_matrix,
    test_regime_significance,
    factor_attribution,
    validate_alpha,
)

__all__ = [
    "bootstrap_sharpe_ci",
    "bootstrap_sharpe_difference_test",
    "compute_regime_transition_matrix",
    "test_regime_significance",
    "factor_attribution",
    "validate_alpha",
]

