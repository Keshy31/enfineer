# Simons-Dalio Regime Engine
"""
A quantitative trading system for market regime detection using
Multi-Modal LSTM-Autoencoder and Gaussian Mixture Models.

Modules:
    data: Data fetching, feature engineering, walk-forward validation
    models: LSTM-Autoencoder (HedgeFundBrain)
    backtest: Transaction cost modeling, capacity analysis
    analysis: Statistical tests, factor attribution, alpha validation
"""

__version__ = "1.2.0"

# Lazy imports to avoid circular dependencies
def get_data_manager():
    from .data import DataManager
    return DataManager

def get_walk_forward_gmm():
    from .data import WalkForwardGMM
    return WalkForwardGMM

def get_hedge_fund_brain():
    from .models import HedgeFundBrain
    return HedgeFundBrain

def get_cost_model():
    from .backtest import CostModel
    return CostModel

def get_statistical_tests():
    from .analysis import (
        bootstrap_sharpe_ci,
        factor_attribution,
        validate_alpha,
    )
    return {
        "bootstrap_sharpe_ci": bootstrap_sharpe_ci,
        "factor_attribution": factor_attribution,
        "validate_alpha": validate_alpha,
    }

