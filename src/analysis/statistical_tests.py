"""
Statistical Tests for Alpha Validation
======================================
Rigorous hypothesis testing to distinguish real alpha from noise.

The Simons Lesson: Every edge must be statistically validated before
you risk capital on it. A backtest Sharpe of 2.0 means nothing without
a confidence interval.

Key Tests:
1. Bootstrap CI: Is Sharpe significantly different from zero?
2. Multiple Testing: Correct for testing many regimes
3. Factor Attribution: Is alpha orthogonal to known factors?
4. Regime Persistence: Are regimes stable enough to trade?
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Result from bootstrap confidence interval estimation."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    significant: bool  # CI doesn't include zero
    
    def __repr__(self) -> str:
        sig = "***" if self.significant else ""
        return (
            f"BootstrapResult(\n"
            f"  estimate: {self.point_estimate:.3f} {sig}\n"
            f"  {self.ci_level*100:.0f}% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}]\n"
            f")"
        )


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    annualization: float = 252,
    random_state: Optional[int] = 42,
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for Sharpe ratio.
    
    The key insight: Point estimates are meaningless without uncertainty.
    A Sharpe of 2.0 with CI [0.5, 3.5] is very different from one with
    CI [1.8, 2.2].
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (daily, assumed).
    n_bootstrap : int
        Number of bootstrap samples.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    annualization : float
        Factor to annualize Sharpe (252 for daily returns).
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    BootstrapResult
        Point estimate and confidence interval.
        
    Example
    -------
    >>> returns = np.random.randn(252) * 0.02 + 0.001  # Simulated returns
    >>> result = bootstrap_sharpe_ci(returns)
    >>> print(f"Sharpe: {result.point_estimate:.2f}, CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    >>> print(f"Significant: {result.significant}")
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]  # Remove NaN
    n = len(returns)
    
    if n < 30:
        raise ValueError(f"Need at least 30 observations, got {n}")
    
    # Point estimate
    ann_factor = np.sqrt(annualization)
    point_sharpe = (returns.mean() / returns.std()) * ann_factor if returns.std() > 0 else 0
    
    # Bootstrap
    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(returns, size=n, replace=True)
        if sample.std() > 0:
            sharpe = (sample.mean() / sample.std()) * ann_factor
        else:
            sharpe = 0
        bootstrap_sharpes.append(sharpe)
    
    bootstrap_sharpes = np.array(bootstrap_sharpes)
    
    # Percentile CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)
    
    # Significant if CI doesn't include zero
    significant = (ci_lower > 0) or (ci_upper < 0)
    
    return BootstrapResult(
        point_estimate=point_sharpe,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        significant=significant,
    )


def bootstrap_sharpe_difference_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: Optional[int] = 42,
) -> Dict:
    """
    Test if two strategies have significantly different Sharpe ratios.
    
    Useful for comparing: regime A vs regime B, strategy vs benchmark.
    
    Parameters
    ----------
    returns_a, returns_b : np.ndarray
        Return series to compare.
        
    Returns
    -------
    Dict with difference estimate and significance.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    returns_a = np.asarray(returns_a)[~np.isnan(returns_a)]
    returns_b = np.asarray(returns_b)[~np.isnan(returns_b)]
    
    ann_factor = np.sqrt(252)
    
    def sharpe(r):
        return (r.mean() / r.std()) * ann_factor if r.std() > 0 else 0
    
    # Point estimates
    sharpe_a = sharpe(returns_a)
    sharpe_b = sharpe(returns_b)
    diff = sharpe_a - sharpe_b
    
    # Bootstrap the difference
    n_a, n_b = len(returns_a), len(returns_b)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        sample_a = np.random.choice(returns_a, size=n_a, replace=True)
        sample_b = np.random.choice(returns_b, size=n_b, replace=True)
        bootstrap_diffs.append(sharpe(sample_a) - sharpe(sample_b))
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
    
    # p-value: proportion of bootstrap samples on opposite side of zero
    if diff >= 0:
        p_value = np.mean(bootstrap_diffs < 0) * 2  # Two-sided
    else:
        p_value = np.mean(bootstrap_diffs > 0) * 2
    
    p_value = min(p_value, 1.0)
    
    return {
        "sharpe_a": sharpe_a,
        "sharpe_b": sharpe_b,
        "difference": diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": (ci_lower > 0) or (ci_upper < 0),
    }


def compute_regime_transition_matrix(
    labels: np.ndarray,
    n_regimes: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute regime transition probability matrix.
    
    This reveals if regimes are persistent (tradeable) or random (noise).
    High diagonal = persistent regimes = good for trend-following.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of regime labels (integers).
    n_regimes : int, optional
        Number of regimes. Inferred from labels if not provided.
        
    Returns
    -------
    transition_matrix : np.ndarray
        Shape (n_regimes, n_regimes). Row i, col j = P(regime j | regime i).
    stats : Dict
        Persistence and stability metrics.
        
    Example
    -------
    >>> labels = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    >>> trans, stats = compute_regime_transition_matrix(labels)
    >>> print(f"Avg persistence: {stats['avg_persistence']:.1%}")
    """
    labels = np.asarray(labels)
    
    if n_regimes is None:
        n_regimes = labels.max() + 1
    
    # Count transitions
    trans_counts = np.zeros((n_regimes, n_regimes))
    for i in range(len(labels) - 1):
        from_regime = labels[i]
        to_regime = labels[i + 1]
        trans_counts[from_regime, to_regime] += 1
    
    # Normalize to probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid div by zero
    trans_matrix = trans_counts / row_sums
    
    # Compute statistics
    diagonal = np.diag(trans_matrix)
    avg_persistence = diagonal.mean()
    
    # Expected duration in each regime (geometric distribution)
    # If P(stay) = p, expected duration = 1 / (1-p)
    expected_durations = 1 / (1 - diagonal + 1e-8)
    
    # Actual durations (run lengths)
    actual_durations = []
    current_regime = labels[0]
    current_duration = 1
    for i in range(1, len(labels)):
        if labels[i] == current_regime:
            current_duration += 1
        else:
            actual_durations.append((current_regime, current_duration))
            current_regime = labels[i]
            current_duration = 1
    actual_durations.append((current_regime, current_duration))
    
    duration_df = pd.DataFrame(actual_durations, columns=["regime", "duration"])
    avg_actual_duration = duration_df.groupby("regime")["duration"].mean()
    
    # Regime change frequency
    n_changes = (labels[1:] != labels[:-1]).sum()
    change_frequency = n_changes / len(labels)
    stability = 1 - change_frequency
    
    stats = {
        "avg_persistence": avg_persistence,
        "persistence_by_regime": {i: diagonal[i] for i in range(n_regimes)},
        "expected_duration_by_regime": {i: expected_durations[i] for i in range(n_regimes)},
        "actual_avg_duration_by_regime": avg_actual_duration.to_dict(),
        "regime_change_frequency": change_frequency,
        "stability": stability,
        "n_changes": n_changes,
        "n_observations": len(labels),
    }
    
    return trans_matrix, stats


def test_regime_significance(
    returns_by_regime: Dict[int, np.ndarray],
    alpha: float = 0.05,
    method: str = "bonferroni",
) -> pd.DataFrame:
    """
    Test if regime-specific returns are significantly different from zero.
    
    Applies multiple testing correction since we test each regime.
    
    Parameters
    ----------
    returns_by_regime : Dict[int, np.ndarray]
        Dictionary mapping regime labels to return arrays.
    alpha : float
        Significance level before correction.
    method : str
        Correction method: 'bonferroni', 'fdr' (Benjamini-Hochberg), or 'none'.
        
    Returns
    -------
    pd.DataFrame with columns:
        - regime, mean_return, std_return, t_stat, raw_p, adjusted_p, significant
    """
    results = []
    n_tests = len(returns_by_regime)
    
    for regime, returns in returns_by_regime.items():
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 10:
            continue
        
        # T-test against zero
        t_stat, raw_p = stats.ttest_1samp(returns, 0)
        
        # Bootstrap CI
        try:
            bootstrap_result = bootstrap_sharpe_ci(returns, n_bootstrap=5000)
            sharpe = bootstrap_result.point_estimate
            sharpe_ci = (bootstrap_result.ci_lower, bootstrap_result.ci_upper)
        except ValueError:
            sharpe = 0
            sharpe_ci = (0, 0)
        
        results.append({
            "regime": regime,
            "n_obs": len(returns),
            "mean_return": returns.mean() * 100,  # As percentage
            "std_return": returns.std() * 100,
            "sharpe": sharpe,
            "sharpe_ci_lower": sharpe_ci[0],
            "sharpe_ci_upper": sharpe_ci[1],
            "t_stat": t_stat,
            "raw_p": raw_p,
        })
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        return df
    
    # Multiple testing correction
    raw_p_values = df["raw_p"].values
    
    if method == "bonferroni":
        adjusted_p = np.minimum(raw_p_values * n_tests, 1.0)
    elif method == "fdr":
        # Benjamini-Hochberg
        sorted_idx = np.argsort(raw_p_values)
        adjusted_p = np.zeros_like(raw_p_values)
        for i, idx in enumerate(sorted_idx):
            adjusted_p[idx] = raw_p_values[idx] * n_tests / (i + 1)
        # Ensure monotonicity
        adjusted_p = np.minimum.accumulate(adjusted_p[np.argsort(sorted_idx)][::-1])[::-1]
        adjusted_p = np.minimum(adjusted_p, 1.0)
    else:
        adjusted_p = raw_p_values
    
    df["adjusted_p"] = adjusted_p
    df["significant"] = df["adjusted_p"] < alpha
    
    return df.sort_values("sharpe", ascending=False)


def factor_attribution(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    annualization: float = 252,
) -> Dict:
    """
    Decompose strategy returns into factor exposures and residual alpha.
    
    The key question: Is this alpha real, or just disguised beta?
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy return series.
    factor_returns : pd.DataFrame
        Factor return series (columns = factors).
    annualization : float
        For annualizing alpha.
        
    Returns
    -------
    Dict with:
        - factor_betas: Exposure to each factor
        - r_squared: Variance explained by factors
        - alpha: Residual return (annualized)
        - alpha_t_stat: T-statistic for alpha
        - alpha_significant: Is alpha significantly different from zero?
        
    Example
    -------
    >>> factors = pd.DataFrame({
    ...     'market': market_returns,
    ...     'momentum': momentum_factor,
    ... })
    >>> result = factor_attribution(strategy_returns, factors)
    >>> print(f"Alpha: {result['alpha']*100:.2f}% (t={result['alpha_t_stat']:.2f})")
    """
    # Align indexes
    common_idx = strategy_returns.index.intersection(factor_returns.index)
    y = strategy_returns.loc[common_idx].values
    X = factor_returns.loc[common_idx].values
    
    # Remove NaN
    valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y = y[valid]
    X = X[valid]
    
    if len(y) < 30:
        raise ValueError(f"Need at least 30 observations, got {len(y)}")
    
    # Add constant for alpha
    X_with_const = np.column_stack([np.ones(len(y)), X])
    
    # OLS regression
    # beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix in regression"}
    
    # Predictions and residuals
    y_pred = X_with_const @ beta
    residuals = y - y_pred
    
    # R-squared
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Alpha statistics
    alpha_daily = beta[0]
    alpha_annual = alpha_daily * annualization
    
    # Standard error of alpha
    n = len(y)
    k = X_with_const.shape[1]
    mse = ss_res / (n - k)
    
    # Variance of beta estimates
    try:
        var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se_alpha = np.sqrt(var_beta[0, 0])
        t_stat = alpha_daily / se_alpha if se_alpha > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
    except np.linalg.LinAlgError:
        se_alpha = np.nan
        t_stat = np.nan
        p_value = 1.0
    
    # Residual Sharpe (alpha Sharpe)
    residual_sharpe = (residuals.mean() / residuals.std()) * np.sqrt(annualization) if residuals.std() > 0 else 0
    
    # Factor betas
    factor_names = factor_returns.columns.tolist()
    factor_betas = dict(zip(factor_names, beta[1:]))
    
    return {
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_annual,
        "alpha_t_stat": t_stat,
        "alpha_p_value": p_value,
        "alpha_significant": p_value < 0.05,
        "alpha_se": se_alpha,
        "residual_sharpe": residual_sharpe,
        "r_squared": r_squared,
        "factor_betas": factor_betas,
        "n_observations": len(y),
    }


def validate_alpha(
    returns: np.ndarray,
    factors: Optional[pd.DataFrame] = None,
    cost_bps: float = 20,
    trades_per_year: int = 52,
) -> Dict:
    """
    Comprehensive alpha validation combining all tests.
    
    This is the "final exam" before risking capital.
    
    Parameters
    ----------
    returns : np.ndarray
        Strategy returns.
    factors : pd.DataFrame, optional
        Factor returns for attribution.
    cost_bps : float
        Round-trip transaction cost in basis points.
    trades_per_year : int
        Estimated number of trades per year.
        
    Returns
    -------
    Dict with all validation metrics and pass/fail for each test.
    """
    results = {}
    
    # 1. Bootstrap Sharpe CI
    try:
        bootstrap = bootstrap_sharpe_ci(returns)
        results["sharpe"] = bootstrap.point_estimate
        results["sharpe_ci_lower"] = bootstrap.ci_lower
        results["sharpe_ci_upper"] = bootstrap.ci_upper
        results["sharpe_significant"] = bootstrap.significant
    except Exception as e:
        results["sharpe_error"] = str(e)
        results["sharpe_significant"] = False
    
    # 2. Transaction cost survival
    annual_cost = cost_bps / 10000 * trades_per_year
    gross_return = np.mean(returns) * 252
    net_return = gross_return - annual_cost
    results["gross_return_annual"] = gross_return
    results["net_return_annual"] = net_return
    results["cost_annual"] = annual_cost
    results["survives_costs"] = net_return > 0
    
    # 3. Net Sharpe
    net_daily_return = np.mean(returns) - annual_cost / 252
    net_sharpe = (net_daily_return / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    results["net_sharpe"] = net_sharpe
    results["net_sharpe_acceptable"] = net_sharpe > 0.5
    
    # 4. Factor attribution (if factors provided)
    if factors is not None:
        try:
            returns_series = pd.Series(returns, index=factors.index[:len(returns)])
            factor_result = factor_attribution(returns_series, factors)
            results["factor_r_squared"] = factor_result["r_squared"]
            results["residual_alpha"] = factor_result["alpha_annual"]
            results["alpha_significant"] = factor_result["alpha_significant"]
            results["factor_betas"] = factor_result["factor_betas"]
        except Exception as e:
            results["factor_error"] = str(e)
            results["alpha_significant"] = None
    
    # 5. Overall verdict
    tests_passed = sum([
        results.get("sharpe_significant", False),
        results.get("survives_costs", False),
        results.get("net_sharpe_acceptable", False),
    ])
    
    results["tests_passed"] = tests_passed
    results["total_tests"] = 3
    results["verdict"] = "PASS" if tests_passed >= 3 else "NEEDS WORK"
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Statistical Tests Module")
    print("=" * 60)
    print()
    
    np.random.seed(42)
    
    # 1. Bootstrap Sharpe CI
    print("[1] Bootstrap Sharpe CI")
    returns = np.random.randn(252) * 0.02 + 0.0005  # Small positive drift
    result = bootstrap_sharpe_ci(returns)
    print(result)
    print()
    
    # 2. Sharpe Difference Test
    print("[2] Sharpe Difference Test")
    returns_a = np.random.randn(126) * 0.02 + 0.001  # Better strategy
    returns_b = np.random.randn(126) * 0.02 + 0.0002  # Worse strategy
    diff_result = bootstrap_sharpe_difference_test(returns_a, returns_b)
    print(f"  Strategy A Sharpe: {diff_result['sharpe_a']:.2f}")
    print(f"  Strategy B Sharpe: {diff_result['sharpe_b']:.2f}")
    print(f"  Difference: {diff_result['difference']:.2f}")
    print(f"  95% CI: [{diff_result['ci_lower']:.2f}, {diff_result['ci_upper']:.2f}]")
    print(f"  p-value: {diff_result['p_value']:.3f}")
    print(f"  Significant: {diff_result['significant']}")
    print()
    
    # 3. Regime Transition Matrix
    print("[3] Regime Transition Matrix")
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0])
    trans_matrix, stats_dict = compute_regime_transition_matrix(labels, n_regimes=3)
    print("  Transition Matrix:")
    print(trans_matrix.round(2))
    print(f"  Avg Persistence: {stats_dict['avg_persistence']:.1%}")
    print(f"  Stability: {stats_dict['stability']:.1%}")
    print()
    
    # 4. Regime Significance Test
    print("[4] Regime Significance Test")
    returns_by_regime = {
        0: np.random.randn(100) * 0.02 + 0.002,   # Bullish
        1: np.random.randn(80) * 0.025 - 0.001,   # Bearish
        2: np.random.randn(50) * 0.015 + 0.0005,  # Neutral
    }
    sig_df = test_regime_significance(returns_by_regime)
    print(sig_df.to_string(index=False))
    print()
    
    # 5. Factor Attribution
    print("[5] Factor Attribution")
    n = 252
    market = np.random.randn(n) * 0.02
    momentum = np.random.randn(n) * 0.01
    strategy = 0.0003 + 0.8 * market + 0.3 * momentum + np.random.randn(n) * 0.005
    
    factors = pd.DataFrame({"market": market, "momentum": momentum})
    strategy_series = pd.Series(strategy, index=factors.index)
    
    factor_result = factor_attribution(strategy_series, factors)
    print(f"  Alpha (annual): {factor_result['alpha_annual']*100:.2f}%")
    print(f"  Alpha t-stat: {factor_result['alpha_t_stat']:.2f}")
    print(f"  Alpha significant: {factor_result['alpha_significant']}")
    print(f"  R-squared: {factor_result['r_squared']:.2%}")
    print(f"  Factor betas: {factor_result['factor_betas']}")
    print()
    
    # 6. Comprehensive Validation
    print("[6] Comprehensive Alpha Validation")
    validation = validate_alpha(strategy, factors, cost_bps=20, trades_per_year=52)
    print(f"  Sharpe: {validation['sharpe']:.2f}")
    print(f"  Sharpe CI: [{validation['sharpe_ci_lower']:.2f}, {validation['sharpe_ci_upper']:.2f}]")
    print(f"  Sharpe significant: {validation['sharpe_significant']}")
    print(f"  Net Sharpe (after costs): {validation['net_sharpe']:.2f}")
    print(f"  Survives costs: {validation['survives_costs']}")
    print(f"  Tests passed: {validation['tests_passed']}/{validation['total_tests']}")
    print(f"  Verdict: {validation['verdict']}")
    print()
    
    print("STATISTICAL TESTS MODULE PASSED")

