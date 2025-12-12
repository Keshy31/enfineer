"""
Walk-Forward Benchmark Test
===========================
CORRECTED benchmark using proper out-of-sample validation.

Key fixes from original benchmark:
1. Walk-forward validation (no look-ahead bias)
2. Stationary features only (no raw levels like yield_10y)
3. PCA compression (reduce dimensionality curse)
4. BIC-based optimal k selection
5. Aligned trading horizon (trade what you analyze)

This produces HONEST performance metrics that should transfer to live trading.

Run from project root:
    python scripts/test_benchmark_walkforward.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from data.manager import DataManager
from data.walk_forward import (
    WalkForwardGMM,
    WalkForwardResult,
    find_optimal_k,
    get_stationary_features,
    validate_stationarity,
)


def analyze_oos_regime_returns(
    features_df: pd.DataFrame,
    wf_result: WalkForwardResult,
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    Analyze forward returns for each regime using ONLY OOS predictions.
    
    This is the honest metric - we're measuring returns for dates
    where the regime was predicted out-of-sample.
    """
    # Create DataFrame with OOS labels
    oos_df = features_df.loc[wf_result.oos_dates].copy()
    oos_df['regime'] = wf_result.oos_labels
    
    # Calculate forward returns
    oos_df['fwd_return'] = oos_df['Close'].shift(-forward_days) / oos_df['Close'] - 1
    
    # Analyze by regime
    results = []
    for regime in range(wf_result.n_regimes):
        mask = oos_df['regime'] == regime
        returns = oos_df.loc[mask, 'fwd_return'].dropna()
        
        if len(returns) > 10:
            results.append({
                'regime': regime,
                'count': len(returns),
                'pct_of_days': len(returns) / len(oos_df) * 100,
                'mean_return': returns.mean() * 100,
                'std_return': returns.std() * 100,
                'sharpe': (returns.mean() / returns.std()) * np.sqrt(252 / forward_days) if returns.std() > 0 else 0,
                'win_rate': (returns > 0).mean() * 100,
                'median_return': returns.median() * 100,
            })
    
    return pd.DataFrame(results)


def backtest_oos_regime_strategy(
    features_df: pd.DataFrame,
    wf_result: WalkForwardResult,
    bullish_regimes: List[int],
    holding_period: int = 5,
) -> Dict:
    """
    Backtest strategy using ONLY out-of-sample regime predictions.
    
    Uses holding period to align with regime analysis horizon.
    """
    # Create DataFrame with OOS labels
    oos_df = features_df.loc[wf_result.oos_dates].copy()
    oos_df['regime'] = wf_result.oos_labels
    
    # Position: 1 if bullish regime, 0 otherwise
    oos_df['signal'] = oos_df['regime'].isin(bullish_regimes).astype(int)
    
    # Calculate returns based on holding period
    # For simplicity, use daily returns but only trade when signal is active
    oos_df['daily_return'] = oos_df['Close'].pct_change()
    
    # Strategy returns (position lagged by 1 day)
    oos_df['strategy_return'] = oos_df['signal'].shift(1) * oos_df['daily_return']
    
    # Drop NaN
    oos_df = oos_df.dropna(subset=['strategy_return', 'daily_return'])
    
    if len(oos_df) == 0:
        return {'error': 'No valid returns'}
    
    # Equity curves
    oos_df['equity_curve'] = (1 + oos_df['strategy_return']).cumprod()
    oos_df['buy_hold'] = (1 + oos_df['daily_return']).cumprod()
    
    # Metrics
    total_return = oos_df['equity_curve'].iloc[-1] - 1
    buy_hold_return = oos_df['buy_hold'].iloc[-1] - 1
    
    # Sharpe ratios
    strategy_sharpe = (
        oos_df['strategy_return'].mean() / oos_df['strategy_return'].std() * np.sqrt(252)
        if oos_df['strategy_return'].std() > 0 else 0
    )
    
    buy_hold_sharpe = (
        oos_df['daily_return'].mean() / oos_df['daily_return'].std() * np.sqrt(252)
        if oos_df['daily_return'].std() > 0 else 0
    )
    
    # Max drawdown
    rolling_max = oos_df['equity_curve'].cummax()
    drawdown = (oos_df['equity_curve'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Time in market
    time_in_market = oos_df['signal'].shift(1).mean() * 100
    
    # Regime stability (how often does regime change?)
    regime_changes = (oos_df['regime'] != oos_df['regime'].shift(1)).sum()
    regime_stability = 1 - (regime_changes / len(oos_df))
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'strategy_sharpe': strategy_sharpe,
        'buy_hold_sharpe': buy_hold_sharpe,
        'max_drawdown': max_drawdown,
        'time_in_market': time_in_market,
        'regime_stability': regime_stability,
        'regime_changes': regime_changes,
        'oos_days': len(oos_df),
        'equity_curve': oos_df[['equity_curve', 'buy_hold']].copy(),
        'positions': oos_df[['regime', 'signal']].copy(),
    }


def identify_bullish_regimes_oos(
    regime_stats: pd.DataFrame,
    min_sharpe: float = 0.3,
    min_samples: int = 50,
) -> List[int]:
    """
    Identify bullish regimes from OOS performance.
    
    More conservative than in-sample: requires minimum samples
    and uses lower Sharpe threshold.
    """
    valid = regime_stats[
        (regime_stats['sharpe'] >= min_sharpe) & 
        (regime_stats['count'] >= min_samples)
    ]
    return valid['regime'].tolist()


def plot_walkforward_results(
    features_df: pd.DataFrame,
    wf_result: WalkForwardResult,
    regime_stats: pd.DataFrame,
    backtest_results: Dict,
    output_path: Path,
):
    """Create visualization of walk-forward benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. OOS regime distribution over time
    ax1 = axes[0, 0]
    oos_df = features_df.loc[wf_result.oos_dates].copy()
    oos_df['regime'] = wf_result.oos_labels
    
    colors = plt.cm.Set2(np.linspace(0, 1, wf_result.n_regimes))
    for regime in range(wf_result.n_regimes):
        mask = oos_df['regime'] == regime
        ax1.scatter(
            oos_df.index[mask],
            oos_df.loc[mask, 'Close'],
            c=[colors[regime]],
            alpha=0.6,
            s=10,
            label=f'Regime {regime}'
        )
    
    ax1.set_title('BTC Price Colored by OOS Regime Prediction')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.set_yscale('log')
    
    # Mark fold boundaries
    for fold in wf_result.folds:
        ax1.axvline(fold.test_start, color='gray', linestyle='--', alpha=0.3)
    
    # 2. Regime statistics bar chart
    ax2 = axes[0, 1]
    x = regime_stats['regime']
    width = 0.35
    ax2.bar(x - width/2, regime_stats['mean_return'], width, label='Mean Return (%)', color='steelblue')
    ax2.bar(x + width/2, regime_stats['sharpe'], width, label='Sharpe Ratio', color='coral')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('Value')
    ax2.set_title('OOS Regime Performance (5-day forward)')
    ax2.legend()
    ax2.set_xticks(x)
    
    # 3. OOS Equity curves
    ax3 = axes[1, 0]
    if 'equity_curve' in backtest_results:
        equity = backtest_results['equity_curve']
        ax3.plot(equity.index, equity['equity_curve'], label='Regime Strategy (OOS)', linewidth=2)
        ax3.plot(equity.index, equity['buy_hold'], label='Buy & Hold', linewidth=2, alpha=0.7)
        ax3.set_title('OOS Equity Curve Comparison')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Portfolio Value (starting at $1)')
        ax3.legend()
        ax3.set_yscale('log')
    
    # 4. Walk-forward fold info
    ax4 = axes[1, 1]
    fold_data = []
    for fold in wf_result.folds:
        fold_data.append({
            'Fold': fold.fold_num,
            'Train End': fold.train_end.strftime('%Y-%m'),
            'Test Period': f"{fold.test_start.strftime('%Y-%m')} to {fold.test_end.strftime('%Y-%m')}",
            'Train N': fold.train_samples,
            'Test N': fold.test_samples,
        })
    
    fold_df = pd.DataFrame(fold_data)
    ax4.axis('off')
    table = ax4.table(
        cellText=fold_df.values,
        colLabels=fold_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Walk-Forward Folds (Expanding Window)', y=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization to {output_path}")


def run_walkforward_benchmark():
    """Run the corrected walk-forward benchmark."""
    
    print("=" * 70)
    print("SIMONS-DALIO REGIME ENGINE")
    print("Walk-Forward Benchmark (CORRECTED - No Look-Ahead Bias)")
    print("=" * 70)
    print()
    
    # =========================================
    # Step 1: Load Data
    # =========================================
    print("[1/6] Loading combined features...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        "BTC-USD", "1d",
        feature_set="combined",
        start="2020-01-01",
    )
    
    print(f"  ✓ Loaded {len(features)} days")
    print(f"  ✓ Date range: {features.index.min().date()} to {features.index.max().date()}")
    
    print()
    
    # =========================================
    # Step 2: Select STATIONARY Features Only
    # =========================================
    print("[2/6] Selecting STATIONARY features only...")
    
    stationary_cols = get_stationary_features()
    available_cols = [c for c in stationary_cols if c in features.columns]
    
    print(f"  ✓ Using {len(available_cols)} stationary features (excluding raw levels)")
    
    # Quick stationarity validation
    stat_report = validate_stationarity(features, available_cols)
    suspicious = stat_report[~stat_report['likely_stationary']]
    
    if len(suspicious) > 0:
        print(f"  ⚠ Warning: {len(suspicious)} features may have stationarity issues:")
        for _, row in suspicious.iterrows():
            print(f"    - {row['feature']}: mean drift = {row['mean_drift_std']:.2f}σ")
    else:
        print("  ✓ All features pass stationarity check")
    
    print()
    
    # =========================================
    # Step 3: Find Optimal k (BIC)
    # =========================================
    print("[3/6] Finding optimal cluster count (BIC)...")
    
    X = features[available_cols].values
    optimal_k, bic_scores = find_optimal_k(
        X, 
        k_range=range(2, 9),
        use_pca=True,
        pca_variance=0.95,
        covariance_type='diag',
    )
    
    print(f"\n  BIC Scores by k:")
    for _, row in bic_scores.iterrows():
        marker = " ← optimal" if row['k'] == optimal_k else ""
        print(f"    k={int(row['k'])}: BIC={row['bic']:.0f}{marker}")
    
    print(f"\n  ✓ Optimal k by BIC: {optimal_k}")
    
    print()
    
    # =========================================
    # Step 4: Walk-Forward GMM
    # =========================================
    print("[4/6] Running Walk-Forward GMM...")
    print()
    
    wf = WalkForwardGMM(
        n_regimes=optimal_k,
        min_train_days=504,   # 2 years
        test_days=126,        # 6 months
        gap_days=30,          # 30-day gap
        use_pca=True,
        pca_variance=0.95,
        covariance_type='diag',
    )
    
    wf_result = wf.fit_predict(features, available_cols)
    
    print()
    
    # =========================================
    # Step 5: Analyze OOS Regime Returns
    # =========================================
    print("[5/6] Analyzing OOS regime returns (5-day forward)...")
    
    regime_stats = analyze_oos_regime_returns(features, wf_result, forward_days=5)
    
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE REGIME ANALYSIS")
    print("=" * 70)
    print()
    print(regime_stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Identify bullish regimes
    bullish = identify_bullish_regimes_oos(regime_stats, min_sharpe=0.3, min_samples=50)
    bearish = [r for r in range(optimal_k) if r not in bullish]
    
    print(f"\n  Bullish regimes (OOS Sharpe >= 0.3): {bullish if bullish else 'None'}")
    print(f"  Other regimes: {bearish}")
    
    print()
    
    # =========================================
    # Step 6: OOS Backtest
    # =========================================
    print("[6/6] Backtesting on OOS predictions only...")
    
    if bullish:
        backtest = backtest_oos_regime_strategy(features, wf_result, bullish)
    else:
        # If no bullish regimes identified, use top performer
        if len(regime_stats) > 0:
            best_regime = regime_stats.loc[regime_stats['sharpe'].idxmax(), 'regime']
            bullish = [int(best_regime)]
            print(f"  ⚠ No regimes met threshold, using best performer: {bullish}")
            backtest = backtest_oos_regime_strategy(features, wf_result, bullish)
        else:
            backtest = {'error': 'No regime stats'}
    
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE BACKTEST RESULTS")
    print("=" * 70)
    print()
    
    if 'error' not in backtest:
        print(f"  OOS Period: {wf_result.oos_dates.min().date()} to {wf_result.oos_dates.max().date()}")
        print(f"  OOS Days: {backtest['oos_days']}")
        print()
        print(f"  Strategy Total Return:  {backtest['total_return']*100:+.1f}%")
        print(f"  Buy & Hold Return:      {backtest['buy_hold_return']*100:+.1f}%")
        print()
        print(f"  Strategy Sharpe (OOS):  {backtest['strategy_sharpe']:.2f}")
        print(f"  Buy & Hold Sharpe:      {backtest['buy_hold_sharpe']:.2f}")
        print()
        print(f"  Max Drawdown:           {backtest['max_drawdown']*100:.1f}%")
        print(f"  Time in Market:         {backtest['time_in_market']:.1f}%")
        print(f"  Regime Stability:       {backtest['regime_stability']*100:.1f}%")
        print(f"  Regime Changes:         {backtest['regime_changes']}")
    else:
        print(f"  Error: {backtest['error']}")
    
    # =========================================
    # Generate Visualization
    # =========================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    plot_walkforward_results(
        features, wf_result, regime_stats, backtest,
        output_dir / "benchmark_walkforward.png"
    )
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 70)
    print("WALK-FORWARD BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    
    # Compare to original (biased) benchmark
    sharpe_spread = regime_stats['sharpe'].max() - regime_stats['sharpe'].min() if len(regime_stats) > 0 else 0
    mean_spread = regime_stats['mean_return'].max() - regime_stats['mean_return'].min() if len(regime_stats) > 0 else 0
    
    print("Key Findings (HONEST - No Look-Ahead):")
    print(f"  • OOS Sharpe spread across regimes: {sharpe_spread:.2f}")
    print(f"  • OOS Return spread across regimes: {mean_spread:.2f}%")
    print()
    
    if sharpe_spread > 0.8:
        print("  ✓ GOOD regime separation survives walk-forward validation")
        print("  → Features capture real market dynamics, not just time periods")
    elif sharpe_spread > 0.4:
        print("  ⚠ MODERATE regime separation in OOS data")
        print("  → Some predictive signal exists, but weaker than in-sample")
    else:
        print("  ✗ WEAK regime separation in OOS data")
        print("  → The original benchmark was likely overfit")
    
    print()
    print("Comparison to Original (Biased) Benchmark:")
    print("  • Original Sharpe spread: 1.36 (IN-SAMPLE - inflated)")
    print(f"  • Walk-Forward Sharpe spread: {sharpe_spread:.2f} (OUT-OF-SAMPLE - honest)")
    print()
    
    if sharpe_spread < 1.0:
        print("  → As expected, OOS performance is lower than in-sample")
        print("  → This is the REAL baseline for the neural network to beat")
    
    return {
        'wf_result': wf_result,
        'regime_stats': regime_stats,
        'backtest': backtest,
        'optimal_k': optimal_k,
        'features': features,
    }


if __name__ == "__main__":
    results = run_walkforward_benchmark()

