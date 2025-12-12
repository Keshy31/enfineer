"""
Benchmark Test: Direct GMM Clustering
======================================
Establishes baseline performance by clustering features directly with GMM.

This answers the critical question: "Do our features capture meaningful regimes?"
If this works, the neural network autoencoder needs to beat it.

Run from project root:
    python scripts/test_benchmark.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from data.manager import DataManager


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get only the engineered feature columns (exclude raw OHLCV and macro OHLCV)."""
    exclude_patterns = [
        'Open', 'High', 'Low', 'Close', 'Volume',  # BTC OHLCV
        'TNX_', 'DXY_', 'GLD_',  # Raw macro OHLCV (but keep derived features)
    ]
    
    feature_cols = []
    for col in df.columns:
        # Skip if it matches any exclude pattern exactly or as prefix
        skip = False
        for pattern in exclude_patterns:
            if col == pattern or (pattern.endswith('_') and col.startswith(pattern)):
                # But keep derived features like dxy_zscore, gold_return, etc.
                if col in ['dxy_return', 'dxy_zscore', 'dxy_momentum_5d', 'dxy_momentum_20d', 'dxy_vs_ma',
                          'gold_return', 'gold_zscore', 'gold_momentum_5d', 'gold_momentum_20d',
                          'yield_10y', 'yield_change_1d', 'yield_change_5d', 'yield_change_20d',
                          'yield_change_zscore', 'yield_vs_ma', 'yield_momentum']:
                    continue
                skip = True
                break
        if not skip:
            feature_cols.append(col)
    
    return feature_cols


def fit_gmm_regimes(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    n_regimes: int = 5,
) -> tuple[np.ndarray, np.ndarray, GaussianMixture, StandardScaler]:
    """
    Fit GMM directly on features (no neural network compression).
    
    Returns
    -------
    labels : np.ndarray
        Regime labels for each row.
    probabilities : np.ndarray
        Soft probabilities for each regime.
    gmm : GaussianMixture
        Fitted GMM model.
    scaler : StandardScaler
        Fitted scaler.
    """
    X = features_df[feature_cols].values
    
    # Scale features (critical for GMM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='full',
        random_state=42,
        n_init=10,
        max_iter=200,
    )
    
    labels = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    return labels, probabilities, gmm, scaler


def analyze_regime_returns(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    Analyze forward returns for each regime.
    
    This is the KEY metric - do different regimes have different expected returns?
    """
    df = features_df.copy()
    df['regime'] = labels
    
    # Calculate forward returns
    df['fwd_return'] = df['Close'].shift(-forward_days) / df['Close'] - 1
    
    # Aggregate stats per regime
    results = []
    for regime in sorted(df['regime'].unique()):
        mask = df['regime'] == regime
        returns = df.loc[mask, 'fwd_return'].dropna()
        
        if len(returns) > 10:  # Need minimum samples
            results.append({
                'regime': regime,
                'count': len(returns),
                'pct_of_days': len(returns) / len(df) * 100,
                'mean_return': returns.mean() * 100,
                'std_return': returns.std() * 100,
                'sharpe': (returns.mean() / returns.std()) * np.sqrt(252 / forward_days) if returns.std() > 0 else 0,
                'win_rate': (returns > 0).mean() * 100,
                'median_return': returns.median() * 100,
            })
    
    return pd.DataFrame(results)


def backtest_regime_strategy(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    bullish_regimes: list[int],
) -> dict:
    """
    Simple backtest: Long in bullish regimes, flat otherwise.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Features with Close prices.
    labels : np.ndarray
        Regime labels.
    bullish_regimes : list[int]
        Which regime labels to consider "bullish" (go long).
        
    Returns
    -------
    dict
        Backtest metrics and equity curve.
    """
    df = features_df.copy()
    df['regime'] = labels
    
    # Position: 1 if bullish regime, 0 otherwise
    df['position'] = df['regime'].isin(bullish_regimes).astype(int)
    
    # Daily returns
    df['daily_return'] = df['Close'].pct_change()
    
    # Strategy returns (position is lagged by 1 day - we see regime today, trade tomorrow)
    df['strategy_return'] = df['position'].shift(1) * df['daily_return']
    
    # Remove NaN
    df = df.dropna()
    
    # Equity curves
    df['equity_curve'] = (1 + df['strategy_return']).cumprod()
    df['buy_hold'] = (1 + df['daily_return']).cumprod()
    
    # Metrics
    total_return = df['equity_curve'].iloc[-1] - 1
    buy_hold_return = df['buy_hold'].iloc[-1] - 1
    
    strategy_sharpe = (
        df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252)
        if df['strategy_return'].std() > 0 else 0
    )
    
    buy_hold_sharpe = (
        df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252)
        if df['daily_return'].std() > 0 else 0
    )
    
    # Max drawdown
    rolling_max = df['equity_curve'].cummax()
    drawdown = (df['equity_curve'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Time in market
    time_in_market = df['position'].shift(1).mean() * 100
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'strategy_sharpe': strategy_sharpe,
        'buy_hold_sharpe': buy_hold_sharpe,
        'max_drawdown': max_drawdown,
        'time_in_market': time_in_market,
        'equity_curve': df[['equity_curve', 'buy_hold']].copy(),
        'positions': df[['regime', 'position']].copy(),
    }


def identify_bullish_regimes(regime_stats: pd.DataFrame, min_sharpe: float = 0.5) -> list[int]:
    """
    Automatically identify which regimes are "bullish" based on returns.
    """
    bullish = regime_stats[regime_stats['sharpe'] >= min_sharpe]['regime'].tolist()
    return bullish


def plot_results(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    regime_stats: pd.DataFrame,
    backtest_results: dict,
    output_path: Path,
):
    """Create visualization of benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Regime distribution over time
    ax1 = axes[0, 0]
    df_plot = features_df.copy()
    df_plot['regime'] = labels
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(regime_stats)))
    for regime in regime_stats['regime']:
        mask = df_plot['regime'] == regime
        ax1.scatter(
            df_plot.index[mask], 
            df_plot.loc[mask, 'Close'],
            c=[colors[regime]], 
            alpha=0.6, 
            s=10,
            label=f'Regime {regime}'
        )
    ax1.set_title('BTC Price Colored by Regime')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.set_yscale('log')
    
    # 2. Regime statistics bar chart
    ax2 = axes[0, 1]
    x = regime_stats['regime']
    width = 0.35
    ax2.bar(x - width/2, regime_stats['mean_return'], width, label='Mean Return (%)', color='steelblue')
    ax2.bar(x + width/2, regime_stats['sharpe'], width, label='Sharpe Ratio', color='coral')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('Value')
    ax2.set_title('Regime Performance (5-day forward)')
    ax2.legend()
    ax2.set_xticks(x)
    
    # 3. Equity curves
    ax3 = axes[1, 0]
    equity = backtest_results['equity_curve']
    ax3.plot(equity.index, equity['equity_curve'], label='Regime Strategy', linewidth=2)
    ax3.plot(equity.index, equity['buy_hold'], label='Buy & Hold', linewidth=2, alpha=0.7)
    ax3.set_title('Equity Curve Comparison')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Portfolio Value (starting at $1)')
    ax3.legend()
    ax3.set_yscale('log')
    
    # 4. Regime time distribution
    ax4 = axes[1, 1]
    ax4.pie(
        regime_stats['count'], 
        labels=[f"R{int(r)}" for r in regime_stats['regime']],
        autopct='%1.1f%%',
        colors=colors[:len(regime_stats)],
    )
    ax4.set_title('Time Spent in Each Regime')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization to {output_path}")


def run_benchmark():
    """Run the full benchmark analysis."""
    
    print("=" * 70)
    print("SIMONS-DALIO REGIME ENGINE")
    print("Benchmark: Direct GMM Clustering (No Neural Network)")
    print("=" * 70)
    print()
    
    # =========================================
    # Step 1: Load Data
    # =========================================
    print("[1/5] Loading combined features...")
    
    dm = DataManager("./data")
    
    # Get combined features (Simons + Dalio)
    features = dm.get_features(
        "BTC-USD", "1d", 
        feature_set="combined",
        start="2020-01-01",
    )
    
    print(f"  ✓ Loaded {len(features)} days of data")
    print(f"  ✓ Date range: {features.index.min().date()} to {features.index.max().date()}")
    print(f"  ✓ Total columns: {len(features.columns)}")
    
    # =========================================
    # Step 2: Select Features
    # =========================================
    print("\n[2/5] Selecting feature columns...")
    
    feature_cols = get_feature_columns(features)
    
    print(f"  ✓ Using {len(feature_cols)} features for clustering:")
    for i, col in enumerate(feature_cols):
        print(f"    {i+1:2d}. {col}")
    
    # Check for NaN
    nan_counts = features[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n  ⚠ Warning: Found NaN values in features:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"    {col}: {nan_counts[col]} NaN")
        print("  Dropping rows with NaN...")
        features = features.dropna(subset=feature_cols)
        print(f"  ✓ {len(features)} rows remaining")
    
    # =========================================
    # Step 3: Fit GMM
    # =========================================
    print("\n[3/5] Fitting GMM with 5 regimes...")
    
    labels, probs, gmm, scaler = fit_gmm_regimes(features, feature_cols, n_regimes=5)
    
    print(f"  ✓ GMM converged in {gmm.n_iter_} iterations")
    print(f"  ✓ Log-likelihood: {gmm.score(scaler.transform(features[feature_cols].values)):.2f}")
    
    # =========================================
    # Step 4: Analyze Regimes
    # =========================================
    print("\n[4/5] Analyzing regime returns (5-day forward)...")
    
    regime_stats = analyze_regime_returns(features, labels, forward_days=5)
    
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS RESULTS")
    print("=" * 70)
    print()
    
    print(regime_stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Identify bullish regimes
    bullish = identify_bullish_regimes(regime_stats, min_sharpe=0.3)
    bearish = [r for r in regime_stats['regime'] if r not in bullish]
    
    print(f"\n  Bullish regimes (Sharpe >= 0.3): {bullish}")
    print(f"  Other regimes: {bearish}")
    
    # =========================================
    # Step 5: Backtest Strategy
    # =========================================
    print("\n[5/5] Backtesting regime-based strategy...")
    
    backtest = backtest_regime_strategy(features, labels, bullish_regimes=bullish)
    
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print()
    
    print(f"  Strategy Total Return:  {backtest['total_return']*100:+.1f}%")
    print(f"  Buy & Hold Return:      {backtest['buy_hold_return']*100:+.1f}%")
    print()
    print(f"  Strategy Sharpe:        {backtest['strategy_sharpe']:.2f}")
    print(f"  Buy & Hold Sharpe:      {backtest['buy_hold_sharpe']:.2f}")
    print()
    print(f"  Max Drawdown:           {backtest['max_drawdown']*100:.1f}%")
    print(f"  Time in Market:         {backtest['time_in_market']:.1f}%")
    
    # =========================================
    # Generate Visualization
    # =========================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    plot_results(
        features, labels, regime_stats, backtest,
        output_dir / "benchmark_gmm_direct.png"
    )
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    
    # Determine if regimes are meaningful
    sharpe_spread = regime_stats['sharpe'].max() - regime_stats['sharpe'].min()
    mean_spread = regime_stats['mean_return'].max() - regime_stats['mean_return'].min()
    
    print("Key Findings:")
    print(f"  • Sharpe spread across regimes: {sharpe_spread:.2f}")
    print(f"  • Return spread across regimes: {mean_spread:.2f}%")
    print()
    
    if sharpe_spread > 1.0:
        print("  ✓ STRONG regime separation - features capture meaningful market states")
        print("  → Neural network should compress this into cleaner representation")
    elif sharpe_spread > 0.5:
        print("  ⚠ MODERATE regime separation - some signal exists")
        print("  → Neural network may improve separation")
    else:
        print("  ✗ WEAK regime separation - features may not capture regimes well")
        print("  → Review feature engineering before building neural network")
    
    print()
    print("This benchmark establishes the baseline. The neural network's job is to:")
    print("  1. Compress features into a cleaner 3D space")
    print("  2. Achieve better regime separation (higher Sharpe spread)")
    print("  3. More stable regime assignments over time")
    
    return {
        'features': features,
        'labels': labels,
        'regime_stats': regime_stats,
        'backtest': backtest,
        'gmm': gmm,
        'scaler': scaler,
    }


if __name__ == "__main__":
    results = run_benchmark()

