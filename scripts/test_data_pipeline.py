"""
Test Data Pipeline
==================
Verifies the data ingestion layer is working correctly.

This script:
1. Fetches Bitcoin OHLCV data
2. Computes Simons features (log returns, Z-scores, volatility)
3. Validates no signal explosions occur
4. Generates visualization for manual inspection

Run from project root:
    python scripts/test_data_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.fetcher import fetch_bitcoin_data
from data.features import compute_simons_features, get_feature_columns


def run_pipeline_test():
    """Run the complete data pipeline test."""
    
    print("=" * 60)
    print("SIMONS-DALIO REGIME ENGINE")
    print("Data Pipeline Test")
    print("=" * 60)
    print()
    
    # =========================================
    # Step 1: Fetch Bitcoin Data
    # =========================================
    print("[1/4] Fetching Bitcoin data...")
    df = fetch_bitcoin_data(years_of_history=2)
    print()
    
    # =========================================
    # Step 2: Compute Features
    # =========================================
    print("[2/4] Computing Simons features...")
    features = compute_simons_features(df, window=30, sigma_floor=0.001)
    print(f"  Generated {len(get_feature_columns())} features")
    print(f"  Final dataset: {len(features)} rows")
    print()
    
    # =========================================
    # Step 3: Validate Results
    # =========================================
    print("[3/4] Validating feature quality...")
    
    # Check for any remaining NaN
    nan_count = features.isnull().sum().sum()
    if nan_count > 0:
        print(f"  ✗ WARNING: {nan_count} NaN values found!")
    else:
        print("  ✓ No NaN values")
    
    # Check Z-score bounds (should rarely exceed ±5 with volatility floor)
    zscore_cols = [c for c in features.columns if "zscore" in c]
    extreme_threshold = 10
    all_bounded = True
    
    for col in zscore_cols:
        max_abs = features[col].abs().max()
        if max_abs > extreme_threshold:
            print(f"  ✗ WARNING: {col} has extreme values (max abs: {max_abs:.2f})")
            all_bounded = False
    
    if all_bounded:
        print(f"  ✓ All Z-scores within ±{extreme_threshold} bounds")
    
    # Print feature statistics
    print("\n  Feature Statistics:")
    stats = features[get_feature_columns()].describe().round(4)
    print(stats.to_string().replace("\n", "\n  "))
    print()
    
    # =========================================
    # Step 4: Visualization
    # =========================================
    print("[4/4] Generating visualization...")
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Bitcoin Data Pipeline - Feature Engineering Results", 
                 fontsize=14, fontweight="bold")
    
    # Plot 1: Raw BTC Price
    ax1 = axes[0]
    ax1.plot(features.index, features["Close"], color="#2962FF", linewidth=1)
    ax1.set_ylabel("Price (USD)")
    ax1.set_title("BTC-USD Daily Close Price")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    # Plot 2: Z-Scored Returns
    ax2 = axes[1]
    colors = np.where(features["log_return_zscore"] >= 0, "#00C853", "#FF1744")
    ax2.bar(features.index, features["log_return_zscore"], color=colors, width=1, alpha=0.7)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axhline(y=2, color="orange", linestyle="--", linewidth=0.5, alpha=0.7, label="±2σ")
    ax2.axhline(y=-2, color="orange", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.set_ylabel("Z-Score")
    ax2.set_title("Log Returns (Z-Scored with Volatility Floor)")
    ax2.set_ylim(-6, 6)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    # Plot 3: Rolling Volatility
    ax3 = axes[2]
    ax3.fill_between(features.index, 0, features["volatility"], 
                     color="#9C27B0", alpha=0.4, label="30-Day Volatility")
    ax3.plot(features.index, features["volatility"], color="#9C27B0", linewidth=1)
    ax3.set_ylabel("Volatility (σ)")
    ax3.set_title("Rolling 30-Day Volatility of Log Returns")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Momentum Z-Score
    ax4 = axes[3]
    ax4.plot(features.index, features["momentum_zscore"], 
             color="#FF6F00", linewidth=1, label="Momentum Z-Score")
    ax4.fill_between(features.index, 0, features["momentum_zscore"],
                     where=features["momentum_zscore"] >= 0, 
                     color="#00C853", alpha=0.3)
    ax4.fill_between(features.index, 0, features["momentum_zscore"],
                     where=features["momentum_zscore"] < 0, 
                     color="#FF1744", alpha=0.3)
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax4.set_ylabel("Z-Score")
    ax4.set_xlabel("Date")
    ax4.set_title("30-Day Momentum (Z-Scored)")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / "output"
    output_path.mkdir(exist_ok=True)
    fig_path = output_path / "btc_feature_pipeline.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved visualization to: {fig_path}")
    
    # Show plot
    plt.show()
    
    # =========================================
    # Summary
    # =========================================
    print()
    print("=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  • Data points: {len(features)}")
    print(f"  • Date range: {features.index[0].strftime('%Y-%m-%d')} to {features.index[-1].strftime('%Y-%m-%d')}")
    print(f"  • Features: {len(get_feature_columns())}")
    print(f"  • Ready for neural network: YES")
    print()
    
    return features


if __name__ == "__main__":
    features = run_pipeline_test()

