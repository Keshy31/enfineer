"""
Dalio Features Module
=====================
Implements macro-economic features inspired by Ray Dalio's approach.

These features provide the "economic coordinates" that contextualize
price action. The neural network's Macro Branch processes these to
understand whether we're in a risk-on or risk-off environment.

Key insight: The same price pattern (e.g., +5% daily move) means
different things in different macro environments:
- Rising yields + strong dollar = likely profit-taking ahead
- Falling yields + weak dollar = more room to run
"""

import numpy as np
import pandas as pd
from typing import Optional

from .features import apply_volatility_floor_zscore


def compute_dalio_features(
    aligned_df: pd.DataFrame,
    window: int = 30,
    sigma_floor: float = 0.001,
) -> pd.DataFrame:
    """
    Compute macro-economic features from aligned market data.
    
    Expects input from align_to_crypto() with columns:
    - BTC OHLCV (Open, High, Low, Close, Volume)
    - TNX_* (Treasury yield)
    - DXY_* (Dollar Index)
    - GLD_* (Gold)
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned DataFrame from align_to_crypto().
    window : int, default=30
        Rolling window for calculations.
    sigma_floor : float, default=0.001
        Volatility floor for Z-score normalization.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus Dalio features.
    """
    features = aligned_df.copy()
    
    # =========================================
    # TREASURY YIELDS (Interest Rate Environment)
    # =========================================
    if "TNX_Close" in features.columns:
        # Yield level (already in percentage points, e.g., 4.5 = 4.5%)
        features["yield_10y"] = features["TNX_Close"]
        
        # Yield changes (momentum in rates)
        features["yield_change_1d"] = features["TNX_Close"].diff(1)
        features["yield_change_5d"] = features["TNX_Close"].diff(5)
        features["yield_change_20d"] = features["TNX_Close"].diff(20)
        
        # Z-scored yield changes
        features["yield_change_zscore"] = apply_volatility_floor_zscore(
            features["yield_change_1d"], window=window, sigma_floor=sigma_floor * 10
        )
        
        # Yield regime: above/below rolling average
        features["yield_vs_ma"] = features["TNX_Close"] - features["TNX_Close"].rolling(
            window=window, min_periods=1
        ).mean()
        
        # Rate of change (momentum)
        features["yield_momentum"] = features["yield_change_5d"].rolling(
            window=window, min_periods=1
        ).sum()
    
    # =========================================
    # DOLLAR INDEX (Currency Strength)
    # =========================================
    if "DXY_Close" in features.columns:
        # Dollar returns
        features["dxy_return"] = np.log(features["DXY_Close"] / features["DXY_Close"].shift(1))
        
        # Z-scored dollar returns
        features["dxy_zscore"] = apply_volatility_floor_zscore(
            features["dxy_return"], window=window, sigma_floor=sigma_floor
        )
        
        # Dollar momentum (5-day trend)
        features["dxy_momentum_5d"] = features["dxy_return"].rolling(
            window=5, min_periods=1
        ).sum()
        
        # Dollar momentum (20-day trend)
        features["dxy_momentum_20d"] = features["dxy_return"].rolling(
            window=20, min_periods=1
        ).sum()
        
        # Dollar vs moving average (trend position)
        features["dxy_vs_ma"] = features["DXY_Close"] / features["DXY_Close"].rolling(
            window=window, min_periods=1
        ).mean() - 1
    
    # =========================================
    # GOLD (Risk-Off Proxy)
    # =========================================
    if "GLD_Close" in features.columns:
        # Gold returns
        features["gold_return"] = np.log(features["GLD_Close"] / features["GLD_Close"].shift(1))
        
        # Z-scored gold returns
        features["gold_zscore"] = apply_volatility_floor_zscore(
            features["gold_return"], window=window, sigma_floor=sigma_floor
        )
        
        # Gold momentum
        features["gold_momentum_5d"] = features["gold_return"].rolling(
            window=5, min_periods=1
        ).sum()
        
        features["gold_momentum_20d"] = features["gold_return"].rolling(
            window=20, min_periods=1
        ).sum()

    # =========================================
    # VOLATILITY (VIX - Fear Gauge)
    # =========================================
    if "VIX_Close" in features.columns:
        features["vix_level"] = features["VIX_Close"]
        
        # VIX Regime: 0=Calm(<20), 1=Fear(20-30), 2=Panic(>30)
        features["vix_regime"] = 0
        features["vix_regime"] = np.where(features["VIX_Close"] >= 20, 1, features["vix_regime"])
        features["vix_regime"] = np.where(features["VIX_Close"] >= 30, 2, features["vix_regime"])

    # =========================================
    # YIELD CURVE (Recession Signal)
    # =========================================
    if "TNX_Close" in features.columns and "IRX_Close" in features.columns:
        # 10Y - 3M Spread (Classic recession indicator)
        features["yield_spread"] = features["TNX_Close"] - features["IRX_Close"]
        
        # Inversion Flag (1 = Inverted/Recession Warning)
        features["curve_inversion"] = (features["yield_spread"] < 0).astype(int)

    # =========================================
    # EQUITY BETA (Nasdaq - Tech Correlation)
    # =========================================
    if "NASDAQ_Close" in features.columns and "Close" in features.columns:
        nasdaq_ret = np.log(features["NASDAQ_Close"] / features["NASDAQ_Close"].shift(1))
        btc_ret = np.log(features["Close"] / features["Close"].shift(1))
        
        # Rolling Correlation
        features["equity_corr"] = btc_ret.rolling(window=window, min_periods=window//2).corr(nasdaq_ret)
        
        # Rolling Beta = Cov(BTC, NQ) / Var(NQ)
        cov = btc_ret.rolling(window=window, min_periods=window//2).cov(nasdaq_ret)
        var = nasdaq_ret.rolling(window=window, min_periods=window//2).var()
        features["equity_beta"] = cov / var

    # =========================================
    # COMMODITIES (Oil - Inflation Proxy)
    # =========================================
    if "OIL_Close" in features.columns:
        features["oil_return"] = np.log(features["OIL_Close"] / features["OIL_Close"].shift(1))
        
        if "Close" in features.columns:
            btc_ret = np.log(features["Close"] / features["Close"].shift(1))
            features["oil_corr"] = btc_ret.rolling(window=window, min_periods=window//2).corr(features["oil_return"])
    
    # =========================================
    # CROSS-ASSET CORRELATIONS
    # =========================================
    if "Close" in features.columns:  # BTC Close
        btc_returns = np.log(features["Close"] / features["Close"].shift(1))
        
        # BTC-Gold correlation (risk correlation regime)
        if "gold_return" in features.columns:
            features["btc_gold_corr"] = btc_returns.rolling(
                window=window, min_periods=window // 2
            ).corr(features["gold_return"])
        
        # BTC-Dollar correlation (inverse relationship expected)
        if "dxy_return" in features.columns:
            features["btc_dxy_corr"] = btc_returns.rolling(
                window=window, min_periods=window // 2
            ).corr(features["dxy_return"])
        
        # BTC-Yield correlation (rate sensitivity)
        if "yield_change_1d" in features.columns:
            features["btc_yield_corr"] = btc_returns.rolling(
                window=window, min_periods=window // 2
            ).corr(features["yield_change_1d"])
    
    # =========================================
    # COMPOSITE INDICATORS
    # =========================================
    
    # Risk-On Score: Combines multiple signals
    # Positive = risk-on environment, Negative = risk-off
    risk_signals = []
    
    if "yield_change_zscore" in features.columns:
        # Falling yields = risk-on (easier financial conditions)
        risk_signals.append(-features["yield_change_zscore"])
    
    if "dxy_zscore" in features.columns:
        # Weak dollar = risk-on (more liquidity for risk assets)
        risk_signals.append(-features["dxy_zscore"])
    
    if "gold_zscore" in features.columns:
        # Weak gold = risk-on (less safe-haven demand)
        risk_signals.append(-features["gold_zscore"])
    
    if "vix_level" in features.columns:
        # Low VIX = Risk-On. Normalize VIX (20 is pivot).
        # (20 - VIX) / 10 -> >0 if VIX<20, <0 if VIX>20
        risk_signals.append((20 - features["vix_level"]) / 10)
        
    if "curve_inversion" in features.columns:
        # Inversion = Risk-Off (-1 penalty)
        risk_signals.append(-features["curve_inversion"])
    
    if risk_signals:
        # Average of normalized signals
        features["risk_on_score"] = sum(risk_signals) / len(risk_signals)
    
    # Macro Regime Indicator
    # Based on yield and dollar direction
    if "yield_momentum" in features.columns and "dxy_momentum_20d" in features.columns:
        # 4 regimes based on yield and dollar trends:
        # 1: Rising yields, Strong dollar (tightening)
        # 2: Rising yields, Weak dollar (inflation concern)
        # 3: Falling yields, Strong dollar (flight to quality)
        # 4: Falling yields, Weak dollar (easing)
        
        yield_rising = (features["yield_momentum"] > 0).astype(int)
        dxy_rising = (features["dxy_momentum_20d"] > 0).astype(int)
        
        # Encode as single number: 0-3
        features["macro_regime"] = yield_rising * 2 + dxy_rising
    
    # =========================================
    # Clean up: Drop warmup period NaNs
    # =========================================
    original_len = len(features)
    features = features.dropna()
    dropped = original_len - len(features)
    
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN (warmup period)")
    
    return features


def get_dalio_feature_columns() -> list[str]:
    """
    Return the list of Dalio feature column names.
    
    Returns
    -------
    list[str]
        Feature column names (excludes raw OHLCV).
    """
    return [
        # Yield features
        "yield_10y",
        "yield_change_1d",
        "yield_change_5d",
        "yield_change_20d",
        "yield_change_zscore",
        "yield_vs_ma",
        "yield_momentum",
        # Dollar features
        "dxy_return",
        "dxy_zscore",
        "dxy_momentum_5d",
        "dxy_momentum_20d",
        "dxy_vs_ma",
        # Gold features
        "gold_return",
        "gold_zscore",
        "gold_momentum_5d",
        "gold_momentum_20d",
        # Volatility features
        "vix_level",
        "vix_regime",
        # Yield Curve features
        "yield_spread",
        "curve_inversion",
        # Equity features
        "equity_beta",
        "equity_corr",
        # Commodity features
        "oil_return",
        "oil_corr",
        # Cross-asset correlations
        "btc_gold_corr",
        "btc_dxy_corr",
        "btc_yield_corr",
        # Composite indicators
        "risk_on_score",
        "macro_regime",
    ]


def compute_combined_features(
    aligned_df: pd.DataFrame,
    window: int = 30,
    sigma_floor: float = 0.001,
) -> pd.DataFrame:
    """
    Compute both Simons and Dalio features on aligned data.
    
    This is the main entry point for the full feature set that
    feeds into the neural network.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned DataFrame from align_to_crypto().
    window : int, default=30
        Rolling window for calculations.
    sigma_floor : float, default=0.001
        Volatility floor for Z-score normalization.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all features:
        - Raw OHLCV
        - Simons features (price-based)
        - Dalio features (macro-based)
    """
    from .features import compute_simons_features
    
    # First, compute Simons features on BTC OHLCV
    # Extract just BTC columns for Simons features
    btc_cols = ["Open", "High", "Low", "Close", "Volume"]
    btc_df = aligned_df[btc_cols].copy()
    
    simons_features = compute_simons_features(btc_df, window=window, sigma_floor=sigma_floor)
    
    # Now compute Dalio features on the full aligned data
    # Need to re-align because Simons drops warmup rows
    aligned_trimmed = aligned_df.loc[simons_features.index].copy()
    
    # Add Simons features
    for col in simons_features.columns:
        if col not in btc_cols:  # Don't duplicate OHLCV
            aligned_trimmed[col] = simons_features[col]
    
    # Compute Dalio features
    combined = compute_dalio_features(aligned_trimmed, window=window, sigma_floor=sigma_floor)
    
    return combined


def get_all_feature_columns() -> list[str]:
    """
    Return all feature column names (Simons + Dalio).
    
    Returns
    -------
    list[str]
        Complete list of feature columns for neural network input.
    """
    from .features import get_feature_columns
    
    simons_cols = get_feature_columns()
    dalio_cols = get_dalio_feature_columns()
    
    # Remove duplicates while preserving order
    all_cols = simons_cols.copy()
    for col in dalio_cols:
        if col not in all_cols:
            all_cols.append(col)
    
    return all_cols


if __name__ == "__main__":
    # Test with synthetic aligned data
    import numpy as np
    
    # Create synthetic aligned data
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    
    aligned_df = pd.DataFrame({
        # BTC OHLCV
        "Open": np.random.randn(100).cumsum() + 40000,
        "High": np.random.randn(100).cumsum() + 40100,
        "Low": np.random.randn(100).cumsum() + 39900,
        "Close": np.random.randn(100).cumsum() + 40000,
        "Volume": np.abs(np.random.randn(100)) * 1e9,
        # Treasury yields (around 4%)
        "TNX_Open": np.random.randn(100).cumsum() * 0.1 + 4,
        "TNX_High": np.random.randn(100).cumsum() * 0.1 + 4.1,
        "TNX_Low": np.random.randn(100).cumsum() * 0.1 + 3.9,
        "TNX_Close": np.random.randn(100).cumsum() * 0.1 + 4,
        "TNX_Volume": np.abs(np.random.randn(100)) * 1e6,
        # Dollar Index (around 105)
        "DXY_Open": np.random.randn(100).cumsum() * 0.5 + 105,
        "DXY_High": np.random.randn(100).cumsum() * 0.5 + 105.5,
        "DXY_Low": np.random.randn(100).cumsum() * 0.5 + 104.5,
        "DXY_Close": np.random.randn(100).cumsum() * 0.5 + 105,
        "DXY_Volume": np.abs(np.random.randn(100)) * 1e6,
        # Gold (around 200)
        "GLD_Open": np.random.randn(100).cumsum() * 2 + 200,
        "GLD_High": np.random.randn(100).cumsum() * 2 + 201,
        "GLD_Low": np.random.randn(100).cumsum() * 2 + 199,
        "GLD_Close": np.random.randn(100).cumsum() * 2 + 200,
        "GLD_Volume": np.abs(np.random.randn(100)) * 1e6,
    }, index=dates)
    aligned_df.index.name = "Date"
    
    print("Input shape:", aligned_df.shape)
    print()
    
    # Compute Dalio features
    print("Computing Dalio features...")
    features = compute_dalio_features(aligned_df)
    
    print()
    print("Output shape:", features.shape)
    print()
    print("Dalio features added:")
    for col in get_dalio_feature_columns():
        if col in features.columns:
            print(f"  ✓ {col}: [{features[col].min():.3f}, {features[col].max():.3f}]")
        else:
            print(f"  ✗ {col}: missing")
    
    print()
    print("Computing combined features...")
    combined = compute_combined_features(aligned_df)
    print(f"Combined shape: {combined.shape}")
    print(f"Total features: {len(get_all_feature_columns())}")

