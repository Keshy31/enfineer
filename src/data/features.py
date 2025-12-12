"""
Feature Engineering Module
==========================
Implements the "Simons Features" from PROJ.md Section 3:
- Log returns for stationarity
- Rolling Z-score normalization  
- Volatility floor fix for quiet market periods
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Convert price series to log returns.
    
    Log returns are preferred over simple returns because:
    - They are time-additive (useful for multi-period analysis)
    - They are approximately normally distributed
    - They ensure stationarity in price data
    
    Parameters
    ----------
    prices : pd.Series
        Price series (e.g., Close prices).
        
    Returns
    -------
    pd.Series
        Log returns: ln(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1))


def apply_volatility_floor_zscore(
    series: pd.Series,
    window: int = 30,
    sigma_floor: float = 0.001,
) -> pd.Series:
    """
    Apply Z-score normalization with volatility floor.
    
    Implements the "Quiet Market Trap Fix" from PROJ.md Section 3.1:
    
        Z_adjusted = (x - μ) / max(σ, σ_floor)
    
    This prevents signal explosions when volatility approaches zero
    during unusually quiet market periods.
    
    Parameters
    ----------
    series : pd.Series
        Input time series to normalize.
    window : int, default=30
        Rolling window size for computing mean and std.
    sigma_floor : float, default=0.001
        Minimum volatility threshold. Prevents division by near-zero values.
        
    Returns
    -------
    pd.Series
        Z-scored series with volatility floor applied.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Apply the volatility floor - this is the key innovation
    adjusted_std = rolling_std.clip(lower=sigma_floor)
    
    z_score = (series - rolling_mean) / adjusted_std
    
    return z_score


def compute_simons_features(
    df: pd.DataFrame,
    window: int = 30,
    sigma_floor: float = 0.001,
) -> pd.DataFrame:
    """
    Compute the full "Simons Features" set for price-based time series analysis.
    
    These features are designed to capture:
    - Momentum (via log returns)
    - Volatility clustering (via rolling measures)
    - Mean reversion signals (via Z-scores)
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: Open, High, Low, Close, Volume.
    window : int, default=30
        Rolling window for all calculations (30 days per PROJ.md).
    sigma_floor : float, default=0.001
        Volatility floor for Z-score normalization.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original data plus engineered features:
        - log_return: Log return of Close
        - log_return_zscore: Z-scored log returns
        - volatility: Rolling standard deviation of log returns
        - volatility_zscore: Z-scored volatility
        - volume_zscore: Z-scored volume
        - high_low_range: (High - Low) / Close (intraday volatility proxy)
        - range_zscore: Z-scored high-low range
        - momentum: Rolling sum of log returns (trend indicator)
        - momentum_zscore: Z-scored momentum
    """
    features = df.copy()
    
    # =========================================
    # CORE: Log Returns & Z-Score
    # =========================================
    features["log_return"] = compute_log_returns(df["Close"])
    features["log_return_zscore"] = apply_volatility_floor_zscore(
        features["log_return"], window=window, sigma_floor=sigma_floor
    )
    
    # =========================================
    # VOLATILITY: Rolling StdDev of Returns
    # =========================================
    features["volatility"] = features["log_return"].rolling(
        window=window, min_periods=1
    ).std()
    features["volatility_zscore"] = apply_volatility_floor_zscore(
        features["volatility"], window=window, sigma_floor=sigma_floor / 10
    )
    
    # =========================================
    # VOLUME: Z-scored trading volume
    # =========================================
    features["volume_zscore"] = apply_volatility_floor_zscore(
        df["Volume"], window=window, sigma_floor=df["Volume"].mean() * 0.01
    )
    
    # =========================================
    # INTRADAY RANGE: High-Low as % of Close
    # =========================================
    features["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    features["range_zscore"] = apply_volatility_floor_zscore(
        features["high_low_range"], window=window, sigma_floor=sigma_floor
    )
    
    # =========================================
    # MOMENTUM: Rolling sum of returns (trend)
    # =========================================
    features["momentum"] = features["log_return"].rolling(
        window=window, min_periods=1
    ).sum()
    features["momentum_zscore"] = apply_volatility_floor_zscore(
        features["momentum"], window=window, sigma_floor=sigma_floor * 10
    )
    
    # =========================================
    # Clean up: Drop rows with NaN from rolling
    # =========================================
    # Keep track of how many rows we're dropping
    original_len = len(features)
    features = features.dropna()
    dropped = original_len - len(features)
    
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN (warmup period)")
    
    return features


def get_feature_columns() -> list[str]:
    """
    Return the list of engineered feature column names.
    
    Useful for selecting only the model input features from the full DataFrame.
    
    Returns
    -------
    list[str]
        List of feature column names (excludes raw OHLCV).
    """
    return [
        "log_return",
        "log_return_zscore",
        "volatility",
        "volatility_zscore",
        "volume_zscore",
        "high_low_range",
        "range_zscore",
        "momentum",
        "momentum_zscore",
    ]


def create_sequences(
    features: pd.DataFrame,
    sequence_length: int = 30,
    feature_cols: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Create sliding window sequences for LSTM input.
    
    This prepares data for the temporal branch of the neural network
    (to be implemented in future phases).
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame from compute_simons_features().
    sequence_length : int, default=30
        Length of each sequence (30 days per PROJ.md).
    feature_cols : list[str], optional
        Which columns to include. If None, uses get_feature_columns().
        
    Returns
    -------
    np.ndarray
        3D array of shape (n_samples, sequence_length, n_features).
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()
    
    data = features[feature_cols].values
    n_samples = len(data) - sequence_length + 1
    n_features = len(feature_cols)
    
    sequences = np.zeros((n_samples, sequence_length, n_features))
    
    for i in range(n_samples):
        sequences[i] = data[i : i + sequence_length]
    
    return sequences


if __name__ == "__main__":
    # Quick test with sample data
    from fetcher import fetch_bitcoin_data
    
    df = fetch_bitcoin_data()
    features = compute_simons_features(df)
    
    print("\nFeature Statistics:")
    print(features[get_feature_columns()].describe().round(3))
    
    # Verify no explosions in Z-scores
    zscore_cols = [c for c in features.columns if "zscore" in c]
    print("\nZ-Score Ranges (should be bounded):")
    for col in zscore_cols:
        print(f"  {col}: [{features[col].min():.2f}, {features[col].max():.2f}]")

