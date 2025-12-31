"""
Training Dataset Module
=======================
Prepares data for the neural network:
- Creates sliding window sequences for LSTM (temporal features)
- Extracts static features for the macro branch
- Handles proper train/val/test separation

The neural network expects:
- X_temporal: [N, sequence_length, n_simons_features] - price patterns over time
- X_macro: [N, n_dalio_features] - economic context (static per sample)
- y: [N,] - reconstruction targets (or labels for supervised tasks)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

from .features import get_feature_columns as get_simons_cols
from .dalio_features import get_dalio_feature_columns as get_dalio_cols


@dataclass
class TrainingDataset:
    """
    Holds prepared training data for the neural network.
    
    Attributes
    ----------
    X_temporal : np.ndarray
        Temporal features for LSTM. Shape: [N, seq_len, n_features]
    X_macro : np.ndarray
        Static macro features. Shape: [N, n_macro_features]
    y : np.ndarray
        Target values (for reconstruction or prediction). Shape: [N,] or [N, n_targets]
    dates : pd.DatetimeIndex
        Dates corresponding to each sample (last date of each sequence).
    feature_names_temporal : list[str]
        Names of temporal features.
    feature_names_macro : list[str]
        Names of macro features.
    """
    X_temporal: np.ndarray
    X_macro: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    feature_names_temporal: list[str]
    feature_names_macro: list[str]
    
    def __repr__(self) -> str:
        return (
            f"TrainingDataset(\n"
            f"  X_temporal: {self.X_temporal.shape} (samples, seq_len, features)\n"
            f"  X_macro:    {self.X_macro.shape} (samples, features)\n"
            f"  y:          {self.y.shape}\n"
            f"  date_range: {self.dates.min().date()} to {self.dates.max().date()}\n"
            f")"
        )
    
    def __len__(self) -> int:
        return len(self.X_temporal)


def get_temporal_feature_cols() -> list[str]:
    """
    Get feature columns for the temporal (LSTM) branch.
    
    These are Simons-style features that capture price dynamics:
    - Log returns and their Z-scores
    - Volatility measures
    - Momentum indicators
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


def get_macro_feature_cols() -> list[str]:
    """
    Get feature columns for the macro (Dense) branch.
    
    These are Dalio-style features that capture economic context:
    - Interest rate environment
    - Dollar strength
    - Gold (risk-off proxy)
    - Cross-asset correlations
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


def create_sequences(
    df: pd.DataFrame,
    temporal_cols: list[str],
    sequence_length: int = 30,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Create sliding window sequences for LSTM input.
    
    For each time step t, creates a sequence from [t-seq_len+1, t].
    The label/output corresponds to time t (the last element of the sequence).
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with DatetimeIndex.
    temporal_cols : list[str]
        Which columns to include in sequences.
    sequence_length : int, default=30
        Length of each sequence (30 days per PROJ.md).
        
    Returns
    -------
    X : np.ndarray
        3D array of shape (n_samples, sequence_length, n_features).
    dates : pd.DatetimeIndex
        Dates corresponding to each sample (last date of sequence).
    """
    # Validate columns exist
    missing = [c for c in temporal_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")
    
    data = df[temporal_cols].values
    n_rows, n_features = data.shape
    
    if n_rows < sequence_length:
        raise ValueError(
            f"Insufficient data: have {n_rows} rows, need at least {sequence_length} for sequences"
        )
    
    n_samples = n_rows - sequence_length + 1
    
    # Pre-allocate array
    X = np.zeros((n_samples, sequence_length, n_features), dtype=np.float32)
    
    # Create sequences using sliding window
    for i in range(n_samples):
        X[i] = data[i : i + sequence_length]
    
    # Dates correspond to the last element of each sequence
    dates = df.index[sequence_length - 1:]
    
    return X, dates


def create_training_dataset(
    features_df: pd.DataFrame,
    sequence_length: int = 30,
    target_col: str = "Close",
    temporal_cols: Optional[list[str]] = None,
    macro_cols: Optional[list[str]] = None,
) -> TrainingDataset:
    """
    Create a complete training dataset for the neural network.
    
    This is the main function for preparing data. It:
    1. Creates sliding window sequences for the temporal (LSTM) branch
    2. Extracts static features for the macro (Dense) branch
    3. Aligns everything to the same dates
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Combined features from DataManager.get_features(feature_set="combined").
    sequence_length : int, default=30
        Length of temporal sequences.
    target_col : str, default="Close"
        Column to use as target (for reconstruction loss).
    temporal_cols : list[str], optional
        Columns for temporal branch. Defaults to Simons features.
    macro_cols : list[str], optional
        Columns for macro branch. Defaults to Dalio features.
        
    Returns
    -------
    TrainingDataset
        Contains X_temporal, X_macro, y, and metadata.
        
    Example
    -------
    >>> dm = DataManager("./data")
    >>> features = dm.get_features("BTC-USD", "1d", feature_set="combined")
    >>> dataset = create_training_dataset(features, sequence_length=30)
    >>> print(dataset)
    TrainingDataset(
      X_temporal: (1000, 30, 9)
      X_macro:    (1000, 21)
      y:          (1000,)
    )
    """
    if temporal_cols is None:
        temporal_cols = get_temporal_feature_cols()
    if macro_cols is None:
        macro_cols = get_macro_feature_cols()
    
    # Filter to available columns
    temporal_cols = [c for c in temporal_cols if c in features_df.columns]
    macro_cols = [c for c in macro_cols if c in features_df.columns]
    
    if len(temporal_cols) == 0:
        raise ValueError("No temporal columns found in DataFrame")
    if len(macro_cols) == 0:
        raise ValueError("No macro columns found in DataFrame")
    
    # Sort by date
    df = features_df.sort_index()
    
    # Create temporal sequences
    X_temporal, dates = create_sequences(df, temporal_cols, sequence_length)
    
    # Extract macro features for the same dates
    # For each sequence, we use the macro state at the END of the sequence
    macro_df = df.loc[dates, macro_cols]
    X_macro = macro_df.values.astype(np.float32)
    
    # Extract target
    if target_col in df.columns:
        y = df.loc[dates, target_col].values.astype(np.float32)
    else:
        # Default to zeros if target not available (unsupervised)
        y = np.zeros(len(dates), dtype=np.float32)
    
    # Check for NaN/Inf
    if np.isnan(X_temporal).any():
        nan_count = np.isnan(X_temporal).sum()
        raise ValueError(f"Found {nan_count} NaN values in temporal data")
    if np.isnan(X_macro).any():
        nan_count = np.isnan(X_macro).sum()
        raise ValueError(f"Found {nan_count} NaN values in macro data")
    if np.isinf(X_temporal).any():
        inf_count = np.isinf(X_temporal).sum()
        raise ValueError(f"Found {inf_count} Inf values in temporal data")
    if np.isinf(X_macro).any():
        inf_count = np.isinf(X_macro).sum()
        raise ValueError(f"Found {inf_count} Inf values in macro data")
    
    return TrainingDataset(
        X_temporal=X_temporal,
        X_macro=X_macro,
        y=y,
        dates=dates,
        feature_names_temporal=temporal_cols,
        feature_names_macro=macro_cols,
    )


def split_dataset(
    dataset: TrainingDataset,
    train_end_date: pd.Timestamp,
    val_end_date: pd.Timestamp,
) -> Tuple[TrainingDataset, TrainingDataset, TrainingDataset]:
    """
    Split a TrainingDataset into train/val/test by date.
    
    Parameters
    ----------
    dataset : TrainingDataset
        Full dataset.
    train_end_date : pd.Timestamp
        Last date for training.
    val_end_date : pd.Timestamp
        Last date for validation.
        
    Returns
    -------
    train, val, test : TrainingDataset
        Split datasets.
    """
    # Create masks
    train_mask = dataset.dates <= train_end_date
    val_mask = (dataset.dates > train_end_date) & (dataset.dates <= val_end_date)
    test_mask = dataset.dates > val_end_date
    
    def subset(mask):
        return TrainingDataset(
            X_temporal=dataset.X_temporal[mask],
            X_macro=dataset.X_macro[mask],
            y=dataset.y[mask],
            dates=dataset.dates[mask],
            feature_names_temporal=dataset.feature_names_temporal,
            feature_names_macro=dataset.feature_names_macro,
        )
    
    return subset(train_mask), subset(val_mask), subset(test_mask)


def normalize_dataset(
    train: TrainingDataset,
    val: TrainingDataset,
    test: TrainingDataset,
) -> Tuple[TrainingDataset, TrainingDataset, TrainingDataset, dict]:
    """
    Normalize datasets using statistics from training set only.
    
    CRITICAL: Always fit normalization on train, apply to val/test.
    
    Parameters
    ----------
    train, val, test : TrainingDataset
        Datasets to normalize.
        
    Returns
    -------
    train_norm, val_norm, test_norm : TrainingDataset
        Normalized datasets.
    stats : dict
        Normalization statistics (for inference).
    """
    # Compute statistics from training data
    # Temporal: mean/std per feature across all timesteps
    temporal_mean = train.X_temporal.mean(axis=(0, 1))
    temporal_std = train.X_temporal.std(axis=(0, 1))
    temporal_std = np.where(temporal_std < 1e-8, 1.0, temporal_std)
    
    # Macro: mean/std per feature
    macro_mean = train.X_macro.mean(axis=0)
    macro_std = train.X_macro.std(axis=0)
    macro_std = np.where(macro_std < 1e-8, 1.0, macro_std)
    
    # Target: mean/std
    y_mean = train.y.mean()
    y_std = train.y.std()
    y_std = y_std if y_std > 1e-8 else 1.0
    
    def normalize_ds(ds: TrainingDataset) -> TrainingDataset:
        return TrainingDataset(
            X_temporal=(ds.X_temporal - temporal_mean) / temporal_std,
            X_macro=(ds.X_macro - macro_mean) / macro_std,
            y=(ds.y - y_mean) / y_std,
            dates=ds.dates,
            feature_names_temporal=ds.feature_names_temporal,
            feature_names_macro=ds.feature_names_macro,
        )
    
    stats = {
        'temporal_mean': temporal_mean,
        'temporal_std': temporal_std,
        'macro_mean': macro_mean,
        'macro_std': macro_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }
    
    return normalize_ds(train), normalize_ds(val), normalize_ds(test), stats


if __name__ == "__main__":
    # Test the training dataset creation
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.manager import DataManager
    from data.splits import create_time_series_split
    
    print("=" * 60)
    print("TRAINING DATASET TEST")
    print("=" * 60)
    print()
    
    # Load features
    print("[1/4] Loading combined features...")
    dm = DataManager("./data")
    features = dm.get_features("BTC-USD", "1d", feature_set="combined", start="2020-01-01")
    print(f"  ✓ Loaded {len(features)} rows with {len(features.columns)} columns")
    
    # Create dataset
    print("\n[2/4] Creating training dataset...")
    dataset = create_training_dataset(features, sequence_length=30)
    print(f"  ✓ {dataset}")
    
    # Split by date
    print("\n[3/4] Splitting train/val/test...")
    split = create_time_series_split(features, gap_days=30)
    
    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        train_end_date=split.train_end,
        val_end_date=split.val_end,
    )
    
    print(f"\n  Train: {train_ds}")
    print(f"  Val:   {val_ds}")
    print(f"  Test:  {test_ds}")
    
    # Normalize
    print("\n[4/4] Normalizing (fit on train only)...")
    train_norm, val_norm, test_norm, stats = normalize_dataset(train_ds, val_ds, test_ds)
    
    print(f"  ✓ Temporal mean shape: {stats['temporal_mean'].shape}")
    print(f"  ✓ Macro mean shape: {stats['macro_mean'].shape}")
    
    # Verify normalization
    print(f"\n  Train temporal mean: {train_norm.X_temporal.mean():.4f} (should be ~0)")
    print(f"  Train temporal std:  {train_norm.X_temporal.std():.4f} (should be ~1)")
    print(f"  Train macro mean:    {train_norm.X_macro.mean():.4f} (should be ~0)")
    print(f"  Train macro std:     {train_norm.X_macro.std():.4f} (should be ~1)")
    
    print("\n" + "=" * 60)
    print("✓ TRAINING DATASET TEST PASSED")
    print("=" * 60)
    print()
    print("Data is ready for neural network training!")
    print(f"  • Temporal input:  {train_norm.X_temporal.shape}")
    print(f"  • Macro input:     {train_norm.X_macro.shape}")
    print(f"  • Reconstruction:  {train_norm.y.shape}")

