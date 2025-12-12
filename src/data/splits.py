"""
Time Series Data Splits Module
==============================
Provides proper train/validation/test splits for time series data.

Critical concepts:
- NO random shuffling (time series must maintain order)
- Purging gaps between sets (prevent look-ahead bias)
- Walk-forward validation for robust evaluation

The gap (purge) is essential because rolling features use past data.
Without a gap, the test set's features would include training data.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Iterator
from pathlib import Path


@dataclass
class TimeSeriesSplit:
    """
    Holds train/validation/test splits with metadata.
    
    Attributes
    ----------
    train : pd.DataFrame
        Training data (oldest).
    val : pd.DataFrame
        Validation data (middle).
    test : pd.DataFrame
        Test data (newest) - only touch for final evaluation!
    train_end : pd.Timestamp
        Last date in training set.
    val_end : pd.Timestamp
        Last date in validation set.
    gap_days : int
        Number of days between sets (purge period).
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    gap_days: int
    
    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplit(\n"
            f"  train: {len(self.train)} rows ({self.train.index.min().date()} to {self.train.index.max().date()})\n"
            f"  val:   {len(self.val)} rows ({self.val.index.min().date()} to {self.val.index.max().date()})\n"
            f"  test:  {len(self.test)} rows ({self.test.index.min().date()} to {self.test.index.max().date()})\n"
            f"  gap_days: {self.gap_days}\n"
            f")"
        )


def create_time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    gap_days: int = 30,
) -> TimeSeriesSplit:
    """
    Split time series data with purging gaps between sets.
    
    The gap prevents data leakage when features use rolling windows.
    For example, with a 30-day window:
    - Training ends: Day 100
    - Gap: Days 101-130 (discarded)
    - Validation starts: Day 131
    
    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame with DatetimeIndex (must be sorted).
    train_ratio : float, default=0.6
        Fraction of data for training.
    val_ratio : float, default=0.2
        Fraction of data for validation (hyperparameter tuning).
    test_ratio : float, default=0.2
        Fraction of data for testing (final evaluation only!).
    gap_days : int, default=30
        Gap between sets (should >= rolling window size).
        
    Returns
    -------
    TimeSeriesSplit
        Contains train, val, test DataFrames and metadata.
        
    Raises
    ------
    ValueError
        If ratios don't sum to 1 or data is insufficient.
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Sort by date
    df = df.sort_index()
    
    n = len(df)
    
    # Calculate target sizes (before gaps)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    # test_size is the remainder
    
    # Calculate indices with gaps
    train_end_idx = train_size
    val_start_idx = train_end_idx + gap_days
    val_end_idx = val_start_idx + val_size
    test_start_idx = val_end_idx + gap_days
    
    # Verify sufficient data
    min_test_size = 50  # Minimum samples for meaningful test
    if test_start_idx + min_test_size > n:
        raise ValueError(
            f"Insufficient data for split with {gap_days}-day gaps.\n"
            f"Have {n} rows, need at least {test_start_idx + min_test_size}.\n"
            f"Consider: reducing gap_days, or using more data."
        )
    
    # Create splits
    train = df.iloc[:train_end_idx].copy()
    val = df.iloc[val_start_idx:val_end_idx].copy()
    test = df.iloc[test_start_idx:].copy()
    
    # Validate splits have data
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("One or more splits are empty. Check your data and ratios.")
    
    split = TimeSeriesSplit(
        train=train,
        val=val,
        test=test,
        train_end=train.index.max(),
        val_end=val.index.max(),
        gap_days=gap_days,
    )
    
    print(f"Time series split with {gap_days}-day purge gaps:")
    print(f"  Train: {train.index.min().date()} to {train.index.max().date()} ({len(train)} rows, {train_ratio*100:.0f}%)")
    print(f"  [GAP:  {gap_days} days]")
    print(f"  Val:   {val.index.min().date()} to {val.index.max().date()} ({len(val)} rows, {val_ratio*100:.0f}%)")
    print(f"  [GAP:  {gap_days} days]")
    print(f"  Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)")
    
    return split


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward validation."""
    fold_num: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_days: int = 365 * 2,
    test_days: int = 90,
    gap_days: int = 30,
    step_days: Optional[int] = None,
) -> list[WalkForwardFold]:
    """
    Generate walk-forward validation folds.
    
    Walk-forward is more robust than a single split because it tests
    on multiple time periods:
    
    Fold 1: [TRAIN========][gap][TEST]
    Fold 2: [TRAIN===============][gap][TEST]
    Fold 3: [TRAIN======================][gap][TEST]
    ...
    
    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame with DatetimeIndex.
    initial_train_days : int, default=730
        Size of initial training period (2 years).
    test_days : int, default=90
        Size of each test period (3 months).
    gap_days : int, default=30
        Gap between train and test.
    step_days : int, optional
        How much to advance between folds. Defaults to test_days.
        
    Returns
    -------
    list[WalkForwardFold]
        List of train/test folds.
    """
    if step_days is None:
        step_days = test_days
    
    df = df.sort_index()
    
    start_date = df.index.min()
    end_date = df.index.max()
    
    folds = []
    fold_num = 1
    
    # Initial training end
    train_end = start_date + pd.Timedelta(days=initial_train_days)
    
    while True:
        # Calculate test window
        test_start = train_end + pd.Timedelta(days=gap_days)
        test_end = test_start + pd.Timedelta(days=test_days)
        
        # Stop if we've run out of data
        if test_end > end_date:
            break
        
        # Extract data
        train = df.loc[:train_end].copy()
        test = df.loc[test_start:test_end].copy()
        
        # Skip if insufficient test data
        if len(test) < 10:
            break
        
        folds.append(WalkForwardFold(
            fold_num=fold_num,
            train=train,
            test=test,
            train_end=train_end,
            test_start=test_start,
            test_end=test.index.max(),
        ))
        
        # Roll forward
        train_end = test_end
        fold_num += 1
    
    if folds:
        print(f"Generated {len(folds)} walk-forward folds:")
        for fold in folds:
            print(f"  Fold {fold.fold_num}: Train {len(fold.train)} rows → Test {len(fold.test)} rows "
                  f"({fold.test_start.date()} to {fold.test_end.date()})")
    else:
        print("Warning: Could not generate any walk-forward folds. Need more data.")
    
    return folds


def verify_no_leakage(split: TimeSeriesSplit) -> bool:
    """
    Verify there's no date overlap between train/val/test.
    
    Parameters
    ----------
    split : TimeSeriesSplit
        The split to verify.
        
    Returns
    -------
    bool
        True if no leakage detected.
        
    Raises
    ------
    ValueError
        If data leakage is detected.
    """
    train_dates = set(split.train.index)
    val_dates = set(split.val.index)
    test_dates = set(split.test.index)
    
    # Check overlaps
    train_val_overlap = train_dates & val_dates
    train_test_overlap = train_dates & test_dates
    val_test_overlap = val_dates & test_dates
    
    if train_val_overlap:
        raise ValueError(f"Data leakage: {len(train_val_overlap)} dates in both train and val")
    if train_test_overlap:
        raise ValueError(f"Data leakage: {len(train_test_overlap)} dates in both train and test")
    if val_test_overlap:
        raise ValueError(f"Data leakage: {len(val_test_overlap)} dates in both val and test")
    
    # Check temporal ordering
    if split.train.index.max() >= split.val.index.min():
        raise ValueError("Train data extends into validation period")
    if split.val.index.max() >= split.test.index.min():
        raise ValueError("Validation data extends into test period")
    
    # Check gap sizes
    train_val_gap = (split.val.index.min() - split.train.index.max()).days
    val_test_gap = (split.test.index.min() - split.val.index.max()).days
    
    print(f"✓ No data leakage detected")
    print(f"  Train→Val gap: {train_val_gap} days")
    print(f"  Val→Test gap:  {val_test_gap} days")
    
    return True


class FeatureScaler:
    """
    Fit scaler on training data, apply to all sets.
    
    CRITICAL: Never fit on validation or test data!
    This prevents information leakage from future data.
    """
    
    def __init__(self, method: str = "standard"):
        """
        Parameters
        ----------
        method : str, default="standard"
            Scaling method: "standard" (z-score) or "minmax".
        """
        self.method = method
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        self.feature_cols_ = None
        self._is_fitted = False
    
    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "FeatureScaler":
        """
        Fit scaler on training data only.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data.
        feature_cols : list[str]
            Columns to scale.
        """
        self.feature_cols_ = feature_cols
        X = train_df[feature_cols].values
        
        if self.method == "standard":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            # Prevent division by zero
            self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        elif self.method == "minmax":
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            # Prevent division by zero
            range_ = self.max_ - self.min_
            range_ = np.where(range_ < 1e-8, 1.0, range_)
            self.max_ = self.min_ + range_
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        result = df.copy()
        X = df[self.feature_cols_].values
        
        if self.method == "standard":
            X_scaled = (X - self.mean_) / self.std_
        elif self.method == "minmax":
            X_scaled = (X - self.min_) / (self.max_ - self.min_)
        
        result[self.feature_cols_] = X_scaled
        return result
    
    def fit_transform(self, train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Fit on train and transform it."""
        return self.fit(train_df, feature_cols).transform(train_df)


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    
    from data.manager import DataManager
    
    print("Testing time series splits...")
    print()
    
    dm = DataManager("./data")
    features = dm.get_features("BTC-USD", "1d", feature_set="combined", start="2020-01-01")
    
    print(f"Total data: {len(features)} rows")
    print()
    
    # Test standard split
    print("=" * 60)
    print("TEST 1: Standard Train/Val/Test Split")
    print("=" * 60)
    
    split = create_time_series_split(features, gap_days=30)
    print()
    verify_no_leakage(split)
    
    print()
    print("=" * 60)
    print("TEST 2: Walk-Forward Validation")
    print("=" * 60)
    
    folds = walk_forward_splits(features, initial_train_days=365*2, test_days=90)
    
    print()
    print("✓ All tests passed!")

