# Data Ingestion Module
"""
Handles data fetching, caching, and feature engineering.

The main interface is DataManager, which provides smart caching
of OHLCV data and computed features in Parquet format with 
SQLite metadata tracking.

Example
-------
>>> from src.data import DataManager
>>> dm = DataManager("./data")
>>> 
>>> # OHLCV data (cached)
>>> df = dm.get_ohlcv("BTC-USD", "1d")
>>> 
>>> # Simons features - price-based (cached by params hash)
>>> features = dm.get_features("BTC-USD", "1d", window=30)
>>> 
>>> # Combined features - price + macro (cached)
>>> features = dm.get_features("BTC-USD", "1d", feature_set="combined")
>>> 
>>> # Aligned data - crypto + macro (for custom analysis)
>>> aligned = dm.get_aligned_data("BTC-USD", start="2020-01-01")
"""

from .fetcher import fetch_bitcoin_data, fetch_from_yfinance
from .features import compute_simons_features, apply_volatility_floor_zscore, get_feature_columns
from .dalio_features import (
    compute_dalio_features,
    compute_combined_features,
    get_dalio_feature_columns,
    get_all_feature_columns,
)
from .alignment import (
    align_to_crypto,
    find_common_date_range,
    trim_to_common_range,
    compute_alignment_quality,
)
from .manager import DataManager, DEFAULT_MACRO_SYMBOLS
from .storage import (
    save_ohlcv, load_ohlcv, parquet_exists,
    save_features, load_features, compute_params_hash,
)
from .metadata import MetadataDB, CoverageInfo, FeatureCacheInfo
from .splits import (
    TimeSeriesSplit,
    create_time_series_split,
    walk_forward_splits,
    verify_no_leakage,
    WalkForwardFold,
    FeatureScaler,
)
from .training import (
    TrainingDataset,
    create_training_dataset,
    create_sequences,
    split_dataset,
    normalize_dataset,
    get_temporal_feature_cols,
    get_macro_feature_cols,
)

__all__ = [
    # High-level interface
    "DataManager",
    "DEFAULT_MACRO_SYMBOLS",
    # Fetching
    "fetch_bitcoin_data",
    "fetch_from_yfinance",
    # Simons Features (price-based)
    "compute_simons_features", 
    "apply_volatility_floor_zscore",
    "get_feature_columns",
    # Dalio Features (macro-based)
    "compute_dalio_features",
    "compute_combined_features",
    "get_dalio_feature_columns",
    "get_all_feature_columns",
    # Alignment
    "align_to_crypto",
    "find_common_date_range",
    "trim_to_common_range",
    "compute_alignment_quality",
    # Storage (low-level)
    "save_ohlcv",
    "load_ohlcv",
    "parquet_exists",
    "save_features",
    "load_features",
    "compute_params_hash",
    # Metadata
    "MetadataDB",
    "CoverageInfo",
    "FeatureCacheInfo",
    # Time Series Splits
    "TimeSeriesSplit",
    "create_time_series_split",
    "walk_forward_splits",
    "verify_no_leakage",
    "WalkForwardFold",
    "FeatureScaler",
    # Training Dataset
    "TrainingDataset",
    "create_training_dataset",
    "create_sequences",
    "split_dataset",
    "normalize_dataset",
    "get_temporal_feature_cols",
    "get_macro_feature_cols",
]
