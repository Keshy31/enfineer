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
>>> # Computed features (cached by params hash)
>>> features = dm.get_features("BTC-USD", "1d", window=30)
"""

from .fetcher import fetch_bitcoin_data, fetch_from_yfinance
from .features import compute_simons_features, apply_volatility_floor_zscore
from .manager import DataManager
from .storage import (
    save_ohlcv, load_ohlcv, parquet_exists,
    save_features, load_features, compute_params_hash,
)
from .metadata import MetadataDB, CoverageInfo, FeatureCacheInfo

__all__ = [
    # High-level interface
    "DataManager",
    # Fetching
    "fetch_bitcoin_data",
    "fetch_from_yfinance",
    # Features
    "compute_simons_features", 
    "apply_volatility_floor_zscore",
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
]

