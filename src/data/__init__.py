# Data Ingestion Module
"""
Handles data fetching, caching, and feature engineering.

The main interface is DataManager, which provides smart caching
of OHLCV data in Parquet format with SQLite metadata tracking.
"""

from .fetcher import fetch_bitcoin_data, fetch_from_yfinance
from .features import compute_simons_features, apply_volatility_floor_zscore
from .manager import DataManager
from .storage import save_ohlcv, load_ohlcv, parquet_exists
from .metadata import MetadataDB, CoverageInfo

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
    # Metadata
    "MetadataDB",
    "CoverageInfo",
]

