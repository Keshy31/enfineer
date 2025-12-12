"""
Storage Module
==============
High-performance Parquet storage for OHLCV financial data.

This module provides the storage layer for the data caching system,
using Apache Parquet for columnar storage with excellent compression
and fast read performance.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
from datetime import datetime


def get_parquet_path(symbol: str, timeframe: str, data_dir: Path) -> Path:
    """
    Get the standardized path for a symbol/timeframe Parquet file.
    
    Convention: data/market/{symbol}/{timeframe}.parquet
    
    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., 'BTC-USD').
    timeframe : str
        Data timeframe (e.g., '1d', '1h', '15m').
    data_dir : Path
        Root data directory.
        
    Returns
    -------
    Path
        Full path to the Parquet file.
    """
    # Sanitize symbol for filesystem (replace special chars)
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")
    return data_dir / "market" / safe_symbol / f"{timeframe}.parquet"


def parquet_exists(symbol: str, timeframe: str, data_dir: Path) -> bool:
    """
    Check if a Parquet file exists for the given symbol/timeframe.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol.
    timeframe : str
        Data timeframe.
    data_dir : Path
        Root data directory.
        
    Returns
    -------
    bool
        True if the Parquet file exists.
    """
    path = get_parquet_path(symbol, timeframe, data_dir)
    return path.exists()


def save_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    data_dir: Path,
) -> Path:
    """
    Save OHLCV DataFrame to Parquet format.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    symbol : str
        Ticker symbol.
    timeframe : str
        Data timeframe.
    data_dir : Path
        Root data directory.
        
    Returns
    -------
    Path
        Path to the saved Parquet file.
        
    Notes
    -----
    - Creates parent directories if they don't exist
    - Uses snappy compression (good balance of speed/size)
    - Preserves DatetimeIndex as a column for filtering
    """
    path = get_parquet_path(symbol, timeframe, data_dir)
    
    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare DataFrame for storage
    df_to_save = df.copy()
    
    # Ensure index is named and reset for storage
    if df_to_save.index.name is None:
        df_to_save.index.name = "Date"
    
    # Reset index to store Date as a column (better for filtering)
    df_to_save = df_to_save.reset_index()
    
    # Convert to PyArrow Table and write
    table = pa.Table.from_pandas(df_to_save, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="snappy",
        # Enable statistics for predicate pushdown
        write_statistics=True,
    )
    
    return path


def load_ohlcv(
    symbol: str,
    timeframe: str,
    data_dir: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from Parquet with optional date filtering.
    
    Uses predicate pushdown for efficient date range queries -
    only reads the relevant row groups from disk.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol.
    timeframe : str
        Data timeframe.
    data_dir : Path
        Root data directory.
    start : str, optional
        Start date filter (inclusive) in 'YYYY-MM-DD' format.
    end : str, optional
        End date filter (inclusive) in 'YYYY-MM-DD' format.
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with DatetimeIndex.
        
    Raises
    ------
    FileNotFoundError
        If the Parquet file doesn't exist.
    """
    path = get_parquet_path(symbol, timeframe, data_dir)
    
    if not path.exists():
        raise FileNotFoundError(
            f"No data found for {symbol}/{timeframe}. "
            f"Expected file: {path}"
        )
    
    # Build filter for predicate pushdown
    filters = []
    if start is not None:
        start_dt = pd.Timestamp(start)
        filters.append(("Date", ">=", start_dt))
    if end is not None:
        end_dt = pd.Timestamp(end)
        filters.append(("Date", "<=", end_dt))
    
    # Read with filters (predicate pushdown)
    if filters:
        table = pq.read_table(path, filters=filters)
    else:
        table = pq.read_table(path)
    
    # Convert to pandas
    df = table.to_pandas()
    
    # Restore DatetimeIndex
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    
    # Sort by date
    df = df.sort_index()
    
    return df


def append_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    data_dir: Path,
) -> Path:
    """
    Append new OHLCV data to existing Parquet file.
    
    Handles deduplication by Date index - if dates overlap,
    the new data takes precedence.
    
    Parameters
    ----------
    df : pd.DataFrame
        New OHLCV data to append.
    symbol : str
        Ticker symbol.
    timeframe : str
        Data timeframe.
    data_dir : Path
        Root data directory.
        
    Returns
    -------
    Path
        Path to the updated Parquet file.
    """
    path = get_parquet_path(symbol, timeframe, data_dir)
    
    if not path.exists():
        # No existing file, just save
        return save_ohlcv(df, symbol, timeframe, data_dir)
    
    # Load existing data
    existing_df = load_ohlcv(symbol, timeframe, data_dir)
    
    # Combine: new data takes precedence on overlapping dates
    combined_df = pd.concat([existing_df, df])
    
    # Remove duplicates, keeping the last (newer) entry
    combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
    
    # Sort by date
    combined_df = combined_df.sort_index()
    
    # Save back
    return save_ohlcv(combined_df, symbol, timeframe, data_dir)


def get_parquet_info(symbol: str, timeframe: str, data_dir: Path) -> dict:
    """
    Get metadata about a Parquet file.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol.
    timeframe : str
        Data timeframe.
    data_dir : Path
        Root data directory.
        
    Returns
    -------
    dict
        File metadata including row count, date range, and file size.
    """
    path = get_parquet_path(symbol, timeframe, data_dir)
    
    if not path.exists():
        return {"exists": False}
    
    # Read metadata without loading full data
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    
    # Get date range from statistics (fast, no full read)
    df = load_ohlcv(symbol, timeframe, data_dir)
    
    return {
        "exists": True,
        "path": str(path),
        "row_count": metadata.num_rows,
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "start_date": df.index.min().strftime("%Y-%m-%d") if len(df) > 0 else None,
        "end_date": df.index.max().strftime("%Y-%m-%d") if len(df) > 0 else None,
        "columns": list(df.columns),
    }


# =========================================
# Feature Storage Functions
# =========================================

def get_feature_path(
    symbol: str,
    timeframe: str,
    feature_set: str,
    params_hash: str,
    data_dir: Path,
) -> Path:
    """
    Get the standardized path for a cached feature file.
    
    Convention: data/features/{symbol}/{feature_set}_{timeframe}_{hash}.parquet
    
    The params_hash ensures different parameter combinations get different files.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol.
    timeframe : str
        Data timeframe.
    feature_set : str
        Name of feature set (e.g., 'simons').
    params_hash : str
        Hash of computation parameters (first 8 chars of MD5).
    data_dir : Path
        Root data directory.
        
    Returns
    -------
    Path
        Full path to the feature Parquet file.
    """
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")
    filename = f"{feature_set}_{timeframe}_{params_hash}.parquet"
    return data_dir / "features" / safe_symbol / filename


def feature_cache_exists(
    symbol: str,
    timeframe: str,
    feature_set: str,
    params_hash: str,
    data_dir: Path,
) -> bool:
    """Check if a cached feature file exists."""
    path = get_feature_path(symbol, timeframe, feature_set, params_hash, data_dir)
    return path.exists()


def save_features(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    feature_set: str,
    params_hash: str,
    data_dir: Path,
) -> Path:
    """
    Save computed features to Parquet.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with DatetimeIndex.
    symbol : str
        Ticker symbol.
    timeframe : str
        Data timeframe.
    feature_set : str
        Name of feature set (e.g., 'simons').
    params_hash : str
        Hash of computation parameters.
    data_dir : Path
        Root data directory.
        
    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    path = get_feature_path(symbol, timeframe, feature_set, params_hash, data_dir)
    
    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare DataFrame for storage
    df_to_save = df.copy()
    
    if df_to_save.index.name is None:
        df_to_save.index.name = "Date"
    
    df_to_save = df_to_save.reset_index()
    
    # Write to Parquet
    table = pa.Table.from_pandas(df_to_save, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="snappy",
        write_statistics=True,
    )
    
    return path


def load_features(
    symbol: str,
    timeframe: str,
    feature_set: str,
    params_hash: str,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Load cached features from Parquet.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol.
    timeframe : str
        Data timeframe.
    feature_set : str
        Name of feature set.
    params_hash : str
        Hash of computation parameters.
    data_dir : Path
        Root data directory.
        
    Returns
    -------
    pd.DataFrame
        Feature DataFrame with DatetimeIndex.
        
    Raises
    ------
    FileNotFoundError
        If the feature cache doesn't exist.
    """
    path = get_feature_path(symbol, timeframe, feature_set, params_hash, data_dir)
    
    if not path.exists():
        raise FileNotFoundError(
            f"No cached features for {symbol}/{timeframe}/{feature_set} "
            f"with params hash {params_hash}"
        )
    
    table = pq.read_table(path)
    df = table.to_pandas()
    
    # Restore DatetimeIndex
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    
    df = df.sort_index()
    
    return df


def compute_params_hash(params: dict) -> str:
    """
    Compute a hash of feature computation parameters.
    
    This is the magic that makes caching work - same params always
    produce the same hash, different params produce different hashes.
    
    Parameters
    ----------
    params : dict
        Dictionary of computation parameters.
        
    Returns
    -------
    str
        First 8 characters of MD5 hash.
        
    Example
    -------
    >>> compute_params_hash({"window": 30, "sigma_floor": 0.001})
    'a1b2c3d4'
    """
    import hashlib
    import json
    
    # Sort keys for consistent hashing
    params_str = json.dumps(params, sort_keys=True)
    full_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    # Return first 8 chars (enough for uniqueness, readable)
    return full_hash[:8]


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    # Create sample data
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Open": np.random.randn(100).cumsum() + 100,
        "High": np.random.randn(100).cumsum() + 101,
        "Low": np.random.randn(100).cumsum() + 99,
        "Close": np.random.randn(100).cumsum() + 100,
        "Volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    df.index.name = "Date"
    
    # Test save/load
    data_dir = Path("./data")
    path = save_ohlcv(df, "TEST-USD", "1d", data_dir)
    print(f"Saved to: {path}")
    
    # Load back
    loaded = load_ohlcv("TEST-USD", "1d", data_dir)
    print(f"Loaded {len(loaded)} rows")
    
    # Get info
    info = get_parquet_info("TEST-USD", "1d", data_dir)
    print(f"Info: {info}")

