"""
Data Manager Module
===================
Orchestrates data fetching, caching, and retrieval.

This is the main interface for accessing market data. It intelligently
manages the local Parquet cache, fetching from yfinance only when
necessary (missing date ranges).

Also provides feature caching - computed features are stored with a
hash of their parameters, so changing params triggers recomputation
while same params load from cache.

Example
-------
>>> dm = DataManager("./data")
>>> df = dm.get_ohlcv("BTC-USD", "1d", start="2023-01-01", end="2024-12-01")
>>> # First call: fetches from yfinance, saves to Parquet
>>> # Second call: reads from Parquet instantly
>>>
>>> # Feature caching
>>> features = dm.get_features("BTC-USD", "1d", window=30)
>>> # First call: computes features, saves to Parquet
>>> # Second call with same params: loads from cache
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Callable
import time

from .storage import (
    save_ohlcv,
    load_ohlcv,
    append_ohlcv,
    parquet_exists,
    get_parquet_path,
    # Feature storage
    save_features,
    load_features,
    feature_cache_exists,
    get_feature_path,
    compute_params_hash,
)
from .metadata import MetadataDB, CoverageInfo, FeatureCacheInfo
from .fetcher import fetch_from_yfinance
from .features import compute_simons_features
from .alignment import align_to_crypto, find_common_date_range, trim_to_common_range
from .dalio_features import compute_dalio_features, compute_combined_features


# Default macro symbols for alignment
DEFAULT_MACRO_SYMBOLS = ["^TNX", "DX-Y.NYB", "GLD"]


class DataManager:
    """
    High-level interface for market data access.
    
    Manages the data caching layer, automatically fetching from
    yfinance when data is missing and reading from local Parquet
    files when available.
    
    Parameters
    ----------
    data_dir : str or Path
        Root directory for data storage.
        
    Attributes
    ----------
    data_dir : Path
        Root data directory.
    metadata : MetadataDB
        SQLite metadata database.
    """
    
    def __init__(self, data_dir: str | Path = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata database
        self.metadata = MetadataDB(self.data_dir / "metadata.db")
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        years_of_history: int = 2,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol, using cache when available.
        
        This is the main method for accessing market data. It:
        1. Checks what data we have cached
        2. Identifies missing date ranges
        3. Fetches only missing data from yfinance
        4. Updates the cache
        5. Returns the full requested range
        
        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g., 'BTC-USD', '^TNX', 'AAPL').
        timeframe : str, default='1d'
            Data timeframe: '1d', '1h', '15m', '5m', etc.
        start : str, optional
            Start date in 'YYYY-MM-DD' format.
            Defaults to `years_of_history` years ago.
        end : str, optional
            End date in 'YYYY-MM-DD' format.
            Defaults to today.
        years_of_history : int, default=2
            Years of history to fetch if start not specified.
        force_refresh : bool, default=False
            If True, re-fetch from API even if cached.
            
        Returns
        -------
        pd.DataFrame
            OHLCV data with DatetimeIndex.
        """
        # Parse dates
        if end is None:
            end_date = date.today()
        else:
            end_date = datetime.strptime(end, "%Y-%m-%d").date()
        
        if start is None:
            start_date = end_date - timedelta(days=365 * years_of_history)
        else:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
        
        # Force refresh - delete existing and re-fetch all
        if force_refresh:
            self.metadata.delete_coverage(symbol, timeframe)
        
        # Check what we need to fetch
        missing_ranges = self.metadata.get_missing_ranges(
            symbol, timeframe, start_date, end_date
        )
        
        if not missing_ranges:
            # Everything cached - fast path
            print(f"✓ Loading {symbol}/{timeframe} from cache")
            return load_ohlcv(
                symbol, timeframe, self.data_dir,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )
        
        # Need to fetch missing data
        print(f"[FETCH] Fetching missing data for {symbol}/{timeframe}...")
        
        for gap_start, gap_end in missing_ranges:
            print(f"  Fetching: {gap_start} to {gap_end}")
            
            start_time = time.time()
            new_data = fetch_from_yfinance(
                symbol=symbol,
                start_date=gap_start.isoformat(),
                end_date=gap_end.isoformat(),
                interval=timeframe,
            )
            fetch_time = time.time() - start_time
            
            if new_data is not None and len(new_data) > 0:
                # Append to Parquet (handles merge with existing)
                file_path = append_ohlcv(new_data, symbol, timeframe, self.data_dir)
                print(f"  ✓ Fetched {len(new_data)} rows in {fetch_time:.2f}s")
            else:
                print(f"  [WARN] No data returned for {gap_start} to {gap_end}")
        
        # Update metadata with full coverage
        if parquet_exists(symbol, timeframe, self.data_dir):
            full_data = load_ohlcv(symbol, timeframe, self.data_dir)
            self.metadata.update_coverage(
                symbol=symbol,
                timeframe=timeframe,
                start_date=full_data.index.min().date(),
                end_date=full_data.index.max().date(),
                row_count=len(full_data),
                file_path=str(get_parquet_path(symbol, timeframe, self.data_dir)),
            )
        
        # Return requested range
        return load_ohlcv(
            symbol, timeframe, self.data_dir,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )
    
    def get_ohlcv_multi(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple symbols.
        
        Parameters
        ----------
        symbols : list[str]
            List of ticker symbols.
        timeframe : str
            Data timeframe.
        start : str, optional
            Start date.
        end : str, optional
            End date.
            
        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping symbol to DataFrame.
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_ohlcv(symbol, timeframe, start, end)
            except Exception as e:
                print(f"✗ Error fetching {symbol}: {e}")
        return result
    
    def list_cached(self) -> list[CoverageInfo]:
        """List all cached data with coverage information."""
        return self.metadata.list_coverage()
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the data cache."""
        coverage_list = self.metadata.list_coverage()
        
        total_rows = sum(c.row_count for c in coverage_list)
        symbols = set(c.symbol for c in coverage_list)
        
        # Calculate total storage size
        total_bytes = 0
        for c in coverage_list:
            path = Path(c.file_path) if c.file_path else None
            if path and path.exists():
                total_bytes += path.stat().st_size
        
        return {
            "symbols_count": len(symbols),
            "timeframes_count": len(coverage_list),
            "total_rows": total_rows,
            "total_size_mb": total_bytes / (1024 * 1024),
            "symbols": list(symbols),
        }
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Clear cached data.
        
        Parameters
        ----------
        symbol : str, optional
            Clear only this symbol. If None, clears all.
        timeframe : str, optional
            Clear only this timeframe.
        """
        if symbol and timeframe:
            # Clear specific symbol/timeframe
            path = get_parquet_path(symbol, timeframe, self.data_dir)
            if path.exists():
                path.unlink()
            self.metadata.delete_coverage(symbol, timeframe)
            print(f"✓ Cleared cache for {symbol}/{timeframe}")
        elif symbol:
            # Clear all timeframes for symbol
            for coverage in self.metadata.list_coverage(symbol):
                path = Path(coverage.file_path)
                if path.exists():
                    path.unlink()
                self.metadata.delete_coverage(symbol, coverage.timeframe)
            print(f"✓ Cleared all cache for {symbol}")
        else:
            # Clear everything
            market_dir = self.data_dir / "market"
            if market_dir.exists():
                import shutil
                shutil.rmtree(market_dir)
            # Re-initialize metadata
            self.metadata = MetadataDB(self.data_dir / "metadata.db")
            print("✓ Cleared all cached data")

    # =========================================
    # Alignment Methods (Crypto + Macro)
    # =========================================
    
    def get_aligned_data(
        self,
        primary_symbol: str = "BTC-USD",
        macro_symbols: Optional[list[str]] = None,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        trim_to_common: bool = True,
    ) -> pd.DataFrame:
        """
        Get aligned data combining crypto with macro assets.
        
        Fetches all symbols and aligns macro data to the crypto's
        7-day trading schedule using forward-fill for weekends.
        
        Parameters
        ----------
        primary_symbol : str, default='BTC-USD'
            Primary crypto symbol (trades 7 days/week).
        macro_symbols : list[str], optional
            Macro symbols to include. Defaults to ['^TNX', 'DX-Y.NYB', 'GLD'].
        timeframe : str, default='1d'
            Data timeframe.
        start : str, optional
            Start date in 'YYYY-MM-DD' format.
        end : str, optional
            End date in 'YYYY-MM-DD' format.
        trim_to_common : bool, default=True
            If True, trim to date range where all symbols have data.
            
        Returns
        -------
        pd.DataFrame
            Aligned DataFrame with crypto OHLCV and macro columns.
            Macro columns are prefixed (e.g., 'TNX_Close', 'DXY_Close').
            
        Example
        -------
        >>> dm = DataManager("./data")
        >>> aligned = dm.get_aligned_data("BTC-USD", start="2020-01-01")
        >>> aligned.columns
        Index(['Open', 'High', 'Low', 'Close', 'Volume',
               'TNX_Close', 'DXY_Close', 'GLD_Close', ...])
        """
        if macro_symbols is None:
            macro_symbols = DEFAULT_MACRO_SYMBOLS.copy()
        
        print(f"[FETCH] Fetching aligned data: {primary_symbol} + {macro_symbols}")
        
        # Fetch primary (crypto) data
        crypto_df = self.get_ohlcv(primary_symbol, timeframe, start=start, end=end)
        
        # Fetch macro data
        macro_dfs = {}
        for symbol in macro_symbols:
            try:
                macro_dfs[symbol] = self.get_ohlcv(symbol, timeframe, start=start, end=end)
            except Exception as e:
                print(f"  [WARN] Could not fetch {symbol}: {e}")
        
        if not macro_dfs:
            print("  [WARN] No macro data available, returning crypto only")
            return crypto_df
        
        # Optionally trim to common date range
        if trim_to_common:
            all_dfs = {primary_symbol: crypto_df, **macro_dfs}
            common_start, common_end = find_common_date_range(all_dfs)
            print(f"  Common date range: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
            
            crypto_df = crypto_df.loc[common_start:common_end]
            macro_dfs = {k: v.loc[common_start:common_end] for k, v in macro_dfs.items()}
        
        # Align macro to crypto dates
        print("  Aligning macro data to crypto dates...")
        aligned = align_to_crypto(crypto_df, macro_dfs)
        
        print(f"  ✓ Aligned data: {len(aligned)} rows, {len(aligned.columns)} columns")
        
        return aligned
    
    # =========================================
    # Feature Caching Methods
    # =========================================
    
    def get_features(
        self,
        symbol: str,
        timeframe: str = "1d",
        feature_set: str = "simons",
        window: int = 30,
        sigma_floor: float = 0.001,
        start: Optional[str] = None,
        end: Optional[str] = None,
        macro_symbols: Optional[list[str]] = None,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """
        Get computed features, using cache when available.
        
        This is the smart caching layer for feature engineering:
        
        1. Hash the computation parameters (window, sigma_floor, etc.)
        2. Check if features with that hash exist in cache
        3. If YES → load from Parquet (fast!)
        4. If NO → compute features, save to cache, return
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str, default='1d'
            Data timeframe.
        feature_set : str, default='simons'
            Which feature set to compute:
            - 'simons': Price-based features only (log returns, momentum, volatility)
            - 'dalio': Macro features only (yields, dollar, gold, correlations)
            - 'combined': Both Simons and Dalio features (full neural network input)
        window : int, default=30
            Rolling window for feature calculations.
        sigma_floor : float, default=0.001
            Volatility floor for Z-score calculations.
        start : str, optional
            Start date for the OHLCV data.
        end : str, optional
            End date for the OHLCV data.
        macro_symbols : list[str], optional
            Macro symbols for 'dalio' or 'combined' feature sets.
            Defaults to ['^TNX', 'DX-Y.NYB', 'GLD'].
        force_recompute : bool, default=False
            If True, recompute even if cached.
            
        Returns
        -------
        pd.DataFrame
            Feature DataFrame with DatetimeIndex.
            
        Example
        -------
        >>> dm = DataManager("./data")
        >>> # Simons features (price-based)
        >>> features = dm.get_features("BTC-USD", window=30)
        >>> 
        >>> # Combined features (price + macro)
        >>> features = dm.get_features("BTC-USD", feature_set="combined")
        >>> 
        >>> # Different params: computes and caches separately
        >>> features = dm.get_features("BTC-USD", window=20)
        """
        if macro_symbols is None:
            macro_symbols = DEFAULT_MACRO_SYMBOLS.copy()
        
        # Build parameters dict for hashing
        # Include everything that affects the output
        params = {
            "feature_set": feature_set,
            "window": window,
            "sigma_floor": sigma_floor,
            "version": "1.1",  # Bumped for combined features support
        }
        
        # Include macro symbols in hash for combined/dalio features
        if feature_set in ("combined", "dalio"):
            params["macro_symbols"] = sorted(macro_symbols)
        
        params_hash = compute_params_hash(params)
        
        # Check cache (unless force recompute)
        if not force_recompute:
            cache_info = self.metadata.get_feature_cache(
                symbol, timeframe, feature_set, params_hash
            )
            
            if cache_info is not None:
                # Cache HIT - load from Parquet
                feature_path = get_feature_path(
                    symbol, timeframe, feature_set, params_hash, self.data_dir
                )
                if feature_path.exists():
                    print(f"✓ Loading {feature_set} features from cache (hash: {params_hash})")
                    return load_features(
                        symbol, timeframe, feature_set, params_hash, self.data_dir
                    )
        
        # Cache MISS - need to compute
        print(f"[COMPUTE] Computing {feature_set} features for {symbol}/{timeframe}...")
        print(f"  Params: window={window}, sigma_floor={sigma_floor}")
        
        start_time = time.time()
        
        # Step 1: Get data (different paths for different feature sets)
        if feature_set == "simons":
            # Simple path: just need OHLCV
            ohlcv = self.get_ohlcv(symbol, timeframe, start=start, end=end)
            features = compute_simons_features(
                ohlcv, window=window, sigma_floor=sigma_floor
            )
            
        elif feature_set == "dalio":
            # Need aligned data
            aligned = self.get_aligned_data(
                primary_symbol=symbol,
                macro_symbols=macro_symbols,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            features = compute_dalio_features(
                aligned, window=window, sigma_floor=sigma_floor
            )
            
        elif feature_set == "combined":
            # Need aligned data + both feature sets
            aligned = self.get_aligned_data(
                primary_symbol=symbol,
                macro_symbols=macro_symbols,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            features = compute_combined_features(
                aligned, window=window, sigma_floor=sigma_floor
            )
            
        else:
            raise ValueError(
                f"Unknown feature set: {feature_set}. "
                f"Valid options: 'simons', 'dalio', 'combined'"
            )
        
        compute_time = time.time() - start_time
        
        # Step 2: Save to cache
        file_path = save_features(
            features, symbol, timeframe, feature_set, params_hash, self.data_dir
        )
        
        # Step 3: Update metadata
        self.metadata.update_feature_cache(
            symbol=symbol,
            timeframe=timeframe,
            feature_set=feature_set,
            params_hash=params_hash,
            params=params,
            row_count=len(features),
            file_path=str(file_path),
        )
        
        print(f"  ✓ Computed {len(features)} rows in {compute_time:.2f}s")
        print(f"  ✓ Cached with hash: {params_hash}")
        
        return features
    
    def list_cached_features(
        self,
        symbol: Optional[str] = None,
        feature_set: Optional[str] = None,
    ) -> list[FeatureCacheInfo]:
        """
        List all cached feature computations.
        
        Parameters
        ----------
        symbol : str, optional
            Filter by symbol.
        feature_set : str, optional
            Filter by feature set.
            
        Returns
        -------
        list[FeatureCacheInfo]
            List of cached feature records.
        """
        return self.metadata.list_feature_cache(symbol, feature_set)
    
    def clear_feature_cache(
        self,
        symbol: Optional[str] = None,
        feature_set: Optional[str] = None,
    ):
        """
        Clear cached features.
        
        Parameters
        ----------
        symbol : str, optional
            Clear only this symbol's features.
        feature_set : str, optional
            Clear only this feature set.
        """
        if symbol:
            # Get list of cached features to delete files
            cached = self.metadata.list_feature_cache(symbol, feature_set)
            for info in cached:
                path = Path(info.file_path)
                if path.exists():
                    path.unlink()
            
            # Delete metadata
            deleted = self.metadata.delete_feature_cache(symbol, feature_set=feature_set)
            print(f"✓ Cleared {deleted} cached feature record(s)")
        else:
            # Clear all feature cache
            features_dir = self.data_dir / "features"
            if features_dir.exists():
                import shutil
                shutil.rmtree(features_dir)
            print("✓ Cleared all cached features")


if __name__ == "__main__":
    # Quick test
    dm = DataManager("./data")
    
    # Fetch some data
    print("\n" + "=" * 50)
    print("First fetch (should hit API):")
    print("=" * 50)
    df = dm.get_ohlcv("BTC-USD", "1d", start="2024-01-01", end="2024-06-01")
    print(f"Got {len(df)} rows")
    print(df.tail())
    
    print("\n" + "=" * 50)
    print("Second fetch (should use cache):")
    print("=" * 50)
    df2 = dm.get_ohlcv("BTC-USD", "1d", start="2024-01-01", end="2024-06-01")
    print(f"Got {len(df2)} rows")
    
    print("\n" + "=" * 50)
    print("Cache stats:")
    print("=" * 50)
    print(dm.get_cache_stats())

