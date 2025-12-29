"""
Data Alignment Module
=====================
Handles alignment between crypto (24/7) and traditional market (5-day) data.

The core challenge: Bitcoin trades every day including weekends and holidays,
while Treasury yields, Dollar Index, and Gold only trade on market days.

Solution: Forward-fill macro data to fill gaps, using Friday's values for
Saturday/Sunday. This reflects reality - macro conditions don't change
over the weekend, traders just use the last known values.
"""

import pandas as pd
import numpy as np
from typing import Optional


def align_to_crypto(
    crypto_df: pd.DataFrame,
    macro_dfs: dict[str, pd.DataFrame],
    fill_method: str = "ffill",
    max_gap_days: int = 5,
) -> pd.DataFrame:
    """
    Align macro data to crypto's 7-day trading schedule.
    
    Takes Bitcoin's date index as the master and reindexes all macro
    data to match, forward-filling weekend/holiday gaps.
    
    Parameters
    ----------
    crypto_df : pd.DataFrame
        Crypto OHLCV data with DatetimeIndex (trades 7 days/week).
    macro_dfs : dict[str, pd.DataFrame]
        Dictionary of macro OHLCV DataFrames, keyed by symbol.
        Example: {"^TNX": df_tnx, "DX-Y.NYB": df_dxy, "GLD": df_gld}
    fill_method : str, default='ffill'
        How to fill missing values:
        - 'ffill': Forward fill (use last known value)
        - 'bfill': Backward fill (use next known value)
        - 'interpolate': Linear interpolation
    max_gap_days : int, default=5
        Maximum consecutive days to fill. Gaps larger than this
        will remain as NaN (indicates data quality issue).
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with crypto OHLCV and aligned macro columns.
        Columns are prefixed with symbol (e.g., 'TNX_Close', 'DXY_Close').
        
    Example
    -------
    >>> aligned = align_to_crypto(btc_df, {"^TNX": tnx_df, "GLD": gld_df})
    >>> aligned.columns
    Index(['Open', 'High', 'Low', 'Close', 'Volume',
           'TNX_Open', 'TNX_High', 'TNX_Low', 'TNX_Close', 'TNX_Volume',
           'GLD_Open', 'GLD_High', 'GLD_Low', 'GLD_Close', 'GLD_Volume'])
    """
    # Start with crypto data as base
    result = crypto_df.copy()
    
    # Clean symbol names for column prefixes
    symbol_prefixes = {
        "^TNX": "TNX",
        "DX-Y.NYB": "DXY",
        "GLD": "GLD",
    }
    
    for symbol, macro_df in macro_dfs.items():
        if macro_df is None or len(macro_df) == 0:
            print(f"  [WARN] Skipping {symbol}: No data")
            continue
            
        # Get clean prefix for column names
        prefix = symbol_prefixes.get(symbol, symbol.replace("^", "").replace("-", "_").replace(".", "_"))
        
        # Reindex to crypto dates
        aligned_macro = macro_df.reindex(result.index)
        
        # Apply fill method
        if fill_method == "ffill":
            aligned_macro = aligned_macro.ffill(limit=max_gap_days)
        elif fill_method == "bfill":
            aligned_macro = aligned_macro.bfill(limit=max_gap_days)
        elif fill_method == "interpolate":
            aligned_macro = aligned_macro.interpolate(method="linear", limit=max_gap_days)
        
        # Rename columns with prefix
        aligned_macro.columns = [f"{prefix}_{col}" for col in aligned_macro.columns]
        
        # Join to result
        result = result.join(aligned_macro)
        
        # Report alignment stats
        original_count = len(macro_df)
        aligned_count = aligned_macro.notna().any(axis=1).sum()
        filled_count = aligned_count - len(macro_df.reindex(result.index).dropna())
        
        print(f"  [OK] {symbol} -> {prefix}_*: {aligned_count} aligned rows ({filled_count} filled)")
    
    return result


def find_common_date_range(
    dataframes: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Find the overlapping date range across all DataFrames.
    
    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Dictionary of DataFrames with DatetimeIndex.
        
    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        (start_date, end_date) of the common range.
    """
    starts = []
    ends = []
    
    for symbol, df in dataframes.items():
        if df is not None and len(df) > 0:
            starts.append(df.index.min())
            ends.append(df.index.max())
    
    if not starts:
        raise ValueError("No valid DataFrames provided")
    
    common_start = max(starts)
    common_end = min(ends)
    
    return common_start, common_end


def trim_to_common_range(
    dataframes: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Trim all DataFrames to their common date range.
    
    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Dictionary of DataFrames with DatetimeIndex.
        
    Returns
    -------
    dict[str, pd.DataFrame]
        Trimmed DataFrames.
    """
    common_start, common_end = find_common_date_range(dataframes)
    
    print(f"  Common date range: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
    
    trimmed = {}
    for symbol, df in dataframes.items():
        if df is not None:
            trimmed[symbol] = df.loc[common_start:common_end].copy()
    
    return trimmed


def compute_alignment_quality(
    aligned_df: pd.DataFrame,
    macro_columns: list[str],
) -> dict:
    """
    Compute quality metrics for aligned data.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output from align_to_crypto().
    macro_columns : list[str]
        List of macro column names to check.
        
    Returns
    -------
    dict
        Quality metrics including NaN counts and fill ratios.
    """
    total_rows = len(aligned_df)
    
    metrics = {
        "total_rows": total_rows,
        "columns": {},
    }
    
    for col in macro_columns:
        if col in aligned_df.columns:
            nan_count = aligned_df[col].isna().sum()
            metrics["columns"][col] = {
                "nan_count": nan_count,
                "nan_pct": (nan_count / total_rows) * 100,
                "valid_count": total_rows - nan_count,
            }
    
    # Overall quality score
    total_nan = sum(m["nan_count"] for m in metrics["columns"].values())
    total_cells = total_rows * len(macro_columns)
    metrics["overall_nan_pct"] = (total_nan / total_cells) * 100 if total_cells > 0 else 0
    metrics["quality_score"] = 100 - metrics["overall_nan_pct"]
    
    return metrics


def get_weekend_mask(index: pd.DatetimeIndex) -> pd.Series:
    """
    Get a boolean mask for weekend days (Saturday=5, Sunday=6).
    
    Parameters
    ----------
    index : pd.DatetimeIndex
        DateTime index to check.
        
    Returns
    -------
    pd.Series
        Boolean Series where True = weekend.
    """
    return pd.Series(index.dayofweek >= 5, index=index)


def report_alignment_gaps(
    aligned_df: pd.DataFrame,
    macro_prefix: str,
) -> pd.DataFrame:
    """
    Report gaps in aligned macro data.
    
    Useful for debugging alignment issues.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output from align_to_crypto().
    macro_prefix : str
        Prefix of macro columns to check (e.g., 'TNX').
        
    Returns
    -------
    pd.DataFrame
        DataFrame showing gap locations and sizes.
    """
    close_col = f"{macro_prefix}_Close"
    
    if close_col not in aligned_df.columns:
        raise ValueError(f"Column {close_col} not found")
    
    # Find NaN locations
    is_nan = aligned_df[close_col].isna()
    
    if not is_nan.any():
        return pd.DataFrame(columns=["start", "end", "gap_days"])
    
    # Find gap starts and ends
    gap_starts = is_nan & ~is_nan.shift(1, fill_value=False)
    gap_ends = is_nan & ~is_nan.shift(-1, fill_value=False)
    
    gaps = []
    starts = aligned_df.index[gap_starts]
    ends = aligned_df.index[gap_ends]
    
    for start, end in zip(starts, ends):
        gap_days = (end - start).days + 1
        gaps.append({
            "start": start,
            "end": end,
            "gap_days": gap_days,
        })
    
    return pd.DataFrame(gaps)


if __name__ == "__main__":
    # Quick test with synthetic data
    import numpy as np
    
    # Create crypto data (7 days/week)
    crypto_dates = pd.date_range("2024-01-01", periods=30, freq="D")
    crypto_df = pd.DataFrame({
        "Open": np.random.randn(30).cumsum() + 100,
        "High": np.random.randn(30).cumsum() + 101,
        "Low": np.random.randn(30).cumsum() + 99,
        "Close": np.random.randn(30).cumsum() + 100,
        "Volume": np.random.randint(1000, 10000, 30),
    }, index=crypto_dates)
    crypto_df.index.name = "Date"
    
    # Create macro data (5 days/week - skip weekends)
    macro_dates = pd.bdate_range("2024-01-01", periods=22)  # Business days only
    macro_df = pd.DataFrame({
        "Open": np.random.randn(22).cumsum() + 4,
        "High": np.random.randn(22).cumsum() + 4.1,
        "Low": np.random.randn(22).cumsum() + 3.9,
        "Close": np.random.randn(22).cumsum() + 4,
        "Volume": np.random.randint(100, 1000, 22),
    }, index=macro_dates)
    macro_df.index.name = "Date"
    
    print("Crypto data shape:", crypto_df.shape)
    print("Macro data shape:", macro_df.shape)
    print()
    
    # Align
    print("Aligning macro to crypto dates:")
    aligned = align_to_crypto(crypto_df, {"^TNX": macro_df})
    
    print()
    print("Aligned shape:", aligned.shape)
    print("Columns:", list(aligned.columns))
    
    # Quality check
    print()
    quality = compute_alignment_quality(aligned, ["TNX_Close"])
    print(f"Quality score: {quality['quality_score']:.1f}%")

