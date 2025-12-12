"""
Data Fetcher Module
===================
Fetches OHLCV data from yfinance for Bitcoin and other assets.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


def fetch_bitcoin_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years_of_history: int = 2,
) -> pd.DataFrame:
    """
    Fetch BTC-USD daily OHLCV data from yfinance.
    
    Parameters
    ----------
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, defaults to `years_of_history` years ago.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, defaults to today.
    years_of_history : int, default=2
        Number of years of historical data to fetch if start_date is not provided.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex.
        
    Raises
    ------
    ValueError
        If no data is returned or data validation fails.
        
    Example
    -------
    >>> df = fetch_bitcoin_data()
    >>> print(df.head())
    """
    # Default date range: 2 years of history
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=365 * years_of_history)
        start_date = start_dt.strftime("%Y-%m-%d")
    
    # Fetch data from yfinance
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    # Validate data
    if df.empty:
        raise ValueError(
            f"No data returned for BTC-USD from {start_date} to {end_date}. "
            "Check your internet connection or date range."
        )
    
    # Clean up columns - keep only OHLCV
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[required_cols].copy()
    
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)  # Remove timezone for simplicity
    df.index.name = "Date"
    
    # Check for missing data
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    if missing_pct > 5:
        raise ValueError(f"Too much missing data: {missing_pct:.1f}%")
    
    # Forward fill small gaps (weekends don't apply to crypto, but just in case)
    df = df.ffill()
    
    # Sort by date ascending
    df = df.sort_index()
    
    print(f"✓ Fetched {len(df)} days of BTC-USD data")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: ${df['Close'].min():,.2f} - ${df['Close'].max():,.2f}")
    
    return df


def fetch_multi_asset(
    symbols: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple symbols.
    
    Useful for future macro feature integration (e.g., ^TNX, DX-Y.NYB).
    
    Parameters
    ----------
    symbols : list[str]
        List of ticker symbols to fetch.
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format.
        
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping symbol to DataFrame.
    """
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data[symbol] = df
                print(f"✓ Fetched {len(df)} days for {symbol}")
            else:
                print(f"✗ No data for {symbol}")
        except Exception as e:
            print(f"✗ Error fetching {symbol}: {e}")
    
    return data


if __name__ == "__main__":
    # Quick test
    df = fetch_bitcoin_data()
    print("\nSample data:")
    print(df.tail())

