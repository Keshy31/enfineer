"""
Data Collection Script
======================
Fetches full historical data for Bitcoin and macro assets.

This script populates the data cache with:
- BTC-USD: Bitcoin (2014-present, 24/7 trading)
- ^TNX: 10-Year Treasury Yield (interest rate environment)
- DX-Y.NYB: US Dollar Index (dollar strength)
- GLD: Gold ETF (risk-off proxy)

Run from project root:
    python scripts/collect_data.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.manager import DataManager


# Asset configuration
ASSETS = {
    "BTC-USD": {
        "name": "Bitcoin USD",
        "asset_class": "crypto",
        "start_date": "2014-09-17",  # BTC-USD available from this date on yfinance
        "description": "Primary trading target - 24/7 crypto market",
    },
    "^TNX": {
        "name": "10-Year Treasury Yield",
        "asset_class": "macro",
        "start_date": "2014-01-01",
        "description": "Interest rate environment - rising yields = tighter conditions",
    },
    "DX-Y.NYB": {
        "name": "US Dollar Index",
        "asset_class": "macro",
        "start_date": "2014-01-01",
        "description": "Dollar strength - strong dollar often negative for risk assets",
    },
    "GLD": {
        "name": "SPDR Gold Shares ETF",
        "asset_class": "macro",
        "start_date": "2014-01-01",
        "description": "Gold proxy - risk-off sentiment indicator",
    },
}


def collect_all_data(data_dir: str = "./data", force_refresh: bool = False):
    """
    Collect full historical data for all configured assets.
    
    Parameters
    ----------
    data_dir : str
        Directory for data storage.
    force_refresh : bool
        If True, re-fetch all data even if cached.
    """
    print("=" * 70)
    print("SIMONS-DALIO REGIME ENGINE")
    print("Data Collection")
    print("=" * 70)
    print()
    
    dm = DataManager(data_dir)
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    results = {}
    
    for symbol, config in ASSETS.items():
        print(f"\n{'─' * 70}")
        print(f"Fetching: {symbol} ({config['name']})")
        print(f"  Asset Class: {config['asset_class']}")
        print(f"  Date Range: {config['start_date']} to {end_date}")
        print(f"  Purpose: {config['description']}")
        print("─" * 70)
        
        try:
            df = dm.get_ohlcv(
                symbol=symbol,
                timeframe="1d",
                start=config["start_date"],
                end=end_date,
                force_refresh=force_refresh,
            )
            
            if len(df) > 0:
                results[symbol] = {
                    "success": True,
                    "rows": len(df),
                    "start": df.index.min().strftime("%Y-%m-%d"),
                    "end": df.index.max().strftime("%Y-%m-%d"),
                    "columns": list(df.columns),
                }
                print(f"\n  ✓ Success: {len(df)} rows")
                print(f"  ✓ Actual range: {results[symbol]['start']} to {results[symbol]['end']}")
            else:
                results[symbol] = {"success": False, "error": "No data returned"}
                print(f"\n  ✗ Failed: No data returned")
                
        except Exception as e:
            results[symbol] = {"success": False, "error": str(e)}
            print(f"\n  ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results.values() if r.get("success"))
    total = len(results)
    
    print(f"\nAssets collected: {successful}/{total}")
    print()
    
    for symbol, result in results.items():
        if result.get("success"):
            print(f"  ✓ {symbol:12} | {result['rows']:,} rows | {result['start']} to {result['end']}")
        else:
            print(f"  ✗ {symbol:12} | Error: {result.get('error', 'Unknown')}")
    
    # Cache stats
    print()
    stats = dm.get_cache_stats()
    print(f"Total cached data:")
    print(f"  • Symbols: {stats['symbols_count']}")
    print(f"  • Total rows: {stats['total_rows']:,}")
    print(f"  • Storage size: {stats['total_size_mb']:.2f} MB")
    
    print("\n" + "=" * 70)
    
    if successful == total:
        print("DATA COLLECTION COMPLETE")
        print("All assets fetched successfully. Ready for feature engineering.")
    else:
        print("DATA COLLECTION INCOMPLETE")
        print(f"Failed to fetch {total - successful} asset(s). Check errors above.")
    
    print("=" * 70)
    
    return results


def show_coverage(data_dir: str = "./data"):
    """Show current data coverage without fetching."""
    dm = DataManager(data_dir)
    
    print("=" * 70)
    print("CURRENT DATA COVERAGE")
    print("=" * 70)
    
    coverage_list = dm.list_cached()
    
    if not coverage_list:
        print("\nNo cached data found. Run collection first.")
        return
    
    print()
    for cov in coverage_list:
        print(f"  {cov.symbol:12} | {cov.timeframe:4} | {cov.row_count:,} rows | "
              f"{cov.start_date} to {cov.end_date}")
    
    print()
    stats = dm.get_cache_stats()
    print(f"Total: {stats['total_rows']:,} rows, {stats['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect market data for the Regime Engine")
    parser.add_argument("--data-dir", default="./data", help="Data storage directory")
    parser.add_argument("--force", action="store_true", help="Force re-fetch all data")
    parser.add_argument("--status", action="store_true", help="Show current coverage only")
    
    args = parser.parse_args()
    
    if args.status:
        show_coverage(args.data_dir)
    else:
        collect_all_data(args.data_dir, force_refresh=args.force)

