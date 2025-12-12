# Technical Architecture: Data Layer

**Version 1.0**

**Date: December 12, 2025**

---

## Overview

This document describes the technical architecture of the data caching layer for the Simons-Dalio Regime Engine. The system uses a tiered storage approach with **Apache Parquet** for OHLCV data and **SQLite** for metadata tracking, enabling high-performance backtesting without redundant API calls.

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                    DATA LAYER ARCHITECTURE                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐         ┌──────────────────────────────────────┐│
│  │   yfinance   │────────▶│         DataManager                  ││
│  │   (or any    │  fetch  │                                      ││
│  │    API)      │  once   │  • Check if data exists locally      ││
│  └──────────────┘         │  • Fetch only missing date ranges    ││
│                           │  • Upsert to Parquet                 ││
│                           │  • Update metadata in SQLite         ││
│                           └──────────────┬───────────────────────┘│
│                                          │                        │
│                    ┌─────────────────────┴────────────────────┐   │
│                    │                                          │   │
│                    ▼                                          ▼   │
│  ┌─────────────────────────────────┐    ┌────────────────────────┐│
│  │     PARQUET FILES               │    │   SQLITE (metadata)    ││
│  │     (Columnar Storage)          │    │   (Bookkeeping)        ││
│  │                                 │    │                        ││
│  │  data/                          │    │  • symbols table       ││
│  │  └── market/                    │    │  • data_coverage table ││
│  │      ├── BTC-USD/               │    │  • last_updated times  ││
│  │      │   ├── 1d.parquet         │    │                        ││
│  │      │   ├── 1h.parquet         │    └────────────────────────┘│
│  │      │   └── 15m.parquet        │                              │
│  │      ├── ETH-USD/               │                              │
│  │      │   └── ...                │                              │
│  │      └── ^TNX/                  │                              │
│  │          └── 1d.parquet         │                              │
│  └─────────────────────────────────┘                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/data/
├── __init__.py      # Public exports
├── fetcher.py       # Raw yfinance API calls
├── storage.py       # Parquet read/write operations
├── metadata.py      # SQLite metadata tracking
├── manager.py       # High-level orchestration (main interface)
└── features.py      # Feature engineering (unchanged)
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `fetcher.py` | Raw API calls to yfinance. Handles rate limiting, error handling, data normalization. |
| `storage.py` | Parquet file I/O. Handles compression, predicate pushdown, deduplication. |
| `metadata.py` | SQLite database for tracking what data we have and when it was updated. |
| `manager.py` | **Main interface**. Orchestrates caching logic, incremental fetching, and provides simple API. |

---

## Why Parquet?

Apache Parquet is a **columnar file format** optimized for analytical workloads:

### Row-Based vs Columnar Storage

```
Row-Based (CSV/SQLite):          Columnar (Parquet):
┌────┬───────┬───────┬───────┐   ┌────────────────────────┐
│Date│ Open  │ High  │ Close │   │ Date: [d1,d2,d3,d4...] │
├────┼───────┼───────┼───────┤   │ Open: [o1,o2,o3,o4...] │
│ d1 │  o1   │  h1   │  c1   │   │ High: [h1,h2,h3,h4...] │
│ d2 │  o2   │  h2   │  c2   │   │Close: [c1,c2,c3,c4...] │
│ d3 │  o3   │  h3   │  c3   │   └────────────────────────┘
└────┴───────┴───────┴───────┘
```

### Benefits

1. **Compression**: 10-50x smaller than CSV (similar values compress well)
2. **Column Pruning**: Only read columns you need
3. **Predicate Pushdown**: Filter rows without loading entire file
4. **Memory Efficiency**: Memory-map large files without loading fully
5. **Schema Preservation**: Types and column names are stored in metadata

### Performance Benchmarks

Typical for 2 years of 1-minute data (~1M rows):

| Format | File Size | Read Time |
|--------|-----------|-----------|
| CSV | 85 MB | 2.1 sec |
| SQLite | 45 MB | 0.8 sec |
| **Parquet** | **8 MB** | **0.05 sec** |

---

## SQLite Schema

The metadata database tracks what data is cached and when it was last updated.

### Tables

```sql
-- Symbol registry
CREATE TABLE symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,       -- e.g., 'BTC-USD'
    name TEXT,                          -- e.g., 'Bitcoin USD'
    asset_class TEXT,                   -- 'crypto', 'equity', 'macro'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data coverage tracking
CREATE TABLE data_coverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    timeframe TEXT NOT NULL,            -- '1d', '1h', '15m'
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    row_count INTEGER,
    file_path TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_source TEXT DEFAULT 'yfinance',
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(symbol_id, timeframe)
);
```

### Indexes

```sql
CREATE INDEX idx_symbols_symbol ON symbols(symbol);
CREATE INDEX idx_coverage_symbol_tf ON data_coverage(symbol_id, timeframe);
```

---

## Data Flow: Smart Incremental Fetching

The key innovation is **gap detection** - only fetching date ranges that are missing from the cache.

```
User Request: get_ohlcv("BTC-USD", "1d", start="2023-01-01", end="2024-12-01")
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Check Metadata Coverage     │
                    │                               │
                    │   Cached: 2024-01-01 to       │
                    │           2024-06-01          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Calculate Missing Ranges    │
                    │                               │
                    │   Gap 1: 2023-01-01 to        │
                    │          2023-12-31           │
                    │   Gap 2: 2024-06-02 to        │
                    │          2024-12-01           │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │  Fetch Gap 1 from │           │  Fetch Gap 2 from │
        │     yfinance      │           │     yfinance      │
        └─────────┬─────────┘           └─────────┬─────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────────┐
                    │   Merge with Existing Data    │
                    │   (Deduplicate by Date)       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Save to Parquet             │
                    │   Update Metadata             │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Return Full Range from      │
                    │   Parquet (with date filter)  │
                    └───────────────────────────────┘
```

---

## API Reference

### DataManager (Primary Interface)

```python
from src.data import DataManager

dm = DataManager(data_dir="./data")

# Basic usage - automatically caches
df = dm.get_ohlcv("BTC-USD", "1d", start="2023-01-01", end="2024-12-01")

# Force refresh from API
df = dm.get_ohlcv("BTC-USD", "1d", force_refresh=True)

# Multiple symbols
data = dm.get_ohlcv_multi(
    ["BTC-USD", "ETH-USD", "^TNX"],
    timeframe="1d",
    start="2024-01-01"
)

# Cache statistics
stats = dm.get_cache_stats()
# {'symbols_count': 3, 'total_rows': 2100, 'total_size_mb': 0.15}

# Clear cache
dm.clear_cache()  # All
dm.clear_cache(symbol="BTC-USD")  # Specific symbol
dm.clear_cache(symbol="BTC-USD", timeframe="1h")  # Specific
```

### Supported Timeframes

| Timeframe | yfinance Limit | Recommended Use |
|-----------|----------------|-----------------|
| `1m` | Max 7 days | Real-time monitoring |
| `5m` | Max 60 days | Short-term analysis |
| `15m` | Max 60 days | Intraday patterns |
| `1h` | Max 730 days | Swing trading |
| `1d` | Full history | **Primary for backtesting** |
| `1wk` | Full history | Long-term trends |

---

## File Paths

### Convention

```
data/
├── market/                    # OHLCV data
│   └── {SYMBOL}/              # Symbol directory (/ -> _)
│       └── {timeframe}.parquet
└── metadata.db                # SQLite tracking
```

### Examples

| Symbol | Timeframe | Path |
|--------|-----------|------|
| BTC-USD | 1d | `data/market/BTC-USD/1d.parquet` |
| ^TNX | 1d | `data/market/^TNX/1d.parquet` |
| DX-Y.NYB | 1h | `data/market/DX-Y.NYB/1h.parquet` |

---

## Error Handling

### API Errors

```python
# yfinance returns None/empty for invalid symbols or date ranges
# DataManager handles gracefully:
df = dm.get_ohlcv("INVALID-SYMBOL", "1d")  # Returns empty DataFrame
```

### Missing Data

```python
# If Parquet file is missing but metadata exists, re-fetches
# If metadata is missing but file exists, rebuilds metadata
```

### Network Failures

```python
# Partial fetches are saved - can resume later
# Existing cached data is never lost
```

---

## Performance Optimization

### 1. Predicate Pushdown

```python
# Only reads rows matching the date filter
df = load_ohlcv("BTC-USD", "1d", data_dir, start="2024-06-01", end="2024-07-01")
# Reads ~30 rows instead of 730
```

### 2. Column Pruning

```python
# Future: specify columns to reduce I/O
df = load_ohlcv("BTC-USD", "1d", data_dir, columns=["Close", "Volume"])
```

### 3. Memory Mapping

PyArrow uses memory-mapped I/O for large files, enabling access to datasets larger than RAM.

---

## Future Enhancements

### Planned

1. **Feature Caching**: Store pre-computed Simons features
2. **DuckDB Queries**: Cross-asset analytics with SQL
3. **Data Quality**: Validation, gap detection, repair

### DuckDB Integration (Future)

```python
import duckdb

# Query across all symbols without loading into memory
result = duckdb.query("""
    SELECT symbol, date, close
    FROM 'data/market/*/1d.parquet'
    WHERE date >= '2024-01-01'
      AND close > LAG(close) OVER (PARTITION BY symbol ORDER BY date)
""").df()
```

---

## Testing

Run the data layer test suite:

```bash
python scripts/test_data_layer.py
```

Expected output:

```
[1/4] First fetch (yfinance -> Parquet)...
  ✓ Fetched 92 rows from yfinance in 2.15s
  ✓ Saved to market/BTC-USD/1d.parquet (4.2 KB)

[2/4] Second fetch (Parquet only)...
  ✓ Loaded 92 rows from Parquet in 0.02s
  ✓ 107x faster than API fetch

[3/4] Incremental update test...
  ✓ Extended coverage: 92 -> 153 rows
  ✓ Fetched 61 new rows in 1.87s

[4/4] Data integrity check...
  ✓ Round-trip verified: 50 rows preserved
  ✓ Values match within tolerance

DATA LAYER TEST PASSED
```

---

## Dependencies

```
pyarrow>=14.0.0    # Parquet read/write
duckdb>=0.9.0      # Query engine (optional, for future)
```

Both are pure Python wheels with no external dependencies.

