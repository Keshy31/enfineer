# Technical Architecture: Data Layer

**Version 1.1**

**Date: December 12, 2025**

---

## Overview

This document describes the technical architecture of the data caching layer for the Simons-Dalio Regime Engine. The system uses a tiered storage approach with **Apache Parquet** for both OHLCV data and computed features, with **SQLite** for metadata tracking. This enables high-performance backtesting without redundant API calls or feature recomputation.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      DATA LAYER ARCHITECTURE                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐         ┌────────────────────────────────────────────┐ │
│  │   yfinance   │────────▶│              DataManager                   │ │
│  │   (or any    │  fetch  │                                            │ │
│  │    API)      │  once   │  get_ohlcv()    │    get_features()        │ │
│  └──────────────┘         │  • Smart fetch  │    • Param-aware cache   │ │
│                           │  • Date gaps    │    • Hash-based lookup   │ │
│                           └────────┬────────┴──────────┬───────────────┘ │
│                                    │                   │                 │
│         ┌──────────────────────────┴───────────────────┴──────────┐      │
│         │                                                         │      │
│         ▼                                                         ▼      │
│  ┌─────────────────────────────────┐    ┌────────────────────────────┐   │
│  │     PARQUET FILES               │    │   SQLITE (metadata)        │   │
│  │     (Columnar Storage)          │    │   (Bookkeeping)            │   │
│  │                                 │    │                            │   │
│  │  data/                          │    │  • symbols table           │   │
│  │  ├── market/        (OHLCV)     │    │  • data_coverage table     │   │
│  │  │   ├── BTC-USD/               │    │  • feature_cache table     │   │
│  │  │   │   └── 1d.parquet         │    │                            │   │
│  │  │   └── ETH-USD/               │    │  Tracks:                   │   │
│  │  │       └── 1d.parquet         │    │  • What data we have       │   │
│  │  │                              │    │  • Date range coverage     │   │
│  │  └── features/      (Computed)  │    │  • Feature params hash     │   │
│  │      └── BTC-USD/               │    │  • Last update times       │   │
│  │          ├── simons_1d_a1b2.pq  │    └────────────────────────────┘   │
│  │          └── simons_1d_c3d4.pq  │                                     │
│  │              (different params)  │                                     │
│  └─────────────────────────────────┘                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
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
├── market/                              # Raw OHLCV data
│   └── {SYMBOL}/
│       └── {timeframe}.parquet
├── features/                            # Computed features
│   └── {SYMBOL}/
│       └── {feature_set}_{timeframe}_{hash}.parquet
└── metadata.db                          # SQLite tracking
```

### OHLCV Examples

| Symbol | Timeframe | Path |
|--------|-----------|------|
| BTC-USD | 1d | `data/market/BTC-USD/1d.parquet` |
| ^TNX | 1d | `data/market/^TNX/1d.parquet` |
| DX-Y.NYB | 1h | `data/market/DX-Y.NYB/1h.parquet` |

### Feature Cache Examples

| Symbol | Params | Path |
|--------|--------|------|
| BTC-USD | window=30 | `data/features/BTC-USD/simons_1d_a1b2c3d4.parquet` |
| BTC-USD | window=20 | `data/features/BTC-USD/simons_1d_e5f6g7h8.parquet` |

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

## Feature Caching

Feature caching prevents redundant computation by storing computed features (e.g., Simons features) with a hash of their parameters.

### The Problem

```python
# Without caching: recomputes every time
for _ in range(100):  # 100 backtest iterations
    features = compute_simons_features(ohlcv, window=30)  # Slow!
```

### The Solution: Parameter Hashing

```python
# Hash the parameters
params = {"window": 30, "sigma_floor": 0.001, "version": "1.0"}
params_hash = hashlib.md5(json.dumps(params)).hexdigest()[:8]
# Result: "a1b2c3d4"
```

Same parameters → same hash → cache hit → instant load.
Different parameters → different hash → cache miss → compute and cache.

### API

```python
dm = DataManager("./data")

# First call: computes and caches
features = dm.get_features("BTC-USD", "1d", window=30, sigma_floor=0.001)

# Second call (same params): loads from cache instantly
features = dm.get_features("BTC-USD", "1d", window=30, sigma_floor=0.001)

# Different params: computes and caches separately
features = dm.get_features("BTC-USD", "1d", window=20, sigma_floor=0.001)

# Force recompute (e.g., after code change)
features = dm.get_features("BTC-USD", "1d", window=30, force_recompute=True)

# List cached features
dm.list_cached_features("BTC-USD")
# [FeatureCacheInfo(params_hash='a1b2c3d4', ...), ...]

# Clear feature cache
dm.clear_feature_cache(symbol="BTC-USD")
```

### SQLite Schema

```sql
CREATE TABLE feature_cache (
    id INTEGER PRIMARY KEY,
    symbol_id INTEGER REFERENCES symbols(id),
    timeframe TEXT NOT NULL,
    feature_set TEXT NOT NULL,       -- 'simons', 'dalio', etc.
    params_hash TEXT NOT NULL,       -- MD5 hash (first 8 chars)
    params_json TEXT,                -- Full params for reference
    row_count INTEGER,
    file_path TEXT,
    computed_at TIMESTAMP,
    UNIQUE(symbol_id, timeframe, feature_set, params_hash)
);
```

### Cache Flow

```
Request: get_features("BTC-USD", "1d", window=30)
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Compute params_hash         │
         │  {"window":30,...} → "a1b2"  │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Check metadata.db           │
         │  SELECT * FROM feature_cache │
         │  WHERE params_hash = 'a1b2'  │
         └──────────────┬───────────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
         [Cache HIT]         [Cache MISS]
              │                   │
              ▼                   ▼
    ┌─────────────────┐  ┌─────────────────┐
    │  Load from      │  │  Compute        │
    │  Parquet        │  │  features       │
    │  (instant)      │  │  Save to cache  │
    └─────────────────┘  └─────────────────┘
```

---

## Future Enhancements

### Planned

1. ~~**Feature Caching**: Store pre-computed Simons features~~ ✅ Implemented
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

The test covers 7 scenarios:

| Test | Description |
|------|-------------|
| 1. First fetch | Download from yfinance, save to Parquet |
| 2. Cache speed | Load from Parquet (should be faster) |
| 3. Incremental | Detect gaps, fetch only missing data |
| 4. Integrity | Verify round-trip data preservation |
| 5. Feature compute | Compute Simons features, cache them |
| 6. Feature cache | Load cached features (should be instant) |
| 7. Feature params | Different params = different cache entry |

Expected output:

```
[1/7] First fetch (yfinance -> Parquet)...
  ✓ Fetched 92 rows from yfinance in 2.15s

[2/7] Second fetch (Parquet only)...
  ✓ Loaded 92 rows from Parquet in 0.02s
  ✓ 107x faster than API fetch

...

[5/7] Feature computation (first time)...
  ✓ Computed 92 rows with 14 features
  ✓ Z-scores bounded (max: 4.23)

[6/7] Feature cache hit (same params)...
  ✓ Loaded from cache in 0.0012s
  ✓ 150x faster than computation

[7/7] Feature param change (different params)...
  ✓ Different params produce different hashes
  ✓ Multiple cache entries: 2

DATA LAYER TEST PASSED
```

---

## Dependencies

```
pyarrow>=14.0.0    # Parquet read/write
duckdb>=0.9.0      # Query engine (optional, for future)
torch>=2.0.0       # Neural network (CUDA recommended)
```

Both are pure Python wheels with no external dependencies.

---

## Walk-Forward Validation Framework

### The Problem with Standard Backtesting

Standard ML backtesting creates look-ahead bias:

```python
# WRONG: Fits on future data
scaler.fit(all_data)           # Sees 2024 statistics
gmm.fit(all_data)              # Knows 2024 clusters
labels = gmm.predict(all_data) # "Predicts" what it already saw
```

### Walk-Forward Solution

```python
# CORRECT: Expanding window
for fold in folds:
    scaler.fit(train_only)     # Only past data
    gmm.fit(train_only)        # Only past clusters
    labels = gmm.predict(test) # Truly unseen future
```

### Implementation

The `WalkForwardGMM` class in `src/data/walk_forward.py` provides:

```python
from src.data import WalkForwardGMM, get_stationary_features

wf = WalkForwardGMM(
    n_regimes=8,           # BIC-optimal
    min_train_days=504,    # 2 years minimum
    test_days=126,         # 6 months per fold
    gap_days=30,           # Purge period
    use_pca=True,          # Reduce dimensions
    pca_variance=0.95,     # Keep 95% variance
)

results = wf.fit_predict(features, get_stationary_features())
```

### Key Features

1. **Stationary Features Only**: `get_stationary_features()` excludes raw levels
2. **PCA Compression**: Reduces 28 features to ~17 dimensions
3. **BIC Cluster Selection**: `find_optimal_k()` chooses optimal regime count
4. **Gap/Purge Period**: 30-day gap prevents feature leakage

---

## Validated Results (December 2025)

### Walk-Forward GMM Baseline

| Metric | Value |
|--------|-------|
| Out-of-Sample Days | 1,260 |
| Optimal Clusters (BIC) | 8 |
| PCA Dimensions | 17 |
| Sharpe Spread | 3.50 |
| Best Regime Sharpe | 2.29 (Regime 7) |
| Worst Regime Sharpe | -1.21 (Regime 0) |
| Strategy Sharpe | 1.03 |
| Strategy Return | +506% |
| Buy & Hold Return | +253% |

### LSTM-Autoencoder (3D Latent)

| Metric | Value |
|--------|-------|
| Latent Dimensions | 3 |
| Sharpe Spread | 0.79 |
| Clusters Detected | 3 |
| Training Time (GPU) | ~4 minutes |

**Conclusion**: 3D latent space loses too much signal. Recommend 8D.

