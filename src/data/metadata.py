"""
Metadata Module
===============
SQLite-based metadata tracking for the data caching layer.

Tracks what data we have, date range coverage, and when
data was last updated. This lightweight bookkeeping enables
smart incremental fetching.
"""

import sqlite3
from pathlib import Path
from datetime import date, datetime
from typing import Optional, NamedTuple
from contextlib import contextmanager


class CoverageInfo(NamedTuple):
    """Data coverage information for a symbol/timeframe."""
    symbol: str
    timeframe: str
    start_date: date
    end_date: date
    row_count: int
    file_path: str
    last_updated: datetime


class FeatureCacheInfo(NamedTuple):
    """Information about a cached feature set."""
    symbol: str
    timeframe: str
    feature_set: str      # e.g., 'simons'
    params_hash: str      # MD5 hash of parameters
    params: dict          # The actual parameters
    row_count: int
    file_path: str
    computed_at: datetime


class MetadataDB:
    """
    SQLite database for tracking data coverage metadata.
    
    This class manages a lightweight SQLite database that tracks:
    - Registered symbols and their asset classes
    - Date range coverage per symbol/timeframe
    - Last update timestamps
    
    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
        
    Example
    -------
    >>> db = MetadataDB(Path("./data/metadata.db"))
    >>> db.register_symbol("BTC-USD", "crypto")
    >>> db.update_coverage("BTC-USD", "1d", date(2023, 1, 1), date(2024, 12, 1), 700)
    >>> coverage = db.get_coverage("BTC-USD", "1d")
    >>> print(coverage.start_date, coverage.end_date)
    """
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema if not exists."""
        with self._connection() as conn:
            conn.executescript("""
                -- Symbol registry
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    name TEXT,
                    asset_class TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Data coverage tracking (raw OHLCV)
                CREATE TABLE IF NOT EXISTS data_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    row_count INTEGER,
                    file_path TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT DEFAULT 'yfinance',
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    UNIQUE(symbol_id, timeframe)
                );
                
                -- Feature cache tracking (computed features)
                -- The params_hash is the key: same params = same hash = cache hit
                CREATE TABLE IF NOT EXISTS feature_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    feature_set TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    params_json TEXT,
                    row_count INTEGER,
                    file_path TEXT,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    UNIQUE(symbol_id, timeframe, feature_set, params_hash)
                );
                
                -- Indexes for fast lookups
                CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol);
                CREATE INDEX IF NOT EXISTS idx_coverage_symbol_tf ON data_coverage(symbol_id, timeframe);
                CREATE INDEX IF NOT EXISTS idx_feature_cache_lookup 
                    ON feature_cache(symbol_id, timeframe, feature_set, params_hash);
            """)
    
    def register_symbol(
        self,
        symbol: str,
        asset_class: str = "unknown",
        name: Optional[str] = None,
    ) -> int:
        """
        Register a new symbol or get existing symbol ID.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g., 'BTC-USD').
        asset_class : str
            Asset class ('crypto', 'equity', 'macro', etc.).
        name : str, optional
            Human-readable name.
            
        Returns
        -------
        int
            Symbol ID in the database.
        """
        with self._connection() as conn:
            # Try to insert, ignore if exists
            conn.execute(
                """
                INSERT OR IGNORE INTO symbols (symbol, asset_class, name)
                VALUES (?, ?, ?)
                """,
                (symbol, asset_class, name),
            )
            
            # Get the ID
            row = conn.execute(
                "SELECT id FROM symbols WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            
            return row["id"]
    
    def get_symbol_id(self, symbol: str) -> Optional[int]:
        """Get symbol ID if registered, None otherwise."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id FROM symbols WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            return row["id"] if row else None
    
    def update_coverage(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        row_count: int,
        file_path: Optional[str] = None,
    ) -> None:
        """
        Update or insert data coverage for a symbol/timeframe.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str
            Data timeframe ('1d', '1h', etc.).
        start_date : date
            First date with data.
        end_date : date
            Last date with data.
        row_count : int
            Number of rows/bars in the dataset.
        file_path : str, optional
            Path to the Parquet file.
        """
        # Ensure symbol is registered
        symbol_id = self.register_symbol(symbol)
        
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO data_coverage 
                    (symbol_id, timeframe, start_date, end_date, row_count, file_path, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol_id, timeframe) DO UPDATE SET
                    start_date = excluded.start_date,
                    end_date = excluded.end_date,
                    row_count = excluded.row_count,
                    file_path = excluded.file_path,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (symbol_id, timeframe, start_date.isoformat(), end_date.isoformat(), row_count, file_path),
            )
    
    def get_coverage(self, symbol: str, timeframe: str) -> Optional[CoverageInfo]:
        """
        Get coverage information for a symbol/timeframe.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str
            Data timeframe.
            
        Returns
        -------
        CoverageInfo or None
            Coverage details if data exists, None otherwise.
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 
                    s.symbol,
                    dc.timeframe,
                    dc.start_date,
                    dc.end_date,
                    dc.row_count,
                    dc.file_path,
                    dc.last_updated
                FROM data_coverage dc
                JOIN symbols s ON s.id = dc.symbol_id
                WHERE s.symbol = ? AND dc.timeframe = ?
                """,
                (symbol, timeframe),
            ).fetchone()
            
            if row is None:
                return None
            
            return CoverageInfo(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                start_date=date.fromisoformat(row["start_date"]),
                end_date=date.fromisoformat(row["end_date"]),
                row_count=row["row_count"],
                file_path=row["file_path"],
                last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None,
            )
    
    def get_missing_ranges(
        self,
        symbol: str,
        timeframe: str,
        requested_start: date,
        requested_end: date,
    ) -> list[tuple[date, date]]:
        """
        Calculate date ranges that are missing from cached data.
        
        This is the core logic for incremental fetching - it figures out
        which date ranges need to be fetched from the API.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str
            Data timeframe.
        requested_start : date
            Start of requested date range.
        requested_end : date
            End of requested date range.
            
        Returns
        -------
        list[tuple[date, date]]
            List of (start, end) tuples for missing date ranges.
            Empty list if all data is cached.
            
        Examples
        --------
        >>> # Cached: 2024-01-01 to 2024-06-01
        >>> # Requested: 2023-06-01 to 2024-12-01
        >>> # Returns: [(2023-06-01, 2023-12-31), (2024-06-02, 2024-12-01)]
        """
        coverage = self.get_coverage(symbol, timeframe)
        
        if coverage is None:
            # No cached data - fetch entire range
            return [(requested_start, requested_end)]
        
        missing = []
        
        # Check for gap before cached data
        if requested_start < coverage.start_date:
            # Need data before what we have
            gap_end = min(
                date.fromordinal(coverage.start_date.toordinal() - 1),
                requested_end
            )
            if gap_end >= requested_start:
                missing.append((requested_start, gap_end))
        
        # Check for gap after cached data
        if requested_end > coverage.end_date:
            # Need data after what we have
            gap_start = max(
                date.fromordinal(coverage.end_date.toordinal() + 1),
                requested_start
            )
            if gap_start <= requested_end:
                missing.append((gap_start, requested_end))
        
        return missing
    
    def list_symbols(self) -> list[dict]:
        """List all registered symbols with their metadata."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT s.symbol, s.asset_class, s.name, s.created_at,
                       COUNT(dc.id) as coverage_count
                FROM symbols s
                LEFT JOIN data_coverage dc ON s.id = dc.symbol_id
                GROUP BY s.id
                ORDER BY s.symbol
                """
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def list_coverage(self, symbol: Optional[str] = None) -> list[CoverageInfo]:
        """
        List all coverage records, optionally filtered by symbol.
        
        Parameters
        ----------
        symbol : str, optional
            Filter to specific symbol.
            
        Returns
        -------
        list[CoverageInfo]
            List of coverage records.
        """
        with self._connection() as conn:
            if symbol:
                rows = conn.execute(
                    """
                    SELECT s.symbol, dc.timeframe, dc.start_date, dc.end_date,
                           dc.row_count, dc.file_path, dc.last_updated
                    FROM data_coverage dc
                    JOIN symbols s ON s.id = dc.symbol_id
                    WHERE s.symbol = ?
                    ORDER BY dc.timeframe
                    """,
                    (symbol,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT s.symbol, dc.timeframe, dc.start_date, dc.end_date,
                           dc.row_count, dc.file_path, dc.last_updated
                    FROM data_coverage dc
                    JOIN symbols s ON s.id = dc.symbol_id
                    ORDER BY s.symbol, dc.timeframe
                    """
                ).fetchall()
            
            return [
                CoverageInfo(
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    start_date=date.fromisoformat(row["start_date"]),
                    end_date=date.fromisoformat(row["end_date"]),
                    row_count=row["row_count"],
                    file_path=row["file_path"],
                    last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None,
                )
                for row in rows
            ]
    
    def delete_coverage(self, symbol: str, timeframe: str) -> bool:
        """Delete coverage record (useful for forcing re-fetch)."""
        with self._connection() as conn:
            symbol_id = self.get_symbol_id(symbol)
            if symbol_id is None:
                return False
            
            result = conn.execute(
                "DELETE FROM data_coverage WHERE symbol_id = ? AND timeframe = ?",
                (symbol_id, timeframe),
            )
            return result.rowcount > 0

    # =========================================
    # Feature Cache Methods
    # =========================================
    
    def update_feature_cache(
        self,
        symbol: str,
        timeframe: str,
        feature_set: str,
        params_hash: str,
        params: dict,
        row_count: int,
        file_path: str,
    ) -> None:
        """
        Record a cached feature computation.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str
            Data timeframe.
        feature_set : str
            Name of feature set (e.g., 'simons').
        params_hash : str
            Hash of computation parameters.
        params : dict
            The actual parameters (stored as JSON for reference).
        row_count : int
            Number of rows in the feature DataFrame.
        file_path : str
            Path to the Parquet file.
        """
        import json
        
        symbol_id = self.register_symbol(symbol)
        params_json = json.dumps(params, sort_keys=True)
        
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO feature_cache 
                    (symbol_id, timeframe, feature_set, params_hash, params_json, 
                     row_count, file_path, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol_id, timeframe, feature_set, params_hash) DO UPDATE SET
                    params_json = excluded.params_json,
                    row_count = excluded.row_count,
                    file_path = excluded.file_path,
                    computed_at = CURRENT_TIMESTAMP
                """,
                (symbol_id, timeframe, feature_set, params_hash, params_json, 
                 row_count, file_path),
            )
    
    def get_feature_cache(
        self,
        symbol: str,
        timeframe: str,
        feature_set: str,
        params_hash: str,
    ) -> Optional[FeatureCacheInfo]:
        """
        Check if features are cached for the given parameters.
        
        This is the cache lookup - if it returns a result, the features
        exist and can be loaded from the file_path.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timeframe : str
            Data timeframe.
        feature_set : str
            Name of feature set (e.g., 'simons').
        params_hash : str
            Hash of computation parameters.
            
        Returns
        -------
        FeatureCacheInfo or None
            Cache info if found, None if cache miss.
        """
        import json
        
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 
                    s.symbol,
                    fc.timeframe,
                    fc.feature_set,
                    fc.params_hash,
                    fc.params_json,
                    fc.row_count,
                    fc.file_path,
                    fc.computed_at
                FROM feature_cache fc
                JOIN symbols s ON s.id = fc.symbol_id
                WHERE s.symbol = ? 
                  AND fc.timeframe = ? 
                  AND fc.feature_set = ?
                  AND fc.params_hash = ?
                """,
                (symbol, timeframe, feature_set, params_hash),
            ).fetchone()
            
            if row is None:
                return None
            
            return FeatureCacheInfo(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                feature_set=row["feature_set"],
                params_hash=row["params_hash"],
                params=json.loads(row["params_json"]) if row["params_json"] else {},
                row_count=row["row_count"],
                file_path=row["file_path"],
                computed_at=datetime.fromisoformat(row["computed_at"]) if row["computed_at"] else None,
            )
    
    def list_feature_cache(
        self, 
        symbol: Optional[str] = None,
        feature_set: Optional[str] = None,
    ) -> list[FeatureCacheInfo]:
        """
        List all cached features, optionally filtered.
        
        Parameters
        ----------
        symbol : str, optional
            Filter to specific symbol.
        feature_set : str, optional
            Filter to specific feature set.
            
        Returns
        -------
        list[FeatureCacheInfo]
            List of cached feature records.
        """
        import json
        
        with self._connection() as conn:
            query = """
                SELECT 
                    s.symbol,
                    fc.timeframe,
                    fc.feature_set,
                    fc.params_hash,
                    fc.params_json,
                    fc.row_count,
                    fc.file_path,
                    fc.computed_at
                FROM feature_cache fc
                JOIN symbols s ON s.id = fc.symbol_id
                WHERE 1=1
            """
            params = []
            
            if symbol:
                query += " AND s.symbol = ?"
                params.append(symbol)
            if feature_set:
                query += " AND fc.feature_set = ?"
                params.append(feature_set)
            
            query += " ORDER BY s.symbol, fc.timeframe, fc.feature_set"
            
            rows = conn.execute(query, params).fetchall()
            
            return [
                FeatureCacheInfo(
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    feature_set=row["feature_set"],
                    params_hash=row["params_hash"],
                    params=json.loads(row["params_json"]) if row["params_json"] else {},
                    row_count=row["row_count"],
                    file_path=row["file_path"],
                    computed_at=datetime.fromisoformat(row["computed_at"]) if row["computed_at"] else None,
                )
                for row in rows
            ]
    
    def delete_feature_cache(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        feature_set: Optional[str] = None,
        params_hash: Optional[str] = None,
    ) -> int:
        """
        Delete cached features matching the criteria.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol (required).
        timeframe : str, optional
            Filter by timeframe.
        feature_set : str, optional
            Filter by feature set.
        params_hash : str, optional
            Filter by specific params hash.
            
        Returns
        -------
        int
            Number of records deleted.
        """
        with self._connection() as conn:
            symbol_id = self.get_symbol_id(symbol)
            if symbol_id is None:
                return 0
            
            query = "DELETE FROM feature_cache WHERE symbol_id = ?"
            params = [symbol_id]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
            if feature_set:
                query += " AND feature_set = ?"
                params.append(feature_set)
            if params_hash:
                query += " AND params_hash = ?"
                params.append(params_hash)
            
            result = conn.execute(query, params)
            return result.rowcount


if __name__ == "__main__":
    # Quick test
    from datetime import date
    
    db = MetadataDB(Path("./data/metadata.db"))
    
    # Register symbol
    sym_id = db.register_symbol("BTC-USD", "crypto", "Bitcoin USD")
    print(f"Symbol ID: {sym_id}")
    
    # Update coverage
    db.update_coverage(
        "BTC-USD", "1d",
        date(2023, 1, 1), date(2024, 12, 1),
        row_count=700,
        file_path="market/BTC-USD/1d.parquet",
    )
    
    # Get coverage
    coverage = db.get_coverage("BTC-USD", "1d")
    print(f"Coverage: {coverage}")
    
    # Test missing ranges
    missing = db.get_missing_ranges(
        "BTC-USD", "1d",
        date(2022, 6, 1), date(2025, 1, 1),
    )
    print(f"Missing ranges: {missing}")
    
    # List all
    print(f"Symbols: {db.list_symbols()}")
    print(f"Coverage: {db.list_coverage()}")

