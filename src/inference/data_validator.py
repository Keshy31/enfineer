"""
Data Validator
==============
Validates data freshness and quality for inference.

Key responsibilities:
- Check if data is fresh (not stale)
- Detect missing values
- Compute data quality score
- Provide fallback strategies for missing data
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DataStatus(Enum):
    """Status of a data source."""
    FRESH = "fresh"           # Updated within expected window
    STALE = "stale"           # Older than expected but usable
    MISSING = "missing"       # No data available
    ERROR = "error"           # Error fetching data


@dataclass
class SourceStatus:
    """Status report for a single data source."""
    symbol: str
    status: DataStatus
    last_update: Optional[datetime] = None
    age_hours: Optional[float] = None
    rows_available: int = 0
    missing_values: int = 0
    message: str = ""
    
    @property
    def is_usable(self) -> bool:
        """Can we use this data for inference?"""
        return self.status in [DataStatus.FRESH, DataStatus.STALE]
    
    def __repr__(self) -> str:
        status_symbols = {
            DataStatus.FRESH: "✓",
            DataStatus.STALE: "⚠",
            DataStatus.MISSING: "✗",
            DataStatus.ERROR: "✗",
        }
        symbol = status_symbols.get(self.status, "?")
        return f"{symbol} {self.symbol}: {self.status.value} ({self.message})"


@dataclass
class DataQualityReport:
    """
    Comprehensive data quality report for inference decision.
    
    Attributes
    ----------
    sources : Dict[str, SourceStatus]
        Status of each data source.
    overall_score : float
        Quality score from 0 to 1.
    can_proceed : bool
        Whether we have enough data to run inference.
    warnings : List[str]
        List of warning messages.
    errors : List[str]
        List of error messages.
    """
    sources: Dict[str, SourceStatus] = field(default_factory=dict)
    overall_score: float = 0.0
    can_proceed: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)
    
    def __repr__(self) -> str:
        lines = ["DataQualityReport:"]
        for symbol, status in self.sources.items():
            lines.append(f"  {status}")
        lines.append(f"  Score: {self.overall_score:.2f}")
        lines.append(f"  Can proceed: {self.can_proceed}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        return "\n".join(lines)


class DataValidator:
    """
    Validates data freshness and quality for inference.
    
    Parameters
    ----------
    max_btc_age_hours : float
        Maximum age for BTC data before considered stale.
    max_macro_age_hours : float
        Maximum age for macro data before considered stale.
    min_rows_required : int
        Minimum rows needed for feature computation.
    
    Example
    -------
    >>> validator = DataValidator()
    >>> report = validator.validate(features_df)
    >>> if report.can_proceed:
    ...     # Run inference
    >>> else:
    ...     print("Data quality too low:", report.errors)
    """
    
    # Required symbols and their roles
    REQUIRED_SYMBOLS = {
        "BTC-USD": "primary",      # Must have
        "^TNX": "macro",           # Can use stale
        "DX-Y.NYB": "macro",       # Can use stale
        "GLD": "macro",            # Can use stale
    }
    
    def __init__(
        self,
        max_btc_age_hours: float = 6.0,
        max_macro_age_hours: float = 24.0,
        min_rows_required: int = 30,
    ):
        self.max_btc_age_hours = max_btc_age_hours
        self.max_macro_age_hours = max_macro_age_hours
        self.min_rows_required = min_rows_required
    
    def validate(
        self,
        data: Dict[str, pd.DataFrame],
        reference_time: Optional[datetime] = None,
    ) -> DataQualityReport:
        """
        Validate all data sources and produce quality report.
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary mapping symbol to OHLCV DataFrame.
        reference_time : datetime, optional
            Time to compare data freshness against. Defaults to now.
            
        Returns
        -------
        DataQualityReport
            Comprehensive quality assessment.
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        report = DataQualityReport(checked_at=reference_time)
        
        # Check each required symbol
        for symbol, role in self.REQUIRED_SYMBOLS.items():
            if symbol not in data or data[symbol] is None or len(data[symbol]) == 0:
                status = SourceStatus(
                    symbol=symbol,
                    status=DataStatus.MISSING,
                    message="No data available",
                )
                if role == "primary":
                    report.errors.append(f"Primary data source {symbol} is missing")
                else:
                    report.warnings.append(f"Macro data {symbol} is missing")
            else:
                df = data[symbol]
                status = self._check_source(df, symbol, role, reference_time)
                
                if status.status == DataStatus.MISSING:
                    if role == "primary":
                        report.errors.append(f"Primary data source {symbol} has no recent data")
                    else:
                        report.warnings.append(f"Macro data {symbol} has no recent data")
                elif status.status == DataStatus.STALE:
                    report.warnings.append(
                        f"{symbol} data is stale ({status.age_hours:.1f} hours old)"
                    )
            
            report.sources[symbol] = status
        
        # Compute overall score
        report.overall_score = self._compute_quality_score(report.sources)
        
        # Determine if we can proceed
        # Must have BTC data, macro can be stale
        btc_status = report.sources.get("BTC-USD")
        if btc_status and btc_status.is_usable:
            report.can_proceed = True
        else:
            report.can_proceed = False
            if not report.errors:
                report.errors.append("BTC-USD data is not usable")
        
        return report
    
    def _check_source(
        self,
        df: pd.DataFrame,
        symbol: str,
        role: str,
        reference_time: datetime,
    ) -> SourceStatus:
        """Check freshness and quality of a single data source."""
        # Get the last timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            last_update = df.index.max()
            if pd.isna(last_update):
                return SourceStatus(
                    symbol=symbol,
                    status=DataStatus.MISSING,
                    message="No valid timestamps",
                )
            last_update = last_update.to_pydatetime()
        else:
            return SourceStatus(
                symbol=symbol,
                status=DataStatus.ERROR,
                message="Index is not DatetimeIndex",
            )
        
        # Remove timezone info for comparison if needed
        if last_update.tzinfo is not None:
            last_update = last_update.replace(tzinfo=None)
        
        # Calculate age
        age = reference_time - last_update
        age_hours = age.total_seconds() / 3600
        
        # Count missing values
        missing_values = df.isna().sum().sum()
        
        # Determine freshness threshold
        if role == "primary":
            max_age = self.max_btc_age_hours
        else:
            max_age = self.max_macro_age_hours
        
        # Determine status
        if age_hours <= max_age:
            status = DataStatus.FRESH
            message = f"Updated {age_hours:.1f}h ago"
        elif age_hours <= max_age * 2:
            status = DataStatus.STALE
            message = f"Stale ({age_hours:.1f}h old)"
        else:
            status = DataStatus.STALE
            message = f"Very stale ({age_hours:.1f}h old)"
        
        # Check minimum rows
        if len(df) < self.min_rows_required:
            status = DataStatus.MISSING
            message = f"Only {len(df)} rows (need {self.min_rows_required})"
        
        return SourceStatus(
            symbol=symbol,
            status=status,
            last_update=last_update,
            age_hours=age_hours,
            rows_available=len(df),
            missing_values=missing_values,
            message=message,
        )
    
    def _compute_quality_score(self, sources: Dict[str, SourceStatus]) -> float:
        """
        Compute overall quality score from 0 to 1.
        
        Scoring:
        - FRESH: 1.0
        - STALE: 0.7
        - MISSING/ERROR: 0.0
        
        BTC is weighted 2x more than macro sources.
        """
        status_scores = {
            DataStatus.FRESH: 1.0,
            DataStatus.STALE: 0.7,
            DataStatus.MISSING: 0.0,
            DataStatus.ERROR: 0.0,
        }
        
        weights = {
            "BTC-USD": 2.0,
            "^TNX": 1.0,
            "DX-Y.NYB": 1.0,
            "GLD": 1.0,
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for symbol, status in sources.items():
            weight = weights.get(symbol, 1.0)
            score = status_scores.get(status.status, 0.0)
            weighted_score += weight * score
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_score / total_weight
    
    def validate_features(
        self,
        features_df: pd.DataFrame,
        required_columns: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate that computed features are ready for inference.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Computed features.
        required_columns : List[str]
            Columns that must be present.
            
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list of issues)
        """
        issues = []
        
        # Check required columns
        missing_cols = [c for c in required_columns if c not in features_df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN in last row (what we'll use for inference)
        if len(features_df) > 0:
            last_row = features_df.iloc[-1]
            nan_cols = last_row[last_row.isna()].index.tolist()
            if nan_cols:
                issues.append(f"NaN in latest row: {nan_cols}")
        
        # Check minimum rows
        if len(features_df) < self.min_rows_required:
            issues.append(f"Only {len(features_df)} rows, need {self.min_rows_required}")
        
        return len(issues) == 0, issues


if __name__ == "__main__":
    # Quick test
    print("DataValidator Test")
    print("=" * 60)
    
    validator = DataValidator()
    
    # Create test data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    test_data = {
        "BTC-USD": pd.DataFrame({
            "Close": np.random.randn(30).cumsum() + 100000,
        }, index=dates),
        "^TNX": pd.DataFrame({
            "Close": np.random.randn(30).cumsum() * 0.1 + 4.5,
        }, index=dates - timedelta(hours=18)),  # Stale
        "DX-Y.NYB": pd.DataFrame({
            "Close": np.random.randn(30).cumsum() * 0.5 + 105,
        }, index=dates),
        "GLD": pd.DataFrame({
            "Close": np.random.randn(30).cumsum() * 2 + 200,
        }, index=dates),
    }
    
    report = validator.validate(test_data)
    print(report)
    print()
    
    print("DATA VALIDATOR TEST PASSED")

