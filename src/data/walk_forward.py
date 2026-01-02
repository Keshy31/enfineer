"""
Walk-Forward Validation Module
==============================
Implements proper out-of-sample testing for regime detection.

The critical insight: You CANNOT fit a model on all data then backtest
on the same data. This creates look-ahead bias where the model "knows"
the future while classifying the past.

Walk-Forward Protocol:
1. Train on historical window [0, T]
2. Predict on UNSEEN future [T+gap, T+gap+test_window]
3. Roll forward, expand training window
4. Stitch OOS predictions to build true equity curve
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from datetime import datetime


@dataclass
class WalkForwardFoldResult:
    """Results from a single walk-forward fold."""
    fold_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    # Model artifacts (fitted on train only)
    scaler: StandardScaler
    pca: Optional[PCA]
    gmm: GaussianMixture
    
    # Out-of-sample predictions
    test_labels: np.ndarray
    test_probabilities: np.ndarray
    test_dates: pd.DatetimeIndex
    
    # Metrics
    train_samples: int
    test_samples: int


@dataclass 
class WalkForwardResult:
    """Aggregated results from walk-forward validation."""
    folds: List[WalkForwardFoldResult]
    
    # Stitched OOS results
    oos_labels: np.ndarray
    oos_dates: pd.DatetimeIndex
    oos_probabilities: np.ndarray
    
    # Config
    n_regimes: int
    use_pca: bool
    pca_variance: float
    
    def __repr__(self) -> str:
        return (
            f"WalkForwardResult(\n"
            f"  folds: {len(self.folds)}\n"
            f"  oos_samples: {len(self.oos_labels)}\n"
            f"  date_range: {self.oos_dates.min().date()} to {self.oos_dates.max().date()}\n"
            f"  n_regimes: {self.n_regimes}\n"
            f")"
        )


class WalkForwardGMM:
    """
    Walk-Forward Gaussian Mixture Model for regime detection.
    
    Implements expanding window training to eliminate look-ahead bias:
    - Fit scaler/PCA/GMM ONLY on past data
    - Predict regimes on UNSEEN future data
    - Roll forward and repeat
    
    Parameters
    ----------
    n_regimes : int, default=5
        Number of GMM clusters. Use find_optimal_k() to determine.
    min_train_days : int, default=504
        Minimum training window (2 years of trading days).
    test_days : int, default=126
        Test window size (6 months).
    gap_days : int, default=30
        Gap between train and test to prevent feature leakage.
    use_pca : bool, default=True
        Whether to apply PCA before GMM (reduces overfitting).
    pca_variance : float, default=0.95
        Variance to retain if using PCA.
    covariance_type : str, default='diag'
        GMM covariance type. 'diag' is more robust with limited data.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_regimes: int = 5,
        min_train_days: int = 504,
        test_days: int = 126,
        gap_days: int = 30,
        use_pca: bool = True,
        pca_variance: float = 0.95,
        covariance_type: str = 'diag',
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.gap_days = gap_days
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        self.results_: Optional[WalkForwardResult] = None
    
    def fit_predict(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Feature DataFrame with DatetimeIndex.
        feature_cols : list[str]
            Columns to use for clustering (must be stationary!).
            
        Returns
        -------
        WalkForwardResult
            Contains stitched OOS predictions and fold details.
        """
        df = features_df.sort_index()
        
        # Validate
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        X_full = df[feature_cols].values
        dates = df.index
        
        n_samples = len(X_full)
        
        if n_samples < self.min_train_days + self.gap_days + self.test_days:
            raise ValueError(
                f"Insufficient data: have {n_samples} samples, "
                f"need at least {self.min_train_days + self.gap_days + self.test_days}"
            )
        
        folds = []
        fold_num = 1
        
        # Start with minimum training window
        train_end_idx = self.min_train_days
        
        print(f"Walk-Forward GMM ({self.n_regimes} regimes, PCA={self.use_pca})")
        print(f"  Min train: {self.min_train_days} days, Test: {self.test_days} days, Gap: {self.gap_days} days")
        print()
        
        while True:
            # Calculate indices
            test_start_idx = train_end_idx + self.gap_days
            test_end_idx = test_start_idx + self.test_days
            
            # Stop if we've run out of data
            if test_end_idx > n_samples:
                break
            
            # Extract train/test data
            X_train = X_full[:train_end_idx]
            X_test = X_full[test_start_idx:test_end_idx]
            
            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            
            # Fit scaler on TRAIN ONLY
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Optional PCA on TRAIN ONLY
            pca = None
            if self.use_pca:
                pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
                X_train_reduced = pca.fit_transform(X_train_scaled)
                X_test_reduced = pca.transform(X_test_scaled)
            else:
                X_train_reduced = X_train_scaled
                X_test_reduced = X_test_scaled
            
            # Fit GMM on TRAIN ONLY
            gmm = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=10,
                max_iter=200,
            )
            gmm.fit(X_train_reduced)
            
            # Predict on TEST (out-of-sample!)
            test_labels = gmm.predict(X_test_reduced)
            test_probs = gmm.predict_proba(X_test_reduced)
            
            # Store fold result
            fold = WalkForwardFoldResult(
                fold_num=fold_num,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                scaler=scaler,
                pca=pca,
                gmm=gmm,
                test_labels=test_labels,
                test_probabilities=test_probs,
                test_dates=test_dates,
                train_samples=len(X_train),
                test_samples=len(X_test),
            )
            folds.append(fold)
            
            pca_dims = pca.n_components_ if pca else X_train_scaled.shape[1]
            print(f"  Fold {fold_num}: Train {len(X_train)} -> Test {len(X_test)} "
                  f"({test_dates[0].date()} to {test_dates[-1].date()}) "
                  f"[{pca_dims} dims]")
            
            # Roll forward (expand training window)
            train_end_idx = test_end_idx
            fold_num += 1
        
        if len(folds) == 0:
            raise ValueError("No folds generated. Need more data.")
        
        # Stitch OOS predictions
        oos_labels = np.concatenate([f.test_labels for f in folds])
        oos_dates = pd.DatetimeIndex(np.concatenate([f.test_dates for f in folds]))
        oos_probs = np.vstack([f.test_probabilities for f in folds])
        
        print(f"\n  Total OOS samples: {len(oos_labels)}")
        print(f"  OOS date range: {oos_dates.min().date()} to {oos_dates.max().date()}")
        
        self.results_ = WalkForwardResult(
            folds=folds,
            oos_labels=oos_labels,
            oos_dates=oos_dates,
            oos_probabilities=oos_probs,
            n_regimes=self.n_regimes,
            use_pca=self.use_pca,
            pca_variance=self.pca_variance,
        )
        
        return self.results_


def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    method: str = 'bic',
    covariance_type: str = 'diag',
    use_pca: bool = True,
    pca_variance: float = 0.95,
) -> Tuple[int, pd.DataFrame]:
    """
    Find optimal number of clusters using information criteria.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (already scaled).
    k_range : range
        Range of k values to test.
    method : str
        'bic' (Bayesian Information Criterion) or 'aic' (Akaike).
        Lower is better for both.
    covariance_type : str
        GMM covariance type.
    use_pca : bool
        Whether to apply PCA first.
    pca_variance : float
        Variance to retain if using PCA.
        
    Returns
    -------
    optimal_k : int
        Best number of clusters.
    scores_df : pd.DataFrame
        Scores for each k value.
    """
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optional PCA
    if use_pca:
        pca = PCA(n_components=pca_variance)
        X_reduced = pca.fit_transform(X_scaled)
        print(f"PCA: {X.shape[1]} -> {X_reduced.shape[1]} dimensions ({pca_variance*100:.0f}% variance)")
    else:
        X_reduced = X_scaled
    
    results = []
    
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=10,
            max_iter=200,
            random_state=42,
        )
        gmm.fit(X_reduced)
        
        bic = gmm.bic(X_reduced)
        aic = gmm.aic(X_reduced)
        
        results.append({
            'k': k,
            'bic': bic,
            'aic': aic,
            'converged': gmm.converged_,
            'n_iter': gmm.n_iter_,
        })
    
    scores_df = pd.DataFrame(results)
    
    # Find optimal
    if method == 'bic':
        optimal_k = scores_df.loc[scores_df['bic'].idxmin(), 'k']
    else:
        optimal_k = scores_df.loc[scores_df['aic'].idxmin(), 'k']
    
    return int(optimal_k), scores_df


def get_stationary_features() -> List[str]:
    """
    Return ONLY stationary features safe for regime clustering.
    
    Excludes:
    - Raw price levels (Close, Open, etc.)
    - Raw yield levels (yield_10y)
    - Raw index levels
    
    Includes:
    - Returns (log_return, dxy_return, gold_return)
    - Z-scores (all *_zscore features)
    - Momentum (relative measures)
    - Correlations (bounded -1 to 1)
    """
    return [
        # Price-based (Simons) - all stationary
        "log_return",
        "log_return_zscore",
        "volatility",
        "volatility_zscore",
        "volume_zscore",
        "high_low_range",
        "range_zscore",
        "momentum",
        "momentum_zscore",
        
        # Yield features - ONLY differenced/relative
        # "yield_10y",        # EXCLUDED: non-stationary level
        "yield_change_1d",    # Differenced: stationary
        "yield_change_5d",    # Differenced: stationary  
        "yield_change_20d",   # Differenced: stationary
        "yield_change_zscore",# Z-scored: stationary
        "yield_vs_ma",        # Relative to MA: stationary
        "yield_momentum",     # Rate of change: stationary
        
        # Dollar features - ONLY returns/relative
        "dxy_return",         # Return: stationary
        "dxy_zscore",         # Z-scored: stationary
        "dxy_momentum_5d",    # Momentum: stationary
        "dxy_momentum_20d",   # Momentum: stationary
        "dxy_vs_ma",          # Relative to MA: stationary
        
        # Gold features - ONLY returns/relative  
        "gold_return",        # Return: stationary
        "gold_zscore",        # Z-scored: stationary
        "gold_momentum_5d",   # Momentum: stationary
        "gold_momentum_20d",  # Momentum: stationary
        
        # Cross-asset correlations - bounded [-1, 1]: stationary
        "btc_gold_corr",
        "btc_dxy_corr",
        "btc_yield_corr",
        
        # Composite indicators
        "risk_on_score",      # Derived from z-scores: stationary
        # "macro_regime",     # EXCLUDED: categorical, not continuous
    ]


def validate_stationarity(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Quick stationarity check using rolling statistics.
    
    A properly stationary feature should have relatively stable
    mean and variance over time.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature data.
    feature_cols : list[str]
        Columns to check.
    threshold : float
        Max acceptable drift in mean (as fraction of std).
        
    Returns
    -------
    pd.DataFrame
        Stationarity report for each feature.
    """
    df = features_df[feature_cols].copy()
    
    # Split into halves
    mid = len(df) // 2
    first_half = df.iloc[:mid]
    second_half = df.iloc[mid:]
    
    results = []
    for col in feature_cols:
        first_mean = first_half[col].mean()
        second_mean = second_half[col].mean()
        overall_std = df[col].std()
        
        # Mean drift as fraction of std
        mean_drift = abs(second_mean - first_mean) / (overall_std + 1e-8)
        
        # Variance ratio
        var_ratio = second_half[col].var() / (first_half[col].var() + 1e-8)
        
        results.append({
            'feature': col,
            'first_half_mean': first_mean,
            'second_half_mean': second_mean,
            'mean_drift_std': mean_drift,
            'variance_ratio': var_ratio,
            'likely_stationary': mean_drift < threshold and 0.5 < var_ratio < 2.0,
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Quick test
    print("Walk-Forward GMM Module Test")
    print("=" * 60)
    
    # Import here to avoid circular dependency
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.manager import DataManager
    
    # Load data
    dm = DataManager("./data")
    features = dm.get_features("BTC-USD", "1d", feature_set="combined", start="2020-01-01")
    
    # Get stationary features only
    feature_cols = get_stationary_features()
    available_cols = [c for c in feature_cols if c in features.columns]
    
    print(f"\nUsing {len(available_cols)} stationary features:")
    for col in available_cols:
        print(f"  • {col}")
    
    # Validate stationarity
    print("\nStationarity Check:")
    stat_report = validate_stationarity(features, available_cols)
    non_stationary = stat_report[~stat_report['likely_stationary']]
    if len(non_stationary) > 0:
        print("  ⚠ Potentially non-stationary features:")
        for _, row in non_stationary.iterrows():
            print(f"    - {row['feature']}: drift={row['mean_drift_std']:.2f}σ")
    else:
        print("  ✓ All features appear stationary")
    
    # Find optimal k
    print("\nFinding optimal cluster count...")
    X = features[available_cols].values
    optimal_k, scores = find_optimal_k(X, k_range=range(2, 9), use_pca=True)
    print(f"\nBIC Scores:")
    print(scores.to_string(index=False))
    print(f"\n  ✓ Optimal k by BIC: {optimal_k}")
    
    # Run walk-forward
    print("\n" + "=" * 60)
    print("Walk-Forward Validation")
    print("=" * 60 + "\n")
    
    wf = WalkForwardGMM(
        n_regimes=optimal_k,
        min_train_days=504,  # 2 years
        test_days=126,       # 6 months
        gap_days=30,
        use_pca=True,
    )
    
    results = wf.fit_predict(features, available_cols)
    print(f"\n{results}")

