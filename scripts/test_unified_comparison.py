"""
Unified Walk-Forward Comparison Framework
==========================================
Fair comparison of regime detection methods using identical walk-forward splits.

This is the CORRECT way to compare methods:
- All methods use the SAME train/test splits
- All methods are evaluated on the SAME OOS dates
- No in-sample fitting on test data

Methods Compared:
1. Random Baseline - sanity check (should have ~0 Sharpe spread)
2. PCA + GMM - linear dimensionality reduction baseline
3. AE + GMM - nonlinear learned representation (configurable latent dim)

Run from project root:
    python scripts/test_unified_comparison.py
    python scripts/test_unified_comparison.py --latent_dims 8,12
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data.manager import DataManager
from data.training import (
    create_training_dataset,
    get_temporal_feature_cols,
    get_macro_feature_cols,
)
from data.walk_forward import get_stationary_features
from models.autoencoder import HedgeFundBrain, AutoencoderConfig
from models.trainer import AutoencoderTrainer, TrainingConfig, create_data_loaders
from analysis.statistical_tests import bootstrap_sharpe_ci, compute_regime_transition_matrix
from backtest.costs import COST_PRESETS, analyze_cost_impact


@dataclass
class MethodResult:
    """Results from a single method on a single fold."""
    method_name: str
    fold_num: int
    test_labels: np.ndarray
    test_dates: pd.DatetimeIndex
    n_regimes: int
    
    # Computed after returns are added
    sharpe_spread: Optional[float] = None
    persistence: Optional[float] = None
    n_significant: Optional[int] = None


@dataclass 
class UnifiedComparisonResult:
    """Aggregated results from unified comparison."""
    method_results: Dict[str, List[MethodResult]]  # method_name -> list of fold results
    
    # Aggregated OOS results per method
    oos_labels: Dict[str, np.ndarray]
    oos_dates: pd.DatetimeIndex
    
    # Comparison metrics
    comparison_df: pd.DataFrame
    
    def __repr__(self) -> str:
        methods = list(self.method_results.keys())
        n_folds = len(self.method_results[methods[0]]) if methods else 0
        return (
            f"UnifiedComparisonResult(\n"
            f"  methods: {methods}\n"
            f"  folds: {n_folds}\n"
            f"  oos_samples: {len(self.oos_dates)}\n"
            f")"
        )


def create_walk_forward_splits(
    n_samples: int,
    min_train: int = 504,
    test_size: int = 126,
    gap: int = 30,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward train/test index splits.
    
    Returns list of (train_indices, test_indices) tuples.
    """
    splits = []
    train_end = min_train
    
    while True:
        test_start = train_end + gap
        test_end = test_start + test_size
        
        if test_end > n_samples:
            break
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
        
        # Expand training window
        train_end = test_end
    
    return splits


def method_random(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_regimes: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """Random baseline - assigns random regime labels."""
    rng = np.random.RandomState(random_state)
    return rng.randint(0, n_regimes, size=len(X_test))


def method_pca_gmm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_regimes: int = 8,
    pca_variance: float = 0.95,
    random_state: int = 42,
) -> np.ndarray:
    """PCA + GMM baseline - linear dimensionality reduction."""
    # Fit scaler on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit PCA on train
    pca = PCA(n_components=pca_variance, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Fit GMM on train
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='diag',
        random_state=random_state,
        n_init=10,
        max_iter=200,
    )
    gmm.fit(X_train_pca)
    
    # Predict on test (OOS!)
    return gmm.predict(X_test_pca)


def method_ae_gmm(
    X_temporal_train: np.ndarray,
    X_macro_train: np.ndarray,
    X_temporal_test: np.ndarray,
    X_macro_test: np.ndarray,
    latent_dim: int = 8,
    n_regimes: int = 8,
    epochs: int = 50,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autoencoder + GMM - nonlinear learned representation."""
    
    # Normalize using train statistics
    temp_mean = X_temporal_train.mean(axis=(0, 1))
    temp_std = X_temporal_train.std(axis=(0, 1))
    temp_std = np.where(temp_std < 1e-8, 1.0, temp_std)
    
    macro_mean = X_macro_train.mean(axis=0)
    macro_std = X_macro_train.std(axis=0)
    macro_std = np.where(macro_std < 1e-8, 1.0, macro_std)
    
    X_temp_train_norm = ((X_temporal_train - temp_mean) / temp_std).astype(np.float32)
    X_temp_test_norm = ((X_temporal_test - temp_mean) / temp_std).astype(np.float32)
    X_macro_train_norm = ((X_macro_train - macro_mean) / macro_std).astype(np.float32)
    X_macro_test_norm = ((X_macro_test - macro_mean) / macro_std).astype(np.float32)
    
    # Create model
    model_config = AutoencoderConfig(
        temporal_features=X_temporal_train.shape[2],
        macro_features=X_macro_train.shape[1],
        sequence_length=X_temporal_train.shape[1],
        latent_dim=latent_dim,
        lstm_hidden=64,
        lstm_layers=2,
        macro_hidden=32,
        dropout=0.2,
    )
    
    train_config = TrainingConfig(
        epochs=epochs,
        early_stopping_patience=10,
        learning_rate=1e-3,
        batch_size=32,
        lambda_macro=2.0,
    )
    
    model = HedgeFundBrain(model_config)
    trainer = AutoencoderTrainer(model, train_config)
    
    # Create data loaders
    train_loader = create_data_loaders(
        X_temp_train_norm, X_macro_train_norm, batch_size=32, shuffle=True
    )
    
    # Use last 20% of train as validation
    val_split = int(len(X_temp_train_norm) * 0.8)
    val_loader = create_data_loaders(
        X_temp_train_norm[val_split:], X_macro_train_norm[val_split:],
        batch_size=32, shuffle=False
    )
    
    # Train
    if verbose:
        print(f"    Training AE({latent_dim}D)...")
    trainer.fit(train_loader, val_loader, fold_num=0, verbose=verbose)
    
    # Encode train and test
    full_train_loader = create_data_loaders(
        X_temp_train_norm, X_macro_train_norm, batch_size=32, shuffle=False
    )
    test_loader = create_data_loaders(
        X_temp_test_norm, X_macro_test_norm, batch_size=32, shuffle=False
    )
    
    latents_train = trainer.encode_dataset(full_train_loader)
    latents_test = trainer.encode_dataset(test_loader)
    
    # Fit GMM on train latents
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='diag',  # Use diag for higher dims
        random_state=random_state,
        n_init=10,
        max_iter=200,
    )
    gmm.fit(latents_train)
    
    # Predict on test (OOS!)
    return gmm.predict(latents_test), latents_test


def compute_regime_metrics(
    labels: np.ndarray,
    returns: np.ndarray,
    n_regimes: int,
    forward_days: int = 5,
) -> Dict:
    """Compute regime detection metrics."""
    
    # Group returns by regime
    sharpes = []
    for regime in range(n_regimes):
        mask = labels == regime
        regime_returns = returns[mask]
        if len(regime_returns) > 10:
            sharpe = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252 / forward_days)
            sharpes.append(sharpe)
    
    sharpe_spread = max(sharpes) - min(sharpes) if len(sharpes) >= 2 else 0
    
    # Compute persistence
    _, trans_stats = compute_regime_transition_matrix(labels, n_regimes=n_regimes)
    persistence = trans_stats['avg_persistence']
    
    return {
        'sharpe_spread': sharpe_spread,
        'persistence': persistence,
        'n_regimes_active': len(sharpes),
    }


def run_unified_comparison(
    latent_dims: List[int] = [8, 12],
    n_regimes: int = 8,
    ae_epochs: int = 50,
    verbose: bool = True,
) -> UnifiedComparisonResult:
    """
    Run unified walk-forward comparison of all methods.
    
    Parameters
    ----------
    latent_dims : List[int]
        Latent dimensions to test for autoencoder
    n_regimes : int
        Number of GMM clusters
    ae_epochs : int
        Training epochs per fold for autoencoder
    verbose : bool
        Print progress
        
    Returns
    -------
    UnifiedComparisonResult
    """
    
    print("=" * 70)
    print("UNIFIED WALK-FORWARD COMPARISON")
    print("=" * 70)
    print()
    
    # =========================================
    # Load Data
    # =========================================
    print("[1/4] Loading data...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        "BTC-USD", "1d",
        feature_set="combined",
        start="2020-01-01",  # Effective start after warmup
    )
    
    # Get feature columns
    temporal_cols = get_temporal_feature_cols()
    macro_cols = [c for c in get_macro_feature_cols() if c in features.columns]
    stationary_cols = [c for c in get_stationary_features() if c in features.columns]
    macro_cols = [c for c in macro_cols if c in stationary_cols]
    
    # Create training dataset for AE
    dataset = create_training_dataset(
        features,
        sequence_length=30,
        temporal_cols=temporal_cols,
        macro_cols=macro_cols,
    )
    
    # Get stationary features for PCA+GMM baseline
    X_stationary = features[stationary_cols].values
    
    print(f"  Samples: {len(dataset)}")
    print(f"  Date range: {dataset.dates[0].date()} to {dataset.dates[-1].date()}")
    print(f"  Stationary features: {len(stationary_cols)}")
    print(f"  Temporal features: {len(temporal_cols)}")
    print(f"  Macro features: {len(macro_cols)}")
    
    # =========================================
    # Create Walk-Forward Splits
    # =========================================
    print("\n[2/4] Creating walk-forward splits...")
    
    splits = create_walk_forward_splits(
        n_samples=len(dataset),
        min_train=504,
        test_size=126,
        gap=30,
    )
    
    print(f"  Created {len(splits)} folds")
    
    # =========================================
    # Run All Methods Through Same Splits
    # =========================================
    print("\n[3/4] Running methods through identical splits...")
    print()
    
    # Methods to compare
    method_names = ['Random', 'PCA+GMM'] + [f'AE({d}D)+GMM' for d in latent_dims]
    
    # Store results
    method_results = {name: [] for name in method_names}
    all_test_dates = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        
        # Get test dates for this fold
        test_dates = pd.DatetimeIndex(dataset.dates[test_idx])
        all_test_dates.append(test_dates)
        
        if verbose:
            print(f"  Fold {fold_num}/{len(splits)}: "
                  f"Train {len(train_idx)} -> Test {len(test_idx)} "
                  f"({test_dates[0].date()} to {test_dates[-1].date()})")
        
        # Prepare data
        X_stat_train = X_stationary[train_idx]
        X_stat_test = X_stationary[test_idx]
        
        X_temp_train = dataset.X_temporal[train_idx]
        X_temp_test = dataset.X_temporal[test_idx]
        X_macro_train = dataset.X_macro[train_idx]
        X_macro_test = dataset.X_macro[test_idx]
        
        # Method 1: Random
        labels_random = method_random(X_stat_train, X_stat_test, n_regimes)
        method_results['Random'].append(MethodResult(
            method_name='Random',
            fold_num=fold_num,
            test_labels=labels_random,
            test_dates=test_dates,
            n_regimes=n_regimes,
        ))
        
        # Method 2: PCA + GMM
        labels_pca = method_pca_gmm(X_stat_train, X_stat_test, n_regimes)
        method_results['PCA+GMM'].append(MethodResult(
            method_name='PCA+GMM',
            fold_num=fold_num,
            test_labels=labels_pca,
            test_dates=test_dates,
            n_regimes=n_regimes,
        ))
        
        # Method 3+: AE + GMM for each latent dim
        for latent_dim in latent_dims:
            method_name = f'AE({latent_dim}D)+GMM'
            labels_ae, _ = method_ae_gmm(
                X_temp_train, X_macro_train,
                X_temp_test, X_macro_test,
                latent_dim=latent_dim,
                n_regimes=n_regimes,
                epochs=ae_epochs,
                verbose=False,
            )
            method_results[method_name].append(MethodResult(
                method_name=method_name,
                fold_num=fold_num,
                test_labels=labels_ae,
                test_dates=test_dates,
                n_regimes=n_regimes,
            ))
    
    # Stitch OOS labels
    oos_dates = pd.DatetimeIndex(np.concatenate(all_test_dates))
    oos_labels = {
        name: np.concatenate([r.test_labels for r in results])
        for name, results in method_results.items()
    }
    
    print(f"\n  Total OOS samples: {len(oos_dates)}")
    
    # =========================================
    # Compute Comparison Metrics
    # =========================================
    print("\n[4/4] Computing comparison metrics...")
    
    # Get returns for OOS period
    oos_features = features.loc[oos_dates]
    forward_returns = oos_features['Close'].shift(-5) / oos_features['Close'] - 1
    forward_returns = forward_returns.values
    
    comparison_data = []
    
    for method_name in method_names:
        labels = oos_labels[method_name]
        metrics = compute_regime_metrics(labels, forward_returns, n_regimes)
        
        # Count significant regimes
        n_significant = 0
        for regime in range(n_regimes):
            mask = labels == regime
            regime_returns = forward_returns[mask]
            if len(regime_returns) > 10:
                try:
                    result = bootstrap_sharpe_ci(regime_returns, annualization=252/5, n_bootstrap=1000)
                    if result.significant:
                        n_significant += 1
                except:
                    pass
        
        comparison_data.append({
            'Method': method_name,
            'Sharpe Spread': metrics['sharpe_spread'],
            'Persistence': metrics['persistence'],
            'Significant Regimes': n_significant,
            'Active Regimes': metrics['n_regimes_active'],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Sharpe Spread', ascending=False)
    
    return UnifiedComparisonResult(
        method_results=method_results,
        oos_labels=oos_labels,
        oos_dates=oos_dates,
        comparison_df=comparison_df,
    )


def print_comparison_report(result: UnifiedComparisonResult):
    """Print formatted comparison report."""
    
    print("\n" + "=" * 70)
    print("UNIFIED COMPARISON RESULTS")
    print("=" * 70)
    print()
    
    # Main comparison table
    print("Method Comparison (sorted by Sharpe Spread):")
    print("-" * 70)
    print(result.comparison_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Determine winner
    print("\n" + "-" * 70)
    best_method = result.comparison_df.iloc[0]['Method']
    best_spread = result.comparison_df.iloc[0]['Sharpe Spread']
    
    if 'Random' in best_method:
        print("WARNING: Random baseline is best - regime detection may not be working!")
    elif 'PCA+GMM' in best_method:
        print(f"VERDICT: PCA+GMM baseline wins with Sharpe spread {best_spread:.2f}")
        print("         Autoencoder does NOT outperform linear baseline.")
    else:
        print(f"VERDICT: {best_method} wins with Sharpe spread {best_spread:.2f}")
        pca_spread = result.comparison_df[result.comparison_df['Method'] == 'PCA+GMM']['Sharpe Spread'].values[0]
        improvement = ((best_spread - pca_spread) / pca_spread) * 100
        print(f"         Improvement over PCA+GMM: {improvement:+.1f}%")
    
    print("-" * 70)


def generate_comparison_visualization(
    result: UnifiedComparisonResult,
    output_dir: Path = Path("./output"),
):
    """Generate comparison visualization."""
    
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df = result.comparison_df
    methods = df['Method'].values
    
    # 1. Sharpe Spread comparison
    ax1 = axes[0, 0]
    colors = ['red' if 'Random' in m else 'steelblue' if 'PCA' in m else 'coral' for m in methods]
    bars = ax1.bar(range(len(methods)), df['Sharpe Spread'].values, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Sharpe Spread')
    ax1.set_title('Regime Sharpe Spread by Method')
    ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Threshold (2.0)')
    ax1.legend()
    
    # Add value labels
    for bar, val in zip(bars, df['Sharpe Spread'].values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Persistence comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(methods)), df['Persistence'].values * 100, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Persistence (%)')
    ax2.set_title('Regime Persistence by Method')
    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Threshold (50%)')
    ax2.legend()
    
    # 3. Significant regimes
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(methods)), df['Significant Regimes'].values, color=colors, alpha=0.8)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('Count')
    ax3.set_title('Statistically Significant Regimes')
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary
    best_method = df.iloc[0]['Method']
    table_data = [
        ['Metric', 'Best Method', 'Value'],
        ['Sharpe Spread', df.iloc[0]['Method'], f"{df.iloc[0]['Sharpe Spread']:.2f}"],
        ['Persistence', df.loc[df['Persistence'].idxmax(), 'Method'], 
         f"{df['Persistence'].max()*100:.1f}%"],
        ['Significant', df.loc[df['Significant Regimes'].idxmax(), 'Method'],
         f"{df['Significant Regimes'].max()}/{df['Active Regimes'].max()}"],
    ]
    
    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Summary', y=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "unified_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualization to {output_dir / 'unified_comparison.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run unified walk-forward comparison")
    parser.add_argument(
        "--latent_dims", 
        type=str, 
        default="8,12",
        help="Comma-separated latent dimensions to test (default: 8,12)"
    )
    parser.add_argument(
        "--n_regimes",
        type=int,
        default=8,
        help="Number of GMM clusters (default: 8)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per fold (default: 50)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    latent_dims = [int(d.strip()) for d in args.latent_dims.split(',')]
    
    result = run_unified_comparison(
        latent_dims=latent_dims,
        n_regimes=args.n_regimes,
        ae_epochs=args.epochs,
    )
    
    print_comparison_report(result)
    generate_comparison_visualization(result)

