"""
Corrected Latent Dimension Sweep
================================
Find optimal latent dimension using the CORRECT metric: OOS Sharpe spread.

The original sweep (sweep_latent_dim.py) used reconstruction loss, which
does not necessarily correlate with regime detection quality.

This sweep uses:
- Walk-forward validation (not fixed 80/20 split)
- OOS Sharpe spread as primary metric
- Secondary metrics: persistence, cost survival

Run from project root:
    python scripts/sweep_latent_dim_corrected.py
    python scripts/sweep_latent_dim_corrected.py --dims 4,6,8,10,12,14
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from sklearn.mixture import GaussianMixture

from data.manager import DataManager
from data.training import (
    create_training_dataset,
    get_temporal_feature_cols,
    get_macro_feature_cols,
)
from data.walk_forward import get_stationary_features
from models.autoencoder import HedgeFundBrain, AutoencoderConfig
from models.trainer import AutoencoderTrainer, TrainingConfig, create_data_loaders
from analysis.statistical_tests import compute_regime_transition_matrix


@dataclass
class SweepResult:
    """Results for a single latent dimension."""
    latent_dim: int
    avg_sharpe_spread: float
    std_sharpe_spread: float
    avg_persistence: float
    avg_val_loss: float
    n_folds: int
    fold_sharpe_spreads: List[float]
    fold_persistences: List[float]
    fold_val_losses: List[float]


def create_walk_forward_splits(
    n_samples: int,
    min_train: int = 504,
    test_size: int = 126,
    gap: int = 30,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create walk-forward train/test index splits."""
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
        
        train_end = test_end
    
    return splits


def compute_sharpe_spread(
    labels: np.ndarray,
    returns: np.ndarray,
    n_regimes: int,
    forward_days: int = 5,
) -> float:
    """Compute Sharpe spread from regime labels and returns."""
    sharpes = []
    for regime in range(n_regimes):
        mask = labels == regime
        regime_returns = returns[mask]
        if len(regime_returns) > 10 and regime_returns.std() > 0:
            sharpe = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252 / forward_days)
            sharpes.append(sharpe)
    
    if len(sharpes) >= 2:
        return max(sharpes) - min(sharpes)
    return 0.0


def evaluate_latent_dim(
    latent_dim: int,
    dataset,
    features: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    n_regimes: int = 8,
    epochs: int = 50,
    verbose: bool = True,
) -> SweepResult:
    """
    Evaluate a single latent dimension using walk-forward validation.
    
    Returns sharpe spread, persistence, and val loss for each fold.
    """
    if verbose:
        print(f"\n  Testing latent_dim={latent_dim}...")
    
    fold_sharpe_spreads = []
    fold_persistences = []
    fold_val_losses = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        
        # Get data
        X_temp_train = dataset.X_temporal[train_idx]
        X_temp_test = dataset.X_temporal[test_idx]
        X_macro_train = dataset.X_macro[train_idx]
        X_macro_test = dataset.X_macro[test_idx]
        test_dates = pd.DatetimeIndex(dataset.dates[test_idx])
        
        # Normalize using train statistics
        temp_mean = X_temp_train.mean(axis=(0, 1))
        temp_std = X_temp_train.std(axis=(0, 1))
        temp_std = np.where(temp_std < 1e-8, 1.0, temp_std)
        
        macro_mean = X_macro_train.mean(axis=0)
        macro_std = X_macro_train.std(axis=0)
        macro_std = np.where(macro_std < 1e-8, 1.0, macro_std)
        
        X_temp_train_norm = ((X_temp_train - temp_mean) / temp_std).astype(np.float32)
        X_temp_test_norm = ((X_temp_test - temp_mean) / temp_std).astype(np.float32)
        X_macro_train_norm = ((X_macro_train - macro_mean) / macro_std).astype(np.float32)
        X_macro_test_norm = ((X_macro_test - macro_mean) / macro_std).astype(np.float32)
        
        # Create model
        model_config = AutoencoderConfig(
            temporal_features=X_temp_train.shape[2],
            macro_features=X_macro_train.shape[1],
            sequence_length=X_temp_train.shape[1],
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
        
        # Validation split
        val_split = int(len(X_temp_train_norm) * 0.8)
        val_loader = create_data_loaders(
            X_temp_train_norm[val_split:], X_macro_train_norm[val_split:],
            batch_size=32, shuffle=False
        )
        
        # Train
        history = trainer.fit(train_loader, val_loader, fold_num=fold_num, verbose=False)
        fold_val_losses.append(history.best_val_loss)
        
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
            covariance_type='diag',
            random_state=42,
            n_init=10,
        )
        gmm.fit(latents_train)
        
        # Predict on test (OOS!)
        test_labels = gmm.predict(latents_test)
        
        # Get forward returns for test period
        test_features = features.loc[test_dates]
        forward_returns = (test_features['Close'].shift(-5) / test_features['Close'] - 1).values
        
        # Compute Sharpe spread
        sharpe_spread = compute_sharpe_spread(test_labels, forward_returns, n_regimes)
        fold_sharpe_spreads.append(sharpe_spread)
        
        # Compute persistence
        _, trans_stats = compute_regime_transition_matrix(test_labels, n_regimes=n_regimes)
        fold_persistences.append(trans_stats['avg_persistence'])
        
        if verbose:
            print(f"    Fold {fold_num}: Sharpe Spread={sharpe_spread:.2f}, "
                  f"Persistence={trans_stats['avg_persistence']:.1%}, "
                  f"Val Loss={history.best_val_loss:.4f}")
    
    return SweepResult(
        latent_dim=latent_dim,
        avg_sharpe_spread=np.mean(fold_sharpe_spreads),
        std_sharpe_spread=np.std(fold_sharpe_spreads),
        avg_persistence=np.mean(fold_persistences),
        avg_val_loss=np.mean(fold_val_losses),
        n_folds=len(splits),
        fold_sharpe_spreads=fold_sharpe_spreads,
        fold_persistences=fold_persistences,
        fold_val_losses=fold_val_losses,
    )


def run_corrected_sweep(
    latent_dims: List[int] = [4, 6, 8, 10, 12, 14],
    n_regimes: int = 8,
    epochs: int = 50,
) -> Tuple[List[SweepResult], int]:
    """
    Run corrected hyperparameter sweep.
    
    Returns list of results and the optimal latent dimension.
    """
    
    print("=" * 70)
    print("CORRECTED LATENT DIMENSION SWEEP")
    print("Primary Metric: OOS Sharpe Spread (not reconstruction loss)")
    print("=" * 70)
    print()
    
    # =========================================
    # Load Data
    # =========================================
    print("[1/3] Loading data...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        "BTC-USD", "1d",
        feature_set="combined",
        start="2020-01-01",
    )
    
    temporal_cols = get_temporal_feature_cols()
    macro_cols = [c for c in get_macro_feature_cols() if c in features.columns]
    stationary_cols = [c for c in get_stationary_features() if c in features.columns]
    macro_cols = [c for c in macro_cols if c in stationary_cols]
    
    dataset = create_training_dataset(
        features,
        sequence_length=30,
        temporal_cols=temporal_cols,
        macro_cols=macro_cols,
    )
    
    print(f"  Samples: {len(dataset)}")
    print(f"  Temporal features: {len(temporal_cols)}")
    print(f"  Macro features: {len(macro_cols)}")
    
    # =========================================
    # Create Walk-Forward Splits
    # =========================================
    print("\n[2/3] Creating walk-forward splits...")
    
    splits = create_walk_forward_splits(
        n_samples=len(dataset),
        min_train=504,
        test_size=126,
        gap=30,
    )
    
    print(f"  Created {len(splits)} folds")
    
    # =========================================
    # Sweep Latent Dimensions
    # =========================================
    print(f"\n[3/3] Sweeping latent dimensions: {latent_dims}")
    
    results = []
    
    for latent_dim in latent_dims:
        result = evaluate_latent_dim(
            latent_dim=latent_dim,
            dataset=dataset,
            features=features,
            splits=splits,
            n_regimes=n_regimes,
            epochs=epochs,
            verbose=True,
        )
        results.append(result)
    
    # Find optimal
    optimal_idx = np.argmax([r.avg_sharpe_spread for r in results])
    optimal_dim = results[optimal_idx].latent_dim
    
    return results, optimal_dim


def print_sweep_report(results: List[SweepResult], optimal_dim: int):
    """Print formatted sweep report."""
    
    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print()
    
    # Create comparison table
    print("Latent Dimension Comparison:")
    print("-" * 70)
    print(f"{'Dim':>6} | {'Sharpe Spread':>14} | {'Persistence':>12} | {'Val Loss':>10} | {'Rank':>6}")
    print("-" * 70)
    
    # Sort by sharpe spread for ranking
    sorted_results = sorted(results, key=lambda x: x.avg_sharpe_spread, reverse=True)
    rankings = {r.latent_dim: i+1 for i, r in enumerate(sorted_results)}
    
    for r in results:
        rank = rankings[r.latent_dim]
        marker = " ***" if r.latent_dim == optimal_dim else ""
        print(f"{r.latent_dim:>6} | {r.avg_sharpe_spread:>7.2f} +/- {r.std_sharpe_spread:<4.2f} | "
              f"{r.avg_persistence:>11.1%} | {r.avg_val_loss:>10.4f} | {rank:>4}{marker}")
    
    print("-" * 70)
    print(f"\nOPTIMAL LATENT DIMENSION: {optimal_dim}")
    
    optimal_result = [r for r in results if r.latent_dim == optimal_dim][0]
    print(f"  - Average OOS Sharpe Spread: {optimal_result.avg_sharpe_spread:.2f}")
    print(f"  - Average Persistence: {optimal_result.avg_persistence:.1%}")
    print(f"  - Average Val Loss: {optimal_result.avg_val_loss:.4f}")
    
    # Compare to reconstruction-optimal
    loss_optimal = min(results, key=lambda r: r.avg_val_loss)
    if loss_optimal.latent_dim != optimal_dim:
        print(f"\n  NOTE: Reconstruction-optimal dimension ({loss_optimal.latent_dim}D) differs from")
        print(f"        regime-detection-optimal dimension ({optimal_dim}D)!")


def generate_sweep_visualization(
    results: List[SweepResult],
    optimal_dim: int,
    output_dir: Path = Path("./output"),
):
    """Generate sweep visualization."""
    
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    dims = [r.latent_dim for r in results]
    sharpe_spreads = [r.avg_sharpe_spread for r in results]
    sharpe_stds = [r.std_sharpe_spread for r in results]
    persistences = [r.avg_persistence * 100 for r in results]
    val_losses = [r.avg_val_loss for r in results]
    
    # 1. Sharpe Spread (primary metric)
    ax1 = axes[0, 0]
    bars = ax1.bar(dims, sharpe_spreads, color='steelblue', alpha=0.8)
    ax1.errorbar(dims, sharpe_spreads, yerr=sharpe_stds, fmt='none', color='black', capsize=5)
    ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Threshold (2.0)')
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('OOS Sharpe Spread')
    ax1.set_title('PRIMARY METRIC: OOS Sharpe Spread')
    
    # Highlight optimal
    opt_idx = dims.index(optimal_dim)
    bars[opt_idx].set_color('coral')
    bars[opt_idx].set_alpha(1.0)
    ax1.legend()
    
    # 2. Persistence
    ax2 = axes[0, 1]
    ax2.plot(dims, persistences, 'o-', color='steelblue', markersize=8)
    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Threshold (50%)')
    ax2.set_xlabel('Latent Dimension')
    ax2.set_ylabel('Persistence (%)')
    ax2.set_title('Regime Persistence')
    ax2.scatter([optimal_dim], [persistences[opt_idx]], color='coral', s=150, zorder=5)
    ax2.legend()
    
    # 3. Validation Loss (old metric - for comparison)
    ax3 = axes[1, 0]
    ax3.plot(dims, val_losses, 'o-', color='steelblue', markersize=8)
    ax3.set_xlabel('Latent Dimension')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('OLD METRIC: Reconstruction Loss (for reference)')
    
    # Mark loss-optimal
    loss_opt_idx = np.argmin(val_losses)
    ax3.scatter([dims[loss_opt_idx]], [val_losses[loss_opt_idx]], color='red', s=150, 
                zorder=5, label=f'Loss-optimal ({dims[loss_opt_idx]}D)')
    ax3.scatter([optimal_dim], [val_losses[opt_idx]], color='coral', s=150, 
                zorder=5, label=f'Regime-optimal ({optimal_dim}D)')
    ax3.legend()
    
    # 4. Summary comparison
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find reconstruction-optimal
    loss_opt_dim = dims[loss_opt_idx]
    
    summary_text = f"""
    SWEEP SUMMARY
    ============
    
    Regime-Detection Optimal: {optimal_dim}D
    - OOS Sharpe Spread: {sharpe_spreads[opt_idx]:.2f}
    - Persistence: {persistences[opt_idx]:.1f}%
    
    Reconstruction Optimal: {loss_opt_dim}D
    - Validation Loss: {val_losses[loss_opt_idx]:.4f}
    
    {"Same dimension!" if loss_opt_dim == optimal_dim else "DIFFERENT - use regime-optimal!"}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / "latent_sweep_corrected.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualization to {output_dir / 'latent_sweep_corrected.png'}")


def save_sweep_results(
    results: List[SweepResult],
    optimal_dim: int,
    output_dir: Path = Path("./output"),
):
    """Save sweep results to JSON for future reference."""
    
    output_data = {
        'sweep_date': datetime.now().isoformat(),
        'optimal_latent_dim': optimal_dim,
        'metric': 'OOS Sharpe Spread',
        'results': [asdict(r) for r in results],
    }
    
    output_path = output_dir / "latent_sweep_corrected_results.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run corrected latent dimension sweep")
    parser.add_argument(
        "--dims",
        type=str,
        default="4,6,8,10,12,14",
        help="Comma-separated latent dimensions to test (default: 4,6,8,10,12,14)"
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
    
    latent_dims = [int(d.strip()) for d in args.dims.split(',')]
    
    results, optimal_dim = run_corrected_sweep(
        latent_dims=latent_dims,
        n_regimes=args.n_regimes,
        epochs=args.epochs,
    )
    
    print_sweep_report(results, optimal_dim)
    generate_sweep_visualization(results, optimal_dim)
    save_sweep_results(results, optimal_dim)

