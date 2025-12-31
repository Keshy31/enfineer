"""
Train LSTM-Autoencoder with Walk-Forward Validation
====================================================
Trains the HedgeFundBrain autoencoder using proper out-of-sample
validation to ensure the latent space generalizes.

The key insight: We train on historical data, encode unseen future
data, then cluster in latent space. This ensures no look-ahead bias.

Run from project root:
    python scripts/train_autoencoder.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from data.manager import DataManager
from data.training import (
    create_training_dataset,
    TrainingDataset,
    get_temporal_feature_cols,
    get_macro_feature_cols,
)
from data.walk_forward import get_stationary_features, find_optimal_k
from models.autoencoder import HedgeFundBrain, AutoencoderConfig
from models.trainer import (
    AutoencoderTrainer,
    TrainingConfig,
    create_data_loaders,
    TrainingHistory,
)


@dataclass
class WalkForwardFoldData:
    """Data for a single walk-forward fold."""
    fold_num: int
    train_temporal: np.ndarray
    train_macro: np.ndarray
    train_dates: pd.DatetimeIndex
    test_temporal: np.ndarray
    test_macro: np.ndarray
    test_dates: pd.DatetimeIndex


def create_walk_forward_folds(
    dataset: TrainingDataset,
    min_train_samples: int = 504,
    test_samples: int = 126,
    gap_samples: int = 30,
) -> List[WalkForwardFoldData]:
    """
    Create walk-forward folds from training dataset.
    
    Parameters
    ----------
    dataset : TrainingDataset
        Full dataset.
    min_train_samples : int
        Minimum training samples (2 years).
    test_samples : int
        Test samples per fold (6 months).
    gap_samples : int
        Gap between train and test.
        
    Returns
    -------
    List[WalkForwardFoldData]
    """
    n_samples = len(dataset)
    folds = []
    fold_num = 1
    
    train_end_idx = min_train_samples
    
    while True:
        test_start_idx = train_end_idx + gap_samples
        test_end_idx = test_start_idx + test_samples
        
        if test_end_idx > n_samples:
            break
        
        fold = WalkForwardFoldData(
            fold_num=fold_num,
            train_temporal=dataset.X_temporal[:train_end_idx],
            train_macro=dataset.X_macro[:train_end_idx],
            train_dates=dataset.dates[:train_end_idx],
            test_temporal=dataset.X_temporal[test_start_idx:test_end_idx],
            test_macro=dataset.X_macro[test_start_idx:test_end_idx],
            test_dates=dataset.dates[test_start_idx:test_end_idx],
        )
        folds.append(fold)
        
        # Expand training window
        train_end_idx = test_end_idx
        fold_num += 1
    
    return folds


def normalize_fold_data(
    fold: WalkForwardFoldData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Normalize fold data using training statistics only.
    
    Returns normalized train/test data and normalization stats.
    """
    # Compute stats from training data
    temp_mean = fold.train_temporal.mean(axis=(0, 1))
    temp_std = fold.train_temporal.std(axis=(0, 1))
    temp_std = np.where(temp_std < 1e-8, 1.0, temp_std)
    
    macro_mean = fold.train_macro.mean(axis=0)
    macro_std = fold.train_macro.std(axis=0)
    macro_std = np.where(macro_std < 1e-8, 1.0, macro_std)
    
    # Normalize
    train_temp_norm = (fold.train_temporal - temp_mean) / temp_std
    test_temp_norm = (fold.test_temporal - temp_mean) / temp_std
    
    train_macro_norm = (fold.train_macro - macro_mean) / macro_std
    test_macro_norm = (fold.test_macro - macro_mean) / macro_std
    
    stats = {
        'temp_mean': temp_mean,
        'temp_std': temp_std,
        'macro_mean': macro_mean,
        'macro_std': macro_std,
    }
    
    return (
        train_temp_norm.astype(np.float32),
        train_macro_norm.astype(np.float32),
        test_temp_norm.astype(np.float32),
        test_macro_norm.astype(np.float32),
        stats,
    )


def analyze_latent_regimes(
    latents: np.ndarray,
    dates: pd.DatetimeIndex,
    features_df: pd.DataFrame,
    n_regimes: int,
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    Cluster latent space and analyze regime returns.
    """
    # Fit GMM on latent space
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='full',  # OK for 3D
        random_state=42,
        n_init=10,
    )
    labels = gmm.fit_predict(latents)
    
    # Create DataFrame with labels
    df = features_df.loc[dates].copy()
    df['regime'] = labels
    df['fwd_return'] = df['Close'].shift(-forward_days) / df['Close'] - 1
    
    # Analyze by regime
    results = []
    for regime in range(n_regimes):
        mask = df['regime'] == regime
        returns = df.loc[mask, 'fwd_return'].dropna()
        
        if len(returns) > 10:
            results.append({
                'regime': regime,
                'count': len(returns),
                'pct': len(returns) / len(df) * 100,
                'mean_return': returns.mean() * 100,
                'std_return': returns.std() * 100,
                'sharpe': (returns.mean() / returns.std()) * np.sqrt(252 / forward_days) if returns.std() > 0 else 0,
            })
    
    return pd.DataFrame(results), labels, gmm


def save_model_package(
    trainer: 'AutoencoderTrainer',
    model_config: AutoencoderConfig,
    train_config: TrainingConfig,
    norm_stats: Dict,
    gmm: GaussianMixture,
    regime_stats: pd.DataFrame,
    histories: List[TrainingHistory],
    oos_dates: pd.DatetimeIndex,
    sharpe_spread: float,
    optimal_k: int,
    n_folds: int,
    run_name: Optional[str] = None,
) -> Path:
    """
    Save complete model package for future inference and comparison.
    
    Saves:
    - Model weights (.pt)
    - GMM cluster model (.pkl)
    - Normalization stats (.pkl)
    - Metadata JSON (for comparison)
    
    Parameters
    ----------
    trainer : AutoencoderTrainer
        Trainer with final model.
    model_config : AutoencoderConfig
        Model architecture config.
    train_config : TrainingConfig
        Training hyperparameters.
    norm_stats : Dict
        Normalization statistics from final fold.
    gmm : GaussianMixture
        Fitted GMM for regime clustering.
    regime_stats : pd.DataFrame
        Regime performance statistics.
    histories : List[TrainingHistory]
        Training histories from all folds.
    oos_dates : pd.DatetimeIndex
        Out-of-sample date range.
    sharpe_spread : float
        Key performance metric.
    optimal_k : int
        Number of clusters.
    n_folds : int
        Number of walk-forward folds.
    run_name : Optional[str]
        Custom name for this run. If None, uses timestamp.
        
    Returns
    -------
    Path
        Directory where model package was saved.
    """
    # Create unique run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if run_name:
        run_id = f"{run_name}_{timestamp}"
    else:
        # Include key metrics in name for quick comparison
        latent_dim = model_config.latent_dim
        run_id = f"ae_L{latent_dim}_K{optimal_k}_S{sharpe_spread:.2f}_{timestamp}"
    
    # Create run directory
    checkpoint_dir = Path("./checkpoints")
    run_dir = checkpoint_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model weights
    model_path = run_dir / "model.pt"
    checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'model_config': model_config.__dict__,
        'training_config': {
            k: str(v) if isinstance(v, Path) else v
            for k, v in train_config.__dict__.items()
        },
    }
    torch.save(checkpoint, model_path)
    
    # 2. Save normalization stats (needed for inference)
    norm_path = run_dir / "norm_stats.pkl"
    with open(norm_path, 'wb') as f:
        pickle.dump(norm_stats, f)
    
    # 3. Save GMM model (needed for regime labeling)
    gmm_path = run_dir / "gmm.pkl"
    with open(gmm_path, 'wb') as f:
        pickle.dump(gmm, f)
    
    # 4. Save metadata JSON (human-readable, for comparison)
    # Compute aggregate training stats
    avg_best_epoch = np.mean([h.best_epoch for h in histories])
    avg_best_val_loss = np.mean([h.best_val_loss for h in histories])
    total_training_time = sum(h.training_time for h in histories)
    
    metadata = {
        # Run info
        'run_id': run_id,
        'timestamp': timestamp,
        'created_at': datetime.now().isoformat(),
        
        # Model architecture
        'model': {
            'latent_dim': model_config.latent_dim,
            'lstm_hidden': model_config.lstm_hidden,
            'lstm_layers': model_config.lstm_layers,
            'macro_hidden': model_config.macro_hidden,
            'temporal_features': model_config.temporal_features,
            'macro_features': model_config.macro_features,
            'sequence_length': model_config.sequence_length,
            'dropout': model_config.dropout,
        },
        
        # Training config
        'training': {
            'epochs': train_config.epochs,
            'batch_size': train_config.batch_size,
            'learning_rate': train_config.learning_rate,
            'lambda_macro': train_config.lambda_macro,
            'early_stopping_patience': train_config.early_stopping_patience,
        },
        
        # Walk-forward info
        'walk_forward': {
            'n_folds': n_folds,
            'avg_best_epoch': float(avg_best_epoch),
            'avg_best_val_loss': float(avg_best_val_loss),
            'total_training_time_sec': float(total_training_time),
        },
        
        # Data info
        'data': {
            'oos_start': str(oos_dates.min().date()),
            'oos_end': str(oos_dates.max().date()),
            'oos_samples': len(oos_dates),
        },
        
        # Performance metrics (KEY FOR COMPARISON)
        'performance': {
            'sharpe_spread': float(sharpe_spread),
            'optimal_k': optimal_k,
            'regime_stats': regime_stats.to_dict('records'),
        },
        
        # File manifest
        'files': {
            'model': 'model.pt',
            'norm_stats': 'norm_stats.pkl',
            'gmm': 'gmm.pkl',
            'metadata': 'metadata.json',
        }
    }
    
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return run_dir


def load_model_package(run_dir: Path) -> Dict:
    """
    Load a saved model package for inference.
    
    Parameters
    ----------
    run_dir : Path
        Directory containing the model package.
        
    Returns
    -------
    Dict with keys: 'model', 'norm_stats', 'gmm', 'metadata'
    """
    run_dir = Path(run_dir)
    
    # Load metadata
    with open(run_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load model
    checkpoint = torch.load(run_dir / "model.pt", weights_only=False)
    model_config = AutoencoderConfig(**checkpoint['model_config'])
    model = HedgeFundBrain(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load normalization stats
    with open(run_dir / "norm_stats.pkl", 'rb') as f:
        norm_stats = pickle.load(f)
    
    # Load GMM
    with open(run_dir / "gmm.pkl", 'rb') as f:
        gmm = pickle.load(f)
    
    return {
        'model': model,
        'model_config': model_config,
        'norm_stats': norm_stats,
        'gmm': gmm,
        'metadata': metadata,
    }


def list_saved_models() -> pd.DataFrame:
    """
    List all saved model packages with key metrics for comparison.
    
    Returns
    -------
    pd.DataFrame
        Sorted by sharpe_spread descending.
    """
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        return pd.DataFrame()
    
    runs = []
    for run_dir in checkpoint_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        
        runs.append({
            'run_id': meta['run_id'],
            'created_at': meta['created_at'],
            'latent_dim': meta['model']['latent_dim'],
            'optimal_k': meta['performance']['optimal_k'],
            'sharpe_spread': meta['performance']['sharpe_spread'],
            'avg_val_loss': meta['walk_forward']['avg_best_val_loss'],
            'n_folds': meta['walk_forward']['n_folds'],
            'oos_samples': meta['data']['oos_samples'],
        })
    
    if not runs:
        return pd.DataFrame()
    
    df = pd.DataFrame(runs)
    df = df.sort_values('sharpe_spread', ascending=False)
    return df


def run_walk_forward_training(run_name: Optional[str] = None):
    """
    Run complete walk-forward training pipeline.
    
    Parameters
    ----------
    run_name : Optional[str]
        Custom name for this training run. If None, auto-generates
        a name from latent_dim, optimal_k, and sharpe_spread.
        
    Returns
    -------
    Dict with training results and saved model path.
    """
    
    print("=" * 70)
    print("SIMONS-DALIO REGIME ENGINE")
    print("Walk-Forward Autoencoder Training")
    print("=" * 70)
    print()
    
    # =========================================
    # Step 1: Load Data
    # =========================================
    print("[1/7] Loading combined features...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        "BTC-USD", "1d",
        feature_set="combined",
        start="2020-01-01",
    )
    
    print(f"  ✓ Loaded {len(features)} rows")
    
    # =========================================
    # Step 2: Create Training Dataset
    # =========================================
    print("\n[2/7] Creating training dataset...")
    
    # Get available feature columns
    temporal_cols = get_temporal_feature_cols()
    macro_cols = [c for c in get_macro_feature_cols() if c in features.columns]
    
    # Filter to stationary features for macro
    stationary = get_stationary_features()
    macro_cols = [c for c in macro_cols if c in stationary]
    
    print(f"  Temporal features: {len(temporal_cols)}")
    print(f"  Macro features: {len(macro_cols)}")
    
    dataset = create_training_dataset(
        features,
        sequence_length=30,
        temporal_cols=temporal_cols,
        macro_cols=macro_cols,
    )
    
    print(f"  ✓ Dataset: {len(dataset)} samples")
    print(f"  ✓ Temporal shape: {dataset.X_temporal.shape}")
    print(f"  ✓ Macro shape: {dataset.X_macro.shape}")
    
    # =========================================
    # Step 3: Create Walk-Forward Folds
    # =========================================
    print("\n[3/7] Creating walk-forward folds...")
    
    folds = create_walk_forward_folds(
        dataset,
        min_train_samples=504,
        test_samples=126,
        gap_samples=30,
    )
    
    print(f"  ✓ Created {len(folds)} folds")
    for fold in folds:
        print(f"    Fold {fold.fold_num}: Train {len(fold.train_dates)} "
              f"→ Test {len(fold.test_dates)} ({fold.test_dates[0].date()} to {fold.test_dates[-1].date()})")
    
    # =========================================
    # Step 4: Train Autoencoder (Walk-Forward)
    # =========================================
    print("\n[4/7] Training autoencoder (walk-forward)...")
    print()
    
    # Model config
    # v1.3: Changed latent_dim from 8 to 12 (see EXPERIMENTS.md - Experiment 5)
    # 12D is the "Goldilocks" zone (0.93 val loss vs 1.30 for 8D)
    model_config = AutoencoderConfig(
        temporal_features=len(temporal_cols),
        macro_features=len(macro_cols),
        sequence_length=30,
        lstm_hidden=64,
        lstm_layers=2,
        macro_hidden=32,
        latent_dim=12,  # 12D optimal
        dropout=0.2,
    )
    
    train_config = TrainingConfig(
        epochs=100,
        early_stopping_patience=15,
        learning_rate=1e-3,
        batch_size=32,
        lambda_macro=2.0,
    )
    
    print(f"Model config: {model_config}")
    print()
    
    # Store results
    all_latents = []
    all_dates = []
    all_histories = []
    final_trainer = None
    final_norm_stats = None
    
    for fold in folds:
        print(f"\n{'='*60}")
        print(f"FOLD {fold.fold_num}")
        print(f"{'='*60}")
        
        # Normalize using TRAIN statistics only
        train_temp, train_macro, test_temp, test_macro, norm_stats = normalize_fold_data(fold)
        
        # Create data loaders
        train_loader = create_data_loaders(train_temp, train_macro, batch_size=32, shuffle=True)
        
        # Use last 20% of train as validation for early stopping
        val_split = int(len(train_temp) * 0.8)
        val_loader = create_data_loaders(
            train_temp[val_split:], train_macro[val_split:],
            batch_size=32, shuffle=False
        )
        
        # Create fresh model for each fold
        model = HedgeFundBrain(model_config)
        trainer = AutoencoderTrainer(model, train_config)
        
        # Train
        history = trainer.fit(train_loader, val_loader, fold_num=fold.fold_num)
        all_histories.append(history)
        
        # Encode TEST data (out-of-sample!)
        test_loader = create_data_loaders(test_temp, test_macro, batch_size=32, shuffle=False)
        test_latents = trainer.encode_dataset(test_loader)
        
        all_latents.append(test_latents)
        all_dates.append(fold.test_dates)
        
        # Keep reference to final fold's trainer and norm stats
        # (final fold has most training data = best model for inference)
        final_trainer = trainer
        final_norm_stats = norm_stats
        
        print(f"\n  Test latents shape: {test_latents.shape}")
        print(f"  Latent range: [{test_latents.min():.2f}, {test_latents.max():.2f}]")
    
    # Stitch OOS latents
    oos_latents = np.vstack(all_latents)
    oos_dates = pd.DatetimeIndex(np.concatenate(all_dates))
    
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Total OOS latents: {oos_latents.shape}")
    print(f"  OOS date range: {oos_dates.min().date()} to {oos_dates.max().date()}")
    
    # =========================================
    # Step 5: Cluster Latent Space
    # =========================================
    print("\n[5/7] Clustering latent space (GMM)...")
    
    # Find optimal k for latent space
    optimal_k, bic_scores = find_optimal_k(
        oos_latents,
        k_range=range(2, 9),
        use_pca=False,  # Already 3D
        covariance_type='full',
    )
    
    print(f"  Optimal k by BIC: {optimal_k}")
    
    # Analyze regimes
    regime_stats, labels, gmm = analyze_latent_regimes(
        oos_latents, oos_dates, features, optimal_k
    )
    
    print("\n" + "=" * 70)
    print("OOS LATENT REGIME ANALYSIS")
    print("=" * 70)
    print()
    print(regime_stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    sharpe_spread = regime_stats['sharpe'].max() - regime_stats['sharpe'].min()
    print(f"\n  Sharpe spread: {sharpe_spread:.2f}")
    
    # =========================================
    # Step 6: Visualize Results
    # =========================================
    print("\n[6/7] Generating visualizations...")
    
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 2D Latent Space (PCA projection for 8D → 2D visualization)
    from sklearn.decomposition import PCA
    
    latent_dim = oos_latents.shape[1]
    if latent_dim > 3:
        # Project 8D → 2D for visualization
        pca_viz = PCA(n_components=2)
        latents_2d = pca_viz.fit_transform(oos_latents)
        explained_var = sum(pca_viz.explained_variance_ratio_) * 100
        title = f'{latent_dim}D Latent Space (PCA → 2D, {explained_var:.0f}% var)'
        
        ax1 = fig.add_subplot(2, 2, 1)
        colors = plt.cm.Set2(np.linspace(0, 1, optimal_k))
        for regime in range(optimal_k):
            mask = labels == regime
            ax1.scatter(
                latents_2d[mask, 0],
                latents_2d[mask, 1],
                c=[colors[regime]],
                alpha=0.5,
                s=20,
                label=f'R{regime}'
            )
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title(title)
        ax1.legend(loc='upper right', fontsize=8)
    else:
        # Original 3D visualization for 3D latent
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        colors = plt.cm.Set2(np.linspace(0, 1, optimal_k))
        for regime in range(optimal_k):
            mask = labels == regime
            ax1.scatter(
                oos_latents[mask, 0],
                oos_latents[mask, 1],
                oos_latents[mask, 2],
                c=[colors[regime]],
                alpha=0.5,
                s=20,
                label=f'R{regime}'
            )
        ax1.set_xlabel('X (Latent 1)')
        ax1.set_ylabel('Y (Latent 2)')
        ax1.set_zlabel('Z (Latent 3)')
        ax1.set_title('3D Latent Space (OOS Encoded)')
        ax1.legend()
    
    # 2. Regime performance
    ax2 = fig.add_subplot(2, 2, 2)
    x = regime_stats['regime']
    width = 0.35
    ax2.bar(x - width/2, regime_stats['mean_return'], width, label='Mean Return (%)', color='steelblue')
    ax2.bar(x + width/2, regime_stats['sharpe'], width, label='Sharpe', color='coral')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Latent Regime Performance ({latent_dim}D, 5-day forward)')
    ax2.legend()
    
    # 3. Training curves
    ax3 = fig.add_subplot(2, 2, 3)
    for i, history in enumerate(all_histories):
        ax3.plot(history.train_losses, alpha=0.5, label=f'Fold {i+1} Train' if i == 0 else None)
        ax3.plot(history.val_losses, alpha=0.8, label=f'Fold {i+1} Val' if i == 0 else None)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Curves (All Folds)')
    ax3.legend()
    
    # 4. Latent evolution over time (show first 4 dimensions)
    ax4 = fig.add_subplot(2, 2, 4)
    n_dims_to_show = min(4, latent_dim)
    for i in range(n_dims_to_show):
        ax4.plot(oos_dates, oos_latents[:, i], alpha=0.7, label=f'Latent {i+1}')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Latent Value')
    ax4.set_title(f'Latent Dimensions Over Time (showing {n_dims_to_show}/{latent_dim})')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "autoencoder_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to output/autoencoder_results.png")
    
    # =========================================
    # Step 7: Save Model Package
    # =========================================
    print("\n[7/7] Saving model package...")
    
    run_dir = save_model_package(
        trainer=final_trainer,
        model_config=model_config,
        train_config=train_config,
        norm_stats=final_norm_stats,
        gmm=gmm,
        regime_stats=regime_stats,
        histories=all_histories,
        oos_dates=oos_dates,
        sharpe_spread=sharpe_spread,
        optimal_k=optimal_k,
        n_folds=len(folds),
        run_name=run_name,
    )
    
    print(f"  ✓ Saved model package to {run_dir}")
    print(f"    - model.pt (weights + config)")
    print(f"    - norm_stats.pkl (for inference)")
    print(f"    - gmm.pkl (for regime labeling)")
    print(f"    - metadata.json (for comparison)")
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print()
    
    # Compare to GMM baseline
    print("Comparison to Walk-Forward GMM Baseline:")
    print("  GMM Baseline Sharpe spread: ~3.50")
    print(f"  Autoencoder Sharpe spread: {sharpe_spread:.2f}")
    print()
    
    if sharpe_spread > 2.0:
        print("  ✓ Autoencoder maintains strong regime separation")
    elif sharpe_spread > 1.0:
        print("  ⚠ Moderate regime separation (similar to baseline)")
    else:
        print("  ✗ Weak regime separation (worse than baseline)")
    
    print()
    print(f"Model saved to: {run_dir}")
    print()
    print("To compare models:")
    print("  from train_autoencoder import list_saved_models")
    print("  print(list_saved_models())")
    print()
    print("To load for inference:")
    print("  from train_autoencoder import load_model_package")
    print(f"  pkg = load_model_package('{run_dir}')")
    
    return {
        'oos_latents': oos_latents,
        'oos_dates': oos_dates,
        'labels': labels,
        'regime_stats': regime_stats,
        'histories': all_histories,
        'model_config': model_config,
        'run_dir': run_dir,
    }


if __name__ == "__main__":
    results = run_walk_forward_training()

