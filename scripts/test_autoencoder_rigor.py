"""
8D Autoencoder Statistical Rigor Test
======================================
Apply the same statistical validation to autoencoder results
as we did for the GMM baseline.

Run from project root:
    python scripts/test_autoencoder_rigor.py
    python scripts/test_autoencoder_rigor.py --model checkpoints/my_model
"""

import sys
import json
import pickle
import argparse
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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
from analysis.statistical_tests import (
    bootstrap_sharpe_ci,
    compute_regime_transition_matrix,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run statistical rigor tests on Autoencoder.")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint to validate")
    return parser.parse_args()

def load_trained_model(model_path):
    """Load model and components from checkpoint."""
    path = Path(model_path)
    
    # Load metadata
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)
        
    # Load norm stats
    with open(path / "norm_stats.pkl", "rb") as f:
        norm_stats = pickle.load(f)
        
    # Load GMM
    with open(path / "gmm.pkl", "rb") as f:
        gmm = pickle.load(f)
        
    # Load Model
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    config = AutoencoderConfig(**checkpoint["model_config"])
    model = HedgeFundBrain(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, config, norm_stats, gmm, metadata

def run_autoencoder_comparison(args):
    """Compare 8D autoencoder to GMM baseline with statistical rigor."""
    
    print("=" * 70)
    print("8D AUTOENCODER vs GMM BASELINE")
    print("Statistical Rigor Comparison")
    print("=" * 70)
    print()
    
    # =========================================
    # Load Data
    # =========================================
    print("[1/5] Loading data...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        "BTC-USD", "1d",
        feature_set="combined",
        start="2020-01-01",
        force_recompute=True,
    )
    
    # Get feature columns
    temporal_cols = get_temporal_feature_cols()
    macro_cols = [c for c in get_macro_feature_cols() if c in features.columns]
    stationary = get_stationary_features()
    macro_cols = [c for c in macro_cols if c in stationary]
    
    dataset = create_training_dataset(
        features,
        sequence_length=30,
        temporal_cols=temporal_cols,
        macro_cols=macro_cols,
    )
    
    print(f"  [OK] Loaded {len(dataset)} samples")

    latents_8d = None
    labels_ae = None
    test_dates = None
    
    # =========================================
    # Model Strategy: Load or Train
    # =========================================
    if args.model:
        print(f"\n[2/5] Loading trained model from {args.model}...")
        model, config, norm_stats, gmm_ae, metadata = load_trained_model(args.model)
        
        # Determine OOS period
        oos_start = metadata.get("data", {}).get("oos_start")
        if not oos_start:
            print("  Warning: No OOS start date in metadata, using default split")
            start_idx = int(len(dataset) * 0.8)
        else:
            # Find index closest to oos_start
            print(f"  Using OOS start date: {oos_start}")
            # dataset.dates is a numpy array of datetimes
            start_idx = np.searchsorted(dataset.dates, np.datetime64(oos_start))
            
        # Prepare Test Data
        X_temp_test = dataset.X_temporal[start_idx:]
        X_macro_test = dataset.X_macro[start_idx:]
        test_dates = dataset.dates[start_idx:]
        
        # Normalize using saved stats
        temp_mean = norm_stats['temp_mean']
        temp_std = norm_stats['temp_std']
        macro_mean = norm_stats['macro_mean']
        macro_std = norm_stats['macro_std']
        
        X_temp_test_norm = ((X_temp_test - temp_mean) / temp_std).astype(np.float32)
        X_macro_test_norm = ((X_macro_test - macro_mean) / macro_std).astype(np.float32)
        
        # Encode
        print("  Encoding test data...")
        test_loader = create_data_loaders(X_temp_test_norm, X_macro_test_norm, batch_size=32, shuffle=False)
        trainer = AutoencoderTrainer(model, TrainingConfig()) # Config not used for encoding
        latents_8d = trainer.encode_dataset(test_loader)
        
        # Predict Regimes
        print("  Predicting regimes...")
        labels_ae = gmm_ae.predict(latents_8d)
        n_regimes_ae = gmm_ae.n_components
        
    else:
        print("\n[2/5] Training 8D autoencoder (quick validation)...")
        
        # Use 80/20 split for quick test
        train_size = int(len(dataset) * 0.8)
        
        X_temp_train = dataset.X_temporal[:train_size]
        X_macro_train = dataset.X_macro[:train_size]
        X_temp_test = dataset.X_temporal[train_size:]
        X_macro_test = dataset.X_macro[train_size:]
        test_dates = dataset.dates[train_size:]
        
        # Normalize
        temp_mean = X_temp_train.mean(axis=(0, 1))
        temp_std = np.where(X_temp_train.std(axis=(0, 1)) < 1e-8, 1.0, X_temp_train.std(axis=(0, 1)))
        macro_mean = X_macro_train.mean(axis=0)
        macro_std = np.where(X_macro_train.std(axis=0) < 1e-8, 1.0, X_macro_train.std(axis=0))
        
        X_temp_train_norm = ((X_temp_train - temp_mean) / temp_std).astype(np.float32)
        X_temp_test_norm = ((X_temp_test - temp_mean) / temp_std).astype(np.float32)
        X_macro_train_norm = ((X_macro_train - macro_mean) / macro_std).astype(np.float32)
        X_macro_test_norm = ((X_macro_test - macro_mean) / macro_std).astype(np.float32)
        
        # Model config - 8D latent
        model_config = AutoencoderConfig(
            temporal_features=len(temporal_cols),
            macro_features=len(macro_cols),
            sequence_length=30,
            latent_dim=8,
        )
        
        train_config = TrainingConfig(
            epochs=50,
            early_stopping_patience=10,
            learning_rate=1e-3,
            batch_size=32,
            lambda_macro=2.0,
        )
        
        # Train
        model = HedgeFundBrain(model_config)
        trainer = AutoencoderTrainer(model, train_config)
        
        train_loader = create_data_loaders(X_temp_train_norm, X_macro_train_norm, batch_size=32, shuffle=True)
        val_split = int(len(X_temp_train_norm) * 0.8)
        val_loader = create_data_loaders(
            X_temp_train_norm[val_split:], X_macro_train_norm[val_split:],
            batch_size=32, shuffle=False
        )
        
        trainer.fit(train_loader, val_loader, fold_num=0)
        
        # Encode test data
        test_loader = create_data_loaders(X_temp_test_norm, X_macro_test_norm, batch_size=32, shuffle=False)
        latents_8d = trainer.encode_dataset(test_loader)
        
        # Cluster latent space
        gmm_ae = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
        labels_ae = gmm_ae.fit_predict(latents_8d)
        n_regimes_ae = 5
    
    print(f"  [OK] 8D latents shape: {latents_8d.shape}")
    
    # =========================================
    # Compare to GMM Baseline
    # =========================================
    print("\n[3/5] Running GMM baseline for comparison...")
    
    # Get same test period features for GMM
    test_features = features.loc[test_dates]
    stationary_cols = [c for c in get_stationary_features() if c in test_features.columns]
    X_gmm = test_features[stationary_cols].values
    
    # Scale and PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_gmm_scaled = scaler.fit_transform(X_gmm)
    
    pca = PCA(n_components=0.95)
    X_gmm_pca = pca.fit_transform(X_gmm_scaled)
    
    # Cluster
    gmm_baseline = GaussianMixture(n_components=8, covariance_type='diag', random_state=42, n_init=10)
    labels_gmm = gmm_baseline.fit_predict(X_gmm_pca)
    
    print(f"  [OK] GMM baseline: {pca.n_components_} PCA dims, {8} clusters")
    
    # =========================================
    # Calculate Returns by Regime
    # =========================================
    print("\n[4/5] Analyzing regime returns...")
    
    test_df = test_features.copy()
    test_df['regime_ae'] = labels_ae
    test_df['regime_gmm'] = labels_gmm
    test_df['fwd_return_5d'] = test_df['Close'].shift(-5) / test_df['Close'] - 1
    test_df['daily_return'] = test_df['Close'].pct_change()
    
    # 8D Autoencoder regime analysis
    print("\n  8D AUTOENCODER REGIMES:")
    print("  " + "-" * 60)
    
    returns_by_regime_ae = {}
    for regime in range(n_regimes_ae):
        mask = test_df['regime_ae'] == regime
        returns = test_df.loc[mask, 'fwd_return_5d'].dropna().values
        if len(returns) > 10:
            returns_by_regime_ae[regime] = returns
            try:
                result = bootstrap_sharpe_ci(returns, annualization=252/5)
                sig = "***" if result.significant else ""
                print(f"    R{regime}: n={len(returns):3d}, Sharpe={result.point_estimate:6.2f}, "
                      f"CI=[{result.ci_lower:6.2f}, {result.ci_upper:6.2f}] {sig}")
            except:
                print(f"    R{regime}: n={len(returns):3d}, Sharpe calculation failed")
    
    # GMM baseline regime analysis
    print("\n  GMM BASELINE REGIMES:")
    print("  " + "-" * 60)
    
    returns_by_regime_gmm = {}
    for regime in range(8):
        mask = test_df['regime_gmm'] == regime
        returns = test_df.loc[mask, 'fwd_return_5d'].dropna().values
        if len(returns) > 10:
            returns_by_regime_gmm[regime] = returns
            try:
                result = bootstrap_sharpe_ci(returns, annualization=252/5)
                sig = "***" if result.significant else ""
                print(f"    R{regime}: n={len(returns):3d}, Sharpe={result.point_estimate:6.2f}, "
                      f"CI=[{result.ci_lower:6.2f}, {result.ci_upper:6.2f}] {sig}")
            except:
                print(f"    R{regime}: n={len(returns):3d}, Sharpe calculation failed")
    
    # =========================================
    # Regime Transition Analysis
    # =========================================
    print("\n[5/5] Comparing regime stability...")
    
    trans_ae, stats_ae = compute_regime_transition_matrix(labels_ae, n_regimes=n_regimes_ae)
    trans_gmm, stats_gmm = compute_regime_transition_matrix(labels_gmm, n_regimes=8)
    
    print(f"\n  8D Autoencoder:")
    print(f"    Avg persistence: {stats_ae['avg_persistence']:.1%}")
    print(f"    Stability: {stats_ae['stability']:.1%}")
    
    print(f"\n  GMM Baseline:")
    print(f"    Avg persistence: {stats_gmm['avg_persistence']:.1%}")
    print(f"    Stability: {stats_gmm['stability']:.1%}")
    
    # =========================================
    # Summary Comparison
    # =========================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # Calculate Sharpe spreads
    ae_sharpes = []
    for regime, returns in returns_by_regime_ae.items():
        if len(returns) > 10:
            ae_sharpes.append((returns.mean() / returns.std()) * np.sqrt(252/5) if returns.std() > 0 else 0)
    
    gmm_sharpes = []
    for regime, returns in returns_by_regime_gmm.items():
        if len(returns) > 10:
            gmm_sharpes.append((returns.mean() / returns.std()) * np.sqrt(252/5) if returns.std() > 0 else 0)
    
    ae_spread = max(ae_sharpes) - min(ae_sharpes) if ae_sharpes else 0
    gmm_spread = max(gmm_sharpes) - min(gmm_sharpes) if gmm_sharpes else 0
    
    print("\n  | Metric                | 8D Autoencoder | GMM Baseline |")
    print("  |----------------------|----------------|--------------|")
    print(f"  | Sharpe Spread        | {ae_spread:14.2f} | {gmm_spread:12.2f} |")
    print(f"  | Regime Persistence   | {stats_ae['avg_persistence']*100:13.1f}% | {stats_gmm['avg_persistence']*100:11.1f}% |")
    print(f"  | Stability            | {stats_ae['stability']*100:13.1f}% | {stats_gmm['stability']*100:11.1f}% |")
    print(f"  | Number of Regimes    | {n_regimes_ae:14d} | {8:12d} |")
    
    # Verdict
    print("\n" + "-" * 70)
    if ae_spread > gmm_spread:
        print("  VERDICT: 8D Autoencoder OUTPERFORMS GMM baseline!")
    elif ae_spread > gmm_spread * 0.8:
        print("  VERDICT: 8D Autoencoder is COMPETITIVE with GMM baseline")
    elif ae_spread > 2.0:
        print("  VERDICT: 8D Autoencoder shows STRONG regime separation")
    else:
        print("  VERDICT: GMM baseline still superior, but 8D shows promise")
    print("-" * 70)
    
    # =========================================
    # Generate Comparison Visualization
    # =========================================
    print("\nGenerating comparison visualization...")
    
    output_dir = Path("./output")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sharpe comparison
    ax1 = axes[0, 0]
    models = ['8D Autoencoder', 'GMM Baseline']
    spreads = [ae_spread, gmm_spread]
    colors = ['coral', 'steelblue']
    bars = ax1.bar(models, spreads, color=colors, alpha=0.8)
    ax1.axhline(y=2.0, color='green', linestyle='--', label='Threshold (2.0)')
    ax1.set_ylabel('Sharpe Spread')
    ax1.set_title('Regime Sharpe Spread Comparison')
    ax1.legend()
    for bar, spread in zip(bars, spreads):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{spread:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Regime stability
    ax2 = axes[0, 1]
    metrics = ['Persistence', 'Stability']
    ae_vals = [stats_ae['avg_persistence']*100, stats_ae['stability']*100]
    gmm_vals = [stats_gmm['avg_persistence']*100, stats_gmm['stability']*100]
    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width/2, ae_vals, width, label='8D Autoencoder', color='coral', alpha=0.8)
    ax2.bar(x + width/2, gmm_vals, width, label='GMM Baseline', color='steelblue', alpha=0.8)
    ax2.axhline(y=50, color='green', linestyle='--', label='Threshold (50%)')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Regime Stability Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # 3. 8D Autoencoder latent space (PCA → 2D)
    ax3 = axes[1, 0]
    pca_viz = PCA(n_components=2)
    latents_2d = pca_viz.fit_transform(latents_8d)
    colors_regime = plt.cm.Set2(np.linspace(0, 1, n_regimes_ae))
    for regime in range(n_regimes_ae):
        mask = labels_ae == regime
        ax3.scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                   c=[colors_regime[regime]], alpha=0.5, s=20, label=f'R{regime}')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('8D Autoencoder Latent Space (PCA → 2D)')
    ax3.legend()
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_data = [
        ['Metric', '8D AE', 'GMM', 'Winner'],
        ['Sharpe Spread', f'{ae_spread:.2f}', f'{gmm_spread:.2f}', 'GMM' if gmm_spread > ae_spread else 'AE'],
        ['Persistence', f'{stats_ae["avg_persistence"]*100:.1f}%', f'{stats_gmm["avg_persistence"]*100:.1f}%', 
         'GMM' if stats_gmm['avg_persistence'] > stats_ae['avg_persistence'] else 'AE'],
        ['Stability', f'{stats_ae["stability"]*100:.1f}%', f'{stats_gmm["stability"]*100:.1f}%',
         'GMM' if stats_gmm['stability'] > stats_ae['stability'] else 'AE'],
        ['# Regimes', str(n_regimes_ae), '8', '-'],
    ]
    
    table = ax4.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Head-to-Head Comparison', y=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "autoencoder_8d_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved to output/autoencoder_8d_comparison.png")
    
    return {
        'ae_spread': ae_spread,
        'gmm_spread': gmm_spread,
        'stats_ae': stats_ae,
        'stats_gmm': stats_gmm,
    }


if __name__ == "__main__":
    args = parse_args()
    results = run_autoencoder_comparison(args)
