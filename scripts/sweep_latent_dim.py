"""
Hyperparameter Sweep: Latent Dimension
======================================
Tests different latent space dimensions to find the optimal size
using the "Elbow Method".

Supports incremental runs by saving/loading results from JSON.

Usage:
    python scripts/sweep_latent_dim.py --dims 2 4 8 16 32
    python scripts/sweep_latent_dim.py --dims 10 12 14 --ticker BTC-USD
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Force UTF-8 output for Windows consoles
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.manager import DataManager
from data.training import (
    create_training_dataset,
    get_temporal_feature_cols,
    get_macro_feature_cols,
)
from data.walk_forward import get_stationary_features
from models.autoencoder import HedgeFundBrain, AutoencoderConfig
from models.trainer import (
    AutoencoderTrainer,
    TrainingConfig,
    create_data_loaders,
)

RESULTS_FILE = Path("./output/latent_sweep_results.json")

def load_previous_results() -> Dict[str, Dict]:
    """Load existing results to support incremental sweeps."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_results(results: Dict[str, Dict]):
    """Save updated results to JSON."""
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def run_latent_dim_sweep(dims: List[int], ticker: str):
    print("=" * 70)
    print(f"HYPERPARAMETER SWEEP: LATENT DIMENSION ({ticker})")
    print(f"Testing Dimensions: {dims}")
    print("=" * 70)
    print()

    # Load previous results
    all_results = load_previous_results()
    print(f"Loaded {len(all_results)} previous results.")

    # =========================================
    # 1. Load Data
    # =========================================
    print("[1/4] Loading and preparing data...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        ticker, "1d",
        feature_set="combined",
        start="2020-01-01",
    )
    
    # Filter features
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
    
    # Fixed Split (80/20)
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    
    # Normalize on Train
    train_temporal = dataset.X_temporal[:train_size]
    train_macro = dataset.X_macro[:train_size]
    val_temporal = dataset.X_temporal[train_size:]
    val_macro = dataset.X_macro[train_size:]
    
    temp_mean = train_temporal.mean(axis=(0, 1))
    temp_std = train_temporal.std(axis=(0, 1))
    temp_std = np.where(temp_std < 1e-8, 1.0, temp_std)
    
    macro_mean = train_macro.mean(axis=0)
    macro_std = train_macro.std(axis=0)
    macro_std = np.where(macro_std < 1e-8, 1.0, macro_std)
    
    # Apply normalization
    train_temporal_norm = (train_temporal - temp_mean) / temp_std
    val_temporal_norm = (val_temporal - temp_mean) / temp_std
    
    train_macro_norm = (train_macro - macro_mean) / macro_std
    val_macro_norm = (val_macro - macro_mean) / macro_std
    
    # Create Loaders
    train_loader = create_data_loaders(
        train_temporal_norm, train_macro_norm, 
        batch_size=32, shuffle=True
    )
    val_loader = create_data_loaders(
        val_temporal_norm, val_macro_norm, 
        batch_size=32, shuffle=False
    )
    
    print(f"  Train samples: {len(train_temporal)}")
    print(f"  Val samples:   {len(val_temporal)}")
    
    # =========================================
    # 2. Run Sweep
    # =========================================
    print("\n[2/4] Running sweep...")
    
    train_config = TrainingConfig(
        epochs=100,
        early_stopping_patience=15,
        learning_rate=1e-3,
        batch_size=32,
        lambda_macro=2.0,
        save_best_only=True,
    )
    
    for dim in dims:
        dim_str = str(dim)
        
        # Skip if already exists (unless force re-run needed, user can delete json)
        if dim_str in all_results:
            print(f"  Skipping {dim}D (already exists in results)")
            continue

        print(f"\n  Testing Latent Dim: {dim}")
        
        config = AutoencoderConfig(
            temporal_features=len(temporal_cols),
            macro_features=len(macro_cols),
            sequence_length=30,
            lstm_hidden=64 if dim <= 8 else 128,  # Scale capacity slightly
            macro_hidden=32 if dim <= 8 else 64,
            latent_dim=dim,
            dropout=0.2,
        )
        
        model = HedgeFundBrain(config)
        trainer = AutoencoderTrainer(model, train_config)
        
        # Train (verbose=True for progress visibility)
        history = trainer.fit(train_loader, val_loader, verbose=True)
        
        all_results[dim_str] = {
            'dim': dim,
            'best_val_loss': history.best_val_loss,
            'train_loss': history.train_losses[history.best_epoch - 1],
            'best_epoch': history.best_epoch,
            'params': model.count_parameters(),
        }
        
        # Incremental save
        save_results(all_results)
        
        print(f"    -> Best Val Loss: {history.best_val_loss:.6f} (Epoch {history.best_epoch})")

    # =========================================
    # 3. Analyze & Plot
    # =========================================
    print("\n[3/4] Generating results...")
    
    # Sort by dimension
    sorted_dims = sorted([int(k) for k in all_results.keys()])
    val_losses = [all_results[str(d)]['best_val_loss'] for d in sorted_dims]
    train_losses = [all_results[str(d)]['train_loss'] for d in sorted_dims]
    
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    plt.plot(sorted_dims, val_losses, 'o-', linewidth=2, label='Validation Loss', markersize=8)
    plt.plot(sorted_dims, train_losses, 's--', alpha=0.5, label='Training Loss', markersize=6)
    
    plt.xlabel('Latent Dimension')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Autoencoder Sweep: Latent Dimension ({ticker})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted_dims)  # Show all tested dims on x-axis
    
    # Annotate points
    for d, v in zip(sorted_dims, val_losses):
        plt.annotate(f'{v:.3f}', (d, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    
    plt.savefig(output_dir / "latent_dim_elbow.png", dpi=150)
    plt.close()
    
    # =========================================
    # 4. Summary
    # =========================================
    print("\n[4/4] Sweep Summary")
    print("-" * 65)
    print(f"{'Dim':<6} | {'Val Loss':<10} | {'Train Loss':<10} | {'Epoch':<6} | {'Params':<10}")
    print("-" * 65)
    
    for dim in sorted_dims:
        r = all_results[str(dim)]
        print(f"{dim:<6} | {r['best_val_loss']:.6f}   | {r['train_loss']:.6f}   | {r['best_epoch']:<6} | {r.get('params', 'N/A'):<10}")
    print("-" * 65)
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Plot saved to: {output_dir / 'latent_dim_elbow.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep Latent Dimensions")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16, 32], help="Dimensions to test")
    parser.add_argument("--ticker", type=str, default="BTC-USD", help="Ticker symbol")
    
    args = parser.parse_args()
    
    run_latent_dim_sweep(args.dims, args.ticker)
