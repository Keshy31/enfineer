"""
Hyperparameter Sweep: Latent Dimension
======================================
Tests different latent space dimensions to find the optimal size
using the "Elbow Method".

Dimensions to test: [2, 4, 8, 16, 32]
"""

import sys
from pathlib import Path
# Force UTF-8 output for Windows consoles
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple

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

def run_latent_dim_sweep():
    print("=" * 70)
    print("HYPERPARAMETER SWEEP: LATENT DIMENSION")
    print("=" * 70)
    print()

    # =========================================
    # 1. Load Data
    # =========================================
    print("[1/4] Loading and preparing data...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        "BTC-USD", "1d",
        feature_set="combined",
        start="2020-01-01",
    )
    
    # Filter features
    temporal_cols = get_temporal_feature_cols()
    macro_cols = [c for c in get_macro_feature_cols() if c in features.columns]
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
    
    latent_dims = [2, 4, 8, 16, 32]
    results = {}
    
    train_config = TrainingConfig(
        epochs=100,
        early_stopping_patience=15,
        learning_rate=1e-3,
        batch_size=32,
        lambda_macro=2.0,
        save_best_only=True,
    )
    
    for dim in latent_dims:
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
        
        # Train (verbose=False to reduce noise)
        history = trainer.fit(train_loader, val_loader, verbose=False)
        
        results[dim] = {
            'best_val_loss': history.best_val_loss,
            'train_loss': history.train_losses[history.best_epoch - 1],
            'best_epoch': history.best_epoch,
        }
        
        print(f"    -> Best Val Loss: {history.best_val_loss:.6f} (Epoch {history.best_epoch})")

    # =========================================
    # 3. Analyze & Plot
    # =========================================
    print("\n[3/4] Generating results...")
    
    dims = list(results.keys())
    val_losses = [results[d]['best_val_loss'] for d in dims]
    train_losses = [results[d]['train_loss'] for d in dims]
    
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, val_losses, 'o-', linewidth=2, label='Validation Loss')
    plt.plot(dims, train_losses, 's--', alpha=0.5, label='Training Loss')
    
    plt.xlabel('Latent Dimension')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoencoder Sweep: Latent Dimension "Elbow Curve"')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(dims)
    
    # Annotate points
    for d, v in zip(dims, val_losses):
        plt.annotate(f'{v:.4f}', (d, v), xytext=(0, 10), textcoords='offset points', ha='center')
    
    plt.savefig(output_dir / "latent_dim_elbow.png", dpi=150)
    plt.close()
    
    # =========================================
    # 4. Summary
    # =========================================
    print("\n[4/4] Sweep Summary")
    print("-" * 40)
    print(f"{'Dim':<6} | {'Val Loss':<10} | {'Train Loss':<10} | {'Epoch':<6}")
    print("-" * 40)
    
    for dim in dims:
        r = results[dim]
        print(f"{dim:<6} | {r['best_val_loss']:.6f}   | {r['train_loss']:.6f}   | {r['best_epoch']:<6}")
    print("-" * 40)
    print(f"\nPlot saved to: {output_dir / 'latent_dim_elbow.png'}")

if __name__ == "__main__":
    run_latent_dim_sweep()

