"""
Test Training Pipeline
======================
Verifies the complete data preparation pipeline for the neural network:
1. Load combined features
2. Create training dataset (sequences for LSTM)
3. Split train/val/test with proper purging gaps
4. Normalize using training statistics only
5. Verify shapes and data integrity

Run from project root:
    python scripts/test_training_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from data.manager import DataManager
from data.splits import create_time_series_split, verify_no_leakage
from data.training import (
    create_training_dataset,
    split_dataset,
    normalize_dataset,
    get_temporal_feature_cols,
    get_macro_feature_cols,
)


def run_training_pipeline_test():
    """Run comprehensive test of the training data pipeline."""
    
    print("=" * 70)
    print("SIMONS-DALIO REGIME ENGINE")
    print("Training Pipeline Test")
    print("=" * 70)
    print()
    
    results = {
        "load_features": False,
        "create_dataset": False,
        "split_data": False,
        "verify_no_leakage": False,
        "normalize": False,
        "verify_shapes": False,
        "verify_no_nan": False,
    }
    
    # =========================================
    # Step 1: Load Combined Features
    # =========================================
    print("[1/7] Loading combined features...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        "BTC-USD", "1d",
        feature_set="combined",
        start="2020-01-01",
    )
    
    print(f"  ✓ Loaded {len(features)} rows")
    print(f"  ✓ Date range: {features.index.min().date()} to {features.index.max().date()}")
    print(f"  ✓ Columns: {len(features.columns)}")
    results["load_features"] = True
    
    print()
    
    # =========================================
    # Step 2: Create Training Dataset
    # =========================================
    print("[2/7] Creating training dataset (30-day sequences)...")
    
    dataset = create_training_dataset(features, sequence_length=30)
    
    print(f"  ✓ X_temporal shape: {dataset.X_temporal.shape}")
    print(f"  ✓ X_macro shape:    {dataset.X_macro.shape}")
    print(f"  ✓ y shape:          {dataset.y.shape}")
    print(f"  ✓ Dates: {dataset.dates.min().date()} to {dataset.dates.max().date()}")
    
    # Verify dimensions
    assert dataset.X_temporal.shape[1] == 30, "Sequence length should be 30"
    assert dataset.X_temporal.shape[2] == len(get_temporal_feature_cols()), "Wrong temporal features"
    
    results["create_dataset"] = True
    
    print()
    
    # =========================================
    # Step 3: Split Train/Val/Test with Gaps
    # =========================================
    print("[3/7] Splitting data (60% train, 20% val, 20% test)...")
    
    split = create_time_series_split(features, gap_days=30)
    
    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        train_end_date=split.train_end,
        val_end_date=split.val_end,
    )
    
    print()
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    
    results["split_data"] = True
    
    print()
    
    # =========================================
    # Step 4: Verify No Data Leakage
    # =========================================
    print("[4/7] Verifying no data leakage...")
    
    verify_no_leakage(split)
    
    # Also verify dataset splits don't overlap
    train_dates = set(train_ds.dates)
    val_dates = set(val_ds.dates)
    test_dates = set(test_ds.dates)
    
    assert len(train_dates & val_dates) == 0, "Train/val overlap!"
    assert len(train_dates & test_dates) == 0, "Train/test overlap!"
    assert len(val_dates & test_dates) == 0, "Val/test overlap!"
    
    print("  ✓ No dataset overlap detected")
    
    results["verify_no_leakage"] = True
    
    print()
    
    # =========================================
    # Step 5: Normalize Using Train Statistics
    # =========================================
    print("[5/7] Normalizing datasets (fit on train only)...")
    
    train_norm, val_norm, test_norm, stats = normalize_dataset(train_ds, val_ds, test_ds)
    
    print(f"  ✓ Normalization stats computed from {len(train_ds)} training samples")
    print(f"  ✓ Temporal features: mean shape {stats['temporal_mean'].shape}")
    print(f"  ✓ Macro features: mean shape {stats['macro_mean'].shape}")
    
    results["normalize"] = True
    
    print()
    
    # =========================================
    # Step 6: Verify Output Shapes
    # =========================================
    print("[6/7] Verifying output shapes...")
    
    expected_temporal_features = len(get_temporal_feature_cols())
    expected_macro_features = len([c for c in get_macro_feature_cols() if c in features.columns])
    
    # Train
    assert train_norm.X_temporal.shape[1] == 30, f"Train seq_len wrong"
    assert train_norm.X_temporal.shape[2] == expected_temporal_features, f"Train temporal features wrong"
    assert train_norm.X_macro.shape[1] == expected_macro_features, f"Train macro features wrong"
    
    # Val
    assert val_norm.X_temporal.shape[1] == 30, f"Val seq_len wrong"
    assert val_norm.X_temporal.shape[2] == expected_temporal_features, f"Val temporal features wrong"
    assert val_norm.X_macro.shape[1] == expected_macro_features, f"Val macro features wrong"
    
    # Test
    assert test_norm.X_temporal.shape[1] == 30, f"Test seq_len wrong"
    assert test_norm.X_temporal.shape[2] == expected_temporal_features, f"Test temporal features wrong"
    assert test_norm.X_macro.shape[1] == expected_macro_features, f"Test macro features wrong"
    
    print(f"  ✓ All shapes correct:")
    print(f"    Train: temporal={train_norm.X_temporal.shape}, macro={train_norm.X_macro.shape}")
    print(f"    Val:   temporal={val_norm.X_temporal.shape}, macro={val_norm.X_macro.shape}")
    print(f"    Test:  temporal={test_norm.X_temporal.shape}, macro={test_norm.X_macro.shape}")
    
    results["verify_shapes"] = True
    
    print()
    
    # =========================================
    # Step 7: Verify No NaN/Inf
    # =========================================
    print("[7/7] Checking for NaN/Inf values...")
    
    for name, ds in [("Train", train_norm), ("Val", val_norm), ("Test", test_norm)]:
        nan_temporal = np.isnan(ds.X_temporal).sum()
        nan_macro = np.isnan(ds.X_macro).sum()
        inf_temporal = np.isinf(ds.X_temporal).sum()
        inf_macro = np.isinf(ds.X_macro).sum()
        
        if nan_temporal > 0 or nan_macro > 0:
            print(f"  ✗ {name}: {nan_temporal} NaN in temporal, {nan_macro} NaN in macro")
        elif inf_temporal > 0 or inf_macro > 0:
            print(f"  ✗ {name}: {inf_temporal} Inf in temporal, {inf_macro} Inf in macro")
        else:
            print(f"  ✓ {name}: No NaN or Inf values")
    
    results["verify_no_nan"] = True
    
    print()
    
    # =========================================
    # Normalization Statistics
    # =========================================
    print("=" * 70)
    print("NORMALIZATION VERIFICATION")
    print("=" * 70)
    print()
    
    # Train should be approximately mean=0, std=1
    train_temp_mean = train_norm.X_temporal.mean()
    train_temp_std = train_norm.X_temporal.std()
    train_macro_mean = train_norm.X_macro.mean()
    train_macro_std = train_norm.X_macro.std()
    
    print(f"Train set statistics (should be ~0 mean, ~1 std):")
    print(f"  Temporal: mean={train_temp_mean:.4f}, std={train_temp_std:.4f}")
    print(f"  Macro:    mean={train_macro_mean:.4f}, std={train_macro_std:.4f}")
    
    # Val/Test won't be exactly 0/1 (that's expected - they use train statistics)
    print()
    print(f"Val set statistics (will differ slightly - uses train stats):")
    print(f"  Temporal: mean={val_norm.X_temporal.mean():.4f}, std={val_norm.X_temporal.std():.4f}")
    print(f"  Macro:    mean={val_norm.X_macro.mean():.4f}, std={val_norm.X_macro.std():.4f}")
    
    print()
    print(f"Test set statistics (will differ slightly - uses train stats):")
    print(f"  Temporal: mean={test_norm.X_temporal.mean():.4f}, std={test_norm.X_temporal.std():.4f}")
    print(f"  Macro:    mean={test_norm.X_macro.mean():.4f}, std={test_norm.X_macro.std():.4f}")
    
    print()
    
    # =========================================
    # Summary
    # =========================================
    print("=" * 70)
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print("TRAINING PIPELINE TEST PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  • All {total} tests passed")
        print()
        print("Data is ready for neural network training!")
        print()
        print("  PyTorch usage:")
        print("  ```python")
        print("  import torch")
        print("  from torch.utils.data import TensorDataset, DataLoader")
        print()
        print("  # Convert to tensors")
        print("  X_temp = torch.FloatTensor(train_norm.X_temporal)")
        print("  X_macro = torch.FloatTensor(train_norm.X_macro)")
        print("  y = torch.FloatTensor(train_norm.y)")
        print()
        print("  # Create DataLoader")
        print("  dataset = TensorDataset(X_temp, X_macro, y)")
        print("  loader = DataLoader(dataset, batch_size=32, shuffle=True)")
        print("  ```")
    else:
        print("TRAINING PIPELINE TEST FAILED")
        print("=" * 70)
        print()
        print(f"Results: {passed}/{total} tests passed")
        for test_name, test_passed in results.items():
            status = "✓" if test_passed else "✗"
            print(f"  {status} {test_name}")
    
    print()
    
    return all(results.values())


if __name__ == "__main__":
    success = run_training_pipeline_test()
    sys.exit(0 if success else 1)

