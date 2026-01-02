# Experiment Log

**Project:** Simons-Dalio Regime Engine  
**Purpose:** Track experiments, findings, and lessons learned

---

## Experiment 1: Original Benchmark (In-Sample)

**Date:** December 12, 2025  
**Status:** ❌ INVALIDATED

### Hypothesis
Direct GMM clustering on features would reveal market regimes.

### Method
- Fit StandardScaler on ALL data
- Fit GMM on ALL data
- Measure regime performance on SAME data

### Results
- Sharpe Spread: 1.36
- Strategy Return: +707%
- Looked promising!

### Problem Discovered
This was **in-sample testing** disguised as a backtest. The model "knew" future data while classifying past dates. Results are meaningless for live trading.

### Lesson
**Never fit on data you'll test on.** Always use walk-forward validation.

---

## Experiment 2: Walk-Forward GMM Baseline

**Date:** December 12, 2025  
**Status:** ✅ VALIDATED

### Hypothesis
Walk-forward validation would reveal true out-of-sample performance.

### Method
- 504-day minimum training window (2 years)
- 126-day test window (6 months)
- 30-day purge gap (prevents feature leakage)
- **Stationary features only** (removed yield_10y, raw levels)
- PCA to 17 dimensions (95% variance)
- BIC-optimal cluster count: 8

### Results
| Metric | Value |
|--------|-------|
| OOS Sharpe Spread | **3.50** |
| Strategy Sharpe | **1.03** |
| Strategy Return | +506% |
| Buy & Hold Return | +253% |
| Time in Market | 74% |
| Regime Stability | 54.6% |

### Key Findings
1. OOS performance (3.50) is actually BETTER than in-sample (1.36) after removing non-stationary features
2. Features capture real market dynamics, not time periods
3. Clear separation between bullish (Regime 7: Sharpe 2.29) and bearish (Regime 0: Sharpe -1.21)

### Lesson
**Stationarity matters more than model complexity.** Removing raw levels dramatically improved regime detection.

---

## Experiment 3: LSTM-Autoencoder (3D Latent)

**Date:** December 12, 2025  
**Status:** ⚠️ UNDERPERFORMS BASELINE

### Hypothesis
Neural network compression to 3D would maintain regime separation while enabling visualization.

### Method
- LSTM encoder (64 hidden, 2 layers)
- Dense macro encoder (32 hidden)
- 3D latent fusion
- λ_macro = 2.0 (force macro awareness)
- Walk-forward training (10 folds)

### Results
| Metric | Value |
|--------|-------|
| OOS Sharpe Spread | **0.79** |
| Clusters Detected | 3 |
| Training Loss | ~1.3-1.5 |

### Analysis
The 3D constraint is too aggressive:
- PCA baseline retains 17 dimensions (95% variance)
- Autoencoder forces 28 → 3 dimensions
- Lost information = lost regime separation

### Lesson
**Visualization vs Performance is a real tradeoff.** 3D is too constrained for this dataset. Recommend 8D+ latent with t-SNE/UMAP for visualization.

---

## Experiment 4: LSTM-Autoencoder (8D Latent)

**Date:** December 29, 2025  
**Status:** ✅ COMPLETED - SUCCESS

### Hypothesis
8D latent will achieve better regime separation while remaining clusterable.

### Method
- AutoencoderConfig(latent_dim=8)
- 10-fold walk-forward training
- 100 epochs per fold
- λ_macro = 2.0

### Results

| Metric | 3D (Old) | 8D (New) | GMM Baseline | Winner |
|--------|----------|----------|--------------|--------|
| Sharpe Spread | 0.79 | **4.25** | 6.39 | GMM |
| Regime Persistence | - | **71.9%** | 71.1% | **8D AE** |
| Regime Stability | - | **70.3%** | 67.7% | **8D AE** |
| Significant Regimes | 0 | **2** | 2 | Tie |

### Conclusion
8D autoencoder trades some Sharpe spread for significantly better stability. For practical trading, stability matters.

---

## Experiment 5: Hyperparameter Sweep (Latent Dimension)

**Date:** December 31, 2025
**Status:** ✅ COMPLETED - SUCCESS

### Hypothesis
We can find the optimal latent dimension ("The Elbow") that balances reconstruction accuracy with clustering stability (Curse of Dimensionality).

### Method
- Fixed Train/Val Split (80/20) on 2020-2025 data
- Swept dimensions: [2, 4, 8, 10, 12, 14, 16, 32]
- Metric: Best Validation Loss (MSE)

### Results (The "Elbow")

| Dimension | Val Loss | Improvement | Params | Est. Samples/Param |
| :--- | :--- | :--- | :--- | :--- |
| **8D** | 1.3051 | Baseline | 97k | ~14.0x |
| **10D** | 1.0208 | -21% | 270k | ~10.5x |
| **12D** | **0.9301** | **-9%** | 270k | **~7.3x** |
| **14D** | 0.8815 | -5% | 271k | ~5.2x |
| **16D** | 0.8793 | -0.2% | 271k | ~3.7x |

### Analysis
- **Diminishing Returns:** The improvement from 12D to 16D is marginal (0.05), but the risk of overfitting the GMM clustering increases dramatically (samples per parameter drops from 7.3x to 3.7x).
- **The Sweet Spot:** 12D offers a 29% error reduction vs 8D while maintaining a safe clustering ratio.

### Conclusion
**12 Dimensions is the optimal architecture.** It captures substantially more market nuance than 8D without hitting the overfitting wall of 16D.

---

## Experiment 6: Statistical Significance Audit

**Date:** Pending  
**Status:** Infrastructure ready  
**Priority:** HIGH

**Hypothesis:** Bootstrap CI will reveal true confidence in regime performance.

---

## Experiment 7: Transaction Cost Survival

**Date:** Pending  
**Status:** Infrastructure ready  
**Priority:** HIGH

**Hypothesis:** Strategy survives realistic trading friction.

---

## Experiment 8: Factor Attribution

**Date:** Pending  
**Status:** Infrastructure ready  
**Priority:** MEDIUM

**Hypothesis:** Strategy alpha is orthogonal to known factors.

---

## Experiment 9: Feature Engineering Upgrade

**Date:** December 31, 2025  
**Status:** ✅ COMPLETED - IMPLEMENTED

**Hypothesis:** Adding "World Class" macro features (VIX, Oil, Yield Curve, Equity Beta) will provide the neural network with critical context for regime detection.

**Method:**
- Added 4 new data sources: `^VIX` (Fear), `^IXIC` (Tech), `^IRX` (Rates), `CL=F` (Oil).
- Implemented features: `yield_spread`, `curve_inversion`, `equity_beta`, `oil_corr`.

**Impact:**
Model now distinguishes "Risk-On" vs "Risk-Off" with explicit macro drivers.

---

## Experiment 10: Methodology Corrections

**Date:** January 2, 2026  
**Status:** 🔧 IN PROGRESS

### Discovery

A comprehensive audit of the testing methodology revealed several critical issues that may invalidate previous autoencoder results.

### Issue 1: In-Sample GMM Baseline in Testing

**Location:** `scripts/test_autoencoder_rigor.py`, lines 230-237

**Problem:** The GMM baseline comparison was fitted **in-sample on test data**:

```python
# WRONG (what the code did):
scaler = StandardScaler()
X_gmm_scaled = scaler.fit_transform(X_test)  # Fits on TEST data!
gmm_baseline = GaussianMixture(...).fit(X_gmm_pca)  # Fits on TEST data!
```

**Impact:** The GMM baseline "cheated" by seeing the test data during fitting, making autoencoder comparison unfair. GMM appeared to perform better than it would in true out-of-sample testing.

**Fix:** Both methods must use identical walk-forward splits where models are trained on TRAIN data and evaluated on TEST data.

### Issue 2: Wrong Metric in Hyperparameter Sweep (Experiment 5)

**Problem:** The latent dimension sweep optimized for **reconstruction loss (MSE)**, not **regime detection quality (Sharpe spread)**.

```python
# What the sweep optimized:
metric = validation_reconstruction_loss  # Lower is "better"

# What actually matters for trading:
metric = out_of_sample_sharpe_spread  # Higher regime separation
```

**Impact:** The "optimal" 12D recommendation may not hold when using the correct metric. A model that reconstructs well might cluster poorly.

**Fix:** Re-run sweep using OOS Sharpe spread as primary metric.

### Issue 3: Data Underutilization

**Problem:** Training used data from 2020-01-01, but bottleneck symbols (^VIX, ^IXIC, etc.) have data available back to 2017.

**Impact:** Lost ~730 days of training history (~33% more data).

**Fix:** Extend data fetch to 2018-01-01 for all symbols.

### Issue 4: Incomplete Data Coverage

**Problem:** The 12D model's OOS period ended at 2025-10-19, but data was available through 2025-12-31.

**Impact:** Missing ~73 trading days of recent validation data.

**Fix:** Ensure walk-forward loop uses all available data.

### Experiments Requiring Re-Validation

| Experiment | Status | Issue |
|------------|--------|-------|
| Exp 3: 3D Autoencoder | ⚠️ Uncertain | Comparison may have been unfair |
| Exp 4: 8D Autoencoder | ⚠️ Uncertain | Comparison may have been unfair |
| Exp 5: Latent Sweep | ❌ Invalid | Used wrong metric |

### Corrective Actions

1. **Create unified test framework** (`test_unified_comparison.py`)
   - Run all methods through identical walk-forward splits
   - Compare: Random, PCA+GMM, AE+GMM
   - Metric: OOS Sharpe spread, persistence, cost survival

2. **Re-run hyperparameter sweep** (`sweep_latent_dim_corrected.py`)
   - Use walk-forward validation (not fixed 80/20)
   - Primary metric: OOS Sharpe spread
   - Test dimensions: [4, 6, 8, 10, 12, 14]

3. **Extend data coverage**
   - Fetch 2018-01-01 to present for all symbols
   - Re-cache combined features

### Lesson Learned

**Apples-to-apples comparison requires identical test conditions.** Different methods must:
- Use the same train/test splits
- Be evaluated on the same OOS dates
- Be measured by downstream performance (trading metrics), not intermediate losses

---

## Experiment 11: Unified Walk-Forward Comparison

**Date:** January 2, 2026  
**Status:** 🔄 PENDING (blocked by Experiment 10 fixes)

### Hypothesis

With corrected methodology, we can determine if the autoencoder provides genuine value over PCA+GMM baseline.

### Method

- Unified walk-forward framework with identical splits
- Extended data: 2018-01-01 to 2025-12-31
- Compare: Random, PCA+GMM, AE(optimal_dim)+GMM
- Metrics: OOS Sharpe spread, persistence, net Sharpe after costs

### Expected Deliverables

1. Fair comparison table showing all methods on equal footing
2. Bootstrap confidence intervals for all metrics
3. Definitive answer: Does AE beat PCA for regime detection?

---

## Experiment 12: Corrected Hyperparameter Sweep

**Date:** January 2, 2026  
**Status:** 🔄 PENDING

### Hypothesis

The true optimal latent dimension (when optimizing for trading performance) may differ from the reconstruction-optimal 12D.

### Method

- Walk-forward sweep (not fixed split)
- Primary metric: OOS Sharpe spread
- Secondary: Regime persistence, cost survival
- Dimensions: [4, 6, 8, 10, 12, 14]

### Expected Output

- True optimal latent dimension based on regime quality
- Comparison showing reconstruction loss vs trading performance correlation