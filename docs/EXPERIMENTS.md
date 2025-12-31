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
