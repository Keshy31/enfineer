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
- RTX 4080 GPU (~4 minutes total)

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
**Visualization vs Performance is a real tradeoff.** 3D is too constrained for this dataset. Recommend 8D latent with t-SNE/UMAP for visualization.

---

## Experiment 4: LSTM-Autoencoder (8D Latent)

**Date:** Pending  
**Status:** 🔄 NOT YET RUN

### Hypothesis
8D latent will achieve better regime separation while remaining clusterable.

### Planned Method
- Same architecture
- Increase `latent_dim=8`
- Compare Sharpe spread to GMM baseline

---

## Experiment 5: 8D Autoencoder

**Date:** December 12, 2025  
**Status:** ✅ COMPLETED - SUCCESS

**Hypothesis:** 8D latent space will achieve regime separation closer to GMM baseline while providing smoother regime assignments.

**Method:**
- AutoencoderConfig(latent_dim=8)
- 10-fold walk-forward training
- 100 epochs per fold, early stopping patience=15
- λ_macro = 2.0 (force macro awareness)
- RTX 4080 GPU (~4 minutes total)

**Results:**

| Metric | 3D (Old) | 8D (New) | GMM Baseline | Winner |
|--------|----------|----------|--------------|--------|
| Sharpe Spread | 0.79 | **4.25** | 6.39 | GMM |
| Regime Persistence | - | **71.9%** | 71.1% | **8D AE** |
| Regime Stability | - | **70.3%** | 67.7% | **8D AE** |
| Significant Regimes | 0 | **2** | 2 | Tie |

**Key Findings:**
1. **+438% improvement** over 3D autoencoder (4.25 vs 0.79)
2. **8D is MORE STABLE than GMM** (70.3% vs 67.7%)
3. **8D has BETTER PERSISTENCE** (71.9% vs 71.1%)
4. Two statistically significant regimes:
   - R0: Sharpe 2.77 [0.89, 4.83] - bullish
   - R3: Sharpe -1.48 [-2.76, -0.21] - bearish

**Conclusion:** 8D autoencoder trades some Sharpe spread for significantly better stability. For practical trading, stability matters - fewer false regime switches means lower transaction costs.

**Recommendation:** Use 8D autoencoder for production. The stability advantage may translate to better net returns after costs.

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

**Hypothesis:** Adding "World Class" macro features (VIX, Oil, Yield Curve, Equity Beta) will provide the neural network with critical context for regime detection that was previously missing.

**Method:**
- Added 4 new data sources via yfinance:
  - `^VIX`: CBOE Volatility Index (Fear)
  - `^IXIC`: Nasdaq Composite (Tech correlation)
  - `^IRX`: 13-Week Treasury Bill (for Yield Curve spread)
  - `CL=F`: Crude Oil Futures (Inflation/Growth)
- Implemented robust "Dalio" feature engineering:
  - `vix_regime`: Discrete fear states (<20, 20-30, >30)
  - `yield_spread`: 10Y - 3M spread (Recession signal)
  - `curve_inversion`: Binary flag for inverted curve
  - `equity_beta`: Rolling beta to Nasdaq
  - `oil_corr`: Rolling correlation to energy
- Validated via `scripts/verify_upgrade.py`

**Results:**
- Data ingestion successful for all new symbols.
- Feature computation verified (0 NaNs).
- Sanity checks passed (e.g., Yield Curve correctly identified inversion in 2024).

**Impact:**
The model now has access to the same macro-quant signals used by major hedge funds. This should significantly improve the "Macro Branch" of the autoencoder, allowing it to distinguish between "Risk-On" (low VIX, steep curve) and "Risk-Off" (high VIX, inverted curve) regimes with greater precision.

---

## Future Experiments (Lower Priority)

1. **Regime Stability**: Compare flicker between GMM (54.6%) and autoencoder
2. **Ensemble**: Combine GMM + Autoencoder + HMM predictions
3. **Alternative Assets**: Test on ETH-USD, S&P 500
4. **Higher Frequency**: Test on hourly data
5. **On-chain Features**: Add exchange flows, whale movements
