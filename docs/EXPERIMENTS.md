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

## Key Learnings Summary

### 1. Look-Ahead Bias is Silent and Deadly
In-sample results looked reasonable (1.36 Sharpe spread). Walk-forward revealed the REAL signal was even stronger (3.50) once we fixed the stationarity issue. Without walk-forward testing, we would have deployed a flawed model.

### 2. Garbage In, Garbage Out (Feature Engineering)
Including `yield_10y` (raw level) caused GMM to cluster time periods instead of regimes. One bad feature can ruin everything.

### 3. Complexity ≠ Performance
The simple GMM + PCA baseline (3.50 Sharpe spread) outperforms the complex neural network (0.79). More parameters = more ways to overfit.

### 4. The Visualization Tax
3D visualization is beautiful but costs ~75% of predictive power. Use projection methods (t-SNE, UMAP) instead of forcing low-dimensional latent spaces.

---

## Future Experiments

### Experiment 5: 8D Autoencoder (NEXT)

**Date:** Pending  
**Status:** Ready to run  
**Priority:** HIGH

**Hypothesis:** 8D latent space will achieve regime separation closer to GMM baseline while providing smoother regime assignments.

**Planned Method:**
- AutoencoderConfig(latent_dim=8)
- Same walk-forward protocol
- Compare Sharpe spread to 3D (0.79) and GMM (3.50)
- Measure regime stability (fewer flips)

**Success Criteria:**
- Sharpe spread > 2.0
- Regime stability > 60%

---

### Experiment 6: Statistical Significance Audit

**Date:** Pending  
**Status:** Infrastructure ready  
**Priority:** HIGH

**Hypothesis:** Bootstrap CI will reveal true confidence in regime performance.

**Planned Method:**
```python
from src.analysis import bootstrap_sharpe_ci, validate_alpha
result = bootstrap_sharpe_ci(regime_returns)
# Require: CI lower bound > 0
```

**Success Criteria:**
- Best regime: 95% CI lower bound > 0
- Worst regime: 95% CI upper bound < 0
- Spread significant after Bonferroni correction

---

### Experiment 7: Transaction Cost Survival

**Date:** Pending  
**Status:** Infrastructure ready  
**Priority:** HIGH

**Hypothesis:** Strategy survives realistic trading friction.

**Planned Method:**
```python
from src.backtest import CostModel, analyze_cost_impact
costs = CostModel(spread_bps=5, slippage_bps=2, commission_bps=3)
result = analyze_cost_impact(returns, positions, costs)
# Require: net_sharpe > 0.5
```

**Success Criteria:**
- Net Sharpe (after 20 bps round-trip) > 0.5
- Alpha survives costs: True

---

### Experiment 8: Factor Attribution

**Date:** Pending  
**Status:** Infrastructure ready  
**Priority:** MEDIUM

**Hypothesis:** Strategy alpha is orthogonal to known factors.

**Planned Method:**
```python
from src.analysis import factor_attribution
factors = pd.DataFrame({
    'market': btc_returns,
    'momentum': momentum_20d,
})
result = factor_attribution(strategy_returns, factors)
# Require: residual alpha significant
```

**Success Criteria:**
- Residual alpha t-stat > 2.0
- R-squared < 0.5 (not just factor exposure)

---

### Future Experiments (Lower Priority)

1. **Regime Stability**: Compare flicker between GMM (54.6%) and autoencoder
2. **Ensemble**: Combine GMM + Autoencoder + HMM predictions
3. **Alternative Assets**: Test on ETH-USD, S&P 500
4. **Higher Frequency**: Test on hourly data
5. **On-chain Features**: Add exchange flows, whale movements

