# Simons-Dalio Regime Engine

A quantitative trading system that fuses geometric price analysis (Simons) with macroeconomic awareness (Dalio) to detect and exploit market regimes.

## Status (December 2025) - v1.2

- [x] **Phase 1:** Data Layer (Parquet caching, feature engineering)
- [x] **Phase 2:** Walk-Forward Validation Framework
- [x] **Phase 3:** GMM Regime Detection (Baseline)
- [x] **Phase 4:** LSTM-Autoencoder (8D recommended)
- [x] **Phase 5:** Statistical Rigor (Bootstrap CI, Factor Attribution)
- [x] **Phase 6:** Transaction Cost Modeling
- [ ] Phase 7: Live Paper Trading

### Validated Results (Out-of-Sample)

| Metric | Walk-Forward GMM |
|--------|------------------|
| Sharpe Spread | 3.50 |
| Strategy Sharpe | 1.03 |
| Strategy Return | +506% |
| Buy & Hold | +253% |
| Test Period | 2021-07 to 2025-09 |

**v1.2 Note:** These results need statistical validation (bootstrap CI, factor attribution) before deploying capital.

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run walk-forward benchmark (honest out-of-sample test)
python scripts/test_benchmark_walkforward.py

# Train autoencoder (requires GPU for speed)
python scripts/train_autoencoder.py

# Test new statistical rigor modules
python src/backtest/costs.py
python src/analysis/statistical_tests.py
```

## Usage

```python
from src.data import DataManager, WalkForwardGMM, get_stationary_features

dm = DataManager("./data")

# Get combined features (Simons + Dalio)
features = dm.get_features("BTC-USD", "1d", feature_set="combined")

# Walk-forward regime detection
wf = WalkForwardGMM(n_regimes=8, use_pca=True)
results = wf.fit_predict(features, get_stationary_features())

print(f"OOS Sharpe spread: {results.regime_stats['sharpe'].max() - results.regime_stats['sharpe'].min():.2f}")
```

### NEW: Statistical Validation (v1.2)

```python
from src.analysis import bootstrap_sharpe_ci, validate_alpha
from src.backtest import CostModel, analyze_cost_impact

# Bootstrap confidence interval for Sharpe
result = bootstrap_sharpe_ci(strategy_returns)
print(f"Sharpe: {result.point_estimate:.2f}, 95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
print(f"Statistically significant: {result.significant}")

# Transaction cost analysis
costs = CostModel(spread_bps=5, slippage_bps=2, commission_bps=3)
impact = analyze_cost_impact(returns, positions, costs)
print(f"Net Sharpe (after costs): {impact['net_sharpe']:.2f}")
print(f"Alpha survives costs: {impact['alpha_survives']}")
```

## Project Structure

```
src/
├── data/           # Data fetching, features, walk-forward validation
├── models/         # LSTM-Autoencoder (HedgeFundBrain)
├── backtest/       # NEW: Transaction cost modeling
└── analysis/       # NEW: Statistical tests, factor attribution
```

## Key Files

| File | Purpose |
|------|---------|
| `src/data/manager.py` | Data fetching and caching |
| `src/data/walk_forward.py` | Walk-forward validation |
| `src/models/autoencoder.py` | LSTM-Autoencoder (HedgeFundBrain) |
| `src/backtest/costs.py` | **NEW:** Transaction cost model |
| `src/analysis/statistical_tests.py` | **NEW:** Bootstrap CI, factor attribution |
| `scripts/test_benchmark_walkforward.py` | Honest regime benchmark |

## Documentation

- [Project Overview](docs/PROJ.md) — Vision, architecture, statistical rigor requirements
- [Technical Spec](docs/TECH.md) — Data layer, walk-forward framework
- [Experiment Log](docs/EXPERIMENTS.md) — What we tested and next experiments

## Success Criteria (v1.2)

Before deploying capital, ALL criteria must pass:

| Test | Threshold | Status |
|------|-----------|--------|
| Bootstrap Sharpe CI > 0 | Lower bound > 0 | Pending |
| Net Sharpe (after costs) | > 0.5 | Pending |
| Factor-adjusted alpha | Significant | Pending |
| Regime persistence | > 3 days avg | Pending |

## Hardware

- **CPU:** Works but slow (~50 minutes for autoencoder)
- **GPU:** Recommended (RTX 4080 → ~4 minutes)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```
