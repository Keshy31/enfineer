# Simons-Dalio Regime Engine

A quantitative trading system that fuses geometric price analysis (Simons) with macroeconomic awareness (Dalio) to detect and exploit market regimes.

## Status (December 2025)

- [x] **Phase 1:** Data Layer (Parquet caching, feature engineering)
- [x] **Phase 2:** Walk-Forward Validation Framework
- [x] **Phase 3:** GMM Regime Detection (Baseline)
- [x] **Phase 4:** LSTM-Autoencoder (3D - needs tuning)
- [ ] Phase 5: Execution Layer
- [ ] Phase 6: Dashboard (Streamlit)

### Validated Results (Out-of-Sample)

| Metric | Walk-Forward GMM |
|--------|------------------|
| Sharpe Spread | 3.50 |
| Strategy Sharpe | 1.03 |
| Strategy Return | +506% |
| Buy & Hold | +253% |
| Test Period | 2021-07 to 2025-09 |

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

## Key Files

| File | Purpose |
|------|---------|
| `src/data/manager.py` | Data fetching and caching |
| `src/data/walk_forward.py` | Walk-forward validation |
| `src/models/autoencoder.py` | LSTM-Autoencoder (HedgeFundBrain) |
| `scripts/test_benchmark_walkforward.py` | Honest regime benchmark |

## Documentation

- [Project Overview](docs/PROJ.md) — Vision, architecture, validated results
- [Technical Spec](docs/TECH.md) — Data layer, walk-forward framework
- [Experiment Log](docs/EXPERIMENTS.md) — What we tested and learned

## Hardware

- **CPU:** Works but slow (~50 minutes for autoencoder)
- **GPU:** Recommended (RTX 4080 → ~4 minutes)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```
