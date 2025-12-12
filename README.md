# Simons-Dalio Regime Engine

A quantitative trading system that fuses geometric price analysis (Simons) with macroeconomic awareness (Dalio) to detect and exploit market regimes.

## Status

- [x] Phase 1: Data Layer (Parquet caching, feature engineering)
- [ ] Phase 2: Neural Network (LSTM-Autoencoder)
- [ ] Phase 3: Regime Classification (GMM)
- [ ] Phase 4: Execution Layer
- [ ] Phase 5: Dashboard (Streamlit)

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Verify data layer
python scripts/test_data_layer.py
```

## Usage

```python
from src.data import DataManager

dm = DataManager("./data")

# Fetch cached OHLCV
df = dm.get_ohlcv("BTC-USD", "1d", start="2020-01-01")

# Get combined features (Simons + Dalio)
features = dm.get_features("BTC-USD", "1d", feature_set="combined")
```

## Documentation

- [Project Overview](docs/PROJ.md) — Vision, architecture, design
- [Technical Spec](docs/TECH.md) — Data layer implementation details
```

---
