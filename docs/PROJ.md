# The Simons-Dalio Regime Engine: Project Overview

**Version 1.1**

**Date: December 12, 2025**

**Status: Validated with Walk-Forward Testing**

---

## Abstract

The Simons-Dalio Regime Engine is a quantitative trading system designed to detect and exploit market regimes in high-frequency financial data. This project fuses the geometric rigor of Jim Simons' approach with the macroeconomic awareness of Ray Dalio, implementing a **Multi-Modal LSTM-Autoencoder** to map complex market states into a stable, low-dimensional phase space. The system provides actionable trading signals by clustering these states into distinct regimes (e.g., "Bull Volatile," "Bear Grind") and filtering them through strict liquidity constraints. This document details the system architecture, the deep learning methodology, and the execution logic that underpins the engine.

---

## 1. Introduction

Traditional technical analysis relies on linear indicators (RSI, MACD) that often fail during regime shifts, while pure fundamental analysis often lacks timing precision. The Simons-Dalio Regime Engine bridges this gap by treating the market as a dynamic system moving through a high-dimensional phase space.

The core objectives of this project are:

* **Geometric Insight:** To visualize the "shape" of the market by compressing 50+ variables into a 3D coordinate system.
* **Regime Detection:** To algorithmically identify hidden market states (regimes) that dictate future probability distributions.
* **Macro-Technical Fusion:** To ensure that price patterns are interpreted within their correct economic context (e.g., rising vs. falling yield environments).
* **Actionability:** To produce clear, liquidity-aware Buy/Sell signals suitable for practical execution.

---

## 2. System Architecture

The engine operates on a four-stage pipeline, moving from raw data to executed trade. This modular design allows for independent optimization of the neural network and the trading logic.



[Image of neural network architecture diagram]


* **Data Ingestion Layer:** Responsible for fetching, cleaning, and normalizing disparate data sources (Price, Volume, Macro). It enforces stationarity via rigorous Z-scoring.
* **Neural Processing Layer:** The core "Brain" of the system. A PyTorch-based Multi-Modal Autoencoder that compresses time-series and static data into a latent representation.
* **Regime Classification Layer:** A Gaussian Mixture Model (GMM) that maps the neural output to discrete market states.
* **Execution Layer:** A final logic gate that applies "Druckenmiller-style" liquidity and risk filters before authorizing a signal.

---

## 3. Data Engineering & Stationarity

Neural networks require stationary inputs to function reliably. The system implements a robust preprocessing pipeline to transform unbounded financial data into a normalized distribution.

### 3.1. The "Quiet Market" Trap Fix

Standard Z-scoring ($Z = \frac{x - \mu}{\sigma}$) fails when volatility ($\sigma$) approaches zero, causing signal explosions during quiet markets. We implement a **Volatility Floor** mechanism:

$$Z_{adjusted} = \frac{x - \mu}{\max(\sigma, \sigma_{floor})}$$

### 3.2. Feature Set

The model consumes a hybrid feature vector:
1.  **Simons Features (Time-Series):** 30-day window of OHLCV data, converted to log-returns and rolling Z-scores.
2.  **Dalio Features (Static Context):** Snapshot of macro drivers, including US 10Y Yields (`^TNX`), Dollar Index (`DX-Y.NYB`), and Gold Correlations.

---

## 4. The Neural Architecture (The "Brain")

The heart of the system is a **Multi-Modal LSTM-Autoencoder**. It is designed to learn a compressed, geometric representation of the market state.

### 4.1. Dual-Branch Encoder

To integrate disparate data types, the encoder splits into two branches:
* **The Temporal Branch (LSTM):** Processes the 30-day sequence of price action. It captures momentum, volatility clusters, and path-dependency.
* **The Macro Branch (Dense):** Processes the static macro snapshot. It provides the "economic coordinates" of the current environment.

### 4.2. Latent Fusion & Loss Function

The two branches merge into a dense layer that compresses the information into a latent space.

To ensure the network pays attention to the slow-moving macro data, we utilize an **Auxiliary Reconstruction Loss**:

$$L_{total} = L_{price\_recon} + \lambda \cdot L_{macro\_recon}$$

Where $\lambda$ is a weighting factor that penalizes the network heavily if it fails to encode the macro state, forcing the latent space to respect economic reality.

### 4.3. Latent Dimensionality (Updated December 2025)

The original design specified **3 latent dimensions** for direct visualization. Empirical walk-forward testing revealed this is too restrictive:

| Configuration | OOS Sharpe Spread | Visualization | Recommendation |
|---------------|-------------------|---------------|----------------|
| 3D (original) | 0.79 | Direct 3D plot | Too constrained |
| 8D | TBD | Via t-SNE/UMAP | **Recommended** |
| 17D (PCA only) | 3.50 | Via t-SNE/UMAP | GMM baseline |

**Key Finding:** PCA analysis shows 17 dimensions explain 95% of feature variance. Compressing to 3D loses critical regime-distinguishing information.

**Recommended Approach:**
- Train with **8 latent dimensions** for optimal regime detection
- Use **t-SNE or UMAP** projection to visualize in 2D/3D
- This preserves predictive power while maintaining interpretability

The 8D choice balances:
1. **Signal preservation**: Retains major market factors
2. **Regularization**: Forces abstraction (discards noise)
3. **GMM compatibility**: Few enough dimensions to cluster robustly

---

## 5. Regime Detection & Strategy Logic

Once the market state is mapped to a 3D point $(x, y, z)$, the system determines the optimal trading action.

### 5.1. Gaussian Mixture Models (GMM)

We utilize GMM to cluster the historical latent space into 5 distinct regimes. Unlike K-Means, GMM allows for "soft" boundaries and elliptical cluster shapes, which better approximate organic market behavior.
* **Regime A (Green):** Bullish / Low Volatility (High Sharpe).
* **Regime B (Red):** Bearish / High Volatility (Crash Risk).
* **Regime C (Grey):** Noise / Mean Reversion (Untradeable).

### 5.2. The Signal Generation

1.  **Inference:** The live market data is passed through the frozen `HedgeFundBrain` to obtain today's coordinate.
2.  **Classification:** The system calculates the probability of the coordinate belonging to a "Bullish" cluster.
3.  **Trigger:** If $P(Bullish) > Threshold$, a preliminary BUY signal is generated.

---

## 6. Execution & Risk Management

A raw signal is insufficient for professional trading. The Execution Layer acts as the final gatekeeper, modeled after the risk management style of Stanley Druckenmiller.

### 6.1. The Liquidity Filter

Trades are rejected if market microstructure is fragile, regardless of the neural network's confidence.
* **Spread Check:** Reject if `Bid-Ask Spread > 0.05%`.
* **Volume Check:** Reject if `Current Volume < 50% of 30-Day Avg`.

### 6.2. Visual Dashboard

The system outputs to a **Streamlit** interface featuring a 3D WebGL scatter plot.
* **Visualization:** Historical regimes appear as colored, transparent clouds.
* **Localization:** The current market state is rendered as a bright star, allowing the trader to visually confirm if the market is entering a "danger zone" (e.g., drifting from a Green cloud into a Red cloud).

---

## 7. Critical Validation: Walk-Forward Testing

A key lesson from initial development: **in-sample backtests are meaningless**. The system was validated using strict walk-forward methodology.

### 7.1. The Look-Ahead Bias Problem

Standard backtesting fits models on all data, then tests on the same data. This creates artificial performance:

```
WRONG (In-Sample):
  Fit GMM on 2020-2025 → Test on 2020-2025 → "Sharpe spread: 1.36"
  
CORRECT (Walk-Forward):
  Fit GMM on 2020-2022 → Test on 2023 (unseen)
  Fit GMM on 2020-2023 → Test on 2024 (unseen)
  Stitch results → "Sharpe spread: 3.50"
```

### 7.2. Stationarity Requirements

Non-stationary features (raw price levels, yield levels) cause GMM to cluster **time periods** rather than **market regimes**. The final system uses ONLY:
- Log returns and differences
- Z-scored values
- Rolling correlations

Excluded: `yield_10y` (raw level), raw price, raw index values.

### 7.3. Validated Baseline Performance (December 2025)

| Metric | Walk-Forward GMM | Notes |
|--------|------------------|-------|
| OOS Sharpe Spread | 3.50 | Regime 7: 2.29, Regime 0: -1.21 |
| Strategy Sharpe | 1.03 | Out-of-sample only |
| Strategy Return | +506% | vs Buy-Hold +253% |
| Time in Market | 74% | Avoided worst regimes |
| Test Period | 2021-07 to 2025-09 | 1,260 trading days |

---

## 8. Conclusion

The Simons-Dalio Regime Engine represents a synthesis of three distinct investment philosophies: geometric modeling, macroeconomic context, and rigorous risk management.

**Key Achievements:**
1. Walk-forward validated regime detection (no look-ahead bias)
2. Features that capture market dynamics, not time periods
3. Clear separation between bullish and bearish regimes

**Current Recommendation:** The walk-forward GMM baseline (17 PCA dimensions, 8 clusters) outperforms the 3D autoencoder. Future work should:
1. Test 8D autoencoder latent space
2. Compare regime stability (autoencoder may reduce "flicker")
3. Integrate transaction costs into backtest

This architecture provides a scalable foundation for future research, including the integration of alternative data (sentiment) and reinforcement learning execution agents.