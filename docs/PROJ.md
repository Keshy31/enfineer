# The Simons-Dalio Regime Engine: Project Overview

**Version 1.3**

**Date: December 31, 2025**

**Status: Optimization & Hyperparameter Tuning**

---

## Abstract

The Simons-Dalio Regime Engine is a quantitative trading system designed to detect and exploit market regimes in high-frequency financial data. This project fuses the geometric rigor of Jim Simons' approach with the macroeconomic awareness of Ray Dalio, implementing a **Multi-Modal LSTM-Autoencoder** to map complex market states into a stable, low-dimensional phase space. The system provides actionable trading signals by clustering these states into distinct regimes (e.g., "Bull Volatile," "Bear Grind") and filtering them through strict liquidity constraints. This document details the system architecture, the deep learning methodology, and the execution logic that underpins the engine.

---

## 1. Introduction

Traditional technical analysis relies on linear indicators (RSI, MACD) that often fail during regime shifts, while pure fundamental analysis often lacks timing precision. The Simons-Dalio Regime Engine bridges this gap by treating the market as a dynamic system moving through a high-dimensional phase space.

The core objectives of this project are:

* **Geometric Insight:** To visualize the "shape" of the market by compressing 50+ variables into a compact coordinate system.
* **Regime Detection:** To algorithmically identify hidden market states (regimes) that dictate future probability distributions.
* **Macro-Technical Fusion:** To ensure that price patterns are interpreted within their correct economic context (e.g., rising vs. falling yield environments).
* **Actionability:** To produce clear, liquidity-aware Buy/Sell signals suitable for practical execution.

---

## 2. System Architecture

The engine operates on a four-stage pipeline, moving from raw data to executed trade. This modular design allows for independent optimization of the neural network and the trading logic.

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
2.  **Dalio Features (Static Context):** Snapshot of macro drivers, including:
    *   **Rates:** US 10Y Yields (`^TNX`) and Yield Curve (10Y-3M).
    *   **Currency:** Dollar Index (`DX-Y.NYB`).
    *   **Risk/Fear:** VIX (`^VIX`) and Gold (`GLD`).
    *   **Growth/Inflation:** Crude Oil (`CL=F`).
    *   **Equity Beta:** Correlation with Nasdaq (`^IXIC`).

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

### 4.3. Latent Dimensionality (Updated December 31, 2025)

Our initial 3D and 8D experiments were superseded by a comprehensive hyperparameter sweep using the Elbow Method.

| Configuration | Validation Loss | Samples/Param | Recommendation |
|---------------|----------------|---------------|----------------|
| **3D** | High (~2.0) | High | Too constrained |
| **8D** | 1.3051 | ~14.0x | Good baseline |
| **12D** | **0.9301** | **~7.3x** | **Optimal** |
| **16D** | 0.8793 | ~3.7x | Diminishing returns |

**Key Finding:** 12 Dimensions represents the "Goldilocks" zone. It reduces validation loss by nearly 30% compared to 8D, capturing significantly more market nuance, while maintaining a safe ratio of ~7.3 samples per parameter for GMM clustering.

**Recommended Approach:**
- Train with **12 latent dimensions** for optimal balance.
- Use **t-SNE or UMAP** projection to visualize in 2D/3D.
- This maintains both predictive power and clustering stability.

---

## 5. Regime Detection & Strategy Logic

Once the market state is mapped to a 12D point, the system determines the optimal trading action.

### 5.1. Gaussian Mixture Models (GMM)

We utilize GMM to cluster the historical latent space into discrete regimes. Unlike K-Means, GMM allows for "soft" boundaries and elliptical cluster shapes, which better approximate organic market behavior.
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

Standard backtesting fits models on all data, then tests on the same data. This creates artificial performance. We instead fit on past data (expanding window) and test on strictly unseen future data.

### 7.2. Stationarity Requirements

Non-stationary features (raw price levels, yield levels) cause GMM to cluster **time periods** rather than **market regimes**. The final system uses ONLY log returns, Z-scores, and rolling correlations.

### 7.3. Validated Baseline Performance (December 2025)

| Metric | Walk-Forward GMM | Notes |
|--------|------------------|-------|
| OOS Sharpe Spread | 3.50 | Regime 7: 2.29, Regime 0: -1.21 |
| Strategy Sharpe | 1.03 | Out-of-sample only |
| Strategy Return | +506% | vs Buy-Hold +253% |
| Time in Market | 74% | Avoided worst regimes |

---

## 8. Statistical Rigor Requirements

**Version 1.2 Update:** A 3.50 Sharpe spread means nothing without confidence intervals. Before claiming "true alpha", we require:

### 8.1. Bootstrap Confidence Intervals

All Sharpe ratios must include 95% CI via bootstrap resampling. CI lower bound must be > 0 to claim statistical significance.

### 8.2. Multiple Testing Correction

With 8 regimes, we run 8+ hypothesis tests. We apply Bonferroni correction to p-value thresholds.

### 8.3. Factor Attribution

Prove alpha is orthogonal to known factors (Market beta, Momentum, Volatility).

### 8.4. Transaction Cost Survival

Alpha must survive realistic trading friction (~16 bps round-trip). Net Sharpe after costs must be > 0.5.

---

## 9. Roadmap to True Alpha

### Phase 1: Statistical Foundation (Current Priority)
| Task | Status | Impact |
|------|--------|--------|
| Bootstrap CI for Sharpe | Implemented | Know if signal is real |
| Transaction cost model | Implemented | Know if tradeable |
| Regime transition matrix | Implemented | Understand dynamics |
| Factor attribution | Implemented | Prove orthogonal alpha |

### Phase 2: Model Improvements
| Task | Status | Impact |
|------|--------|--------|
| 12D Hyperparameter Sweep | ✅ Completed | +30% Accuracy vs 8D |
| Ensemble (GMM + AE + HMM) | Pending | Robustness |
| Uncertainty quantification | Pending | Know when to sit out |

### Phase 3: Feature Expansion
| Task | Status | Impact |
|------|--------|--------|
| VIX, Oil, Yield Curve, Nasdaq | ✅ Completed | World-class macro signals |
| On-chain metrics | Pending | Crypto-native signals |
| Sentiment data | Pending | Contrarian signals |

### Phase 4: Production Infrastructure
| Task | Status | Impact |
|------|--------|--------|
| MLflow experiment tracking | Pending | Reproducibility |
| Model versioning | Pending | Rollback capability |
| Live paper trading | Pending | Real-world validation |

---

## 10. Conclusion

The Simons-Dalio Regime Engine represents a synthesis of three distinct investment philosophies: geometric modeling, macroeconomic context, and rigorous risk management.

**Key Achievements (v1.3):**
1. Walk-forward validated regime detection (no look-ahead bias)
2. Validated optimal architecture (12D latent space)
3. Integration of "Dalio" macro features (Yield Curve, VIX, Oil)

**Current Focus:**
The 12D autoencoder provides the optimal balance of signal capture and clustering stability. Our focus now shifts to final model training and deployment of the live signal pipeline.
