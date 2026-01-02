---
name: Robust Testing Framework
overview: Fix the testing methodology to ensure fair comparisons between Random, PCA+GMM, and Autoencoder+GMM approaches. Extend data coverage to 2018-2025, create unified walk-forward validation, and re-run hyperparameter sweep with correct metrics.
todos:
  - id: docs-tech
    content: Update TECH.md with data coverage and architecture clarification
    status: completed
  - id: docs-experiments
    content: Add Experiment 10 to EXPERIMENTS.md documenting methodology corrections
    status: completed
  - id: docs-proj
    content: Update PROJ.md roadmap and validation section
    status: completed
  - id: data-fetch
    content: Fetch historical data for bottleneck symbols back to 2018-01-01
    status: completed
  - id: data-cache
    content: Regenerate combined features cache with 2018-2025 date range
    status: completed
  - id: unified-test
    content: Create scripts/test_unified_comparison.py for fair method comparison
    status: completed
  - id: corrected-sweep
    content: Create scripts/sweep_latent_dim_corrected.py with OOS Sharpe metric
    status: completed
  - id: deprecate-old
    content: Mark test_autoencoder_rigor.py as deprecated with explanation
    status: completed
  - id: run-sweep
    content: Execute corrected sweep to find true optimal latent dimension
    status: completed
  - id: final-comparison
    content: Run unified comparison and document validated results
    status: completed
  - id: gpu-fix
    content: Ensure venv is activated for GPU-enabled PyTorch (2.6.0+cu124)
    status: completed
---

# Robust Testing Framework & Data Optimization Plan

## Context: What We Discovered

Our analysis revealed several issues with the current testing approach:

1. **Testing Flaw**: `test_autoencoder_rigor.py` fits GMM baseline in-sample on test data, creating unfair comparison
2. **Data Underutilization**: Only using 2020-2025 data when 2018+ is available; missing 2.5 months of recent data
3. **Sweep Metric Mismatch**: Hyperparameter sweep optimized reconstruction loss, not regime detection quality
4. **Architecture Clarification**: Both baseline (PCA+GMM) and autoencoder (AE+GMM) use GMM - we're comparing representation learning methods

---

## Phase 1: Documentation Updates (First Priority)

Update documentation to capture methodology learnings before implementation.

### 1.1 Update [docs/TECH.md](docs/TECH.md)

Add new section "Data Coverage Requirements":

- Document bottleneck symbols (^IRX, ^IXIC, ^VIX, CL=F) and their availability
- Specify optimal date range: 2018-01-01 to present
- Document the ~33% data improvement from extending history

Add new section "Model Architecture Clarification":

- Explain that AE is a representation learner, GMM is the clusterer
- Diagram: `Raw Features → PCA → GMM` vs `Raw Features → AE → GMM`
- Both approaches end with GMM clustering

### 1.2 Update [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

Add "Experiment 10: Methodology Corrections":

- Document the in-sample GMM bug discovered in testing
- Record that Experiment 5 (sweep) used wrong metric
- Note that results from experiments 3-5 may need re-validation

### 1.3 Update [docs/PROJ.md](docs/PROJ.md)

Update Section 7 (Walk-Forward Testing):

- Add data utilization requirements
- Document unified comparison framework
- Update roadmap with current phase

---

## Phase 2: Data Pipeline Optimization

### 2.1 Extend Historical Data Fetch

Modify [src/data/fetcher.py](src/data/fetcher.py) or run data collection:

- Fetch ^IRX, ^IXIC, ^VIX, CL=F from 2018-01-01
- Verify data quality and continuity
- Update cached parquet files

### 2.2 Re-cache Combined Features

Clear and regenerate feature cache:

- Start date: 2018-01-01
- End date: 2025-12-31 (latest available)
- Verify alignment across all macro symbols

Expected output: ~2,900 samples (vs current ~2,168)---

## Phase 3: Unified Walk-Forward Test Framework

### 3.1 Create New Test Script

Create `scripts/test_unified_comparison.py` that:

```python
# Pseudocode for unified comparison
for fold in walk_forward_folds:
    # Same train/test split for ALL methods
    X_train, X_test = split_data(fold)
    
    # Method 1: Random baseline
    random_labels = np.random.randint(0, K, len(X_test))
    
    # Method 2: PCA + GMM (baseline)
    pca = PCA(0.95).fit(X_train)
    gmm_pca = GMM(K).fit(pca.transform(X_train))
    pca_labels = gmm_pca.predict(pca.transform(X_test))  # OOS!
    
    # Method 3: AE(dim) + GMM
    ae = train_autoencoder(X_train)
    latents_train = ae.encode(X_train)
    latents_test = ae.encode(X_test)
    gmm_ae = GMM(K).fit(latents_train)
    ae_labels = gmm_ae.predict(latents_test)  # OOS!
    
    # Compare on SAME test data
    compare_metrics(random_labels, pca_labels, ae_labels, returns[test])
```



### 3.2 Metrics to Compare

For each method, compute:

- OOS Sharpe spread (max regime Sharpe - min regime Sharpe)
- Regime persistence (avg probability of staying in same regime)
- Statistically significant regimes (bootstrap CI excludes 0)
- Net Sharpe after transaction costs

### 3.3 Deprecate Old Test Scripts

Mark as deprecated (but keep for reference):

- `scripts/test_autoencoder_rigor.py` - has in-sample GMM bug
- Keep `scripts/test_statistical_rigor.py` - still valid for GMM-only baseline

---

## Phase 4: Corrected Hyperparameter Sweep

### 4.1 Create New Sweep Script

Create `scripts/sweep_latent_dim_corrected.py`:

- Use walk-forward validation (not fixed 80/20 split)
- Primary metric: **OOS Sharpe spread** (not reconstruction loss)
- Secondary metrics: persistence, cost survival
- Test dimensions: [4, 6, 8, 10, 12, 14]

### 4.2 Sweep Protocol

```javascript
For each latent_dim in [4, 6, 8, 10, 12, 14]:
    results = []
    for fold in walk_forward_folds:
        train AE(latent_dim) on fold.train
        encode fold.test → latents
        fit GMM on train latents
        predict regimes on test latents (OOS)
        compute sharpe_spread, persistence
        results.append(metrics)
    
    avg_sharpe_spread[latent_dim] = mean(results)

optimal_dim = argmax(avg_sharpe_spread)
```

---

## Phase 5: Final Validation & Comparison

### 5.1 Train Final Models

Using optimal latent_dim from sweep:

- Train AE with full walk-forward protocol
- Save model checkpoint with complete metadata

### 5.2 Generate Comparison Report

Run unified test to produce:

- Side-by-side table: Random vs PCA+GMM vs AE+GMM
- Bootstrap confidence intervals for all metrics
- Visualization of regime quality by method

### 5.3 Update Documentation with Results

Record final validated results in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)---

## File Changes Summary

| File | Action ||------|--------|| `docs/TECH.md` | Add data coverage and architecture sections || `docs/EXPERIMENTS.md` | Add Experiment 10 (methodology corrections) || `docs/PROJ.md` | Update roadmap and validation section || `scripts/test_unified_comparison.py` | **NEW** - Fair comparison framework || `scripts/sweep_latent_dim_corrected.py` | **NEW** - Correct sweep metric || `scripts/test_autoencoder_rigor.py` | Mark deprecated || `src/data/walk_forward.py` | Minor: add unified fold generator |---

## Success Criteria

1. All macro data extends to 2018-01-01
2. Unified test runs all methods through identical walk-forward splits
3. Sweep selects optimal latent_dim based on OOS Sharpe spread