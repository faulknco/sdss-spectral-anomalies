# Paper-Inspired Enhancements — Design

Based on Mejri et al. (2024) "Unsupervised Anomaly Detection in Time-series" (arXiv:2212.03637).

## Enhancements

### 1. Multi-Seed Stability Runs
- Add `--n_runs` parameter (default 5) to pipeline
- Each model trains N times with different random seeds
- Store all score arrays, report mean +/- std per spectrum
- Dashboard shows stability metrics and error bars

### 2. Parameter Count Reporting
- Compute parameter counts for all models after construction
- PyTorch: `sum(p.numel() for p in model.parameters())`
- sklearn: PCA components * features + forest estimators
- Save to results JSON, display in dashboard sidebar

### 3. DAGMM (3rd Model)
- New `src/models/dagmm.py`
- Architecture: autoencoder encoder -> latent code + reconstruction error features -> GMM estimation network
- Anomaly score: negative log-likelihood under learned GMM
- Catches density-based anomalies other models miss

### 4. OC-SVM Baseline (4th Model)
- New class in `src/models/classical.py` or separate file
- PCA dimensionality reduction -> One-Class SVM (RBF kernel)
- Quick sklearn-based baseline

### 5. Anomaly Type Categorization
- New `src/features/categorize.py`
- Classify top anomalies into categories:
  - Continuum anomaly: unusual overall shape (high PCA reconstruction error)
  - Line anomaly: unusual absorption/emission lines (high local variance in residuals)
  - Noise anomaly: high variance across wavelength
  - Classification anomaly: spectrum doesn't match labeled spectral type
- Add category column to results

### 6. Dashboard Updates
- New "Model Stability" tab with error bars and std heatmap
- Parameter count display in sidebar
- Anomaly type column in browser tab
- All 4 models in comparison view (scatter matrix)
