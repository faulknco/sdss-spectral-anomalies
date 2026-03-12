# Changelog

## Unreleased

### Added
- **Conditional Normalizing Flow** (`src/models/normalizing_flow.py`) — RealNVP with FiLM-modulated coupling blocks. Learns `p(flux_pca | Teff, log g, [Fe/H], sn_median)` and uses NLL as the anomaly score. More principled than reconstruction error: a proper conditional density estimate rather than a proxy.
  - PCA compression (3500 bins → 50 components) before the flow for tractability
  - FiLM conditioning prevents the flow from ignoring metadata
  - `nll_score(spectra, metadata)` interface; `param_count()` consistent with other models
  - Wired into pipeline as Step 5d; scores saved as `flow_scores.npy` and `flow_losses.npy`
  - Added to `all_scores` comparison and `param_counts.json`
  - 4 unit tests + integration test coverage

## [0.1.0] — Prior work

### Added
- **Conditional Autoencoder + Conformal Calibration** — metadata-conditioned conv autoencoder (Teff, log g, [Fe/H], sn_median injected via FiLM into the bottleneck). Split conformal calibrator converts raw scores to p-values with finite-sample guarantees.
- **SDSS data pipeline improvements** — SPALL extension support, PLUG_RA fallback, SDSS-II and BOSS FITS format handling, focused anomaly review tooling.
- **Dashboard** — Streamlit app with Anomaly Browser, Model Comparison, PCA Explorer, and Stability tabs. Conditional AE scores and conformal p-values surfaced.
- **DAGMM** (`src/models/dagmm.py`) — Deep Autoencoding Gaussian Mixture Model anomaly detector.
- **OC-SVM** (`src/models/ocsvm.py`) — One-Class SVM with PCA preprocessing.
- **Multi-seed stability analysis** (`src/models/stability.py`) — runs models across seeds and reports mean/std scores.
- **Anomaly categorization** (`src/features/categorize.py`) — labels anomalies as continuum, line, noise, or normal based on PCA residual structure.
- **N-model comparison framework** (`src/models/compare.py`) — rank aggregation across arbitrary score dicts; `adaptive_top_n` for dataset-size-aware top-N selection.
- **Parameter counting** — all models expose `param_count() -> int`.
- **Spectrum-to-RGB color conversion** (`src/features/`) — perceived star color from flux.
- **Elodie stellar parameters** persisted into `spectra_metadata.parquet` (Teff, log g, [Fe/H]).
- **Convolutional Autoencoder** (`src/models/autoencoder.py`) — 1D conv encoder/decoder.
- **PCA + Isolation Forest** (`src/models/classical.py`) — classical baseline.
- **SDSS download module** (`src/data/download.py`) — query builder, FITS parser, multi-transport download (rsync, HTTPS, astroquery).
- **Preprocessing module** (`src/data/preprocess.py`) — resample, normalize, batch pipeline.
- **Pipeline orchestrator** (`src/run_pipeline.py`) — end-to-end CLI entry point.
- **Streamlit dashboard** scaffolded.
- **Integration test** covering full synthetic pipeline.
- Project scaffolded with uv, pyproject.toml, and directory structure.
