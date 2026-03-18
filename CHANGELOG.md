# Changelog

## 2026-03-17 — PBH Signature Detection

Added physics-motivated feature extractors for primordial black hole (PBH) candidate screening. These are exploratory screens for follow-up, not confirmations — many mundane astrophysical phenomena produce similar spectral features.

### Scripts

- **`scripts/rescore_pbh.py`** — Rescore PBH features on existing spectra without rerunning the full pipeline. Use after a pipeline run completes to apply updated scoring code (e.g. accretion continuum floor fix). Run with `uv run python scripts/rescore_pbh.py`.

### New Modules

- **`src/features/spectral_lines.py`** — Shared line measurement utilities (flux, continuum, equivalent width, line profile asymmetry) for 7 stellar absorption lines (Ca K, Ca H, H-gamma, H-beta, Mg b, Na D, H-alpha)
- **`src/features/microlensing.py`** — Gravitational microlensing score based on continuum-to-line flux ratio deviations from spectral-class expectations and achromatic residual analysis
- **`src/features/accretion.py`** — Accretion signature score combining unexpected emission line detection, blue/UV excess vs Planck shape, and power-law continuum fitting
- **`src/features/line_asymmetry.py`** — Line profile asymmetry scoring (skewness, kurtosis, blue/red EW ratio) for detecting differential limb magnification from microlensing
- **`src/features/multiepoch.py`** — Multi-epoch variability search grouping repeat SDSS observations by sky position and scoring achromatic vs chromatic flux changes
- **`src/features/gaia_crossmatch.py`** — Gaia DR3 astrometric cross-match for RUWE, excess noise, and IPD harmonic amplitude anomalies suggesting unseen companions
- **`src/features/photometric_crossmatch.py`** — SDSS ugriz photometric cross-match with dereddened color analysis and blue excess detection

### Modified

- **`src/features/categorize.py`** — Added `categorize_pbh_candidates()` (original `categorize_anomalies` untouched)
- **`src/run_pipeline.py`** — Added Steps 6b-6e (PBH scoring, multi-epoch, Gaia, photometry) with `--skip-pbh` flag; PBH scores in `comparison.parquet` but excluded from unsupervised rank aggregation
- **`src/dashboard/app.py`** — Added PBH Candidates tab with candidate tables, cross-correlation scatter, spectrum+template overlay, multi-epoch variability, Gaia astrometry, photometric colors, and PBH sort options in Anomaly Browser

### Bug Fixes

- Fixed false-positive accretion scores for M-dwarf spectra where near-zero blue continuum flux produced extreme emission ratios (added 0.1 continuum floor)
- Fixed Gaia batch query returning empty results for sky-scattered candidates (switched to per-source cone searches for wide footprints)
- Fixed WD subclass parsing in photometric cross-match ("WDhotter" no longer misparses as O-type)

### Tests

- 7 new test files: `test_spectral_lines.py`, `test_microlensing.py`, `test_accretion.py`, `test_line_asymmetry.py`, `test_multiepoch.py`, `test_gaia_crossmatch.py`, `test_photometric_crossmatch.py`
- Updated `test_categorize.py` and `test_integration.py` with PBH coverage
- Total: 175 tests passing

### Initial Results (599 spectra)

- 30 microlensing candidates, 30 accretion candidates at 95th percentile threshold
- 10 candidates overlap with 5+ unsupervised model consensus
- Top candidate: spec-1246-54478-0258 (K7) — high microlensing (1.41), line asymmetry (1.55), 7/7 model agreement
- Accretion candidates dominated by WD-classified objects with mismatched Teff (likely composite/binary systems)
- Gaia cross-match pending (ESA archive degraded); photometric cross-match pending SDSS query

## 2026-03-18 — 50K Scaling, Evaluation Harness, and Review Labeling

### 50K Streaming Pipeline (Spec C)

- **Stream-and-discard architecture** — downloads FITS in batches of 500 with 8 parallel workers, preprocesses immediately into a memory-mapped float32 numpy array, deletes raw FITS. Keeps disk usage under 2 GB for 50K spectra.
- **`stream_and_preprocess()`** in `src/data/download.py` — parallel `ThreadPoolExecutor` downloads with checkpoint/resume via JSON. If the pipeline crashes, restart picks up from the last completed batch.
- **`create_memmap()` / `write_to_memmap()`** in `src/data/preprocess.py` — incremental .npy file writing compatible with `np.load(mmap_mode='r')`.
- **OC-SVM subsampling** — `max_train_samples=5000` parameter avoids O(N^2) kernel matrix at 50K. PCA/scaler fit on full data, SVM trains on subsample.
- **Step 9: Keep top anomaly FITS** — re-downloads top 200 FITS by combined rank to `data/raw_kept/` for dashboard inspection.
- **Dashboard memmap loading** — `@st.cache_resource` for large spectra arrays, `kept_dir` parameter for FITS lookup.
- **CLI**: `--streaming`, `--batch-size`, `--download-workers`, `--keep-top-n`
- **Bug fix**: `--download-mode skip` now loads existing processed spectra instead of re-reading FITS from `data/raw/`

### Conditional CVAE and Normalizing Flow (Spec A)

- **Conditional VAE** (`src/models/cvae.py`) — variational bottleneck with beta-warmup KL annealing. Anomaly score = negative ELBO (sum of reconstruction + KL divergence). Density proxy conditioned on stellar parameters.
- **Conditional MAF** (`src/models/conditional_flow.py`) — Masked Autoregressive Flow on PCA-compressed spectra. 8 MADE blocks with alternating orderings, batch normalization, early stopping. True log-likelihood `p(flux_pca | Teff, log g, [Fe/H], S/N)`.
- Both models expose `anomaly_score()` and `param_count()`, plug into `compare_n_models` and `SplitConformalCalibrator`.
- Pipeline now has **7 models**: IF, AE, OC-SVM, DAGMM, Conditional AE, CVAE, Conditional Flow.

### Line-Window Preprocessing (Spec A)

- **`src/features/line_windows.py`** — shared 11-line spectral catalog (Ca K, Ca H, H-gamma, H-beta, MgH, Na D, H-alpha, TiO, Ca II triplet). `DISPLAY_LINES` subset for dashboard annotations.
- **`compute_derivative_spectra()`** — dF/dlambda via finite differences, normalized by median absolute value.
- **`extract_line_features()`** — 44-dimensional feature vector (11 lines x 4 features: equivalent width proxy, local depth, asymmetry, derivative variance).

### Semi-Synthetic Evaluation (Spec A)

- **`src/features/synthetic_anomalies.py`** — injects 5 anomaly types into clean spectra: emission line insertion, line broadening, continuum tilt, wavelength shift, missing band segment. Deterministic with seed control.
- **`src/features/evaluate_retrieval.py`** — precision@k, recall@k, AUROC, AUPRC, per-anomaly-type recall. Runs against any model's scores.
- **Dashboard Tab 7: Evaluation** — AUROC/AUPRC comparison table, precision@k bar chart, per-type recall heatmap. Powered by `evaluation_results.json`.

### Human Review Labeling (Spec B)

- **`src/features/review_labels.py`** — `LABEL_CODES`, `LABEL_DISPLAY_NAMES`, `load_labels()`, `save_labels()`. Five categories: artifact, low S/N, plausible oddity, known rare subtype, unclear.
- **Dashboard Focused Review tab** — label selectbox, notes text input, progress bar with category breakdown. Labels held in session state, persisted via sidebar "Save All Labels" button to `review_labels.parquet`.
- **Pipeline** merges existing labels into `focused_review.parquet` on re-run (left join on filename, survives re-runs).
- `review_labels.parquet` excluded from `.gitignore` (human-generated, not auto-generated).

### Tests

- 175 tests passing (up from 78 at start of session)
- New test files: `test_line_windows.py`, `test_cvae.py`, `test_conditional_flow.py`, `test_synthetic_anomalies.py`, `test_evaluate_retrieval.py`, `test_review_labels.py`
- Updated: `test_integration.py`, `test_download.py`, `test_preprocess.py`, `test_ocsvm.py`

### Evaluation Results (599 spectra)

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Autoencoder | 0.788 | 0.410 |
| Conditional AE | 0.742 | 0.317 |
| OC-SVM | 0.645 | 0.309 |
| Isolation Forest | 0.678 | 0.263 |
| DAGMM | 0.678 | 0.230 |
| CVAE | 0.679 | 0.190 |
| Conditional Flow | 0.602 | 0.187 |

50K run pending (downloads completed, training to resume).

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
