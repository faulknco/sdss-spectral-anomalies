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

## Unreleased

### Added
- **Conditional Normalizing Flow** (`src/models/normalizing_flow.py`) — RealNVP with FiLM-modulated coupling blocks. Learns `p(flux_pca | Teff, log g, [Fe/H], sn_median)` and uses NLL as the anomaly score. More principled than reconstruction error: a proper conditional density estimate rather than a proxy.
  - PCA compression (3500 bins -> 50 components) before the flow for tractability
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
