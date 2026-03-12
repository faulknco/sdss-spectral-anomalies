# SDSS Spectral Anomalies

Anomaly detection in SDSS stellar spectra using classical and deep learning models, with an interactive Streamlit dashboard for exploration.

## What it does

Downloads stellar spectra from SDSS DR18, preprocesses them onto a common wavelength grid, runs six anomaly detectors in parallel, and ranks spectra by how unusual they are. Anomalies agreed upon by multiple models are surfaced for review.

## Models

| Model | Type | Score |
|-------|------|-------|
| PCA + Isolation Forest | Classical | Isolation score |
| Convolutional Autoencoder | Deep learning | Reconstruction error |
| One-Class SVM | Classical | Decision function |
| DAGMM | Deep learning (generative) | Energy score |
| Conditional Autoencoder | Deep learning, metadata-conditioned | Reconstruction error given stellar params |
| **Conditional Normalizing Flow** | Deep learning, generative, metadata-conditioned | **Negative log-likelihood given stellar params** |

The normalizing flow is the most principled density estimator: it learns `p(flux | Teff, log g, [Fe/H], S/N)` and flags spectra with low probability under their expected stellar class.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone https://github.com/faulknco/sdss-spectral-anomalies
cd sdss-spectral-anomalies
uv sync
```

## Usage

### Run the full pipeline

```bash
uv run sdss-anomalies --n-spectra 5000 --sn-min 10
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-spectra` | 5000 | Number of spectra to query from SDSS |
| `--sn-min` | 10.0 | Minimum signal-to-noise ratio |
| `--download-mode` | `missing` | `missing` downloads new FITS; `skip` uses local only |
| `--metadata-only` | off | Query metadata without downloading spectra |

### Launch the dashboard

```bash
uv run streamlit run src/dashboard/app.py
```

### Run tests

```bash
uv run python -m pytest tests/ -v
```

## Project structure

```
src/
  data/
    download.py        # SDSS metadata query and FITS download
    preprocess.py      # Resampling, normalization, metadata extraction
  models/
    classical.py       # PCA + Isolation Forest
    autoencoder.py     # Convolutional autoencoder
    ocsvm.py           # One-Class SVM with PCA preprocessing
    dagmm.py           # Deep Autoencoding Gaussian Mixture Model
    conditional_autoencoder.py  # Metadata-conditioned autoencoder
    normalizing_flow.py         # Conditional RealNVP normalizing flow
    conformal.py       # Split conformal calibration for p-values
    stability.py       # Multi-seed stability analysis
    compare.py         # N-model comparison and rank aggregation
  features/
    categorize.py      # Anomaly type categorization (continuum/line/noise/normal)
  dashboard/
    app.py             # Streamlit dashboard
  run_pipeline.py      # Pipeline orchestrator
tests/                 # 82 tests, all passing
docs/
  plans/               # Implementation plans
  superpowers/specs/   # Design specs
```

## Output

Pipeline results are written to `data/results/`:

| File | Contents |
|------|----------|
| `if_scores.npy` | Isolation Forest anomaly scores |
| `ae_scores.npy` | Autoencoder reconstruction errors |
| `ocsvm_scores.npy` | OC-SVM scores |
| `dagmm_scores.npy` | DAGMM energy scores |
| `conditional_ae_scores.npy` | Conditional AE reconstruction errors |
| `conditional_ae_pvalues.npy` | Conformal p-values |
| `flow_scores.npy` | Normalizing flow NLL scores |
| `comparison.parquet` | All scores merged with metadata |
| `top_anomalies_agreed.parquet` | Anomalies agreed by 3+ models |
| `focused_review.parquet` | Top 10 anomalies for manual review |
| `param_counts.json` | Parameter counts per model |
