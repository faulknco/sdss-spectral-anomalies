# SDSS Spectral Anomalies

Anomaly detection in SDSS stellar spectra using classical and deep learning models, with an interactive Streamlit dashboard for exploration.

## What it does

Downloads stellar spectra from SDSS DR18, preprocesses them onto a common wavelength grid, runs seven anomaly detectors (including conditional density models), and ranks spectra by how unusual they are. Includes semi-synthetic evaluation, conformal p-value calibration, and a human review labeling workflow. Scales to 50K+ spectra via a stream-and-discard download pipeline.

## Models

| Model | Type | Score |
|-------|------|-------|
| PCA + Isolation Forest | Classical | Isolation score |
| Convolutional Autoencoder | Deep learning | Reconstruction error |
| One-Class SVM | Classical (subsampled for scale) | Decision function |
| DAGMM | Deep learning (generative) | Energy score |
| Conditional Autoencoder | Deep learning, metadata-conditioned | Reconstruction error given stellar params |
| **Conditional VAE** | Deep learning, variational, metadata-conditioned | **Negative ELBO (density proxy)** |
| **Conditional MAF** | Normalizing flow, metadata-conditioned | **Negative log-likelihood given stellar params** |

The conditional models ask "is this spectrum strange *for a star with these parameters*?" rather than just "is this spectrum rare in the population?" The MAF provides true log-likelihood; the CVAE provides an ELBO lower bound. Both are calibrated into conformal p-values.

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
uv run python -m src.run_pipeline --n-spectra 5000 --sn-min 10
```

### Run at 50K scale (stream-and-discard)

```bash
uv run python -m src.run_pipeline --n-spectra 50000 --streaming
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-spectra` | 5000 | Number of spectra to query from SDSS |
| `--sn-min` | 10.0 | Minimum signal-to-noise ratio |
| `--download-mode` | `missing` | `missing` downloads new FITS; `skip` uses local only |
| `--metadata-only` | off | Query metadata without downloading spectra |
| `--streaming` | off | Stream-and-discard mode (auto-enabled for >1000 spectra) |
| `--batch-size` | 500 | FITS download batch size for streaming |
| `--download-workers` | 8 | Parallel download workers |
| `--keep-top-n` | 200 | Top anomaly FITS to keep for dashboard |

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
    download.py        # SDSS query, FITS download, stream_and_preprocess
    preprocess.py      # Resampling, normalization, memmap helpers
  models/
    classical.py       # PCA + Isolation Forest
    autoencoder.py     # Convolutional autoencoder
    ocsvm.py           # One-Class SVM (subsampled for 50K+)
    dagmm.py           # Deep Autoencoding Gaussian Mixture Model
    conditional_autoencoder.py  # Metadata-conditioned autoencoder
    cvae.py            # Conditional VAE (ELBO scoring)
    conditional_flow.py # Conditional MAF normalizing flow
    conformal.py       # Split conformal calibration for p-values
    stability.py       # Multi-seed stability analysis
    compare.py         # N-model comparison and rank aggregation
  features/
    line_windows.py    # Spectral line catalog, derivative spectra, line features
    categorize.py      # Anomaly type categorization
    synthetic_anomalies.py  # Semi-synthetic anomaly injection (5 types)
    evaluate_retrieval.py   # Precision@k, recall@k, AUROC, AUPRC
    review_labels.py   # Human review label I/O
  dashboard/
    app.py             # Streamlit dashboard (7 tabs + PBH)
  run_pipeline.py      # Pipeline orchestrator
tests/                 # 175 tests, all passing
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
| `flow_scores.npy` | Conditional MAF NLL scores |
| `cvae_scores.npy` | CVAE negative ELBO scores |
| `cvae_pvalues.npy` | CVAE conformal p-values |
| `flow_pvalues.npy` | Flow conformal p-values |
| `evaluation_results.json` | Semi-synthetic retrieval metrics (AUROC, precision@k, per-type recall) |
| `review_labels.parquet` | Human review labels (persisted from dashboard) |
| `comparison.parquet` | All 7 model scores merged with metadata |
| `top_anomalies_agreed.parquet` | Anomalies agreed by 3+ models |
| `focused_review.parquet` | Top candidates for manual review (with labels if available) |
| `param_counts.json` | Parameter counts per model |
