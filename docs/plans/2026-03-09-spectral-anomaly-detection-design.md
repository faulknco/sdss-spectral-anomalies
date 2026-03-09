# SDSS Spectral Anomaly Detection — Design

## Goal

Build an ML pipeline to find genuinely unusual stellar spectra in SDSS data using both classical and deep learning approaches, with an interactive Streamlit dashboard to explore results. Dual purpose: scientific discovery and ML portfolio piece.

## Stack

Python, uv, astroquery, astropy, specutils, scikit-learn, PyTorch, Streamlit, Plotly

## Project Structure

```
sdss-spectral-anomalies/
├── src/
│   ├── data/          # Download & preprocessing pipeline
│   ├── features/      # Spectral feature extraction, PCA
│   ├── models/        # Isolation Forest, Autoencoder
│   └── dashboard/     # Streamlit app to browse anomalies
├── data/
│   ├── raw/           # Downloaded FITS files
│   ├── processed/     # Cleaned numpy arrays / parquet
│   └── results/       # Anomaly scores, labels
├── tests/
├── docs/plans/
└── pyproject.toml
```

## Phases

- **Phase A:** 5-10K spectra — build full pipeline end-to-end
- **Phase B:** Scale to 50-100K spectra with batch downloads and memory-mapped arrays

## Data Pipeline

1. Query SDSS DR18 via `astroquery.sdss` for stellar spectra
2. Filter for S/N > 10, mix of spectral types (O, B, A, F, G, K, M)
3. Store raw FITS in `data/raw/`, metadata as parquet
4. Resample to common wavelength grid (~3800-9200 A)
5. Normalize flux (divide by median)
6. Mask/interpolate bad pixels and sky line residuals
7. Output: numpy array `(n_spectra, n_wavelength_bins)` + metadata dataframe in `data/processed/`

## Models

### Model 1 — PCA + Isolation Forest (baseline)

- PCA: ~3500 bins → ~50 components (retaining ~95% variance)
- Isolation Forest on PCA components, contamination ~5%
- Anomaly scores: isolation forest decision function + PCA reconstruction error

### Model 2 — Convolutional Autoencoder (PyTorch)

- Encoder: 1D convolutions (3-4 layers) → bottleneck 32-64 dims
- Decoder: transposed convolutions back to original shape
- Loss: MSE reconstruction error
- Train 80/20 split
- Anomaly score: per-spectrum reconstruction error

### Comparison

- Rank spectra by anomaly score from each model
- Compute overlap in top-N anomalies
- Flag agreement (high confidence) and disagreement (model-specific insights)

## Dashboard (Streamlit)

1. **Anomaly Browser** — Table of top anomalies, expandable with spectrum plot, metadata (RA, Dec, spectral type, S/N), scores from both models
2. **Comparison View** — Scatter of IF score vs AE reconstruction error, colored by spectral type, click to inspect spectrum
3. **PCA Explorer** — 2D/3D PCA scatter colored by anomaly score

All plots via Plotly. Spectrum plots overlay autoencoder reconstruction on original.
