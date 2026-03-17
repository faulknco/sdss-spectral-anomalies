# Spec A: Conditional Flow, Semi-Synthetic Evaluation, and Line-Window Preprocessing

**Date:** 2026-03-17
**Status:** Approved design, pending implementation plan

## Goal

Add density-based conditional anomaly scoring (CVAE then normalizing flow), line-aware preprocessing, and a semi-synthetic evaluation harness to measure retrieval quality across all models.

## Approach

Sequential build. Each component is independently testable before the next starts:

1. Line-window preprocessing
2. CVAE (density proxy via ELBO)
3. Semi-synthetic anomaly injection + evaluation harness
4. Conditional normalizing flow (true likelihood)
5. Pipeline and dashboard integration

## 1. Line-Window Preprocessing

**New file:** `src/features/line_windows.py`

Two new spectral representations alongside the existing median-normalized spectra:

### Derivative Spectra

`dF/dlambda` via finite differences on the wavelength grid. Emphasizes local line-shape features over broad continuum. Normalized by dividing by median absolute value.

### Line-Window Feature Extraction

For each of 11 spectral lines, extract a ~40 Angstrom window centered on the line. Compute per-window features: equivalent width proxy, local depth, local asymmetry, local derivative variance. Output: a fixed-size feature vector (11 lines x 4 features = 44 dims).

### Line Catalog

Shared constant, also used by dashboard annotations:

| Line | Wavelength (A) |
|------|---------------|
| Ca K | 3933.7 |
| Ca H | 3968.5 |
| H-gamma | 4340.5 |
| H-beta | 4861.3 |
| MgH | 5210.0 |
| Na D | 5892.0 |
| H-alpha | 6562.8 |
| TiO | 7050.0 |
| Ca II a | 8498.0 |
| Ca II b | 8542.0 |
| Ca II c | 8662.0 |

### Integration

The pipeline computes and saves `derivative_spectra.npy` and `line_features.npy` in `data/processed/`. The semi-synthetic assessment uses line windows to verify injected anomalies are detectable in the right spectral regions. The dashboard's hardcoded `SPECTRAL_LINES` list is replaced with an import from the shared catalog. The catalog exports both the full 11-line list and a `DISPLAY_LINES` subset (the original 6: Ca K, Ca H, H-gamma, H-beta, Na D, H-alpha) for annotation overlays where visual clarity matters. The full 11-line list is used for feature extraction and injection targeting.

## 2. Conditional VAE (CVAE)

**New file:** `src/models/cvae.py`

### Architecture

Extends the conditional AE pattern with a variational bottleneck. The encoder outputs `mu` and `log_var`, we sample `z = mu + sigma * epsilon`, and the loss becomes `reconstruction + beta * KL_divergence`.

```
flux (B, 1, L) -> conv_encoder -> (B, 128)
meta (B, meta_dim) -> meta_mlp -> (B, meta_embed_dim)
concat -> fc_mu, fc_logvar -> sample z (B, bottleneck_dim)

z concat meta_emb -> decoder_fc -> deconv -> recon (B, 1, L)
```

The constructor accepts a dynamic `meta_dim` parameter (defaulting to `len(METADATA_FEATURE_COLS)` = 4), following the same pattern as `ConditionalSpectralAutoencoder`. The pipeline passes `metadata.shape[1]` at construction time.

### Key Differences from Conditional AE

- Encoder produces `mu` and `log_var` instead of a deterministic bottleneck.
- Loss: MSE + beta * KL. Beta warms up linearly from 0 over the first 10 epochs to avoid posterior collapse.
- Anomaly score: negative ELBO per spectrum = `reconstruction_error + KL_divergence`. Higher means the spectrum is less likely given its stellar parameters.
- Method is named `anomaly_score()` (not `reconstruction_error()`) because the ELBO includes the KL term, not just reconstruction MSE. The CVAE does not expose a `reconstruction_error()` method to avoid confusion.

### Scoring Interface

`model.anomaly_score(spectra, metadata) -> np.ndarray`. Returns negative ELBO per spectrum. Plugs directly into `compare_n_models` and `SplitConformalCalibrator`.

### Training

Same train/calibration split as the conditional AE (80/20 random, seed 42). Reuses `build_metadata_features` for metadata standardization. Conformal p-values computed on calibration set scores. Fixed 50 epochs (no early stopping) to match the conditional AE. The beta warmup over the first 10 epochs provides implicit regularization against posterior collapse.

## 3. Conditional Normalizing Flow

**New file:** `src/models/conditional_flow.py`

### Architecture

A Masked Autoregressive Flow (MAF) operating on PCA-compressed spectra, conditioned on stellar parameters. MAF over RealNVP because MAF is faster at density computation (the primary use case is scoring, not generation).

```
flux (B, L) -> PCA -> z_pca (B, 50)
meta (B, 4) -> meta_mlp -> context (B, meta_embed_dim)

z_pca, context -> MAF (8 MADE blocks) -> base distribution (standard normal)

score = -log p(z_pca | context)
```

### Key Design Decisions

- **PCA first:** 3500 wavelength bins is too high-dimensional for a flow. Compress to 50 PCA components using the PCA already computed by the classical model.
- **8 MADE blocks** with alternating input ordering, hidden dim 128. Each block: linear -> ReLU -> linear -> affine transform. Batch normalization between blocks.
- **No new PCA fitting** -- reuse `ClassicalAnomalyDetector`'s PCA transform. The flow takes pre-transformed PCA components as input. The pipeline must run the classical model before the flow (already the case).

### Scoring Interface

Two methods:

- `model.log_prob(pca_components, metadata) -> np.ndarray` -- returns log-likelihood per spectrum.
- `model.anomaly_score(pca_components, metadata) -> np.ndarray` -- returns `-log_prob` (higher = more anomalous). This is the method used by `compare_n_models` and conformal calibration, keeping the convention consistent with all other models in the codebase.

### Training

Same 80/20 split. Adam optimizer, learning rate 1e-4. 100 epochs with early stopping on validation loss (10% of training set held out).

### Dependency

Requires PCA components from the classical model. The pipeline already computes and saves these.

## 4. Semi-Synthetic Anomaly Injection

**New file:** `src/features/synthetic_anomalies.py`

Takes clean spectra and injects controlled anomalies with known ground truth. Returns modified spectra + labels for measuring model retrieval performance.

### Five Injection Types

1. **Emission line insertion** -- Gaussian emission peak at a random catalog line position. Amplitude: 2-10x local continuum. Width: 3-15 A FWHM. Mimics unexpected emission in absorption-line stars.

2. **Line broadening** -- Convolves a ~100 A window around a random line with a Gaussian kernel (sigma 5-20 A). Simulates rotational broadening or unresolved binary blending.

3. **Continuum tilt** -- Multiplies the spectrum by `1 + slope * (lambda - lambda_mid)` where slope is drawn from U(-0.0005, 0.0005) per Angstrom. Simulates reddening or flux calibration errors.

4. **Wavelength shift** -- Shifts the spectrum by 5-50 A (reinterpolated onto same grid). Mimics radial velocity outliers or wavelength calibration failures.

5. **Missing band segment** -- Zeros out a contiguous 100-500 A region at a random location. Simulates data reduction dropouts or detector gaps.

### Interface

```python
def inject_anomalies(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Returns (modified_spectra, labels, injection_log).

    labels: 0 = clean, 1 = injected.
    injection_log: list of dicts with index, type, and params for each injection.
    """
```

Each injected spectrum gets exactly one anomaly type, chosen uniformly at random. The injection log records what was done so the assessment can break down retrieval by anomaly type.

### Retrieval Assessment Helper

**New file:** `src/features/evaluate_retrieval.py`

```python
def evaluate_retrieval(
    labels: np.ndarray,
    scores: np.ndarray,
    injection_log: list[dict],
    top_k_list: list[int] = [10, 25, 50, 100],
) -> dict:
    """Returns precision_at_k, recall_at_k, auroc, auprc, per_type_recall_at_50."""
```

Runs against any model's scores, giving a standardized comparison.

## 5. Pipeline and Dashboard Integration

### Pipeline (`src/run_pipeline.py`)

New steps inserted into the existing pipeline. The ordering is:

- Steps 1-4b: unchanged (download, preprocess, classical, OC-SVM, AE, DAGMM)
- Step 5: conditional AE + conformal (unchanged)
- **Step 5d:** Compute derivative spectra and line features. Save to `data/processed/`. (Inserted after Step 5c, before stability runs.)
- **Step 5e:** Train CVAE, score, conformal calibrate. Save `cvae_scores.npy`, `cvae_pvalues.npy`, `cvae_losses.npy`.
- **Step 5f:** Train conditional flow on PCA components (from Step 3). Save `flow_scores.npy`, `flow_pvalues.npy`, `flow_losses.npy`.
- Step 5b: stability runs (unchanged, stays after all model training)
- Steps 6-7: categorization and comparison (unchanged, but comparison now includes 7 models)
- **Step 8:** Semi-synthetic assessment. Inject anomalies into clean spectra, re-project through the already-fitted classical PCA (via `classical.transform(modified_spectra)`) for the flow model, then score with all 7 models. Save `evaluation_results.json`.

Updated model roster for `compare_n_models`:

```python
all_scores = {
    "if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores,
    "dagmm": dagmm_scores, "cond_ae": cond_ae_scores,
    "cvae": cvae_scores, "flow": flow_scores,
}
```

### Dashboard (`src/dashboard/app.py`)

- `load_data()` picks up new score/p-value arrays with the same `exists()` fallback pattern.
- Sort options in Anomaly Browser gain "CVAE" and "Conditional Flow".
- Model Comparison scatter matrix grows to 7 models.
- Sidebar param counts include CVAE and flow.
- **New Tab 7: "Evaluation"** -- Retrieval metrics from `evaluation_results.json`: bar charts of precision_at_k and recall_at_k by model, per-anomaly-type recall heatmap, AUROC/AUPRC comparison table. Only visible when `evaluation_results.json` exists.

### Notes

- **Combined sort:** The Anomaly Browser's "Combined" sort currently sums raw scores without normalization. With 7 models, this is increasingly dominated by whichever model has the largest raw score magnitude. This is a pre-existing issue; we leave it as-is for now but may revisit with rank-based combination later.
- **GPU:** All models are designed to train on CPU (consistent with the existing pipeline). The MAF with 8 MADE blocks on 50-dim PCA input is modest and does not require GPU for reasonable training times.
- **Adaptive top-k:** `evaluate_retrieval`'s `top_k_list` default of `[10, 25, 50, 100]` should be treated as a default. The pipeline integration should use `adaptive_top_n` or similar scaling when the dataset size changes significantly.

### What Doesn't Change

- `SplitConformalCalibrator` -- reused as-is for CVAE and flow scores.
- Existing model training steps -- untouched.
- Stability runner -- not applied to CVAE/flow initially.
- `build_metadata_features` -- reused as-is.

## New Files

| File | Purpose |
|------|---------|
| `src/features/line_windows.py` | Derivative spectra, line-window features, shared line catalog |
| `src/models/cvae.py` | Conditional VAE with ELBO scoring |
| `src/models/conditional_flow.py` | MAF-based conditional normalizing flow |
| `src/features/synthetic_anomalies.py` | Semi-synthetic anomaly injection |
| `src/features/evaluate_retrieval.py` | Retrieval metrics |
| `tests/test_line_windows.py` | Tests for line-window preprocessing |
| `tests/test_cvae.py` | Tests for CVAE |
| `tests/test_conditional_flow.py` | Tests for conditional flow |
| `tests/test_synthetic_anomalies.py` | Tests for anomaly injection |
| `tests/test_evaluate_retrieval.py` | Tests for retrieval harness |

## Modified Files

| File | Change |
|------|--------|
| `src/run_pipeline.py` | Steps 5d-5f, Step 8, updated model roster |
| `src/dashboard/app.py` | New scores, sort options, Tab 7, shared line catalog import |
| `tests/test_integration.py` | CVAE, flow, and assessment steps |

## New Outputs in `data/`

| File | Location |
|------|----------|
| `derivative_spectra.npy` | `data/processed/` |
| `line_features.npy` | `data/processed/` |
| `cvae_scores.npy` | `data/results/` |
| `cvae_pvalues.npy` | `data/results/` |
| `cvae_losses.npy` | `data/results/` |
| `flow_scores.npy` | `data/results/` |
| `flow_pvalues.npy` | `data/results/` |
| `flow_losses.npy` | `data/results/` |
| `evaluation_results.json` | `data/results/` |
