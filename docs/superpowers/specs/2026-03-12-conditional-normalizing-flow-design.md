# Conditional Normalizing Flow for Spectral Anomaly Detection

**Date:** 2026-03-12
**Status:** Approved

## Goal

Add a conditional RealNVP normalizing flow that scores each spectrum by how anomalous it is *given* its stellar parameters (Teff, log g, [Fe/H], sn_median). The flow learns `p(flux_pca | metadata)` and uses negative log-likelihood (NLL) as the anomaly score. This replaces the reconstruction-error proxy used by the conditional autoencoder with a proper density estimate.

## Background

The pipeline currently trains five anomaly detectors:
- Classical (PCA + Isolation Forest)
- Autoencoder (unconditional, conv)
- OC-SVM
- DAGMM
- Conditional Autoencoder (metadata-conditioned, reconstruction error score)

All scores feed into `compare_n_models()` via an `all_scores: dict[str, np.ndarray]` interface. The normalizing flow adds a sixth detector using the same interface with no changes to the comparison or dashboard layers.

## Architecture

### Compression

Raw spectra (3500 bins) are compressed to 50 PCA components before the flow. PCA is fit on the training split only and applied to all spectra at score time. This makes the flow tractable on CPU and reduces noise.

### Model: ConditionalRealNVP

```
spectra (n, 3500)
      │
      ▼
  PCA → z (n, 50)   +   metadata (n, 4)
                              │
                         MetaEmbedder MLP (4 → 32 → 16)
                              │
      ┌───────────────────────┘
      ▼
  8 × CouplingBlock (alternating even/odd dimension masks)
    ├─ split z → z_a, z_b
    ├─ CouplingMLP(z_a) → s_raw, t_raw   [hidden_dim=128]
    ├─ FiLM(meta_embedding) → γ, β applied to MLP hidden layer
    ├─ s = tanh(s_raw)
    ├─ z_b = z_b * exp(s) + t
    └─ log_det += sum(s)
      │
      ▼
  Gaussian base:  NLL = 0.5*||z||² + 0.5*D*log(2π) - sum(log_det)
      │
      ▼
  anomaly score (n,)  ← higher = more anomalous
```

**FiLM conditioning:** each `CouplingBlock` has a small linear layer `meta_embedding → (γ, β)` of shape `(hidden_dim * 2,)`. These modulate the hidden activations of the coupling MLP before the final projection to `(s_raw, t_raw)`. This prevents the flow from ignoring metadata, which is the failure mode of simple concatenation approaches.

### Scoring

`model.nll_score(spectra, metadata) -> np.ndarray` applies PCA transform then runs the forward pass to compute per-spectrum NLL. Higher NLL = more anomalous. This is the value saved to `flow_scores.npy`.

## New File

**`src/models/normalizing_flow.py`**

| Class / Function | Purpose |
|---|---|
| `CouplingBlock` | Single affine coupling layer with FiLM conditioning |
| `ConditionalRealNVP` | Full flow: PCA + 8 coupling blocks + Gaussian NLL |
| `train_normalizing_flow(spectra, metadata, ...)` | Fits PCA, trains flow, returns model |
| `model.nll_score(spectra, metadata)` | Anomaly score interface |
| `model.param_count()` | Total trainable parameters |

## Pipeline Integration

New **Step 5d** in `run_pipeline.py`, inserted after the existing conditional AE step, reusing the same `train_idx` / `cal_idx` split and `meta_features` array:

```python
# Step 5d: Conditional Normalizing Flow
flow_model = train_normalizing_flow(
    spectra[train_idx].astype(np.float32),
    meta_features[train_idx],
    n_components=50,
    epochs=50,
)
flow_scores = flow_model.nll_score(spectra.astype(np.float32), meta_features)
np.save(RESULTS_DIR / "flow_scores.npy", flow_scores)
param_counts["flow"] = flow_model.param_count()
```

`all_scores` dict in Step 7 gains a `"flow"` key. No other changes to comparison, dashboard, or categorization code.

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `n_components` | 50 | PCA dims fed to flow |
| `n_coupling` | 8 | Number of coupling blocks |
| `hidden_dim` | 128 | MLP hidden size per coupling block |
| `meta_embed_dim` | 16 | Metadata embedding dimension |
| `epochs` | 50 | Training epochs |
| `batch_size` | 64 | |
| `lr` | 1e-3 | Adam optimizer |

## Testing

**`tests/test_normalizing_flow.py`** — three tests using fast config (`n_components=10`, `n_coupling=4`, `hidden_dim=32`, `epochs=3`):

1. **`test_flow_nll_shape`** — synthetic `(100, 500)` spectra + `(100, 3)` metadata → scores shape `(100,)`, all finite
2. **`test_flow_anomalies_score_higher`** — spectra shifted +10σ score higher NLL than normal spectra on average
3. **`test_flow_param_count`** — `param_count()` returns a positive int

## What This Does Not Change

- Dashboard code (`src/dashboard/app.py`) — already handles arbitrary score keys
- `compare_n_models()` — already accepts `dict[str, np.ndarray]`
- Conformal calibration — can be applied to `flow_scores` in a future step (item 2/3 on the roadmap)
- Anomaly categorization — unchanged

## Out of Scope

- Masked Autoregressive Flow (MAF) — slower inference, not needed
- Flow directly on 3500-dim spectra — intractable, rejected
- Structured residual score — separate roadmap item
- FDR threshold wiring — separate roadmap item
