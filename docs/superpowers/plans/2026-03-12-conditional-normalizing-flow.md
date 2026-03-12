# Conditional Normalizing Flow Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conditional RealNVP normalizing flow that scores spectra by NLL given stellar parameters, and wire it into the pipeline as a sixth anomaly detector.

**Architecture:** PCA compresses spectra (3500 bins → 50 components). A conditional RealNVP with FiLM-modulated coupling blocks learns `p(z_pca | Teff, logg, FeH, sn_median)`. NLL is the anomaly score — higher means more anomalous given the star's expected physics.

**Tech Stack:** Python 3.12, PyTorch, scikit-learn (PCA, StandardScaler), NumPy, pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/models/normalizing_flow.py` | `CouplingBlock`, `ConditionalRealNVP`, `train_normalizing_flow` |
| Create | `tests/test_normalizing_flow.py` | Unit tests for the flow model |
| Modify | `src/run_pipeline.py` | Step 5d, `param_counts` dict, `all_scores` dict |
| Modify | `tests/test_integration.py` | Add flow assertions to the synthetic pipeline test |

---

## Chunk 1: Model implementation

### Task 1: Write failing tests for the flow

**Files:**
- Create: `tests/test_normalizing_flow.py`

- [ ] **Step 1: Write all four tests**

```python
# tests/test_normalizing_flow.py
"""Tests for conditional RealNVP normalizing flow."""
import numpy as np
import pytest
from src.models.normalizing_flow import train_normalizing_flow

FAST_CONFIG = dict(
    n_components=10,
    n_coupling=4,
    hidden_dim=32,
    meta_embed_dim=8,
    epochs=3,
    batch_size=32,
)


def _spectra(n=100, bins=500, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, bins)).astype(np.float32)


def _meta(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 4)).astype(np.float32)


def test_flow_nll_shape():
    spectra = _spectra()
    meta = _meta()
    model, losses = train_normalizing_flow(spectra, meta, **FAST_CONFIG)
    scores = model.nll_score(spectra, meta)
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))
    assert len(losses) == FAST_CONFIG["epochs"]


def test_flow_anomalies_score_higher():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (90, 500)).astype(np.float32)
    anomalous = rng.normal(0, 1, (10, 500)).astype(np.float32) + 10.0
    meta_normal = rng.normal(0, 1, (90, 4)).astype(np.float32)
    meta_anomalous = rng.normal(0, 1, (10, 4)).astype(np.float32)

    spectra = np.vstack([normal, anomalous])
    meta = np.vstack([meta_normal, meta_anomalous])

    model, _ = train_normalizing_flow(spectra, meta, **FAST_CONFIG)
    scores = model.nll_score(spectra, meta)
    assert np.mean(scores[90:]) > np.mean(scores[:90])


def test_flow_param_count():
    model, _ = train_normalizing_flow(_spectra(50), _meta(50), **FAST_CONFIG)
    count = model.param_count()
    assert isinstance(count, int)
    assert count > 0


def test_flow_handles_nan_metadata():
    spectra = _spectra()
    meta = _meta()
    meta[0, 0] = float("nan")
    meta[5, 2] = float("nan")
    model, _ = train_normalizing_flow(spectra, meta, **FAST_CONFIG)
    scores = model.nll_score(spectra, meta)
    assert np.all(np.isfinite(scores))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_normalizing_flow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.normalizing_flow'`

---

### Task 2: Implement `CouplingBlock`

**Files:**
- Create: `src/models/normalizing_flow.py`

- [ ] **Step 3: Write `CouplingBlock`**

```python
# src/models/normalizing_flow.py
"""Conditional RealNVP normalizing flow for spectral anomaly detection."""
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class CouplingBlock(nn.Module):
    """Affine coupling layer with FiLM conditioning from a metadata embedding.

    The coupling MLP maps z_a (pass-through dims) → hidden activations.
    FiLM (Feature-wise Linear Modulation) applies a per-channel scale+shift
    to those activations using a linear projection of the metadata embedding.
    The modulated hidden state is then projected to (s_raw, t), which define
    the affine transform applied to z_b (transformed dims).
    """

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 128,
        meta_embed_dim: int = 16,
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        in_dim = int(mask.sum().item())
        out_dim = dim - in_dim

        # Maps z_a → hidden activations
        self.net_hidden = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        # FiLM: meta_embedding → (γ, β), each of size hidden_dim
        self.film = nn.Linear(meta_embed_dim, hidden_dim * 2)
        # Maps modulated hidden → (s_raw, t), each of size out_dim
        self.net_out = nn.Linear(hidden_dim, out_dim * 2)
        self._out_dim = out_dim

    def forward(
        self, z: torch.Tensor, meta_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (z_out, log_det_per_sample)."""
        z_a = z[:, self.mask]    # pass-through dims
        z_b = z[:, ~self.mask]   # dims to transform

        h = self.net_hidden(z_a)

        # FiLM modulation
        gamma, beta = self.film(meta_emb).chunk(2, dim=1)
        h = h * gamma + beta

        s_raw, t = self.net_out(h).chunk(2, dim=1)
        s = torch.tanh(s_raw)

        z_out = z.clone()
        z_out[:, ~self.mask] = z_b * torch.exp(s) + t
        log_det = s.sum(dim=1)  # log |det J| for this block
        return z_out, log_det
```

---

### Task 3: Implement `ConditionalRealNVP` and `train_normalizing_flow`

**Files:**
- Modify: `src/models/normalizing_flow.py` (append to same file)

- [ ] **Step 4: Append `ConditionalRealNVP` and `train_normalizing_flow`**

```python
class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP flow: PCA compression + affine coupling blocks with FiLM conditioning.

    Anomaly score = NLL under the Gaussian base distribution.
    The log_det is accumulated in the forward (data→latent) direction, so the
    negative sign in the NLL formula is correct:
        NLL = 0.5*||z||² + 0.5*D*log(2π) - sum(log_det)
    """

    def __init__(
        self,
        n_components: int = 50,
        n_coupling: int = 8,
        hidden_dim: int = 128,
        meta_dim: int = 4,
        meta_embed_dim: int = 16,
    ):
        super().__init__()
        self.n_components = n_components

        self.meta_embedder = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, meta_embed_dim),
            nn.ReLU(),
        )

        self.coupling_blocks = nn.ModuleList()
        for i in range(n_coupling):
            mask = torch.zeros(n_components, dtype=torch.bool)
            if i % 2 == 0:
                mask[::2] = True   # even indices pass through
            else:
                mask[1::2] = True  # odd indices pass through
            self.coupling_blocks.append(
                CouplingBlock(n_components, mask, hidden_dim=hidden_dim, meta_embed_dim=meta_embed_dim)
            )

        # Set by train_normalizing_flow; not nn parameters
        self.pca: PCA | None = None
        self.scaler: StandardScaler | None = None

    def _embed_meta(self, meta: torch.Tensor) -> torch.Tensor:
        meta = meta.clone()
        meta[~torch.isfinite(meta)] = 0.0
        return self.meta_embedder(meta)

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow forward pass. Returns (z_final, total_log_det)."""
        meta_emb = self._embed_meta(meta)
        total_log_det = torch.zeros(z.size(0), device=z.device)
        for block in self.coupling_blocks:
            z, log_det = block(z, meta_emb)
            total_log_det += log_det
        return z, total_log_det

    def _nll(self, z_out: torch.Tensor, total_log_det: torch.Tensor) -> torch.Tensor:
        D = z_out.size(1)
        return 0.5 * (z_out ** 2).sum(dim=1) + 0.5 * D * math.log(2 * math.pi) - total_log_det

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pca_transform(self, spectra: np.ndarray) -> np.ndarray:
        assert self.pca is not None and self.scaler is not None, (
            "Model has no fitted PCA — call train_normalizing_flow first"
        )
        return self.pca.transform(self.scaler.transform(spectra))

    @torch.no_grad()
    def nll_score(self, spectra: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        """Per-spectrum NLL anomaly score. Higher = more anomalous given metadata."""
        was_training = self.training
        self.eval()
        z_np = self._pca_transform(spectra.astype(np.float32))
        z = torch.tensor(z_np, dtype=torch.float32)
        meta = torch.tensor(metadata.copy(), dtype=torch.float32)
        meta[~torch.isfinite(meta)] = 0.0
        z_out, log_det = self.forward(z, meta)
        scores = self._nll(z_out, log_det).numpy()
        if was_training:
            self.train()
        return scores


def train_normalizing_flow(
    spectra: np.ndarray,
    metadata: np.ndarray,
    n_components: int = 50,
    n_coupling: int = 8,
    hidden_dim: int = 128,
    meta_embed_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple["ConditionalRealNVP", list[float]]:
    """Fit PCA on spectra, then train the conditional RealNVP. Returns (model, epoch_losses).

    n_components is clamped to min(n_components, min(spectra.shape)) so small
    datasets and test runs don't raise a PCA error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clamp to valid PCA range
    n_components = min(n_components, min(spectra.shape[0], spectra.shape[1]))

    # Fit PCA on training spectra
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    z_np = pca.fit_transform(scaler.fit_transform(spectra))  # (n, n_components)

    # Zero-fill NaN metadata for training
    meta_clean = metadata.copy().astype(np.float32)
    meta_clean[~np.isfinite(meta_clean)] = 0.0

    model = ConditionalRealNVP(
        n_components=n_components,
        n_coupling=n_coupling,
        hidden_dim=hidden_dim,
        meta_dim=metadata.shape[1],
        meta_embed_dim=meta_embed_dim,
    ).to(device)
    model.pca = pca
    model.scaler = scaler

    z_tensor = torch.tensor(z_np, dtype=torch.float32)
    meta_tensor = torch.tensor(meta_clean, dtype=torch.float32)
    loader = DataLoader(TensorDataset(z_tensor, meta_tensor), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for z_batch, meta_batch in loader:
            z_batch, meta_batch = z_batch.to(device), meta_batch.to(device)
            z_out, log_det = model(z_batch, meta_batch)
            loss = model._nll(z_out, log_det).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        epoch_losses.append(total / n_batches)

    return model.cpu(), epoch_losses
```

- [ ] **Step 5: Run all four tests**

Run: `uv run python -m pytest tests/test_normalizing_flow.py -v`
Expected: All 4 PASS

- [ ] **Step 6: Run full test suite to check for regressions**

Run: `uv run python -m pytest tests/ -v --tb=short`
Expected: All 78 existing tests pass plus 4 new ones (82 total)

- [ ] **Step 7: Commit**

```bash
git add src/models/normalizing_flow.py tests/test_normalizing_flow.py
git commit -m "feat: add conditional RealNVP normalizing flow with FiLM conditioning"
```

---

## Chunk 2: Pipeline integration

> **Precondition:** `src/models/normalizing_flow.py` must exist before any step in this chunk is applied. The integration test import (`from src.models.normalizing_flow import train_normalizing_flow`) is a module-level statement — if the module does not exist, all tests in `test_integration.py` fail with `ModuleNotFoundError`. Commit Chunk 1 first.

### Task 4: Wire flow into `run_pipeline.py`

**Files:**
- Modify: `src/run_pipeline.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add import to `run_pipeline.py`**

In `src/run_pipeline.py`, find the block of model imports (around line 20) and add:

```python
from src.models.normalizing_flow import train_normalizing_flow
```

- [ ] **Step 2: Add Step 5d after the conformal calibration block**

In `src/run_pipeline.py`, find the line:
```python
    # Step 5b: Stability runs
```

Insert the following block immediately before it:

```python
    # Step 5d: Conditional Normalizing Flow
    logger.info("=== Step 5d: Training Conditional Normalizing Flow ===")
    flow_model, flow_losses = train_normalizing_flow(
        spectra[train_idx].astype(np.float32),
        meta_features[train_idx],
        n_components=n_components,
        epochs=50,
    )
    flow_scores = flow_model.nll_score(spectra.astype(np.float32), meta_features)
    np.save(RESULTS_DIR / "flow_scores.npy", flow_scores)
    np.save(RESULTS_DIR / "flow_losses.npy", np.array(flow_losses))

```

- [ ] **Step 3: Update `param_counts` dict to include `"flow"`**

In `src/run_pipeline.py`, find the `param_counts` dict literal (around lines 201–207):

```python
    param_counts = {
        "classical_if": classical.param_count(),
        "autoencoder": model.param_count(),
        "ocsvm": ocsvm.param_count(),
        "dagmm": dagmm_model.param_count(),
        "conditional_ae": cond_ae_model.param_count(),
    }
```

Replace with:

```python
    param_counts = {
        "classical_if": classical.param_count(),
        "autoencoder": model.param_count(),
        "ocsvm": ocsvm.param_count(),
        "dagmm": dagmm_model.param_count(),
        "conditional_ae": cond_ae_model.param_count(),
        "flow": flow_model.param_count(),
    }
```

- [ ] **Step 4: Add `"flow"` to `all_scores` dict**

In `src/run_pipeline.py`, find the `all_scores` dict (around line 213):

```python
    all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores, "cond_ae": cond_ae_scores}
```

Replace with:

```python
    all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores, "cond_ae": cond_ae_scores, "flow": flow_scores}
```

- [ ] **Step 5: Add flow assertions to the integration test**

In `tests/test_integration.py`, add the following import at the top with the other model imports:

```python
from src.models.normalizing_flow import train_normalizing_flow
```

At the end of `test_full_pipeline_synthetic` (after the conformal calibration assertions), append:

```python
    # Normalizing Flow
    flow_model, flow_losses = train_normalizing_flow(
        spectra[train_idx].astype(np.float32),
        meta_features[train_idx],
        n_components=10,
        n_coupling=4,
        hidden_dim=32,
        epochs=3,
        batch_size=16,
    )
    assert len(flow_losses) == 3
    assert flow_model.param_count() > 0
    flow_scores = flow_model.nll_score(spectra.astype(np.float32), meta_features)
    assert flow_scores.shape == (n_spectra,)
    assert np.all(np.isfinite(flow_scores))

    # Verify flow scores integrate with compare_n_models (keeps all_scores in sync with pipeline)
    all_scores_with_flow = {**all_scores, "flow": flow_scores}
    n_comparison_flow = compare_n_models(all_scores_with_flow, top_n=10)
    assert "n_models_agreed" in n_comparison_flow.columns
    assert "combined_rank" in n_comparison_flow.columns
```

- [ ] **Step 6: Run integration test to verify**

Run: `uv run python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `uv run python -m pytest tests/ -v --tb=short`
Expected: All tests pass (82 unit + 1 integration)

- [ ] **Step 8: Commit**

```bash
git add src/run_pipeline.py tests/test_integration.py
git commit -m "feat: wire conditional normalizing flow into pipeline"
```

---

## Notes

- `agreement_fraction` in `focused_review.parquet` and `top_anomalies_agreed.parquet` will change from `/5` to `/6` once the flow is added to `all_scores`. This is expected — the fraction now reflects six models.
- `flow_scores.npy` is saved to `data/results/` alongside all other model scores and will automatically appear in the dashboard's Model Comparison and Anomaly Browser tabs.
- The flow's `nll_score(spectra, metadata)` takes two arguments, unlike unconditional models' `score(spectra)`. Do not use it as a drop-in replacement in generic model loops.
