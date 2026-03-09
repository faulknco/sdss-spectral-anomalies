# Conditional Autoencoder + Conformal Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a metadata-conditioned autoencoder that scores spectra by how anomalous they are *given* their stellar parameters (Teff, log g, [Fe/H], S/N), then calibrate those scores into conformal p-values.

**Architecture:** The existing `SpectralAutoencoder` is unconditional. The new `ConditionalSpectralAutoencoder` concatenates a learned metadata embedding into the bottleneck so it asks "is this spectrum strange for a star with these parameters?". A `SplitConformalCalibrator` converts raw anomaly scores into p-values using a held-out calibration split.

**Tech Stack:** PyTorch (already in project), NumPy, pandas, pytest. No new dependencies required.

---

## Background: What already exists

- `src/data/preprocess.py` — `load_and_preprocess()` reads FITS files. Currently drops `elodieTEff`, `elodieLogG`, `elodieFeH` (SPECOBJ HDU has them but they are not extracted into metadata).
- `src/data/download.py` — `build_sdss_query()` already selects `elodieTEff`, `elodieLogG`, `elodieFeH`, `snmedian`. They land in `metadata.parquet` (step 1) but not in `spectra_metadata.parquet` (step 2).
- `src/models/autoencoder.py` — `SpectralAutoencoder` + `train_autoencoder()`. Conv encoder -> AdaptiveAvgPool -> linear bottleneck -> FC + deconv decoder.
- `src/run_pipeline.py` — Orchestrates all steps. Writes `spectra_metadata.parquet` from `meta_list` (5 fields: filename, ra, dec, subclass, sn_median — no stellar params).
- `src/models/compare.py` — `compare_n_models()` accepts `dict[str, np.ndarray]` of scores. Adding new scores is a one-liner.
- `src/dashboard/app.py` — `load_data()` loads specific `.npy` files. Tabs: Anomaly Browser, Model Comparison, PCA Explorer, Model Stability.
- Tests use small synthetic data (100 spectra, 500 wavelengths). Fast training uses `epochs=3`, `bottleneck_dim=16`.

---

### Task 1: Persist stellar parameters into spectra_metadata

**Problem:** `load_and_preprocess()` in `preprocess.py` builds `meta_list` with 5 fields. The stellar params are in FITS `SPECOBJ` HDU but not extracted. We need them in `spectra_metadata.parquet` for the conditional AE.

**Files:**
- Modify: `src/data/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write a failing test**

Add to `tests/test_preprocess.py`:

```python
def test_metadata_includes_stellar_params():
    from src.data.preprocess import _make_metadata_dict
    meta = _make_metadata_dict(
        filename="spec-0001-50000-0001.fits",
        ra=180.0,
        dec=45.0,
        subclass="G5",
        sn_median=25.0,
        teff=5500.0,
        logg=4.4,
        feh=-0.1,
    )
    assert meta["elodie_teff"] == 5500.0
    assert meta["elodie_logg"] == 4.4
    assert meta["elodie_feh"] == -0.1
```

Run: `uv run pytest tests/test_preprocess.py::test_metadata_includes_stellar_params -v`
Expected: FAIL — `_make_metadata_dict` does not exist.

**Step 2: Add `_make_metadata_dict` helper to `src/data/preprocess.py`**

Add after the imports:

```python
def _make_metadata_dict(
    filename: str,
    ra: float,
    dec: float,
    subclass: str,
    sn_median: float,
    teff: float,
    logg: float,
    feh: float,
) -> dict:
    return {
        "filename": filename,
        "ra": ra,
        "dec": dec,
        "subclass": subclass,
        "sn_median": sn_median,
        "elodie_teff": teff,
        "elodie_logg": logg,
        "elodie_feh": feh,
    }
```

Then update the metadata dict inside `load_and_preprocess` (lines 78-86). Replace the current dict literal with:

```python
                def _safe_float(arr, name, fallback=np.nan):
                    return float(arr[name][0]) if name in arr.dtype.names else fallback

                metadata.append(_make_metadata_dict(
                    filename=fpath.name,
                    ra=float(specobj["RA"][0]),
                    dec=float(specobj["DEC"][0]),
                    subclass=str(specobj["SUBCLASS"][0]).strip(),
                    sn_median=_safe_float(specobj, "SN_MEDIAN_ALL", 0.0),
                    teff=_safe_float(specobj, "ELODIE_TEFF"),
                    logg=_safe_float(specobj, "ELODIE_LOGG"),
                    feh=_safe_float(specobj, "ELODIE_FEH"),
                ))
```

Note: SDSS FITS SPECOBJ field names are `ELODIE_TEFF`, `ELODIE_LOGG`, `ELODIE_FEH`. Use `np.nan` fallback for missing fields.

**Step 3: Run test**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: All PASS.

**Step 4: Commit**

```bash
git add src/data/preprocess.py tests/test_preprocess.py
git commit -m "feat: persist elodie stellar params into spectra metadata"
```

---

### Task 2: Create ConditionalSpectralAutoencoder

**Context:** Architecture flow:

```
flux (B, 1, L) -> conv_encoder -> (B, 128) via AdaptiveAvgPool
meta (B, 3+)   -> meta_mlp    -> (B, meta_embed_dim)
concat -> encoder_fc -> latent (B, bottleneck_dim)

latent -> concat meta_emb -> decoder_fc -> reshape -> deconv -> recon (B, 1, L)
```

The metadata embedding is injected at both encoder and decoder bottleneck.

**Files:**
- Create: `src/models/conditional_autoencoder.py`
- Create: `tests/test_conditional_autoencoder.py`

**Step 1: Write failing tests**

Create `tests/test_conditional_autoencoder.py`:

```python
import numpy as np
import torch
import pytest
from src.models.conditional_autoencoder import (
    ConditionalSpectralAutoencoder,
    train_conditional_autoencoder,
)


def make_meta(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 3)).astype(np.float32)


def test_forward_output_shape():
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=32)
    x = torch.randn(8, 1, 500)
    meta = torch.randn(8, 3)
    out = model(x, meta)
    assert out.shape == (8, 1, 500)


def test_encode_shape():
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=32)
    x = torch.randn(8, 1, 500)
    meta = torch.randn(8, 3)
    z = model.encode(x, meta)
    assert z.shape == (8, 32)


def test_param_count_positive():
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=32)
    assert model.param_count() > 0


def test_reconstruction_error_shape():
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=32)
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((16, 500)).astype(np.float32)
    meta = make_meta(16)
    errors = model.reconstruction_error(spectra, meta)
    assert errors.shape == (16,)
    assert np.all(errors >= 0)


def test_train_reduces_loss():
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((64, 500)).astype(np.float32)
    meta = make_meta(64)
    model, losses = train_conditional_autoencoder(
        spectra, meta, bottleneck_dim=16, epochs=5, batch_size=16, lr=1e-3
    )
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_handles_nan_metadata():
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((8, 500)).astype(np.float32)
    meta = make_meta(8)
    meta[0, 0] = np.nan
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=16)
    errors = model.reconstruction_error(spectra, meta)
    assert errors.shape == (8,)
    assert np.all(np.isfinite(errors))
```

Run: `uv run pytest tests/test_conditional_autoencoder.py -v`
Expected: FAIL — module not found.

**Step 2: Implement `src/models/conditional_autoencoder.py`**

```python
"""Metadata-conditioned autoencoder for spectral anomaly detection."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ConditionalSpectralAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3500,
        meta_dim: int = 3,
        bottleneck_dim: int = 64,
        meta_embed_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.meta_dim = meta_dim
        self.bottleneck_dim = bottleneck_dim
        self.meta_embed_dim = meta_embed_dim

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )  # output: (B, 128)

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, meta_embed_dim),
            nn.ReLU(),
        )  # output: (B, meta_embed_dim)

        self.encoder_fc = nn.Linear(128 + meta_embed_dim, bottleneck_dim)

        self._conv_out_dim = (input_dim + 7) // 8
        self.decoder_fc = nn.Linear(bottleneck_dim + meta_embed_dim, 128 * self._conv_out_dim)

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _embed_meta(self, meta: torch.Tensor) -> torch.Tensor:
        meta = meta.clone()
        meta[~torch.isfinite(meta)] = 0.0
        return self.meta_mlp(meta)

    def encode(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_encoder(x)
        meta_emb = self._embed_meta(meta)
        return self.encoder_fc(torch.cat([conv_out, meta_emb], dim=1))

    def decode(self, z: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        meta_emb = self._embed_meta(meta)
        x = self.decoder_fc(torch.cat([z, meta_emb], dim=1))
        x = x.view(x.size(0), 128, self._conv_out_dim)
        x = self.conv_decoder(x)
        if x.size(2) > self.input_dim:
            x = x[:, :, : self.input_dim]
        elif x.size(2) < self.input_dim:
            x = nn.functional.pad(x, (0, self.input_dim - x.size(2)))
        return x

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x, meta), meta)

    @torch.no_grad()
    def reconstruction_error(self, spectra: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        self.eval()
        x = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        meta = torch.tensor(metadata, dtype=torch.float32)
        recon = self.forward(x, meta)
        return ((x - recon) ** 2).mean(dim=(1, 2)).numpy()


def train_conditional_autoencoder(
    spectra: np.ndarray,
    metadata: np.ndarray,
    bottleneck_dim: int = 64,
    meta_embed_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple["ConditionalSpectralAutoencoder", list[float]]:
    """Train the conditional autoencoder. metadata shape: (n, meta_dim). NaNs -> 0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
    meta = torch.tensor(metadata, dtype=torch.float32)
    meta[~torch.isfinite(meta)] = 0.0

    loader = DataLoader(TensorDataset(x, meta), batch_size=batch_size, shuffle=True)

    model = ConditionalSpectralAutoencoder(
        input_dim=spectra.shape[1],
        meta_dim=metadata.shape[1],
        bottleneck_dim=bottleneck_dim,
        meta_embed_dim=meta_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for x_batch, meta_batch in loader:
            x_batch, meta_batch = x_batch.to(device), meta_batch.to(device)
            loss = criterion(model(x_batch, meta_batch), x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        epoch_losses.append(total / n_batches)

    return model.cpu(), epoch_losses
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_conditional_autoencoder.py -v`
Expected: All 6 PASS.

**Step 4: Commit**

```bash
git add src/models/conditional_autoencoder.py tests/test_conditional_autoencoder.py
git commit -m "feat: add metadata-conditioned autoencoder"
```

---

### Task 3: Create SplitConformalCalibrator

**Context:** Split conformal p-value formula for anomaly score `s_test` given calibration scores `s_1..s_m`:

```
p = (number of calibration scores >= s_test + 1) / (m + 1)
```

This gives a finite-sample valid p-value: if the test spectrum is from the same distribution, p is super-uniform.

**Files:**
- Create: `src/models/conformal.py`
- Create: `tests/test_conformal.py`

**Step 1: Write failing tests**

Create `tests/test_conformal.py`:

```python
import numpy as np
import pytest
from src.models.conformal import SplitConformalCalibrator


def test_fit_stores_calibration_scores():
    cal = SplitConformalCalibrator()
    cal.fit(np.random.default_rng(0).standard_normal(100))
    assert len(cal.calibration_scores_) == 100


def test_pvalues_shape():
    rng = np.random.default_rng(0)
    cal = SplitConformalCalibrator()
    cal.fit(rng.standard_normal(200))
    pvals = cal.pvalues(rng.standard_normal(50))
    assert pvals.shape == (50,)


def test_pvalues_in_unit_interval():
    rng = np.random.default_rng(0)
    cal = SplitConformalCalibrator()
    cal.fit(rng.standard_normal(200))
    pvals = cal.pvalues(rng.standard_normal(50))
    assert np.all(pvals >= 0) and np.all(pvals <= 1)


def test_extreme_outlier_gets_low_pvalue():
    rng = np.random.default_rng(0)
    cal = SplitConformalCalibrator()
    cal.fit(rng.standard_normal(500))
    pvals = cal.pvalues(np.array([100.0]))
    assert pvals[0] < 0.01


def test_typical_inlier_gets_high_pvalue():
    rng = np.random.default_rng(0)
    cal_scores = rng.standard_normal(500)
    cal = SplitConformalCalibrator()
    cal.fit(cal_scores)
    pvals = cal.pvalues(np.array([float(np.median(cal_scores))]))
    assert pvals[0] > 0.3


def test_raises_if_not_fitted():
    cal = SplitConformalCalibrator()
    with pytest.raises(RuntimeError, match="fit"):
        cal.pvalues(np.array([1.0]))


def test_threshold_returns_float():
    rng = np.random.default_rng(0)
    cal = SplitConformalCalibrator()
    cal.fit(rng.standard_normal(500))
    assert isinstance(cal.threshold(alpha=0.05), float)
```

Run: `uv run pytest tests/test_conformal.py -v`
Expected: FAIL — module not found.

**Step 2: Implement `src/models/conformal.py`**

```python
"""Split conformal calibration for anomaly scores."""
import numpy as np


class SplitConformalCalibrator:
    """Convert raw anomaly scores to conformal p-values.

    Usage:
        cal = SplitConformalCalibrator()
        cal.fit(calibration_scores)      # held-out scores; higher = more anomalous
        pvals = cal.pvalues(test_scores)
        thresh = cal.threshold(alpha=0.05)
    """

    def fit(self, calibration_scores: np.ndarray) -> "SplitConformalCalibrator":
        self.calibration_scores_ = np.array(calibration_scores, dtype=np.float64)
        return self

    def pvalues(self, test_scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "calibration_scores_"):
            raise RuntimeError("Call fit() before pvalues()")
        test = np.array(test_scores, dtype=np.float64)
        m = len(self.calibration_scores_)
        counts = (self.calibration_scores_[:, None] >= test[None, :]).sum(axis=0)
        return (counts + 1) / (m + 1)

    def threshold(self, alpha: float = 0.05) -> float:
        if not hasattr(self, "calibration_scores_"):
            raise RuntimeError("Call fit() before threshold()")
        return float(np.quantile(self.calibration_scores_, 1 - alpha))
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_conformal.py -v`
Expected: All 7 PASS.

**Step 4: Commit**

```bash
git add src/models/conformal.py tests/test_conformal.py
git commit -m "feat: add split conformal calibrator for anomaly p-values"
```

---

### Task 4: Add build_metadata_features helper

**Context:** The pipeline will need to convert the `spectra_metadata.parquet` DataFrame into a normalized float32 matrix for the conditional AE. This selects 4 columns, fills NaNs with column medians, and standardizes each column.

**Files:**
- Modify: `src/data/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write failing tests**

Add to `tests/test_preprocess.py`:

```python
def test_build_metadata_features_shape():
    import pandas as pd
    from src.data.preprocess import build_metadata_features
    df = pd.DataFrame({
        "elodie_teff": [5000.0, 6000.0, np.nan, 7000.0],
        "elodie_logg": [4.0, 4.5, 3.5, np.nan],
        "elodie_feh": [-0.5, 0.0, 0.3, -0.2],
        "sn_median": [20.0, 35.0, 15.0, 50.0],
    })
    features = build_metadata_features(df)
    assert features.shape == (4, 4)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_build_metadata_features_standardized():
    import pandas as pd
    from src.data.preprocess import build_metadata_features
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "elodie_teff": rng.uniform(4000, 8000, 100),
        "elodie_logg": rng.uniform(2.0, 5.0, 100),
        "elodie_feh": rng.uniform(-1.5, 0.5, 100),
        "sn_median": rng.uniform(10, 100, 100),
    })
    features = build_metadata_features(df)
    assert np.abs(features.mean(axis=0)).max() < 0.1
```

Run: `uv run pytest tests/test_preprocess.py::test_build_metadata_features_shape tests/test_preprocess.py::test_build_metadata_features_standardized -v`
Expected: FAIL.

**Step 2: Add to `src/data/preprocess.py`** (at end of file):

```python
METADATA_FEATURE_COLS = ["elodie_teff", "elodie_logg", "elodie_feh", "sn_median"]


def build_metadata_features(metadata_df) -> np.ndarray:
    """Extract and standardize stellar metadata into a float32 feature matrix.

    Missing values are filled with column median. Columns are z-score standardized.
    Returns array of shape (n_spectra, len(METADATA_FEATURE_COLS)).
    """
    df = metadata_df[METADATA_FEATURE_COLS].copy().astype(np.float64)
    for col in df.columns:
        median = df[col].median()
        df[col] = df[col].fillna(median if np.isfinite(median) else 0.0)
    mean = df.mean()
    std = df.std().replace(0, 1)
    df = (df - mean) / std
    return df.values.astype(np.float32)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: All PASS.

**Step 4: Commit**

```bash
git add src/data/preprocess.py tests/test_preprocess.py
git commit -m "feat: add build_metadata_features for conditional AE input"
```

---

### Task 5: Wire conditional AE and conformal into run_pipeline.py

**Context:** Add two new steps inside `run()`:
- Step 5: Train conditional AE on 80% of spectra+metadata. Score all.
- Step 5c: Fit conformal calibrator on held-out 20% scores. Save p-values.

Also update `compare_n_models` dict and `param_counts` to include the new model.

**Files:**
- Modify: `src/run_pipeline.py`

**Step 1: Add imports** at top of file after existing imports:

```python
from src.models.conditional_autoencoder import train_conditional_autoencoder
from src.models.conformal import SplitConformalCalibrator
from src.data.preprocess import build_metadata_features, METADATA_FEATURE_COLS
```

**Step 2: Add Steps 5 and 5c** inside `run()`, after the DAGMM block and before the stability block:

```python
    # Step 5: Conditional Autoencoder + Conformal Calibration
    logger.info("=== Step 5: Training Conditional Autoencoder ===")
    meta_df = pd.read_parquet(PROCESSED_DIR / "spectra_metadata.parquet")
    for col in METADATA_FEATURE_COLS:
        if col not in meta_df.columns:
            meta_df[col] = float("nan")
    meta_features = build_metadata_features(meta_df)

    n = len(spectra)
    cal_size = max(1, int(0.2 * n))
    train_idx = np.arange(n - cal_size)
    cal_idx = np.arange(n - cal_size, n)

    cond_ae_model, cond_ae_losses = train_conditional_autoencoder(
        spectra[train_idx].astype(np.float32), meta_features[train_idx], bottleneck_dim=64, epochs=50
    )
    np.save(RESULTS_DIR / "cond_ae_losses.npy", np.array(cond_ae_losses))

    cond_ae_scores = cond_ae_model.reconstruction_error(spectra.astype(np.float32), meta_features)
    np.save(RESULTS_DIR / "conditional_ae_scores.npy", cond_ae_scores)

    logger.info("=== Step 5c: Conformal calibration ===")
    conformal = SplitConformalCalibrator()
    conformal.fit(cond_ae_scores[cal_idx])
    cond_ae_pvalues = conformal.pvalues(cond_ae_scores)
    np.save(RESULTS_DIR / "conditional_ae_pvalues.npy", cond_ae_pvalues)
    with open(RESULTS_DIR / "conformal_threshold.json", "w") as f:
        json.dump({"alpha": 0.05, "threshold": conformal.threshold(alpha=0.05)}, f, indent=2)
```

**Step 3: Update `compare_n_models` call**:

Change:
```python
    all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores}
```
To:
```python
    all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores, "cond_ae": cond_ae_scores}
```

**Step 4: Update `param_counts` dict**:

Add `"conditional_ae": cond_ae_model.param_count()` to the param_counts dict.

**Step 5: Run unit tests** (not integration):

Run: `uv run pytest tests/ -v --ignore=tests/test_integration.py -x`
Expected: All PASS.

**Step 6: Commit**

```bash
git add src/run_pipeline.py
git commit -m "feat: wire conditional AE and conformal calibration into pipeline"
```

---

### Task 6: Update integration test

**Context:** `tests/test_integration.py` exercises the full pipeline on 100 synthetic spectra. Add conditional AE + conformal steps.

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Add imports** at top of test file:

```python
from src.models.conditional_autoencoder import train_conditional_autoencoder
from src.models.conformal import SplitConformalCalibrator
from src.data.preprocess import build_metadata_features
import pandas as pd
```

**Step 2: Add new assertions** at end of `test_full_pipeline_synthetic`, after the categorization block:

```python
    # Conditional Autoencoder
    meta_df = pd.DataFrame({
        "elodie_teff": rng.uniform(4000, 8000, n_spectra),
        "elodie_logg": rng.uniform(2.0, 5.0, n_spectra),
        "elodie_feh": rng.uniform(-1.5, 0.5, n_spectra),
        "sn_median": rng.uniform(10, 100, n_spectra),
    })
    meta_features = build_metadata_features(meta_df)
    assert meta_features.shape == (n_spectra, 4)

    cal_size = 20
    train_idx = np.arange(n_spectra - cal_size)
    cal_idx = np.arange(n_spectra - cal_size, n_spectra)

    cond_ae_model, cond_ae_losses = train_conditional_autoencoder(
        spectra[train_idx].astype(np.float32),
        meta_features[train_idx],
        bottleneck_dim=16, epochs=3, batch_size=16,
    )
    assert len(cond_ae_losses) == 3
    assert cond_ae_model.param_count() > 0

    cond_ae_scores = cond_ae_model.reconstruction_error(spectra.astype(np.float32), meta_features)
    assert cond_ae_scores.shape == (n_spectra,)
    assert np.all(cond_ae_scores >= 0)

    conformal = SplitConformalCalibrator()
    conformal.fit(cond_ae_scores[cal_idx])
    pvals = conformal.pvalues(cond_ae_scores)
    assert pvals.shape == (n_spectra,)
    assert np.all(pvals >= 0) and np.all(pvals <= 1)
    assert isinstance(conformal.threshold(alpha=0.05), float)
```

**Step 3: Run integration test**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS.

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add conditional AE and conformal steps to integration test"
```

---

### Task 7: Update dashboard

**Context:** Load new `.npy` files, add sort options, show metrics, add Tab 5 for p-value histogram.

**Files:**
- Modify: `src/dashboard/app.py`

**Step 1: Update `load_data()` return value**

After loading `categories`, add:

```python
    cond_ae_scores_path = RESULTS_DIR / "conditional_ae_scores.npy"
    cond_ae_pvalues_path = RESULTS_DIR / "conditional_ae_pvalues.npy"
    cond_ae_scores = np.load(cond_ae_scores_path) if cond_ae_scores_path.exists() else np.zeros(len(if_scores))
    cond_ae_pvalues = np.load(cond_ae_pvalues_path) if cond_ae_pvalues_path.exists() else np.ones(len(if_scores))
```

Extend the return tuple to include `cond_ae_scores, cond_ae_pvalues` at the end.

**Step 2: Update unpack in `main()`**

Add `cond_ae_scores, cond_ae_pvalues` to the unpack tuple.

**Step 3: Add Tab 5**

Change `tab1, tab2, tab3, tab4 = st.tabs(...)` to include `tab5` and `"Conformal P-Values"`.

**Step 4: Update Anomaly Browser sort options**

Add `"Conditional AE"` and `"Conformal P-Value (most anomalous)"` to the selectbox list.

Add elif branches:
```python
        elif sort_by == "Conditional AE":
            order = np.argsort(-cond_ae_scores)
        else:  # Conformal P-Value
            order = np.argsort(cond_ae_pvalues)
```

Add `cond_ae_score` and `cond_ae_pvalue` columns to `table_data`.

In the detail view, add metrics:
```python
            col5, col6 = st.columns(2)
            col5.metric("Cond AE Score", f"{cond_ae_scores[spectrum_idx]:.4f}")
            col6.metric("Conformal p-value", f"{cond_ae_pvalues[spectrum_idx]:.4f}")
```

**Step 5: Implement Tab 5 body**

```python
    with tab5:
        st.subheader("Conditional AE: Conformal P-Value Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=cond_ae_pvalues, nbinsx=50, name="p-values",
                                   marker_color="steelblue", opacity=0.75))
        fig.update_layout(xaxis_title="Conformal p-value", yaxis_title="Count",
                          title="Uniform = well-calibrated; spike near 0 = anomalies", height=400)
        st.plotly_chart(fig, use_container_width=True)

        alpha = st.slider("Flag anomalies at p-value <=", 0.01, 0.20, 0.05, step=0.01)
        n_flagged = int((cond_ae_pvalues <= alpha).sum())
        st.metric(f"Spectra flagged at alpha={alpha}", n_flagged)

        flagged_idx = np.where(cond_ae_pvalues <= alpha)[0]
        if len(flagged_idx) > 0:
            flagged_df = metadata.iloc[flagged_idx].copy()
            flagged_df["cond_ae_score"] = cond_ae_scores[flagged_idx]
            flagged_df["cond_ae_pvalue"] = cond_ae_pvalues[flagged_idx]
            st.dataframe(flagged_df.sort_values("cond_ae_pvalue").reset_index(drop=True),
                         use_container_width=True)
```

**Step 6: Run full test suite**

Run: `uv run pytest tests/ -v -x`
Expected: All PASS.

**Step 7: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat: add conditional AE scores and conformal p-values to dashboard"
```

---

## Summary of changes

| New file | Purpose |
|---|---|
| `src/models/conditional_autoencoder.py` | Metadata-conditioned autoencoder |
| `src/models/conformal.py` | Split conformal p-value calibrator |
| `tests/test_conditional_autoencoder.py` | Tests for conditional AE |
| `tests/test_conformal.py` | Tests for conformal calibrator |

| Modified file | Change |
|---|---|
| `src/data/preprocess.py` | `_make_metadata_dict`, stellar param extraction, `build_metadata_features` |
| `src/run_pipeline.py` | Steps 5 + 5c: cond AE training + conformal calibration |
| `src/dashboard/app.py` | New sort options, metrics, Tab 5 |
| `tests/test_preprocess.py` | New metadata field and features tests |
| `tests/test_integration.py` | Cond AE + conformal steps |

New outputs in `data/results/`:
- `conditional_ae_scores.npy`
- `conditional_ae_pvalues.npy`
- `cond_ae_losses.npy`
- `conformal_threshold.json`
