# Conditional Flow, Semi-Synthetic Evaluation, and Line-Window Preprocessing -- Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add density-based conditional anomaly scoring (CVAE then normalizing flow), line-aware preprocessing, and a semi-synthetic evaluation harness to measure retrieval quality across all 7 models.

**Architecture:** Sequential build: line-window preprocessing -> CVAE -> semi-synthetic injection + evaluation -> conditional normalizing flow -> pipeline and dashboard integration. Each component is independently testable. All new models expose `anomaly_score()` returning higher = more anomalous, plugging into the existing `compare_n_models` and `SplitConformalCalibrator`.

**Tech Stack:** Python 3.12+, PyTorch, scikit-learn, NumPy, SciPy, pandas, Streamlit, Plotly, pytest

**Spec:** `docs/superpowers/specs/2026-03-17-conditional-flow-evaluation-design.md`

---

## Chunk 1: Line-Window Preprocessing

### Task 1: Line Catalog and Derivative Spectra

**Files:**
- Create: `src/features/line_windows.py`
- Create: `tests/test_line_windows.py`

- [ ] **Step 1: Write failing tests for line catalog and derivative spectra**

```python
# tests/test_line_windows.py
"""Tests for line-window preprocessing."""
import numpy as np
from src.features.line_windows import (
    SPECTRAL_LINES,
    DISPLAY_LINES,
    compute_derivative_spectra,
)


def test_spectral_lines_has_11_entries():
    assert len(SPECTRAL_LINES) == 11
    for name, wavelength in SPECTRAL_LINES:
        assert isinstance(name, str)
        assert 3800 <= wavelength <= 9200


def test_display_lines_is_subset_of_spectral_lines():
    assert len(DISPLAY_LINES) == 6
    full_names = {name for name, _ in SPECTRAL_LINES}
    for name, _ in DISPLAY_LINES:
        assert name in full_names


def test_derivative_spectra_shape():
    rng = np.random.default_rng(42)
    spectra = rng.normal(1, 0.1, (50, 500))
    wl = np.linspace(3800, 9200, 500)
    deriv = compute_derivative_spectra(spectra, wl)
    assert deriv.shape == (50, 499)  # finite differences lose one bin


def test_derivative_spectra_normalized():
    rng = np.random.default_rng(42)
    spectra = rng.normal(1, 0.1, (50, 500))
    wl = np.linspace(3800, 9200, 500)
    deriv = compute_derivative_spectra(spectra, wl)
    # median absolute value per spectrum should be near 1
    median_abs = np.median(np.abs(deriv), axis=1)
    assert np.all(np.isfinite(deriv))
    # after normalization, median |value| should be approximately 1
    np.testing.assert_allclose(median_abs, 1.0, atol=0.5)


def test_derivative_spectra_all_zero_handled():
    spectra = np.zeros((5, 500))
    wl = np.linspace(3800, 9200, 500)
    deriv = compute_derivative_spectra(spectra, wl)
    assert deriv.shape == (5, 499)
    assert np.all(np.isfinite(deriv))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_line_windows.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write line catalog and derivative spectra implementation**

```python
# src/features/line_windows.py
"""Line-window preprocessing: derivative spectra, line-window features, and shared line catalog."""
import numpy as np

# Full 11-line catalog used for feature extraction and injection targeting.
SPECTRAL_LINES: list[tuple[str, float]] = [
    ("Ca K", 3933.7),
    ("Ca H", 3968.5),
    ("H-gamma", 4340.5),
    ("H-beta", 4861.3),
    ("MgH", 5210.0),
    ("Na D", 5892.0),
    ("H-alpha", 6562.8),
    ("TiO", 7050.0),
    ("Ca II a", 8498.0),
    ("Ca II b", 8542.0),
    ("Ca II c", 8662.0),
]

# 6-line subset for dashboard annotation overlays where visual clarity matters.
# Uses Unicode Greek letters to match the existing dashboard convention.
DISPLAY_LINES: list[tuple[str, float]] = [
    ("Ca K", 3933.7),
    ("Ca H", 3968.5),
    ("\u0048\u03b3", 4340.5),
    ("\u0048\u03b2", 4861.3),
    ("Na D", 5892.0),
    ("\u0048\u03b1", 6562.8),
]


def compute_derivative_spectra(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
) -> np.ndarray:
    """Compute dF/dlambda via finite differences, normalized by median absolute value.

    Args:
        spectra: Array of shape (n_spectra, n_wavelengths).
        wavelength_grid: Array of shape (n_wavelengths,).

    Returns:
        Array of shape (n_spectra, n_wavelengths - 1).
    """
    dlambda = np.diff(wavelength_grid)
    dflux = np.diff(spectra, axis=1)
    derivatives = dflux / dlambda[np.newaxis, :]

    # Normalize each spectrum's derivative by its median absolute value
    med_abs = np.median(np.abs(derivatives), axis=1, keepdims=True)
    med_abs = np.where(med_abs > 0, med_abs, 1.0)
    return derivatives / med_abs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_line_windows.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/features/line_windows.py tests/test_line_windows.py
git commit -m "feat: add line catalog and derivative spectra computation"
```

---

### Task 2: Line-Window Feature Extraction

**Files:**
- Modify: `src/features/line_windows.py`
- Modify: `tests/test_line_windows.py`

- [ ] **Step 1: Write failing tests for line-window features**

Append to `tests/test_line_windows.py`:

```python
from src.features.line_windows import extract_line_features


def test_extract_line_features_shape():
    rng = np.random.default_rng(42)
    spectra = rng.normal(1, 0.1, (50, 500))
    wl = np.linspace(3800, 9200, 500)
    features = extract_line_features(spectra, wl)
    # 11 lines x 4 features = 44 dims
    assert features.shape == (50, 44)
    assert features.dtype == np.float32 or features.dtype == np.float64


def test_extract_line_features_finite():
    rng = np.random.default_rng(42)
    spectra = rng.normal(1, 0.1, (20, 500))
    wl = np.linspace(3800, 9200, 500)
    features = extract_line_features(spectra, wl)
    assert np.all(np.isfinite(features))


def test_extract_line_features_detects_emission():
    """A spectrum with a strong emission line should have different features than a flat spectrum."""
    wl = np.linspace(3800, 9200, 500)
    flat = np.ones((2, 500))
    # Add emission at H-alpha (6562.8 A)
    emission = flat.copy()
    halpha_idx = np.argmin(np.abs(wl - 6562.8))
    emission[1, halpha_idx - 2 : halpha_idx + 3] += 5.0

    features = extract_line_features(emission, wl)
    # H-alpha is the 7th line (index 6), 4 features per line
    halpha_features_flat = features[0, 24:28]
    halpha_features_emission = features[1, 24:28]
    # At least one feature should differ significantly
    assert np.max(np.abs(halpha_features_emission - halpha_features_flat)) > 0.1
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_line_windows.py::test_extract_line_features_shape -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement line-window feature extraction**

Append to `src/features/line_windows.py`:

```python
def _extract_window(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
    center: float,
    half_width: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a wavelength window centered on a line.

    Returns (window_flux, window_wavelengths) for the region
    [center - half_width, center + half_width].
    """
    mask = (wavelength_grid >= center - half_width) & (wavelength_grid <= center + half_width)
    return spectrum[mask], wavelength_grid[mask]


def _window_features(flux: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """Compute 4 features for a single spectral window.

    Features:
        0: equivalent width proxy (sum of 1 - flux/continuum)
        1: local depth (max deviation below continuum)
        2: local asymmetry (skewness of flux in window)
        3: local derivative variance (variance of flux differences)
    """
    if len(flux) < 3:
        return np.zeros(4)

    continuum = 0.5 * (flux[0] + flux[-1])
    if continuum == 0:
        continuum = 1.0

    normalized = flux / continuum

    # Equivalent width proxy
    dlambda = np.median(np.diff(wavelength)) if len(wavelength) > 1 else 1.0
    ew_proxy = float(np.sum(1.0 - normalized) * dlambda)

    # Local depth
    depth = float(np.max(np.abs(1.0 - normalized)))

    # Asymmetry: difference between mean of left half and right half
    mid = len(flux) // 2
    left_mean = float(np.mean(flux[:mid])) if mid > 0 else 0.0
    right_mean = float(np.mean(flux[mid:])) if mid < len(flux) else 0.0
    asymmetry = left_mean - right_mean

    # Derivative variance
    dflux = np.diff(flux)
    deriv_var = float(np.var(dflux)) if len(dflux) > 0 else 0.0

    return np.array([ew_proxy, depth, asymmetry, deriv_var])


def extract_line_features(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    half_width: float = 20.0,
) -> np.ndarray:
    """Extract line-window features for all spectra.

    For each of 11 spectral lines, extracts a ~40 Angstrom window and computes
    4 features per window: equivalent width proxy, local depth, local asymmetry,
    local derivative variance.

    Args:
        spectra: Array of shape (n_spectra, n_wavelengths).
        wavelength_grid: Array of shape (n_wavelengths,).
        half_width: Half-width of extraction window in Angstroms.

    Returns:
        Array of shape (n_spectra, 44) -- 11 lines x 4 features.
    """
    n = len(spectra)
    n_lines = len(SPECTRAL_LINES)
    features = np.zeros((n, n_lines * 4))

    for i in range(n):
        for j, (_, center) in enumerate(SPECTRAL_LINES):
            flux_win, wl_win = _extract_window(spectra[i], wavelength_grid, center, half_width)
            features[i, j * 4 : (j + 1) * 4] = _window_features(flux_win, wl_win)

    return features
```

- [ ] **Step 4: Run all line_windows tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_line_windows.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/features/line_windows.py tests/test_line_windows.py
git commit -m "feat: add line-window feature extraction for 11 spectral lines"
```

---

## Chunk 2: Conditional VAE

### Task 3: CVAE Model

**Files:**
- Create: `src/models/cvae.py`
- Create: `tests/test_cvae.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cvae.py
"""Tests for Conditional VAE."""
import numpy as np
import torch
from src.models.cvae import ConditionalVAE, train_cvae


def make_meta(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_forward_output_shape():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=32)
    x = torch.randn(8, 1, 500)
    meta = torch.randn(8, 4)
    recon, mu, log_var = model(x, meta)
    assert recon.shape == (8, 1, 500)
    assert mu.shape == (8, 32)
    assert log_var.shape == (8, 32)


def test_encode_returns_mu_logvar():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=32)
    x = torch.randn(8, 1, 500)
    meta = torch.randn(8, 4)
    mu, log_var = model.encode(x, meta)
    assert mu.shape == (8, 32)
    assert log_var.shape == (8, 32)


def test_param_count_positive():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=32)
    assert model.param_count() > 0


def test_anomaly_score_shape():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=32)
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((16, 500)).astype(np.float32)
    meta = make_meta(16)
    scores = model.anomaly_score(spectra, meta)
    assert scores.shape == (16,)
    assert np.all(np.isfinite(scores))


def test_anomaly_score_nonnegative():
    """Negative ELBO should be non-negative (reconstruction + KL are both >= 0)."""
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=16)
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((16, 500)).astype(np.float32)
    meta = make_meta(16)
    scores = model.anomaly_score(spectra, meta)
    # KL can be slightly negative numerically; allow small tolerance
    assert np.all(scores >= -0.1)


def test_train_reduces_loss():
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((64, 500)).astype(np.float32)
    meta = make_meta(64)
    model, losses = train_cvae(
        spectra, meta, bottleneck_dim=16, epochs=5, batch_size=16, lr=1e-3,
        beta_warmup_epochs=0,
    )
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_handles_nan_metadata():
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((8, 500)).astype(np.float32)
    meta = make_meta(8)
    meta[0, 0] = np.nan
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=16)
    scores = model.anomaly_score(spectra, meta)
    assert scores.shape == (8,)
    assert np.all(np.isfinite(scores))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_cvae.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement the CVAE**

```python
# src/models/cvae.py
"""Metadata-conditioned Variational Autoencoder for spectral anomaly detection."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 3500,
        meta_dim: int = 4,
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
        )

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, meta_embed_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128 + meta_embed_dim, bottleneck_dim)
        self.fc_log_var = nn.Linear(128 + meta_embed_dim, bottleneck_dim)

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

    def encode(self, x: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv_encoder(x)
        meta_emb = self._embed_meta(meta)
        h = torch.cat([conv_out, meta_emb], dim=1)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

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

    def forward(
        self, x: torch.Tensor, meta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x, meta)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, meta), mu, log_var

    @torch.no_grad()
    def anomaly_score(self, spectra: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        """Return negative ELBO per spectrum (higher = more anomalous)."""
        was_training = self.training
        self.eval()
        x = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        meta = torch.tensor(metadata, dtype=torch.float32)
        recon, mu, log_var = self.forward(x, meta)
        recon_loss = ((x - recon) ** 2).sum(dim=(1, 2))
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
        scores = (recon_loss + kl).numpy()  # negative ELBO with consistent sum reduction
        if was_training:
            self.train()
        return scores


def train_cvae(
    spectra: np.ndarray,
    metadata: np.ndarray,
    bottleneck_dim: int = 64,
    meta_embed_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta_warmup_epochs: int = 10,
) -> tuple["ConditionalVAE", list[float]]:
    """Train the CVAE. metadata shape: (n, meta_dim). NaNs -> 0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
    meta = torch.tensor(metadata, dtype=torch.float32)
    meta[~torch.isfinite(meta)] = 0.0

    loader = DataLoader(TensorDataset(x, meta), batch_size=batch_size, shuffle=True)

    model = ConditionalVAE(
        input_dim=spectra.shape[1],
        meta_dim=metadata.shape[1],
        bottleneck_dim=bottleneck_dim,
        meta_embed_dim=meta_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        beta = min(1.0, epoch / max(beta_warmup_epochs, 1))
        total, n_batches = 0.0, 0
        for x_batch, meta_batch in loader:
            x_batch, meta_batch = x_batch.to(device), meta_batch.to(device)
            recon, mu, log_var = model(x_batch, meta_batch)
            recon_loss = nn.functional.mse_loss(recon, x_batch)
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
            loss = recon_loss + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        epoch_losses.append(total / n_batches)

    return model.cpu(), epoch_losses
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_cvae.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/models/cvae.py tests/test_cvae.py
git commit -m "feat: add conditional VAE with ELBO-based anomaly scoring"
```

---

## Chunk 3: Semi-Synthetic Anomaly Injection and Evaluation

### Task 4: Anomaly Injection

**Files:**
- Create: `src/features/synthetic_anomalies.py`
- Create: `tests/test_synthetic_anomalies.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_synthetic_anomalies.py
"""Tests for semi-synthetic anomaly injection."""
import numpy as np
from src.features.synthetic_anomalies import inject_anomalies, ANOMALY_TYPES


def make_spectra(n=100, n_wl=500):
    rng = np.random.default_rng(42)
    spectra = rng.normal(1.0, 0.05, (n, n_wl))
    wl = np.linspace(3800, 9200, n_wl)
    return spectra, wl


def test_inject_anomalies_output_shapes():
    spectra, wl = make_spectra()
    modified, labels, log = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    assert modified.shape == spectra.shape
    assert labels.shape == (100,)
    assert set(labels).issubset({0, 1})
    assert sum(labels) == 10  # 10% of 100


def test_inject_anomalies_modifies_only_labeled():
    spectra, wl = make_spectra()
    modified, labels, log = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    clean_mask = labels == 0
    np.testing.assert_array_equal(modified[clean_mask], spectra[clean_mask])
    injected_mask = labels == 1
    assert not np.allclose(modified[injected_mask], spectra[injected_mask])


def test_inject_anomalies_log_has_all_types():
    spectra, wl = make_spectra(n=500)
    _, _, log = inject_anomalies(spectra, wl, fraction=0.2, seed=42)
    types_seen = {entry["type"] for entry in log}
    assert types_seen == set(ANOMALY_TYPES)


def test_inject_anomalies_log_structure():
    spectra, wl = make_spectra()
    _, _, log = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    assert len(log) == 10
    for entry in log:
        assert "index" in entry
        assert "type" in entry
        assert "params" in entry
        assert entry["type"] in ANOMALY_TYPES


def test_inject_anomalies_deterministic():
    spectra, wl = make_spectra()
    m1, l1, _ = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    m2, l2, _ = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(l1, l2)


def test_inject_anomalies_zero_fraction():
    spectra, wl = make_spectra()
    modified, labels, log = inject_anomalies(spectra, wl, fraction=0.0, seed=42)
    np.testing.assert_array_equal(modified, spectra)
    assert sum(labels) == 0
    assert len(log) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_synthetic_anomalies.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement anomaly injection**

```python
# src/features/synthetic_anomalies.py
"""Semi-synthetic anomaly injection for evaluation."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from src.features.line_windows import SPECTRAL_LINES

ANOMALY_TYPES = [
    "emission_line",
    "line_broadening",
    "continuum_tilt",
    "wavelength_shift",
    "missing_band",
]


def _inject_emission_line(
    spectrum: np.ndarray, wl: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Add a Gaussian emission peak at a random catalog line."""
    line_idx = rng.integers(len(SPECTRAL_LINES))
    _, center = SPECTRAL_LINES[line_idx]
    fwhm = rng.uniform(3.0, 15.0)
    sigma = fwhm / 2.355
    local_idx = np.argmin(np.abs(wl - center))
    amplitude = rng.uniform(2.0, 10.0) * max(abs(spectrum[local_idx]), 0.01)
    gaussian = amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2)
    result = spectrum + gaussian
    params = {"center": center, "fwhm": fwhm, "amplitude": amplitude}
    return result, params


def _inject_line_broadening(
    spectrum: np.ndarray, wl: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Convolve a ~100 A window around a random line with a broadening kernel."""
    line_idx = rng.integers(len(SPECTRAL_LINES))
    _, center = SPECTRAL_LINES[line_idx]
    sigma_a = rng.uniform(5.0, 20.0)
    dlambda = np.median(np.diff(wl))
    sigma_pix = sigma_a / dlambda
    half_window = 50.0  # Angstroms
    mask = (wl >= center - half_window) & (wl <= center + half_window)
    result = spectrum.copy()
    if mask.sum() > 3:
        result[mask] = gaussian_filter1d(spectrum[mask], sigma=sigma_pix)
    params = {"center": center, "sigma_angstrom": sigma_a}
    return result, params


def _inject_continuum_tilt(
    spectrum: np.ndarray, wl: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Multiply spectrum by a linear ramp."""
    slope = rng.uniform(-0.0005, 0.0005)
    mid = 0.5 * (wl[0] + wl[-1])
    ramp = 1.0 + slope * (wl - mid)
    result = spectrum * ramp
    params = {"slope": slope, "midpoint": mid}
    return result, params


def _inject_wavelength_shift(
    spectrum: np.ndarray, wl: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Shift spectrum by 5-50 Angstroms, reinterpolated onto same grid."""
    shift = rng.uniform(5.0, 50.0) * rng.choice([-1, 1])
    f = interp1d(wl + shift, spectrum, kind="linear", bounds_error=False, fill_value=0.0)
    result = f(wl)
    params = {"shift_angstrom": shift}
    return result, params


def _inject_missing_band(
    spectrum: np.ndarray, wl: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Zero out a contiguous 100-500 Angstrom region."""
    band_width = rng.uniform(100.0, 500.0)
    max_start = wl[-1] - band_width
    start = rng.uniform(wl[0], max(wl[0], max_start))
    end = start + band_width
    mask = (wl >= start) & (wl <= end)
    result = spectrum.copy()
    result[mask] = 0.0
    params = {"start": start, "end": end}
    return result, params


_INJECTORS = {
    "emission_line": _inject_emission_line,
    "line_broadening": _inject_line_broadening,
    "continuum_tilt": _inject_continuum_tilt,
    "wavelength_shift": _inject_wavelength_shift,
    "missing_band": _inject_missing_band,
}


def inject_anomalies(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Inject synthetic anomalies into a fraction of spectra.

    Args:
        spectra: Array of shape (n, L).
        wavelength_grid: Array of shape (L,).
        fraction: Fraction of spectra to modify (0 to 1).
        seed: Random seed for reproducibility.

    Returns:
        modified_spectra: Array of shape (n, L) with injected anomalies.
        labels: Array of shape (n,), 0 = clean, 1 = injected.
        injection_log: List of dicts with {index, type, params}.
    """
    rng = np.random.default_rng(seed)
    n = len(spectra)
    n_inject = int(round(n * fraction))

    modified = spectra.copy()
    labels = np.zeros(n, dtype=int)
    injection_log = []

    if n_inject == 0:
        return modified, labels, injection_log

    inject_indices = rng.choice(n, size=n_inject, replace=False)
    inject_indices.sort()

    for idx in inject_indices:
        anomaly_type = rng.choice(ANOMALY_TYPES)
        injector = _INJECTORS[anomaly_type]
        modified[idx], params = injector(spectra[idx].copy(), wavelength_grid, rng)
        labels[idx] = 1
        injection_log.append({"index": int(idx), "type": anomaly_type, "params": params})

    return modified, labels, injection_log
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_synthetic_anomalies.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/features/synthetic_anomalies.py tests/test_synthetic_anomalies.py
git commit -m "feat: add semi-synthetic anomaly injection with 5 anomaly types"
```

---

### Task 5: Retrieval Evaluation Helper

**Files:**
- Create: `src/features/evaluate_retrieval.py`
- Create: `tests/test_evaluate_retrieval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluate_retrieval.py
"""Tests for retrieval evaluation metrics."""
import numpy as np
from src.features.evaluate_retrieval import evaluate_retrieval


def test_evaluate_retrieval_basic():
    labels = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.8, 0.3, 0.7, 0.05, 0.12, 0.11])
    log = [
        {"index": 3, "type": "emission_line", "params": {}},
        {"index": 4, "type": "continuum_tilt", "params": {}},
        {"index": 6, "type": "missing_band", "params": {}},
    ]
    result = evaluate_retrieval(labels, scores, log, top_k_list=[3, 5])
    assert "precision_at_k" in result
    assert "recall_at_k" in result
    assert "auroc" in result
    assert "auprc" in result
    assert "per_type_recall_at_k" in result
    assert 3 in result["precision_at_k"]
    assert 5 in result["recall_at_k"]


def test_evaluate_retrieval_perfect_scores():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([1.0, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    log = [
        {"index": 0, "type": "emission_line", "params": {}},
        {"index": 1, "type": "emission_line", "params": {}},
        {"index": 2, "type": "emission_line", "params": {}},
    ]
    result = evaluate_retrieval(labels, scores, log, top_k_list=[3])
    assert result["precision_at_k"][3] == 1.0
    assert result["recall_at_k"][3] == 1.0
    assert result["auroc"] > 0.99


def test_evaluate_retrieval_per_type():
    labels = np.zeros(20, dtype=int)
    scores = np.random.default_rng(42).random(20)
    labels[:4] = 1
    scores[:4] = [0.95, 0.90, 0.85, 0.80]
    log = [
        {"index": 0, "type": "emission_line", "params": {}},
        {"index": 1, "type": "emission_line", "params": {}},
        {"index": 2, "type": "continuum_tilt", "params": {}},
        {"index": 3, "type": "continuum_tilt", "params": {}},
    ]
    result = evaluate_retrieval(labels, scores, log, top_k_list=[5])
    assert "emission_line" in result["per_type_recall_at_k"]
    assert "continuum_tilt" in result["per_type_recall_at_k"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_evaluate_retrieval.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement evaluation helper**

```python
# src/features/evaluate_retrieval.py
"""Retrieval evaluation metrics for semi-synthetic anomaly detection."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_retrieval(
    labels: np.ndarray,
    scores: np.ndarray,
    injection_log: list[dict],
    top_k_list: list[int] | None = None,
) -> dict:
    """Compute retrieval metrics for anomaly detection.

    Args:
        labels: Ground truth, 0 = clean, 1 = injected. Shape (n,).
        scores: Anomaly scores (higher = more anomalous). Shape (n,).
        injection_log: List of {index, type, params} from inject_anomalies.
        top_k_list: List of k values for precision/recall at k.

    Returns:
        Dict with keys: precision_at_k, recall_at_k, auroc, auprc, per_type_recall_at_k.
    """
    if top_k_list is None:
        top_k_list = [10, 25, 50, 100]

    n_positive = int(labels.sum())
    ranked = np.argsort(-scores)

    precision_at_k = {}
    recall_at_k = {}
    for k in top_k_list:
        k_capped = min(k, len(labels))
        top_k_labels = labels[ranked[:k_capped]]
        tp = int(top_k_labels.sum())
        precision_at_k[k] = tp / k_capped if k_capped > 0 else 0.0
        recall_at_k[k] = tp / n_positive if n_positive > 0 else 0.0

    auroc = float(roc_auc_score(labels, scores)) if n_positive > 0 and n_positive < len(labels) else 0.0
    auprc = float(average_precision_score(labels, scores)) if n_positive > 0 else 0.0

    # Per-type recall at the largest k
    max_k = max(top_k_list)
    max_k_capped = min(max_k, len(labels))
    top_k_set = set(ranked[:max_k_capped].tolist())
    type_to_indices: dict[str, list[int]] = {}
    for entry in injection_log:
        t = entry["type"]
        type_to_indices.setdefault(t, []).append(entry["index"])

    per_type_recall: dict[str, float] = {}
    for t, indices in type_to_indices.items():
        found = sum(1 for idx in indices if idx in top_k_set)
        per_type_recall[t] = found / len(indices) if indices else 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "auroc": auroc,
        "auprc": auprc,
        "per_type_recall_at_k": per_type_recall,
        "top_k_for_per_type": max_k,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_evaluate_retrieval.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/features/evaluate_retrieval.py tests/test_evaluate_retrieval.py
git commit -m "feat: add retrieval evaluation metrics for semi-synthetic anomalies"
```

---

## Chunk 4: Conditional Normalizing Flow

### Task 6: MADE Block and MAF

**Files:**
- Create: `src/models/conditional_flow.py`
- Create: `tests/test_conditional_flow.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_conditional_flow.py
"""Tests for conditional normalizing flow (MAF)."""
import numpy as np
import torch
from src.models.conditional_flow import ConditionalMAF, train_conditional_flow


def make_pca(n: int, dim: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, dim)).astype(np.float32)


def make_meta(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_log_prob_shape():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    pca = make_pca(16)
    meta = make_meta(16)
    log_probs = model.log_prob(pca, meta)
    assert log_probs.shape == (16,)
    assert np.all(np.isfinite(log_probs))


def test_anomaly_score_shape():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    pca = make_pca(16)
    meta = make_meta(16)
    scores = model.anomaly_score(pca, meta)
    assert scores.shape == (16,)
    assert np.all(np.isfinite(scores))


def test_anomaly_score_is_negative_log_prob():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    pca = make_pca(16)
    meta = make_meta(16)
    lp = model.log_prob(pca, meta)
    scores = model.anomaly_score(pca, meta)
    np.testing.assert_allclose(scores, -lp, atol=1e-5)


def test_param_count_positive():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    assert model.param_count() > 0


def test_train_reduces_loss():
    pca = make_pca(100)
    meta = make_meta(100)
    model, losses = train_conditional_flow(
        pca, meta, n_blocks=4, hidden_dim=64, epochs=5, batch_size=32, lr=1e-3
    )
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_handles_nan_metadata():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    pca = make_pca(8)
    meta = make_meta(8)
    meta[0, 0] = np.nan
    scores = model.anomaly_score(pca, meta)
    assert scores.shape == (8,)
    assert np.all(np.isfinite(scores))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_conditional_flow.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement the conditional MAF**

```python
# src/models/conditional_flow.py
"""Conditional Masked Autoregressive Flow for spectral anomaly detection.

Operates on PCA-compressed spectra conditioned on stellar parameters.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MaskedLinear(nn.Module):
    """Linear layer with a binary mask for autoregressive connectivity."""

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.linear.weight * self.mask, self.linear.bias)


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation, conditioned on context."""

    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int, reverse: bool = False):
        super().__init__()
        self.input_dim = input_dim

        # Assign ordering
        if reverse:
            input_order = np.arange(input_dim - 1, -1, -1)
        else:
            input_order = np.arange(input_dim)
        hidden_order = np.arange(hidden_dim) % input_dim

        # Build masks
        mask1 = torch.tensor(
            (hidden_order[:, None] >= input_order[None, :]).astype(np.float32)
        )
        mask2 = torch.tensor(
            (input_order[:, None] > hidden_order[None, :]).astype(np.float32)
        )

        # Context is fully connected to hidden
        self.context_fc = nn.Linear(context_dim, hidden_dim)
        self.masked1 = MaskedLinear(input_dim, hidden_dim, mask1)
        self.masked2_mu = MaskedLinear(hidden_dim, input_dim, mask2)
        self.masked2_log_s = MaskedLinear(hidden_dim, input_dim, mask2)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.masked1(x) + self.context_fc(context))
        mu = self.masked2_mu(h)
        log_s = self.masked2_log_s(h)
        return mu, log_s


class ConditionalMAF(nn.Module):
    """Masked Autoregressive Flow conditioned on stellar metadata.

    Each block applies an autoregressive affine transform.
    """

    def __init__(
        self,
        input_dim: int = 50,
        context_dim: int = 4,
        n_blocks: int = 8,
        hidden_dim: int = 128,
        meta_embed_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.meta_mlp = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, meta_embed_dim),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            MADE(input_dim, hidden_dim, meta_embed_dim, reverse=(i % 2 == 1))
            for i in range(n_blocks)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(input_dim) for _ in range(n_blocks)
        ])

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _embed_meta(self, meta: torch.Tensor) -> torch.Tensor:
        meta = meta.clone()
        meta[~torch.isfinite(meta)] = 0.0
        return self.meta_mlp(meta)

    def forward(
        self, x: torch.Tensor, meta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform x to base space, return (z, sum_log_det)."""
        context = self._embed_meta(meta)
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        z = x
        for block, bn in zip(self.blocks, self.batch_norms):
            mu, log_s = block(z, context)
            z = (z - mu) * torch.exp(-log_s)
            log_det_sum -= log_s.sum(dim=1)
            z = bn(z)
        return z, log_det_sum

    def _log_prob_tensor(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x, meta)
        log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(dim=1)
        return log_pz + log_det

    @torch.no_grad()
    def log_prob(self, pca_components: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        """Return log-likelihood per spectrum."""
        was_training = self.training
        self.eval()
        x = torch.tensor(pca_components, dtype=torch.float32)
        meta = torch.tensor(metadata, dtype=torch.float32)
        result = self._log_prob_tensor(x, meta).numpy()
        if was_training:
            self.train()
        return result

    @torch.no_grad()
    def anomaly_score(self, pca_components: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        """Return -log_prob per spectrum (higher = more anomalous)."""
        return -self.log_prob(pca_components, metadata)


def train_conditional_flow(
    pca_components: np.ndarray,
    metadata: np.ndarray,
    n_blocks: int = 8,
    hidden_dim: int = 128,
    meta_embed_dim: int = 16,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 10,
    val_fraction: float = 0.1,
) -> tuple["ConditionalMAF", list[float]]:
    """Train the conditional MAF.

    Uses early stopping on a validation split (taken from the training data).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(pca_components)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    x_train = torch.tensor(pca_components[train_idx], dtype=torch.float32)
    meta_train = torch.tensor(metadata[train_idx], dtype=torch.float32)
    meta_train[~torch.isfinite(meta_train)] = 0.0
    x_val = torch.tensor(pca_components[val_idx], dtype=torch.float32)
    meta_val = torch.tensor(metadata[val_idx], dtype=torch.float32)
    meta_val[~torch.isfinite(meta_val)] = 0.0

    loader = DataLoader(
        TensorDataset(x_train, meta_train), batch_size=batch_size, shuffle=True,
        drop_last=True,
    )

    model = ConditionalMAF(
        input_dim=pca_components.shape[1],
        context_dim=metadata.shape[1],
        n_blocks=n_blocks,
        hidden_dim=hidden_dim,
        meta_embed_dim=meta_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for x_batch, meta_batch in loader:
            x_batch, meta_batch = x_batch.to(device), meta_batch.to(device)
            loss = -model._log_prob_tensor(x_batch, meta_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        train_loss = total / n_batches
        epoch_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = -model._log_prob_tensor(
                x_val.to(device), meta_val.to(device)
            ).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), epoch_losses
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_conditional_flow.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/models/conditional_flow.py tests/test_conditional_flow.py
git commit -m "feat: add conditional MAF normalizing flow for density-based anomaly scoring"
```

---

## Chunk 5: Pipeline Integration

### Task 7: Add New Steps to Pipeline

**Files:**
- Modify: `src/run_pipeline.py`

- [ ] **Step 1: Read the current pipeline**

Read `src/run_pipeline.py` to understand exact insertion points.

- [ ] **Step 2: Add imports**

Add after the existing imports at the top of `src/run_pipeline.py`:

```python
from src.features.line_windows import compute_derivative_spectra, extract_line_features
from src.models.cvae import train_cvae
from src.models.conditional_flow import train_conditional_flow
from src.features.synthetic_anomalies import inject_anomalies
from src.features.evaluate_retrieval import evaluate_retrieval
```

- [ ] **Step 3: Add Step 5d (line features) after conformal calibration block, before stability runs**

Insert after the `conformal_threshold.json` write block (after the `json.dump` for `conformal_threshold.json`), before the `# Step 5b: Stability runs` comment:

```python
    # Step 5d: Derivative spectra and line features
    logger.info("=== Step 5d: Computing derivative spectra and line features ===")
    derivative_spectra = compute_derivative_spectra(spectra, DEFAULT_GRID)
    np.save(PROCESSED_DIR / "derivative_spectra.npy", derivative_spectra)
    line_features = extract_line_features(spectra, DEFAULT_GRID)
    np.save(PROCESSED_DIR / "line_features.npy", line_features)
```

- [ ] **Step 4: Add Step 5e (CVAE) after Step 5d**

```python
    # Step 5e: Conditional VAE
    logger.info("=== Step 5e: Training Conditional VAE ===")
    cvae_model, cvae_losses = train_cvae(
        spectra[train_idx].astype(np.float32), meta_features[train_idx],
        bottleneck_dim=64, epochs=50,
    )
    np.save(RESULTS_DIR / "cvae_losses.npy", np.array(cvae_losses))
    cvae_scores = cvae_model.anomaly_score(spectra.astype(np.float32), meta_features)
    np.save(RESULTS_DIR / "cvae_scores.npy", cvae_scores)

    cvae_conformal = SplitConformalCalibrator()
    cvae_conformal.fit(cvae_scores[cal_idx])
    cvae_pvalues = cvae_conformal.pvalues(cvae_scores)
    np.save(RESULTS_DIR / "cvae_pvalues.npy", cvae_pvalues)
```

- [ ] **Step 5: Add Step 5f (conditional flow) after Step 5e**

```python
    # Step 5f: Conditional Normalizing Flow
    logger.info("=== Step 5f: Training Conditional Normalizing Flow ===")
    flow_model, flow_losses = train_conditional_flow(
        pca_components[train_idx], meta_features[train_idx],
        n_blocks=8, hidden_dim=128, epochs=100,
    )
    np.save(RESULTS_DIR / "flow_losses.npy", np.array(flow_losses))
    flow_scores = flow_model.anomaly_score(pca_components, meta_features)
    np.save(RESULTS_DIR / "flow_scores.npy", flow_scores)

    flow_conformal = SplitConformalCalibrator()
    flow_conformal.fit(flow_scores[cal_idx])
    flow_pvalues = flow_conformal.pvalues(flow_scores)
    np.save(RESULTS_DIR / "flow_pvalues.npy", flow_pvalues)
```

- [ ] **Step 6: Update param_counts dict**

Change the `param_counts` dict to include the new models:

```python
    param_counts = {
        "classical_if": classical.param_count(),
        "autoencoder": model.param_count(),
        "ocsvm": ocsvm.param_count(),
        "dagmm": dagmm_model.param_count(),
        "conditional_ae": cond_ae_model.param_count(),
        "cvae": cvae_model.param_count(),
        "conditional_flow": flow_model.param_count(),
    }
```

- [ ] **Step 7: Update compare_n_models call**

Change the `all_scores` dict:

```python
    all_scores = {
        "if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores,
        "dagmm": dagmm_scores, "cond_ae": cond_ae_scores,
        "cvae": cvae_scores, "flow": flow_scores,
    }
```

- [ ] **Step 8: Add Step 8 (semi-synthetic evaluation) at end of run(), before the final logger.info**

```python
    # Step 8: Semi-synthetic evaluation
    logger.info("=== Step 8: Semi-synthetic evaluation ===")
    modified_spectra, eval_labels, injection_log = inject_anomalies(
        spectra, DEFAULT_GRID, fraction=0.1, seed=42,
    )

    eval_scoring = {
        "if": classical.score(modified_spectra),
        "ae": model.reconstruction_error(modified_spectra.astype(np.float32)),
        "ocsvm": ocsvm.score(modified_spectra),
        "dagmm": dagmm_model.anomaly_score(modified_spectra.astype(np.float32)),
        "cond_ae": cond_ae_model.reconstruction_error(modified_spectra.astype(np.float32), meta_features),
        "cvae": cvae_model.anomaly_score(modified_spectra.astype(np.float32), meta_features),
        "flow": flow_model.anomaly_score(classical.transform(modified_spectra), meta_features),
    }

    eval_results = {}
    for model_name, model_scores in eval_scoring.items():
        eval_results[model_name] = evaluate_retrieval(
            eval_labels, model_scores, injection_log,
            top_k_list=[10, 25, 50, min(100, len(spectra))],
        )

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=float)
```

- [ ] **Step 9: Run existing tests to verify no regressions**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/ -v --ignore=tests/test_integration.py -x`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add src/run_pipeline.py
git commit -m "feat: wire CVAE, flow, line features, and evaluation into pipeline"
```

---

## Chunk 6: Dashboard Integration

### Task 8: Update Dashboard

**Files:**
- Modify: `src/dashboard/app.py`

- [ ] **Step 1: Read current dashboard**

Read `src/dashboard/app.py` to understand exact structure.

- [ ] **Step 2: Replace hardcoded SPECTRAL_LINES import**

Change the import section. Replace:

```python
from src.features.color import spectrum_to_rgb, rgb_to_hex
```

With:

```python
from src.features.color import spectrum_to_rgb, rgb_to_hex
from src.features.line_windows import DISPLAY_LINES
```

Remove the hardcoded `SPECTRAL_LINES = [...]` block (lines 25-32 of current file).

Then replace all uses of `SPECTRAL_LINES` in the file with `DISPLAY_LINES`.

- [ ] **Step 3: Add new score arrays to load_data()**

After the existing `focused_review` loading block, add:

```python
    cvae_scores_path = RESULTS_DIR / "cvae_scores.npy"
    cvae_pvalues_path = RESULTS_DIR / "cvae_pvalues.npy"
    flow_scores_path = RESULTS_DIR / "flow_scores.npy"
    flow_pvalues_path = RESULTS_DIR / "flow_pvalues.npy"
    eval_results_path = RESULTS_DIR / "evaluation_results.json"

    cvae_scores = np.load(cvae_scores_path) if cvae_scores_path.exists() else np.zeros(len(if_scores))
    cvae_pvalues = np.load(cvae_pvalues_path) if cvae_pvalues_path.exists() else np.ones(len(if_scores))
    flow_scores = np.load(flow_scores_path) if flow_scores_path.exists() else np.zeros(len(if_scores))
    flow_pvalues = np.load(flow_pvalues_path) if flow_pvalues_path.exists() else np.ones(len(if_scores))
    eval_results = None
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            eval_results = json.load(f)
```

Add `cvae_scores, cvae_pvalues, flow_scores, flow_pvalues, eval_results` to the return tuple.

- [ ] **Step 4: Update main() unpack, tabs, sort, scatter matrix, and Focused Review**

Update the unpack at the top of `main()` to include `cvae_scores, cvae_pvalues, flow_scores, flow_pvalues, eval_results`.

Change the `st.tabs(...)` call to add `"Evaluation"` as tab7:

```python
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Anomaly Browser", "Model Comparison", "PCA Explorer",
        "Model Stability", "Conformal P-Values", "Focused Review", "Evaluation",
    ])
```

Add `"CVAE"` and `"Conditional Flow"` to the sort_by selectbox options.

Update the Combined sort to include all 7 models:

```python
        if sort_by == "Combined":
            order = np.argsort(-(if_scores + ae_scores + ocsvm_scores + dagmm_scores + cond_ae_scores + cvae_scores + flow_scores))
```

Add elif branches for the new sort options:

```python
        elif sort_by == "CVAE":
            order = np.argsort(-cvae_scores)
        elif sort_by == "Conditional Flow":
            order = np.argsort(-flow_scores)
```

Add `cvae_score` and `flow_score` columns to `table_data`.

In the Anomaly Browser detail view (after the existing Cond AE / Conformal p-value metrics), add:

```python
            col7, col8 = st.columns(2)
            col7.metric("CVAE Score", f"{cvae_scores[spectrum_idx]:.4f}")
            col8.metric("Flow Score", f"{flow_scores[spectrum_idx]:.4f}")
```

Update Tab 2 scatter matrix to include CVAE and Flow:

```python
        scatter_df = pd.DataFrame({
            "IF Score": if_scores,
            "AE Score": ae_scores,
            "OC-SVM Score": ocsvm_scores,
            "DAGMM Score": dagmm_scores,
            "Conditional AE Score": cond_ae_scores,
            "CVAE Score": cvae_scores,
            "Flow Score": flow_scores,
        })
        fig = px.scatter_matrix(
            scatter_df,
            dimensions=list(scatter_df.columns),
            opacity=0.3, height=900,
            title="Pairwise Model Score Comparisons",
        )
```

Update Tab 6 Focused Review `rank_df` to include CVAE and Flow (if columns exist):

```python
            models = [("IF", "if"), ("AE", "ae"), ("OC-SVM", "ocsvm"),
                      ("DAGMM", "dagmm"), ("Cond AE", "cond_ae"),
                      ("CVAE", "cvae"), ("Flow", "flow")]
            rank_rows = []
            for label, key in models:
                rank_col = f"{key}_rank"
                score_col = f"{key}_score"
                if rank_col in candidate.index and score_col in candidate.index:
                    rank_rows.append({"model": label, "rank": int(candidate[rank_col]), "score": candidate[score_col]})
            rank_df = pd.DataFrame(rank_rows).sort_values("rank")
```

Update the agreement fraction to use dynamic model count:

```python
            n_total_models = len([c for c in candidate.index if c.endswith("_rank")])
            ...
            f"{int(candidate['n_models_agreed'])}/{n_total_models}",
```

- [ ] **Step 5: Implement Tab 7 (Evaluation)**

```python
    # --- Tab 7: Evaluation ---
    with tab7:
        st.subheader("Semi-Synthetic Evaluation Results")
        if eval_results is None:
            st.info("No evaluation results found. Run the pipeline to generate them.")
        else:
            model_names = list(eval_results.keys())

            # AUROC / AUPRC comparison
            st.subheader("Overall Metrics")
            metrics_df = pd.DataFrame({
                "Model": model_names,
                "AUROC": [eval_results[m]["auroc"] for m in model_names],
                "AUPRC": [eval_results[m]["auprc"] for m in model_names],
            }).sort_values("AUROC", ascending=False)
            st.dataframe(metrics_df.reset_index(drop=True), use_container_width=True)

            # Precision at k bar chart
            st.subheader("Precision @ k")
            k_values = sorted(eval_results[model_names[0]]["precision_at_k"].keys(), key=lambda x: int(x))
            prec_data = []
            for m in model_names:
                for k in k_values:
                    prec_data.append({"Model": m, "k": int(k), "Precision": eval_results[m]["precision_at_k"][str(k)]})
            prec_df = pd.DataFrame(prec_data)
            fig = px.bar(prec_df, x="k", y="Precision", color="Model", barmode="group", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Per-type recall heatmap
            st.subheader("Per Anomaly Type Recall")
            type_data = {}
            for m in model_names:
                ptr = eval_results[m].get("per_type_recall_at_k", {})
                for t, val in ptr.items():
                    type_data.setdefault(t, {})[m] = val
            if type_data:
                type_df = pd.DataFrame(type_data).T
                type_df.index.name = "Anomaly Type"
                fig = px.imshow(
                    type_df.values,
                    x=list(type_df.columns),
                    y=list(type_df.index),
                    color_continuous_scale="Blues",
                    text_auto=".2f",
                    height=400,
                    title="Recall by Anomaly Type and Model",
                )
                st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 6: Run all tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/ -v --ignore=tests/test_integration.py -x`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat: add CVAE, flow scores, and evaluation tab to dashboard"
```

---

## Chunk 7: Integration Test and Final Verification

### Task 9: Update Integration Test

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add imports**

Add to the top of `tests/test_integration.py`:

```python
from src.features.line_windows import compute_derivative_spectra, extract_line_features
from src.models.cvae import train_cvae
from src.models.conditional_flow import train_conditional_flow
from src.features.synthetic_anomalies import inject_anomalies
from src.features.evaluate_retrieval import evaluate_retrieval
```

- [ ] **Step 2: Add CVAE, flow, and evaluation blocks**

Append to the end of `test_full_pipeline_synthetic`, after the existing conformal block:

```python
    # Line-window preprocessing
    derivative = compute_derivative_spectra(spectra, target_grid)
    assert derivative.shape == (n_spectra, n_wavelengths - 1)
    line_feats = extract_line_features(spectra, target_grid)
    assert line_feats.shape == (n_spectra, 44)

    # CVAE
    cvae_model, cvae_losses = train_cvae(
        spectra[train_idx].astype(np.float32),
        meta_features[train_idx],
        bottleneck_dim=16, epochs=3, batch_size=16,
    )
    assert len(cvae_losses) == 3
    assert cvae_model.param_count() > 0
    cvae_scores = cvae_model.anomaly_score(spectra.astype(np.float32), meta_features)
    assert cvae_scores.shape == (n_spectra,)
    assert np.all(np.isfinite(cvae_scores))

    cvae_conformal = SplitConformalCalibrator()
    cvae_conformal.fit(cvae_scores[cal_idx])
    cvae_pvals = cvae_conformal.pvalues(cvae_scores)
    assert cvae_pvals.shape == (n_spectra,)
    assert np.all(cvae_pvals >= 0) and np.all(cvae_pvals <= 1)

    # Conditional flow
    pca_components = classical.transform(spectra)
    flow_model, flow_losses = train_conditional_flow(
        pca_components[train_idx].astype(np.float32),
        meta_features[train_idx],
        n_blocks=4, hidden_dim=64, epochs=5, batch_size=32,
    )
    assert len(flow_losses) == 5
    assert flow_model.param_count() > 0
    flow_scores = flow_model.anomaly_score(
        pca_components.astype(np.float32), meta_features
    )
    assert flow_scores.shape == (n_spectra,)
    assert np.all(np.isfinite(flow_scores))

    # N-model comparison with 7 models
    all_scores_7 = {
        "if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores,
        "dagmm": dagmm_scores, "cond_ae": cond_ae_scores,
        "cvae": cvae_scores, "flow": flow_scores,
    }
    comp_7 = compare_n_models(all_scores_7, top_n=10)
    assert "n_models_agreed" in comp_7.columns
    assert len(comp_7) == n_spectra

    # Semi-synthetic evaluation (smoke test: verifies interface, not detection quality,
    # since if_scores are from original spectra not modified ones)
    modified, labels, log = inject_anomalies(spectra, target_grid, fraction=0.1, seed=42)
    assert modified.shape == spectra.shape
    assert sum(labels) == 10

    eval_result = evaluate_retrieval(labels, if_scores, log, top_k_list=[5, 10])
    assert "auroc" in eval_result
    assert "precision_at_k" in eval_result

    # Verify flow can score PCA-projected modified spectra (critical Step 8 path)
    modified_pca = classical.transform(modified)
    flow_modified_scores = flow_model.anomaly_score(
        modified_pca.astype(np.float32), meta_features
    )
    assert flow_modified_scores.shape == (n_spectra,)
    assert np.all(np.isfinite(flow_modified_scores))
```

- [ ] **Step 3: Run integration test**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add CVAE, flow, line features, and evaluation to integration test"
```
