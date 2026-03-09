# Paper-Inspired Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OC-SVM, DAGMM, multi-seed stability, parameter counting, anomaly categorization, and dashboard updates inspired by Mejri et al. (2024).

**Architecture:** Four models (IF, AE, OC-SVM, DAGMM) with multi-seed stability runs, anomaly type categorization, and an updated N-model comparison framework. Dashboard gains stability tab, parameter counts, and category columns.

**Tech Stack:** Python, PyTorch, scikit-learn, NumPy, pandas, Streamlit, Plotly

---

### Task 1: OC-SVM Baseline Model

**Files:**
- Create: `src/models/ocsvm.py`
- Test: `tests/test_ocsvm.py`

**Step 1: Write the failing test**

```python
# tests/test_ocsvm.py
"""Tests for OC-SVM anomaly detector."""
import numpy as np
from src.models.ocsvm import OCSVMDetector


def test_ocsvm_fit_score():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (100, 500))
    detector = OCSVMDetector(n_components=20)
    detector.fit(spectra)
    scores = detector.score(spectra)
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))


def test_ocsvm_anomalies_score_higher():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (90, 500))
    anomalous = rng.normal(0, 1, (10, 500)) + 10
    spectra = np.vstack([normal, anomalous])
    detector = OCSVMDetector(n_components=20)
    detector.fit(spectra)
    scores = detector.score(spectra)
    assert np.mean(scores[90:]) > np.mean(scores[:90])


def test_ocsvm_param_count():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 500))
    detector = OCSVMDetector(n_components=20)
    detector.fit(spectra)
    count = detector.param_count()
    assert isinstance(count, int)
    assert count > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_ocsvm.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/models/ocsvm.py
"""One-Class SVM anomaly detector with PCA preprocessing."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class OCSVMDetector:
    def __init__(
        self,
        n_components: int = 50,
        kernel: str = "rbf",
        nu: float = 0.05,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.svm = OneClassSVM(kernel=kernel, nu=nu)

    def fit(self, spectra: np.ndarray) -> "OCSVMDetector":
        scaled = self.scaler.fit_transform(spectra)
        components = self.pca.fit_transform(scaled)
        self.svm.fit(components)
        return self

    def score(self, spectra: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(spectra)
        components = self.pca.transform(scaled)
        return -self.svm.decision_function(components)

    def param_count(self) -> int:
        n_sv = self.svm.support_vectors_.shape[0]
        n_features = self.svm.support_vectors_.shape[1]
        pca_params = self.pca.components_.size + self.pca.mean_.size
        return n_sv * n_features + pca_params
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_ocsvm.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/models/ocsvm.py tests/test_ocsvm.py
git commit -m "feat: add OC-SVM baseline anomaly detector"
```

---

### Task 2: DAGMM Model

**Files:**
- Create: `src/models/dagmm.py`
- Test: `tests/test_dagmm.py`

**Step 1: Write the failing test**

```python
# tests/test_dagmm.py
"""Tests for DAGMM anomaly detector."""
import numpy as np
from src.models.dagmm import DAGMM, train_dagmm


def test_dagmm_forward_shape():
    import torch
    model = DAGMM(input_dim=500, latent_dim=16, n_gmm=4)
    x = torch.randn(32, 500)
    x_hat, z, gamma = model(x)
    assert x_hat.shape == (32, 500)
    assert gamma.shape[0] == 32
    assert gamma.shape[1] == 4


def test_dagmm_train_and_score():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (100, 500)).astype(np.float32)
    model = train_dagmm(spectra, latent_dim=8, n_gmm=3, epochs=3, batch_size=32)
    scores = model.anomaly_score(spectra)
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))


def test_dagmm_anomalies_score_higher():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (90, 500)).astype(np.float32)
    anomalous = (rng.normal(0, 1, (10, 500)) + 10).astype(np.float32)
    spectra = np.vstack([normal, anomalous])
    model = train_dagmm(spectra, latent_dim=8, n_gmm=3, epochs=10, batch_size=32)
    scores = model.anomaly_score(spectra)
    assert np.mean(scores[90:]) > np.mean(scores[:90])


def test_dagmm_param_count():
    model = DAGMM(input_dim=500, latent_dim=16, n_gmm=4)
    count = model.param_count()
    assert isinstance(count, int)
    assert count > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_dagmm.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/models/dagmm.py
"""Deep Autoencoding Gaussian Mixture Model for anomaly detection."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DAGMM(nn.Module):
    def __init__(self, input_dim: int = 3500, latent_dim: int = 16, n_gmm: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_gmm = n_gmm

        # Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

        # Estimation network: latent_dim + 2 (recon error features) -> n_gmm
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, n_gmm),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)

        # Reconstruction error features
        recon_mse = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)
        recon_cos = 1 - nn.functional.cosine_similarity(x, x_hat, dim=1).unsqueeze(1)
        z = torch.cat([z_c, recon_mse, recon_cos], dim=1)

        gamma = self.estimation(z)
        return x_hat, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = z.size(0)
        gamma_sum = gamma.sum(dim=0)  # (K,)
        phi = gamma_sum / N  # mixture weights

        # Weighted means
        mu = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(dim=0) / gamma_sum.unsqueeze(1)

        # Weighted covariances
        z_centered = z.unsqueeze(1) - mu.unsqueeze(0)  # (N, K, D)
        cov = torch.zeros(self.n_gmm, z.size(1), z.size(1), device=z.device)
        for k in range(self.n_gmm):
            diff = z_centered[:, k]  # (N, D)
            weighted = gamma[:, k].unsqueeze(1) * diff  # (N, D)
            cov[k] = (weighted.T @ diff) / gamma_sum[k]
            cov[k] += 1e-6 * torch.eye(z.size(1), device=z.device)

        return phi, mu, cov

    def compute_energy(self, z, phi, mu, cov):
        K, D = mu.shape
        energy = torch.zeros(z.size(0), device=z.device)

        for k in range(K):
            diff = z - mu[k]  # (N, D)
            cov_inv = torch.linalg.inv(cov[k])
            cov_det = torch.linalg.det(cov[k]).clamp(min=1e-12)
            exp_term = -0.5 * (diff @ cov_inv * diff).sum(dim=1)
            norm_const = torch.sqrt((2 * torch.pi) ** D * cov_det)
            energy += phi[k] * torch.exp(exp_term) / norm_const

        return -torch.log(energy.clamp(min=1e-12))

    @torch.no_grad()
    def anomaly_score(self, spectra: np.ndarray) -> np.ndarray:
        self.train(False)
        x = torch.tensor(spectra, dtype=torch.float32)
        _, z, gamma = self.forward(x)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        energy = self.compute_energy(z, phi, mu, cov)
        return energy.numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_dagmm(
    spectra: np.ndarray,
    latent_dim: int = 16,
    n_gmm: int = 4,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_energy: float = 0.1,
    lambda_cov: float = 0.005,
) -> DAGMM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(spectra, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DAGMM(input_dim=spectra.shape[1], latent_dim=latent_dim, n_gmm=n_gmm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, z, gamma = model(batch)

            recon_loss = ((batch - x_hat) ** 2).mean()
            phi, mu, cov = model.compute_gmm_params(z, gamma)
            energy = model.compute_energy(z, phi, mu, cov).mean()
            cov_diag = sum(1.0 / cov[k].diag().sum() for k in range(model.n_gmm))

            loss = recon_loss + lambda_energy * energy + lambda_cov * cov_diag
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model = model.cpu()
    return model
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_dagmm.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/models/dagmm.py tests/test_dagmm.py
git commit -m "feat: add DAGMM anomaly detector"
```

---

### Task 3: Parameter Count for Existing Models

**Files:**
- Modify: `src/models/classical.py` (add `param_count` method)
- Modify: `src/models/autoencoder.py` (add `param_count` method)
- Test: `tests/test_param_count.py`

**Step 1: Write the failing test**

```python
# tests/test_param_count.py
"""Tests for parameter counting across all models."""
import numpy as np
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import SpectralAutoencoder


def test_classical_param_count():
    detector = ClassicalAnomalyDetector(n_components=20)
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 500))
    detector.fit(spectra)
    count = detector.param_count()
    assert isinstance(count, int)
    assert count > 0


def test_autoencoder_param_count():
    model = SpectralAutoencoder(input_dim=500, bottleneck_dim=16)
    count = model.param_count()
    assert isinstance(count, int)
    assert count > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_param_count.py -v`
Expected: FAIL with AttributeError (no param_count method)

**Step 3: Write minimal implementation**

Add to `src/models/classical.py` (after `reconstruction_error` method):

```python
    def param_count(self) -> int:
        pca_params = self.pca.components_.size + self.pca.mean_.size
        n_estimators = self.iforest.n_estimators
        n_features = self.pca.n_components
        forest_params = n_estimators * n_features * 10  # approximate tree params
        return pca_params + forest_params
```

Add to `src/models/autoencoder.py` (in `SpectralAutoencoder` class):

```python
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_param_count.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/models/classical.py src/models/autoencoder.py tests/test_param_count.py
git commit -m "feat: add parameter counting to classical and autoencoder models"
```

---

### Task 4: Multi-Seed Stability Runner

**Files:**
- Create: `src/models/stability.py`
- Test: `tests/test_stability.py`

**Step 1: Write the failing test**

```python
# tests/test_stability.py
"""Tests for multi-seed stability runner."""
import numpy as np
from src.models.stability import stability_run


def _mock_train_fn(spectra, seed):
    rng = np.random.default_rng(seed)
    return rng.random(len(spectra))


def test_stability_run_shape():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 100))
    result = stability_run(_mock_train_fn, spectra, n_runs=5)
    assert result["mean_scores"].shape == (50,)
    assert result["std_scores"].shape == (50,)
    assert result["all_scores"].shape == (5, 50)


def test_stability_run_std_nonzero():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 100))
    result = stability_run(_mock_train_fn, spectra, n_runs=5)
    assert np.any(result["std_scores"] > 0)


def test_stability_run_single_run():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 100))
    result = stability_run(_mock_train_fn, spectra, n_runs=1)
    assert result["all_scores"].shape == (1, 50)
    assert np.allclose(result["std_scores"], 0)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_stability.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/models/stability.py
"""Multi-seed stability runner for anomaly detection models."""
import numpy as np
from typing import Callable


def stability_run(
    train_and_score_fn: Callable[[np.ndarray, int], np.ndarray],
    spectra: np.ndarray,
    n_runs: int = 5,
    base_seed: int = 42,
) -> dict:
    """Run a model N times with different seeds, return stability metrics.

    Args:
        train_and_score_fn: Callable(spectra, seed) -> scores array of shape (n,)
        spectra: Input spectra array of shape (n, features)
        n_runs: Number of independent runs
        base_seed: Starting seed (incremented per run)

    Returns:
        Dict with keys: mean_scores, std_scores, all_scores
    """
    all_scores = []
    for i in range(n_runs):
        seed = base_seed + i
        scores = train_and_score_fn(spectra, seed)
        all_scores.append(scores)

    all_scores = np.array(all_scores)  # (n_runs, n_spectra)
    return {
        "mean_scores": all_scores.mean(axis=0),
        "std_scores": all_scores.std(axis=0),
        "all_scores": all_scores,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_stability.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/models/stability.py tests/test_stability.py
git commit -m "feat: add multi-seed stability runner"
```

---

### Task 5: Anomaly Type Categorization

**Files:**
- Create: `src/features/categorize.py`
- Test: `tests/test_categorize.py`

**Step 1: Write the failing test**

```python
# tests/test_categorize.py
"""Tests for anomaly type categorization."""
import numpy as np
from src.features.categorize import categorize_anomalies


def test_categorize_returns_labels():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 500))
    pca_errors = rng.random(50)
    categories = categorize_anomalies(spectra, pca_errors)
    assert len(categories) == 50
    valid = {"continuum", "line", "noise", "normal"}
    assert all(c in valid for c in categories)


def test_categorize_noise_detection():
    rng = np.random.default_rng(42)
    # High-variance spectrum should be categorized as noise
    spectra = np.zeros((10, 500))
    spectra[0] = rng.normal(0, 10, 500)  # very noisy
    spectra[1:] = rng.normal(0, 0.1, (9, 500))  # quiet
    pca_errors = np.zeros(10)
    pca_errors[0] = 0.5
    categories = categorize_anomalies(spectra, pca_errors)
    assert categories[0] == "noise"


def test_categorize_continuum_detection():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 0.1, (10, 500))
    pca_errors = np.zeros(10)
    pca_errors[0] = 100.0  # extreme PCA error -> continuum anomaly
    categories = categorize_anomalies(spectra, pca_errors)
    assert categories[0] == "continuum"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_categorize.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/features/categorize.py
"""Categorize anomalies by type based on spectral properties."""
import numpy as np


def categorize_anomalies(
    spectra: np.ndarray,
    pca_errors: np.ndarray,
    noise_threshold_percentile: float = 90,
    continuum_threshold_percentile: float = 90,
    line_threshold_percentile: float = 90,
) -> list[str]:
    """Classify each spectrum into an anomaly category.

    Categories:
        - noise: high overall variance across wavelength
        - continuum: unusual overall shape (high PCA reconstruction error)
        - line: unusual local features (high local variance in residuals)
        - normal: none of the above

    Args:
        spectra: Array of shape (n, wavelengths)
        pca_errors: PCA reconstruction errors of shape (n,)

    Returns:
        List of category strings, one per spectrum.
    """
    n = len(spectra)

    # Noise metric: variance across wavelength per spectrum
    spectral_variance = np.var(spectra, axis=1)
    noise_thresh = np.percentile(spectral_variance, noise_threshold_percentile)

    # Continuum metric: PCA reconstruction error
    continuum_thresh = np.percentile(pca_errors, continuum_threshold_percentile)

    # Line metric: local variance (std of differences between adjacent bins)
    diffs = np.diff(spectra, axis=1)
    local_variance = np.var(diffs, axis=1)
    line_thresh = np.percentile(local_variance, line_threshold_percentile)

    categories = []
    for i in range(n):
        if spectral_variance[i] > noise_thresh:
            categories.append("noise")
        elif pca_errors[i] > continuum_thresh:
            categories.append("continuum")
        elif local_variance[i] > line_thresh:
            categories.append("line")
        else:
            categories.append("normal")

    return categories
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_categorize.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/features/__init__.py src/features/categorize.py tests/test_categorize.py
git commit -m "feat: add anomaly type categorization"
```

---

### Task 6: Update compare.py for N Models

**Files:**
- Modify: `src/models/compare.py`
- Test: `tests/test_compare_n_models.py`

**Step 1: Write the failing test**

```python
# tests/test_compare_n_models.py
"""Tests for N-model comparison."""
import numpy as np
from src.models.compare import compare_n_models


def test_compare_n_models_basic():
    rng = np.random.default_rng(42)
    scores = {
        "if": rng.random(100),
        "ae": rng.random(100),
        "ocsvm": rng.random(100),
        "dagmm": rng.random(100),
    }
    result = compare_n_models(scores, top_n=10)
    assert len(result) == 100
    for name in scores:
        assert f"{name}_score" in result.columns
        assert f"{name}_rank" in result.columns
    assert "n_models_agreed" in result.columns
    assert "combined_rank" in result.columns


def test_compare_n_models_agreement():
    # Make all models agree on first 5 being top anomalies
    scores = {}
    for name in ["a", "b", "c"]:
        s = np.zeros(50)
        s[:5] = 10.0  # top 5 are clearly anomalous
        scores[name] = s
    result = compare_n_models(scores, top_n=10)
    assert result["n_models_agreed"].iloc[0] == 3
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_compare_n_models.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `src/models/compare.py`:

```python
def compare_n_models(
    scores_dict: dict[str, np.ndarray],
    top_n: int = 100,
) -> pd.DataFrame:
    """Compare anomaly rankings across N models.

    Args:
        scores_dict: Dict mapping model name -> scores array
        top_n: Number of top anomalies to consider for agreement

    Returns:
        DataFrame with per-model scores, ranks, agreement count, combined rank
    """
    df = pd.DataFrame()
    rank_cols = []

    for name, scores in scores_dict.items():
        df[f"{name}_score"] = scores
        ranks = rankdata(-scores, method="ordinal")
        df[f"{name}_rank"] = ranks
        rank_cols.append(f"{name}_rank")

    # Count how many models put each spectrum in top_n
    df["n_models_agreed"] = sum(
        (df[col] <= top_n).astype(int) for col in rank_cols
    )

    # Combined rank: average of all model ranks
    df["combined_rank"] = df[rank_cols].mean(axis=1)

    return df
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_compare_n_models.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/models/compare.py tests/test_compare_n_models.py
git commit -m "feat: add N-model comparison framework"
```

---

### Task 7: Update Pipeline

**Files:**
- Modify: `src/run_pipeline.py`

**Step 1: Read the current pipeline**

Read `src/run_pipeline.py` to understand the current structure.

**Step 2: Update the pipeline to include all 4 models, stability, categorization**

Update `src/run_pipeline.py` to:
1. Import new models (OCSVMDetector, train_dagmm) and new modules (stability_run, categorize_anomalies, compare_n_models)
2. Add OC-SVM step after classical model
3. Add DAGMM step after autoencoder
4. Add stability runs for IF and AE (OC-SVM and DAGMM are deterministic-ish, skip for speed)
5. Add anomaly categorization step
6. Use compare_n_models instead of compare_anomaly_scores
7. Save all new results (ocsvm_scores, dagmm_scores, stability, categories, param_counts)

Key additions to `run()`:

```python
from src.models.ocsvm import OCSVMDetector
from src.models.dagmm import train_dagmm
from src.models.stability import stability_run
from src.features.categorize import categorize_anomalies
from src.models.compare import compare_n_models

# After Step 3 (Classical model):
# Step 3b: OC-SVM
logger.info("=== Step 3b: Training OC-SVM ===")
ocsvm = OCSVMDetector(n_components=50)
ocsvm.fit(spectra)
ocsvm_scores = ocsvm.score(spectra)
np.save(RESULTS_DIR / "ocsvm_scores.npy", ocsvm_scores)

# After Step 4 (Autoencoder):
# Step 4b: DAGMM
logger.info("=== Step 4b: Training DAGMM ===")
dagmm_model = train_dagmm(spectra.astype(np.float32), latent_dim=16, n_gmm=4, epochs=50)
dagmm_scores = dagmm_model.anomaly_score(spectra.astype(np.float32))
np.save(RESULTS_DIR / "dagmm_scores.npy", dagmm_scores)

# Step 5: Stability
logger.info("=== Step 5: Multi-seed stability runs ===")
def if_train_fn(spectra, seed):
    det = ClassicalAnomalyDetector(n_components=50, contamination=0.05, random_state=seed)
    det.fit(spectra)
    return det.score(spectra)

if_stability = stability_run(if_train_fn, spectra, n_runs=5)
np.save(RESULTS_DIR / "if_stability_mean.npy", if_stability["mean_scores"])
np.save(RESULTS_DIR / "if_stability_std.npy", if_stability["std_scores"])

# Step 6: Anomaly categorization
logger.info("=== Step 6: Categorizing anomalies ===")
categories = categorize_anomalies(spectra, pca_errors)
np.save(RESULTS_DIR / "categories.npy", np.array(categories))

# Step 7: Parameter counts
param_counts = {
    "classical_if": classical.param_count(),
    "autoencoder": model.param_count(),
    "ocsvm": ocsvm.param_count(),
    "dagmm": dagmm_model.param_count(),
}
import json
with open(RESULTS_DIR / "param_counts.json", "w") as f:
    json.dump(param_counts, f, indent=2)

# Step 8: Compare all models
logger.info("=== Step 8: Comparing all models ===")
all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores}
comparison = compare_n_models(all_scores, top_n=100)
```

**Step 3: Commit**

```bash
git add src/run_pipeline.py
git commit -m "feat: update pipeline with all 4 models, stability, and categorization"
```

---

### Task 8: Update Dashboard

**Files:**
- Modify: `src/dashboard/app.py`

**Step 1: Read the current dashboard**

Read `src/dashboard/app.py` to understand the current structure.

**Step 2: Update the dashboard**

Updates to make:
1. `load_data()`: Load new score arrays (ocsvm_scores, dagmm_scores), stability data, categories, param_counts.json
2. Sidebar: Display parameter counts for all 4 models
3. Tab 1 (Anomaly Browser): Add sort options for OC-SVM/DAGMM, add category column to table
4. Tab 2 (Model Comparison): Update to scatter matrix of all 4 models
5. New Tab 4 (Model Stability): Error bars chart, std heatmap

Key changes to `load_data()`:

```python
@st.cache_data
def load_data():
    spectra = np.load(PROCESSED_DIR / "spectra.npy")
    metadata = pd.read_parquet(PROCESSED_DIR / "spectra_metadata.parquet")
    comparison = pd.read_parquet(RESULTS_DIR / "comparison.parquet")
    pca_components = np.load(RESULTS_DIR / "pca_components.npy")
    if_scores = np.load(RESULTS_DIR / "if_scores.npy")
    ae_scores = np.load(RESULTS_DIR / "ae_scores.npy")
    wavelength_grid = np.linspace(3800, 9200, spectra.shape[1])

    # New data
    ocsvm_scores = np.load(RESULTS_DIR / "ocsvm_scores.npy")
    dagmm_scores = np.load(RESULTS_DIR / "dagmm_scores.npy")
    if_stability_std = np.load(RESULTS_DIR / "if_stability_std.npy")
    categories = list(np.load(RESULTS_DIR / "categories.npy"))
    import json
    with open(RESULTS_DIR / "param_counts.json") as f:
        param_counts = json.load(f)

    return (spectra, metadata, comparison, pca_components,
            if_scores, ae_scores, ocsvm_scores, dagmm_scores,
            if_stability_std, categories, param_counts, wavelength_grid)
```

New Tab 4 (Model Stability):

```python
with tab4:
    st.subheader("Model Stability (IF multi-seed)")
    fig = go.Figure()
    sorted_idx = np.argsort(-if_scores)[:200]
    fig.add_trace(go.Bar(
        x=list(range(len(sorted_idx))),
        y=if_scores[sorted_idx],
        error_y=dict(type="data", array=if_stability_std[sorted_idx]),
        name="IF Score +/- std",
    ))
    fig.update_layout(title="Top 200 Anomalies: IF Score with Stability Error Bars",
                      xaxis_title="Rank", yaxis_title="IF Score", height=500)
    st.plotly_chart(fig, use_container_width=True)
```

Sidebar param counts:

```python
st.sidebar.header("Model Parameters")
for model_name, count in param_counts.items():
    st.sidebar.metric(model_name, f"{count:,}")
```

**Step 3: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat: update dashboard with 4 models, stability tab, and categories"
```

---

### Task 9: Update Integration Test

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Update integration test to cover all 4 models and new features**

```python
# tests/test_integration.py
"""Integration test using synthetic data to verify full pipeline works."""
import numpy as np
from src.data.preprocess import preprocess_spectra
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import train_autoencoder
from src.models.ocsvm import OCSVMDetector
from src.models.dagmm import train_dagmm
from src.models.compare import compare_anomaly_scores, compare_n_models
from src.models.stability import stability_run
from src.features.categorize import categorize_anomalies


def test_full_pipeline_synthetic():
    """Run the full pipeline on synthetic spectra."""
    rng = np.random.default_rng(42)
    n_spectra = 100
    n_wavelengths = 500

    wavelengths = [np.linspace(3800, 9200, n_wavelengths) for _ in range(n_spectra)]
    normal_flux = [rng.normal(10, 1, n_wavelengths) for _ in range(90)]
    anomalous_flux = [
        rng.normal(10, 1, n_wavelengths) + 5 * np.sin(np.linspace(0, 10, n_wavelengths))
        for _ in range(10)
    ]
    fluxes = normal_flux + anomalous_flux

    target_grid = np.linspace(3800, 9200, n_wavelengths)

    # Preprocess
    spectra = preprocess_spectra(wavelengths, fluxes, target_grid)
    assert spectra.shape == (100, n_wavelengths)

    # Classical model
    classical = ClassicalAnomalyDetector(n_components=20, contamination=0.1)
    classical.fit(spectra)
    if_scores = classical.score(spectra)
    pca_errors = classical.reconstruction_error(spectra)
    assert if_scores.shape == (100,)

    # Autoencoder
    model, losses = train_autoencoder(
        spectra.astype(np.float32),
        bottleneck_dim=16, epochs=3, batch_size=32,
    )
    ae_scores = model.reconstruction_error(spectra.astype(np.float32))
    assert ae_scores.shape == (100,)

    # OC-SVM
    ocsvm = OCSVMDetector(n_components=20)
    ocsvm.fit(spectra)
    ocsvm_scores = ocsvm.score(spectra)
    assert ocsvm_scores.shape == (100,)

    # DAGMM
    dagmm_model = train_dagmm(
        spectra.astype(np.float32),
        latent_dim=8, n_gmm=3, epochs=3, batch_size=32,
    )
    dagmm_scores = dagmm_model.anomaly_score(spectra.astype(np.float32))
    assert dagmm_scores.shape == (100,)

    # Compare (original 2-model)
    comparison = compare_anomaly_scores(if_scores, ae_scores, top_n=10)
    assert len(comparison) == 100
    assert "agreed" in comparison.columns

    # Compare (N-model)
    all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores}
    n_comparison = compare_n_models(all_scores, top_n=10)
    assert "n_models_agreed" in n_comparison.columns
    assert "combined_rank" in n_comparison.columns

    # Param counts
    assert classical.param_count() > 0
    assert model.param_count() > 0
    assert ocsvm.param_count() > 0
    assert dagmm_model.param_count() > 0

    # Stability
    def mock_train(s, seed):
        rng2 = np.random.default_rng(seed)
        return rng2.random(len(s))

    stability = stability_run(mock_train, spectra, n_runs=3)
    assert stability["mean_scores"].shape == (100,)

    # Categorization
    categories = categorize_anomalies(spectra, pca_errors)
    assert len(categories) == 100
    valid = {"continuum", "line", "noise", "normal"}
    assert all(c in valid for c in categories)
```

**Step 2: Run test to verify it passes**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: update integration test to cover all 4 models and new features"
```
