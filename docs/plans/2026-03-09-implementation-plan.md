# SDSS Spectral Anomaly Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an ML pipeline that downloads SDSS stellar spectra, detects anomalies using PCA+Isolation Forest and a convolutional autoencoder, and provides a Streamlit dashboard to explore results.

**Architecture:** Data flows through download -> preprocess -> two parallel model paths (classical + deep) -> comparison -> dashboard. All managed as a uv Python project with src layout.

**Tech Stack:** Python 3.12+, uv, astroquery, astropy, numpy, pandas, scikit-learn, PyTorch, Streamlit, Plotly, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/features/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/dashboard/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Step 1: Initialize git repo**

```bash
cd /Users/faulknco/Projects/sdss-spectral-anomalies
git init
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "sdss-spectral-anomalies"
version = "0.1.0"
description = "Anomaly detection in SDSS stellar spectra"
requires-python = ">=3.12"
dependencies = [
    "astroquery>=0.4",
    "astropy>=6.0",
    "numpy>=1.26",
    "pandas>=2.1",
    "pyarrow>=14.0",
    "scikit-learn>=1.4",
    "torch>=2.2",
    "streamlit>=1.30",
    "plotly>=5.18",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.2",
]

[tool.ruff]
line-length = 100
```

**Step 3: Create .gitignore**

```
data/raw/
data/processed/
data/results/
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
.firecrawl/
```

**Step 4: Create directory structure**

```bash
mkdir -p src/data src/features src/models src/dashboard tests data/raw data/processed data/results
touch src/__init__.py src/data/__init__.py src/features/__init__.py src/models/__init__.py src/dashboard/__init__.py tests/__init__.py
```

**Step 5: Initialize uv environment and install deps**

```bash
uv venv
uv sync
uv pip install -e ".[dev]"
```

**Step 6: Verify setup**

```bash
uv run python -c "import astroquery; import torch; import streamlit; print('All imports OK')"
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold project with uv, dependencies, and directory structure"
```

---

### Task 2: Data Download Module

**Files:**
- Create: `src/data/download.py`
- Create: `tests/test_download.py`

**Step 1: Write the failing test**

```python
# tests/test_download.py
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.download import build_sdss_query, parse_spectrum_fits


def test_build_sdss_query_returns_sql_string():
    sql = build_sdss_query(limit=100, sn_min=10.0)
    assert "SELECT" in sql.upper()
    assert "specobjall" in sql.lower() or "specobj" in sql.lower()
    assert "100" in sql
    assert "sn_median" in sql.lower() or "snmedian" in sql.lower()


def test_parse_spectrum_fits_extracts_flux_and_wavelength():
    """Test that we correctly extract flux and wavelength from an SDSS FITS HDU structure."""
    mock_data = MagicMock()
    mock_data["loglam"] = np.array([3.58, 3.59, 3.60])
    mock_data["flux"] = np.array([10.0, 12.0, 11.0])

    result = parse_spectrum_fits(mock_data)
    assert "wavelength" in result
    assert "flux" in result
    np.testing.assert_allclose(result["wavelength"], 10 ** np.array([3.58, 3.59, 3.60]))
    np.testing.assert_array_equal(result["flux"], np.array([10.0, 12.0, 11.0]))
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/data/download.py
"""Download stellar spectra from SDSS via astroquery."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astroquery.sdss import SDSS

logger = logging.getLogger(__name__)

SPECTRAL_CLASSES = ["O", "B", "A", "F", "G", "K", "M"]


def build_sdss_query(limit: int = 10000, sn_min: float = 10.0) -> str:
    """Build SQL query for SDSS stellar spectra with S/N filtering."""
    return f"""
    SELECT TOP {limit}
        s.specobjid, s.plate, s.mjd, s.fiberid,
        s.ra, s.dec, s.class, s.subclass,
        s.z, s.zerr, s.snmedian,
        s.elodieTEff, s.elodieLogG, s.elodieFeH
    FROM SpecObj AS s
    WHERE s.class = 'STAR'
        AND s.snmedian > {sn_min}
        AND s.zwarning = 0
    ORDER BY NEWID()
    """


def query_stellar_metadata(limit: int = 10000, sn_min: float = 10.0) -> pd.DataFrame:
    """Query SDSS for stellar spectra metadata."""
    sql = build_sdss_query(limit=limit, sn_min=sn_min)
    logger.info(f"Querying SDSS for {limit} stellar spectra (S/N > {sn_min})...")
    result = SDSS.query_sql(sql)
    if result is None:
        raise RuntimeError("SDSS query returned no results")
    df = result.to_pandas()
    logger.info(f"Retrieved metadata for {len(df)} spectra")
    return df


def parse_spectrum_fits(coadd_data) -> dict:
    """Extract wavelength and flux arrays from SDSS FITS COADD extension data."""
    loglam = np.array(coadd_data["loglam"], dtype=np.float64)
    flux = np.array(coadd_data["flux"], dtype=np.float64)
    return {
        "wavelength": 10**loglam,
        "flux": flux,
    }


def download_spectra(
    metadata: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 50,
) -> list[Path]:
    """Download SDSS spectra FITS files for given metadata rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for start in range(0, len(metadata), batch_size):
        batch = metadata.iloc[start : start + batch_size]
        logger.info(
            f"Downloading batch {start // batch_size + 1} "
            f"({len(batch)} spectra)..."
        )

        for _, row in batch.iterrows():
            plate = int(row["plate"])
            mjd = int(row["mjd"])
            fiberid = int(row["fiberid"])
            fname = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
            fpath = output_dir / fname

            if fpath.exists():
                downloaded.append(fpath)
                continue

            try:
                sp = SDSS.get_spectra(
                    plate=plate, mjd=mjd, fiberID=fiberid
                )
                if sp and len(sp) > 0:
                    sp[0].writeto(fpath, overwrite=True)
                    downloaded.append(fpath)
            except Exception as e:
                logger.warning(f"Failed to download {fname}: {e}")

    logger.info(f"Downloaded {len(downloaded)} spectra to {output_dir}")
    return downloaded
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/download.py tests/test_download.py
git commit -m "feat: add SDSS spectral data download module with SQL query and FITS parsing"
```

---

### Task 3: Preprocessing Module

**Files:**
- Create: `src/data/preprocess.py`
- Create: `tests/test_preprocess.py`

**Step 1: Write the failing test**

```python
# tests/test_preprocess.py
import numpy as np
from src.data.preprocess import (
    resample_spectrum,
    normalize_spectrum,
    preprocess_spectra,
)


def test_resample_spectrum_to_common_grid():
    wavelength = np.linspace(3800, 9200, 500)
    flux = np.sin(wavelength / 1000)
    target_grid = np.linspace(3800, 9200, 3500)

    resampled = resample_spectrum(wavelength, flux, target_grid)
    assert resampled.shape == (3500,)
    assert not np.any(np.isnan(resampled))


def test_normalize_spectrum_divides_by_median():
    flux = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    normalized = normalize_spectrum(flux)
    expected = flux / np.median(flux)
    np.testing.assert_allclose(normalized, expected)


def test_normalize_spectrum_handles_zero_median():
    flux = np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    normalized = normalize_spectrum(flux)
    assert normalized.shape == flux.shape
    assert np.all(np.isfinite(normalized))


def test_preprocess_spectra_returns_correct_shape():
    n_spectra = 5
    n_original = 500
    target_grid = np.linspace(3800, 9200, 3500)

    wavelengths = [np.linspace(3800, 9200, n_original) for _ in range(n_spectra)]
    fluxes = [np.random.randn(n_original) + 10 for _ in range(n_spectra)]

    result = preprocess_spectra(wavelengths, fluxes, target_grid)
    assert result.shape == (n_spectra, 3500)
    assert not np.any(np.isnan(result))
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/data/preprocess.py
"""Preprocess SDSS spectra: resample, normalize, clean."""
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

DEFAULT_GRID = np.linspace(3800, 9200, 3500)


def resample_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    target_grid: np.ndarray,
) -> np.ndarray:
    """Resample a spectrum onto a common wavelength grid via linear interpolation."""
    mask = np.isfinite(flux) & np.isfinite(wavelength)
    if mask.sum() < 10:
        return np.full(len(target_grid), np.nan)

    f = interp1d(
        wavelength[mask],
        flux[mask],
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return f(target_grid)


def normalize_spectrum(flux: np.ndarray) -> np.ndarray:
    """Normalize flux by dividing by the median. Handles zero median."""
    median = np.median(flux)
    if median == 0 or not np.isfinite(median):
        return np.zeros_like(flux)
    return flux / median


def preprocess_spectra(
    wavelengths: list[np.ndarray],
    fluxes: list[np.ndarray],
    target_grid: np.ndarray = DEFAULT_GRID,
) -> np.ndarray:
    """Resample and normalize a list of spectra to a common grid."""
    processed = []
    for wl, fl in zip(wavelengths, fluxes):
        resampled = resample_spectrum(wl, fl, target_grid)
        normalized = normalize_spectrum(resampled)
        processed.append(normalized)
    return np.array(processed)


def load_and_preprocess(
    fits_dir: Path,
    target_grid: np.ndarray = DEFAULT_GRID,
) -> tuple[np.ndarray, list[dict]]:
    """Load FITS files from a directory and return preprocessed spectra + metadata."""
    from astropy.io import fits as astro_fits
    from src.data.download import parse_spectrum_fits

    fits_files = sorted(fits_dir.glob("spec-*.fits"))
    logger.info(f"Loading {len(fits_files)} FITS files from {fits_dir}")

    wavelengths = []
    fluxes = []
    metadata = []

    for fpath in fits_files:
        try:
            with astro_fits.open(fpath) as hdul:
                parsed = parse_spectrum_fits(hdul["COADD"].data)
                wavelengths.append(parsed["wavelength"])
                fluxes.append(parsed["flux"])

                specobj = hdul["SPECOBJ"].data
                metadata.append({
                    "filename": fpath.name,
                    "ra": float(specobj["RA"][0]),
                    "dec": float(specobj["DEC"][0]),
                    "subclass": str(specobj["SUBCLASS"][0]).strip(),
                    "sn_median": float(specobj["SN_MEDIAN_ALL"][0])
                    if "SN_MEDIAN_ALL" in specobj.dtype.names
                    else 0.0,
                })
        except Exception as e:
            logger.warning(f"Failed to load {fpath.name}: {e}")

    spectra = preprocess_spectra(wavelengths, fluxes, target_grid)
    logger.info(f"Preprocessed {spectra.shape[0]} spectra to shape {spectra.shape}")
    return spectra, metadata
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/preprocess.py tests/test_preprocess.py
git commit -m "feat: add spectral preprocessing with resampling and normalization"
```

---

### Task 4: PCA + Isolation Forest Model

**Files:**
- Create: `src/models/classical.py`
- Create: `tests/test_classical.py`

**Step 1: Write the failing test**

```python
# tests/test_classical.py
import numpy as np
from src.models.classical import ClassicalAnomalyDetector


def test_fit_and_score():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (100, 200))
    anomalous = rng.normal(10, 5, (5, 200))
    data = np.vstack([normal, anomalous])

    detector = ClassicalAnomalyDetector(n_components=20, contamination=0.05)
    detector.fit(data)
    scores = detector.score(data)

    assert scores.shape == (105,)
    normal_mean = np.mean(scores[:100])
    anomalous_mean = np.mean(scores[100:])
    assert anomalous_mean > normal_mean


def test_pca_reconstruction_error():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (50, 200))

    detector = ClassicalAnomalyDetector(n_components=20)
    detector.fit(data)
    errors = detector.reconstruction_error(data)

    assert errors.shape == (50,)
    assert np.all(errors >= 0)


def test_get_pca_components():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (50, 200))

    detector = ClassicalAnomalyDetector(n_components=20)
    detector.fit(data)
    components = detector.transform(data)

    assert components.shape == (50, 20)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classical.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/models/classical.py
"""PCA + Isolation Forest anomaly detection pipeline."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class ClassicalAnomalyDetector:
    def __init__(
        self,
        n_components: int = 50,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.iforest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )

    def fit(self, spectra: np.ndarray) -> "ClassicalAnomalyDetector":
        """Fit scaler, PCA, and Isolation Forest on spectra."""
        scaled = self.scaler.fit_transform(spectra)
        components = self.pca.fit_transform(scaled)
        self.iforest.fit(components)
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """Project spectra into PCA space."""
        scaled = self.scaler.transform(spectra)
        return self.pca.transform(scaled)

    def score(self, spectra: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        components = self.transform(spectra)
        return -self.iforest.decision_function(components)

    def reconstruction_error(self, spectra: np.ndarray) -> np.ndarray:
        """Compute PCA reconstruction error per spectrum."""
        scaled = self.scaler.transform(spectra)
        components = self.pca.transform(scaled)
        reconstructed = self.pca.inverse_transform(components)
        return np.mean((scaled - reconstructed) ** 2, axis=1)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_classical.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/classical.py tests/test_classical.py
git commit -m "feat: add PCA + Isolation Forest anomaly detector"
```

---

### Task 5: Convolutional Autoencoder Model

**Files:**
- Create: `src/models/autoencoder.py`
- Create: `tests/test_autoencoder.py`

**Step 1: Write the failing test**

```python
# tests/test_autoencoder.py
import numpy as np
import torch
from src.models.autoencoder import SpectralAutoencoder, train_autoencoder


def test_autoencoder_forward_shape():
    model = SpectralAutoencoder(input_dim=3500, bottleneck_dim=64)
    x = torch.randn(8, 1, 3500)
    reconstructed = model(x)
    assert reconstructed.shape == (8, 1, 3500)


def test_autoencoder_encode_shape():
    model = SpectralAutoencoder(input_dim=3500, bottleneck_dim=64)
    x = torch.randn(8, 1, 3500)
    latent = model.encode(x)
    assert latent.shape[0] == 8
    assert latent.shape[1] == 64


def test_train_autoencoder_reduces_loss():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (64, 3500)).astype(np.float32)

    model, losses = train_autoencoder(
        data, bottleneck_dim=32, epochs=5, batch_size=16, lr=1e-3
    )

    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_reconstruction_error():
    model = SpectralAutoencoder(input_dim=3500, bottleneck_dim=64)
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (16, 3500)).astype(np.float32)

    errors = model.reconstruction_error(data)
    assert errors.shape == (16,)
    assert np.all(errors >= 0)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_autoencoder.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/models/autoencoder.py
"""1D Convolutional Autoencoder for spectral anomaly detection."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SpectralAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 3500, bottleneck_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, bottleneck_dim),
        )

        self._conv_out_dim = (input_dim + 7) // 8

        self.decoder_fc = nn.Linear(bottleneck_dim, 128 * self._conv_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 128, self._conv_out_dim)
        x = self.decoder(x)
        if x.size(2) > self.input_dim:
            x = x[:, :, : self.input_dim]
        elif x.size(2) < self.input_dim:
            pad = self.input_dim - x.size(2)
            x = nn.functional.pad(x, (0, pad))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    @torch.no_grad()
    def reconstruction_error(self, spectra: np.ndarray) -> np.ndarray:
        """Compute per-spectrum MSE reconstruction error."""
        self.train(False)
        x = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        reconstructed = self.forward(x)
        mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
        return mse.numpy()


def train_autoencoder(
    spectra: np.ndarray,
    bottleneck_dim: int = 64,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple["SpectralAutoencoder", list[float]]:
    """Train autoencoder on spectra array of shape (n, wavelength_bins)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpectralAutoencoder(
        input_dim=spectra.shape[1], bottleneck_dim=bottleneck_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)

    model = model.cpu()
    return model, epoch_losses
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_autoencoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/autoencoder.py tests/test_autoencoder.py
git commit -m "feat: add convolutional autoencoder for spectral anomaly detection"
```

---

### Task 6: Model Comparison Module

**Files:**
- Create: `src/models/compare.py`
- Create: `tests/test_compare.py`

**Step 1: Write the failing test**

```python
# tests/test_compare.py
import numpy as np
import pandas as pd
from src.models.compare import compare_anomaly_scores, rank_anomalies


def test_compare_anomaly_scores():
    if_scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
    ae_scores = np.array([0.2, 0.8, 0.4, 0.7, 0.1])

    result = compare_anomaly_scores(if_scores, ae_scores, top_n=2)
    assert "if_rank" in result.columns
    assert "ae_rank" in result.columns
    assert "agreed" in result.columns
    assert len(result) == 5


def test_rank_anomalies():
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    metadata = [{"filename": f"spec-{i}.fits"} for i in range(5)]

    ranked = rank_anomalies(scores, metadata, top_n=3)
    assert len(ranked) == 3
    assert ranked.iloc[0]["anomaly_score"] == 0.9
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_compare.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/models/compare.py
"""Compare anomaly scores from multiple models."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_anomalies(
    scores: np.ndarray,
    metadata: list[dict],
    top_n: int = 100,
) -> pd.DataFrame:
    """Rank spectra by anomaly score, return top N."""
    df = pd.DataFrame(metadata)
    df["anomaly_score"] = scores
    df = df.sort_values("anomaly_score", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def compare_anomaly_scores(
    if_scores: np.ndarray,
    ae_scores: np.ndarray,
    top_n: int = 100,
) -> pd.DataFrame:
    """Compare Isolation Forest and Autoencoder anomaly rankings."""
    if_ranks = rankdata(-if_scores, method="ordinal")
    ae_ranks = rankdata(-ae_scores, method="ordinal")

    df = pd.DataFrame({
        "if_score": if_scores,
        "ae_score": ae_scores,
        "if_rank": if_ranks,
        "ae_rank": ae_ranks,
    })

    df["agreed"] = (df["if_rank"] <= top_n) & (df["ae_rank"] <= top_n)
    df["combined_rank"] = (df["if_rank"] + df["ae_rank"]) / 2

    return df
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_compare.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/compare.py tests/test_compare.py
git commit -m "feat: add model comparison and anomaly ranking"
```

---

### Task 7: Run Pipeline Script

**Files:**
- Create: `src/run_pipeline.py`

**Step 1: Write the pipeline orchestrator**

```python
# src/run_pipeline.py
"""Main pipeline: download, preprocess, train models, compare, save results."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import query_stellar_metadata, download_spectra
from src.data.preprocess import load_and_preprocess, DEFAULT_GRID
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import train_autoencoder
from src.models.compare import compare_anomaly_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def run(n_spectra: int = 5000, sn_min: float = 10.0):
    """Run the full anomaly detection pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    logger.info("=== Step 1: Downloading spectra ===")
    metadata_df = query_stellar_metadata(limit=n_spectra, sn_min=sn_min)
    metadata_df.to_parquet(PROCESSED_DIR / "metadata.parquet")
    download_spectra(metadata_df, RAW_DIR)

    # Step 2: Preprocess
    logger.info("=== Step 2: Preprocessing spectra ===")
    spectra, meta_list = load_and_preprocess(RAW_DIR, DEFAULT_GRID)
    np.save(PROCESSED_DIR / "spectra.npy", spectra)
    pd.DataFrame(meta_list).to_parquet(PROCESSED_DIR / "spectra_metadata.parquet")

    # Step 3: Classical model
    logger.info("=== Step 3: Training PCA + Isolation Forest ===")
    classical = ClassicalAnomalyDetector(n_components=50, contamination=0.05)
    classical.fit(spectra)
    if_scores = classical.score(spectra)
    pca_errors = classical.reconstruction_error(spectra)
    pca_components = classical.transform(spectra)
    np.save(RESULTS_DIR / "if_scores.npy", if_scores)
    np.save(RESULTS_DIR / "pca_errors.npy", pca_errors)
    np.save(RESULTS_DIR / "pca_components.npy", pca_components)

    # Step 4: Autoencoder
    logger.info("=== Step 4: Training Autoencoder ===")
    model, losses = train_autoencoder(spectra, bottleneck_dim=64, epochs=50)
    ae_scores = model.reconstruction_error(spectra)
    np.save(RESULTS_DIR / "ae_scores.npy", ae_scores)
    np.save(RESULTS_DIR / "ae_losses.npy", np.array(losses))

    # Step 5: Compare
    logger.info("=== Step 5: Comparing models ===")
    comparison = compare_anomaly_scores(if_scores, ae_scores, top_n=100)
    comparison_with_meta = pd.concat(
        [comparison, pd.DataFrame(meta_list)], axis=1
    )
    comparison_with_meta.to_parquet(RESULTS_DIR / "comparison.parquet")

    top_agreed = comparison_with_meta[comparison_with_meta["agreed"]].sort_values(
        "combined_rank"
    )
    top_agreed.to_parquet(RESULTS_DIR / "top_anomalies_agreed.parquet")

    logger.info(f"Pipeline complete. {len(top_agreed)} agreed anomalies found.")
    logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
```

**Step 2: Commit**

```bash
git add src/run_pipeline.py
git commit -m "feat: add main pipeline orchestrator"
```

---

### Task 8: Streamlit Dashboard

**Files:**
- Create: `src/dashboard/app.py`

**Step 1: Write the dashboard**

```python
# src/dashboard/app.py
"""Streamlit dashboard for exploring spectral anomalies."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


@st.cache_data
def load_data():
    spectra = np.load(PROCESSED_DIR / "spectra.npy")
    metadata = pd.read_parquet(PROCESSED_DIR / "spectra_metadata.parquet")
    comparison = pd.read_parquet(RESULTS_DIR / "comparison.parquet")
    pca_components = np.load(RESULTS_DIR / "pca_components.npy")
    if_scores = np.load(RESULTS_DIR / "if_scores.npy")
    ae_scores = np.load(RESULTS_DIR / "ae_scores.npy")
    wavelength_grid = np.linspace(3800, 9200, spectra.shape[1])
    return spectra, metadata, comparison, pca_components, if_scores, ae_scores, wavelength_grid


def plot_spectrum(spectra, wavelength_grid, idx, title="Spectrum"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelength_grid, y=spectra[idx],
        mode="lines", name="Observed", line=dict(color="steelblue", width=1),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (Angstroms)",
        yaxis_title="Normalized Flux",
        height=350,
    )
    return fig


def main():
    st.set_page_config(page_title="SDSS Spectral Anomalies", layout="wide")
    st.title("SDSS Spectral Anomaly Explorer")

    spectra, metadata, comparison, pca_components, if_scores, ae_scores, wl = load_data()

    tab1, tab2, tab3 = st.tabs(["Anomaly Browser", "Model Comparison", "PCA Explorer"])

    # --- Tab 1: Anomaly Browser ---
    with tab1:
        st.subheader("Top Anomalies")
        sort_by = st.selectbox("Sort by", ["Combined", "Isolation Forest", "Autoencoder"])
        top_n = st.slider("Show top N", 10, 500, 100)

        if sort_by == "Combined":
            order = np.argsort(-(if_scores + ae_scores))
        elif sort_by == "Isolation Forest":
            order = np.argsort(-if_scores)
        else:
            order = np.argsort(-ae_scores)

        top_indices = order[:top_n]
        table_data = metadata.iloc[top_indices].copy()
        table_data["if_score"] = if_scores[top_indices]
        table_data["ae_score"] = ae_scores[top_indices]
        table_data["index"] = top_indices

        selected = st.dataframe(
            table_data.reset_index(drop=True),
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        if selected and selected.selection and selected.selection.rows:
            row_idx = selected.selection.rows[0]
            spectrum_idx = int(table_data.iloc[row_idx]["index"])
            meta = table_data.iloc[row_idx]
            st.plotly_chart(
                plot_spectrum(spectra, wl, spectrum_idx, f"Spectrum: {meta.get('filename', spectrum_idx)}"),
                use_container_width=True,
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("IF Score", f"{if_scores[spectrum_idx]:.4f}")
            col2.metric("AE Score", f"{ae_scores[spectrum_idx]:.4f}")
            col3.metric("Spectral Type", meta.get("subclass", "N/A"))

    # --- Tab 2: Model Comparison ---
    with tab2:
        st.subheader("Isolation Forest vs Autoencoder Scores")
        scatter_df = pd.DataFrame({
            "IF Score": if_scores,
            "AE Score": ae_scores,
            "Spectral Type": metadata.get("subclass", "unknown"),
        })
        fig = px.scatter(
            scatter_df, x="IF Score", y="AE Score", color="Spectral Type",
            opacity=0.5, height=600,
            title="Model Agreement: IF vs AE Anomaly Scores",
        )
        st.plotly_chart(fig, use_container_width=True)

        n_agreed = int(comparison["agreed"].sum()) if "agreed" in comparison.columns else 0
        st.metric("Spectra flagged by BOTH models (top 100)", n_agreed)

    # --- Tab 3: PCA Explorer ---
    with tab3:
        st.subheader("PCA Latent Space")
        color_by = st.selectbox("Color by", ["IF Score", "AE Score", "Spectral Type"])

        pca_df = pd.DataFrame({
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "PC3": pca_components[:, 2] if pca_components.shape[1] > 2 else 0,
            "IF Score": if_scores,
            "AE Score": ae_scores,
            "Spectral Type": metadata.get("subclass", "unknown"),
        })

        dim_choice = st.radio("Dimensions", ["2D", "3D"], horizontal=True)
        if dim_choice == "2D":
            fig = px.scatter(
                pca_df, x="PC1", y="PC2", color=color_by,
                opacity=0.5, height=600, title="PCA 2D Projection",
            )
        else:
            fig = px.scatter_3d(
                pca_df, x="PC1", y="PC2", z="PC3", color=color_by,
                opacity=0.4, height=700, title="PCA 3D Projection",
            )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
```

**Step 2: Test the dashboard locally**

```bash
uv run streamlit run src/dashboard/app.py
```

Expected: Opens browser at localhost:8501 with three tabs.

**Step 3: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat: add Streamlit dashboard with anomaly browser, model comparison, and PCA explorer"
```

---

### Task 9: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test using synthetic data to verify full pipeline works."""
import numpy as np
from src.data.preprocess import preprocess_spectra
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import train_autoencoder
from src.models.compare import compare_anomaly_scores


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
    assert if_scores.shape == (100,)

    # Autoencoder
    model, losses = train_autoencoder(
        spectra.astype(np.float32),
        bottleneck_dim=16, epochs=3, batch_size=32,
    )
    ae_scores = model.reconstruction_error(spectra.astype(np.float32))
    assert ae_scores.shape == (100,)

    # Compare
    comparison = compare_anomaly_scores(if_scores, ae_scores, top_n=10)
    assert len(comparison) == 100
    assert "agreed" in comparison.columns
```

**Step 2: Run test**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration test with synthetic spectra"
```

---

### Task 10: Run Full Pipeline on Real Data

**Step 1: Run the pipeline**

```bash
uv run python -m src.run_pipeline
```

Takes 10-30 minutes depending on download speed.

**Step 2: Launch the dashboard**

```bash
uv run streamlit run src/dashboard/app.py
```

**Step 3: Verify results exist**

```bash
ls -la data/results/
```

Expected files: `if_scores.npy`, `ae_scores.npy`, `pca_components.npy`, `comparison.parquet`, `top_anomalies_agreed.parquet`

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: verify pipeline runs end-to-end on real SDSS data"
```
