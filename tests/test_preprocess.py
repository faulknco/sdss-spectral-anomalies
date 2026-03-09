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


def test_preprocess_spectra_returns_correct_shape():
    n_spectra = 5
    n_original = 500
    target_grid = np.linspace(3800, 9200, 3500)

    wavelengths = [np.linspace(3800, 9200, n_original) for _ in range(n_spectra)]
    fluxes = [np.random.randn(n_original) + 10 for _ in range(n_spectra)]

    result = preprocess_spectra(wavelengths, fluxes, target_grid)
    assert result.shape == (n_spectra, 3500)
    assert not np.any(np.isnan(result))


def test_load_and_preprocess_empty_dir_returns_empty(tmp_path):
    """load_and_preprocess on an empty directory returns empty arrays."""
    from src.data.preprocess import load_and_preprocess, DEFAULT_GRID
    spectra, metadata = load_and_preprocess(tmp_path, DEFAULT_GRID)
    assert spectra.shape[0] == 0
    assert metadata == []


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
