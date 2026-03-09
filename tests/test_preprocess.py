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
