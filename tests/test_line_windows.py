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
    assert deriv.shape == (50, 499)


def test_derivative_spectra_normalized():
    rng = np.random.default_rng(42)
    spectra = rng.normal(1, 0.1, (50, 500))
    wl = np.linspace(3800, 9200, 500)
    deriv = compute_derivative_spectra(spectra, wl)
    median_abs = np.median(np.abs(deriv), axis=1)
    assert np.all(np.isfinite(deriv))
    np.testing.assert_allclose(median_abs, 1.0, atol=0.5)


def test_derivative_spectra_all_zero_handled():
    spectra = np.zeros((5, 500))
    wl = np.linspace(3800, 9200, 500)
    deriv = compute_derivative_spectra(spectra, wl)
    assert deriv.shape == (5, 499)
    assert np.all(np.isfinite(deriv))
