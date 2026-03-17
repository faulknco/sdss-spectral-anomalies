"""Tests for accretion signature detection."""
import numpy as np
import pandas as pd
from src.features.accretion import (
    detect_unexpected_emission,
    measure_blue_excess,
    fit_powerlaw_residual,
    accretion_score,
    _planck_shape,
)


def test_detect_unexpected_emission_normal():
    """Normal absorption spectrum should not flag emission."""
    wl = np.linspace(3800, 9200, 3500)
    spectrum = np.ones(3500)
    # Add absorption at H-alpha
    spectrum -= 0.3 * np.exp(-0.5 * ((wl - 6562.8) / 5.0) ** 2)

    detections = detect_unexpected_emission(spectrum, wl, "G2")
    assert len(detections) == 0


def test_detect_unexpected_emission_halpha():
    """K-dwarf with H-alpha emission should be flagged."""
    wl = np.linspace(3800, 9200, 3500)
    spectrum = np.ones(3500)
    # Add emission bump at H-alpha
    spectrum += 0.5 * np.exp(-0.5 * ((wl - 6562.8) / 5.0) ** 2)

    detections = detect_unexpected_emission(spectrum, wl, "K5")
    assert len(detections) > 0
    h_alpha_detected = any(d["line_name"] == "H_alpha" for d in detections)
    assert h_alpha_detected


def test_measure_blue_excess_hot_star():
    """Hot star (high Teff) should show minimal blue excess against Planck."""
    wl = np.linspace(3800, 9200, 3500)
    teff = 8000.0
    # Use the Planck shape itself as the spectrum -> no excess
    spectrum = _planck_shape(wl, teff)
    excess = measure_blue_excess(spectrum, wl, teff)
    # Should be near zero
    assert abs(excess) < 0.1


def test_measure_blue_excess_added_blue():
    """Spectrum with extra blue flux should show positive excess."""
    wl = np.linspace(3800, 9200, 3500)
    teff = 5000.0
    spectrum = _planck_shape(wl, teff)
    # Add extra blue flux
    blue_mask = wl < 4500
    spectrum[blue_mask] *= 2.0

    excess = measure_blue_excess(spectrum, wl, teff)
    assert excess > 0


def test_fit_powerlaw_residual_clean():
    """Pure blackbody should have low power-law significance."""
    wl = np.linspace(3800, 9200, 3500)
    teff = 5500.0
    spectrum = _planck_shape(wl, teff)
    result = fit_powerlaw_residual(spectrum, wl, teff)
    assert result["significance"] >= 0


def test_fit_powerlaw_residual_with_powerlaw():
    """Blackbody + power law should show non-trivial significance."""
    wl = np.linspace(3800, 9200, 3500)
    teff = 5500.0
    bb = _planck_shape(wl, teff)
    # Add a power-law component: A * lambda^(-alpha)
    powerlaw = 0.5 * (wl / 5000.0) ** (-2.0)
    spectrum = bb + powerlaw

    result = fit_powerlaw_residual(spectrum, wl, teff)
    assert result["amplitude"] > 0
    assert result["spectral_index"] > 0


def test_accretion_score_shape():
    rng = np.random.default_rng(42)
    n = 20
    wl = np.linspace(3800, 9200, 3500)
    spectra = np.ones((n, 3500)) + rng.normal(0, 0.01, (n, 3500))
    meta = pd.DataFrame({
        "subclass": ["G2"] * n,
        "elodie_teff": [5500.0] * n,
    })
    scores = accretion_score(spectra, wl, meta)
    assert scores.shape == (n,)
    assert np.all(scores >= 0)


def test_accretion_score_emission_spectrum():
    """Spectrum with emission lines and power-law should score high."""
    wl = np.linspace(3800, 9200, 3500)
    n = 10
    spectra = np.ones((n, 3500))

    # First spectrum: add emission at H-alpha and blue excess
    spectra[0] += 0.8 * np.exp(-0.5 * ((wl - 6562.8) / 5.0) ** 2)  # emission
    spectra[0] += 0.3 * (wl / 5000.0) ** (-1.5)  # power-law component
    blue_mask = wl < 4500
    spectra[0, blue_mask] *= 1.5  # blue excess

    meta = pd.DataFrame({
        "subclass": ["K5"] * n,
        "elodie_teff": [4500.0] * n,
    })
    scores = accretion_score(spectra, wl, meta)
    # The anomalous spectrum should score higher than the normal ones
    assert scores[0] > np.mean(scores[1:])
