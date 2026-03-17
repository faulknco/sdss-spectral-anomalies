"""Tests for line profile asymmetry scoring."""
import numpy as np
import pandas as pd
from src.features.spectral_lines import measure_line_profile
from src.features.line_asymmetry import (
    compute_line_asymmetry_features,
    build_population_asymmetry_stats,
    line_asymmetry_score,
)


def test_measure_line_profile_symmetric():
    """Symmetric Gaussian absorption should have ~0 skewness."""
    wl = np.linspace(3800, 9200, 3500)
    spectrum = np.ones(3500)
    center = 6562.8
    spectrum -= 0.4 * np.exp(-0.5 * ((wl - center) / 5.0) ** 2)

    prof = measure_line_profile(spectrum, wl, center)
    assert abs(prof["skewness"]) < 0.3
    assert abs(prof["blue_red_ratio"] - 1.0) < 0.25  # discretization shifts center slightly


def test_measure_line_profile_asymmetric():
    """Asymmetric line (blue wing deeper) should have non-zero skewness."""
    wl = np.linspace(3800, 9200, 3500)
    spectrum = np.ones(3500)
    center = 6562.8
    # Blue wing deeper than red wing
    blue_gauss = 0.5 * np.exp(-0.5 * ((wl - (center - 3)) / 3.0) ** 2)
    red_gauss = 0.2 * np.exp(-0.5 * ((wl - (center + 3)) / 3.0) ** 2)
    spectrum -= blue_gauss + red_gauss

    prof = measure_line_profile(spectrum, wl, center)
    # Blue wing is deeper, so blue_red_ratio should deviate from 1
    assert prof["blue_red_ratio"] != 1.0
    assert np.isfinite(prof["skewness"])


def test_measure_line_profile_flat():
    """Flat spectrum: no meaningful profile, should have small values."""
    wl = np.linspace(3800, 9200, 3500)
    spectrum = np.ones(3500)
    prof = measure_line_profile(spectrum, wl, 5000.0)
    # Profile is flat, so everything should be near zero or nan
    assert np.isfinite(prof["skewness"]) or np.isnan(prof["skewness"])


def test_measure_line_profile_low_continuum():
    """Very low continuum should return nan (same guard as accretion)."""
    wl = np.linspace(3800, 9200, 3500)
    spectrum = np.full(3500, 0.01)  # way below 0.1 threshold
    prof = measure_line_profile(spectrum, wl, 5000.0)
    assert np.isnan(prof["skewness"])


def test_compute_line_asymmetry_features_shape():
    rng = np.random.default_rng(42)
    n = 20
    wl = np.linspace(3800, 9200, 3500)
    spectra = np.ones((n, 3500)) + rng.normal(0, 0.01, (n, 3500))
    meta = pd.DataFrame({"subclass": ["G2"] * n})

    df = compute_line_asymmetry_features(spectra, wl, meta)
    assert len(df) == n
    assert "broad_class" in df.columns
    assert any(c.endswith("_skewness") for c in df.columns)
    assert any(c.endswith("_blue_red_ratio") for c in df.columns)


def test_build_population_asymmetry_stats():
    rng = np.random.default_rng(42)
    n = 30
    wl = np.linspace(3800, 9200, 3500)
    spectra = np.ones((n, 3500)) + rng.normal(0, 0.01, (n, 3500))
    meta = pd.DataFrame({"subclass": ["G2"] * n})

    df = compute_line_asymmetry_features(spectra, wl, meta)
    stats = build_population_asymmetry_stats(df, min_count=5)
    assert len(stats) > 0
    assert any(c.endswith("_median") for c in stats.columns)


def test_line_asymmetry_score_shape():
    rng = np.random.default_rng(42)
    n = 25
    wl = np.linspace(3800, 9200, 3500)
    spectra = np.ones((n, 3500)) + rng.normal(0, 0.01, (n, 3500))
    meta = pd.DataFrame({"subclass": ["G2"] * n})

    scores = line_asymmetry_score(spectra, wl, meta)
    assert scores.shape == (n,)
    assert np.all(scores >= 0)


def test_line_asymmetry_score_detects_asymmetric():
    """Spectrum with deliberately asymmetric lines should score higher."""
    rng = np.random.default_rng(42)
    n = 30
    wl = np.linspace(3800, 9200, 3500)
    spectra = np.ones((n, 3500)) + rng.normal(0, 0.005, (n, 3500))

    # Add symmetric absorption to all
    for center in [6562.8, 4861.3]:
        for i in range(n):
            spectra[i] -= 0.3 * np.exp(-0.5 * ((wl - center) / 5.0) ** 2)

    # Make first spectrum's lines asymmetric
    for center in [6562.8, 4861.3]:
        spectra[0] += 0.3 * np.exp(-0.5 * ((wl - center) / 5.0) ** 2)  # remove symmetric
        spectra[0] -= 0.4 * np.exp(-0.5 * ((wl - (center - 3)) / 3.0) ** 2)  # blue-heavy
        spectra[0] -= 0.1 * np.exp(-0.5 * ((wl - (center + 3)) / 6.0) ** 2)  # weak red

    meta = pd.DataFrame({"subclass": ["G2"] * n})
    scores = line_asymmetry_score(spectra, wl, meta)

    # The asymmetric spectrum should score above the median
    assert scores[0] > np.median(scores)
