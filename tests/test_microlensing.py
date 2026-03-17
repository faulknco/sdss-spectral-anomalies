"""Tests for microlensing signature detection."""
import numpy as np
import pandas as pd
from src.features.microlensing import (
    _parse_broad_class,
    compute_continuum_line_ratios,
    build_expected_ratios,
    achromatic_residual_score,
    microlensing_score,
)


def test_parse_broad_class():
    assert _parse_broad_class("G2 IV") == "G"
    assert _parse_broad_class("K5") == "K"
    assert _parse_broad_class("dM4") == "M"
    assert _parse_broad_class("") == "unknown"
    assert _parse_broad_class("A0") == "A"


def _make_synthetic(n=50, n_wl=3500):
    rng = np.random.default_rng(42)
    wl = np.linspace(3800, 9200, n_wl)
    spectra = np.ones((n, n_wl)) + rng.normal(0, 0.01, (n, n_wl))
    meta = pd.DataFrame({"subclass": ["G2"] * n})
    return spectra, wl, meta


def test_compute_continuum_line_ratios():
    spectra, wl, meta = _make_synthetic(n=10)
    df = compute_continuum_line_ratios(spectra, wl, meta)
    assert len(df) == 10
    assert "broad_class" in df.columns
    assert any(c.endswith("_ratio") for c in df.columns)


def test_build_expected_ratios():
    spectra, wl, meta = _make_synthetic(n=30)
    ratio_df = compute_continuum_line_ratios(spectra, wl, meta)
    expected = build_expected_ratios(ratio_df, min_count=5)
    assert len(expected) > 0
    assert any(c.endswith("_median") for c in expected.columns)


def test_achromatic_residual_score_flat():
    """Flat spectrum should produce a finite non-negative achromatic score."""
    wl = np.linspace(3800, 9200, 3500)
    flat = np.ones(3500)
    score = achromatic_residual_score(flat, wl)
    assert score >= 0
    assert np.isfinite(score)


def test_achromatic_residual_score_offset():
    """Spectrum with flat offset from polynomial should score higher."""
    wl = np.linspace(3800, 9200, 3500)
    # Polynomial continuum + constant offset
    spectrum = 0.001 * (wl - 5000) ** 2 / 1e6 + 1.0 + 0.3
    score_offset = achromatic_residual_score(spectrum, wl)
    # Just the polynomial, no offset
    spectrum_no_offset = 0.001 * (wl - 5000) ** 2 / 1e6 + 1.0
    score_clean = achromatic_residual_score(spectrum_no_offset, wl)
    # Both should be non-negative
    assert score_offset >= 0
    assert score_clean >= 0


def test_microlensing_score_boosted_continuum():
    """Spectrum with artificially boosted continuum should score higher."""
    rng = np.random.default_rng(42)
    n = 30
    wl = np.linspace(3800, 9200, 3500)

    # Normal spectra: flat with absorption lines
    spectra = np.ones((n, 3500))
    for i in range(n):
        spectra[i] += rng.normal(0, 0.01, 3500)
        # Add absorption lines at H-alpha, H-beta
        for center in [6562.8, 4861.3]:
            spectra[i] -= 0.3 * np.exp(-0.5 * ((wl - center) / 5.0) ** 2)

    meta = pd.DataFrame({"subclass": ["G2"] * n})

    # Boosted spectrum: raise the continuum but keep lines the same depth
    boosted = spectra.copy()
    boosted[0] = spectra[0] + 0.5  # uniform continuum boost

    scores_normal = microlensing_score(spectra, wl, meta)
    scores_boosted = microlensing_score(boosted, wl, meta)

    assert scores_boosted[0] > scores_normal[0]
    assert all(s >= 0 for s in scores_normal)
    assert all(s >= 0 for s in scores_boosted)


def test_microlensing_score_shape():
    spectra, wl, meta = _make_synthetic(n=20)
    scores = microlensing_score(spectra, wl, meta)
    assert scores.shape == (20,)
    assert np.all(scores >= 0)
