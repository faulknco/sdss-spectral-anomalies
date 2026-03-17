"""Tests for Gaia DR3 cross-match module.

These tests use synthetic data only — no actual Gaia TAP queries are made.
"""
import numpy as np
import pandas as pd
from src.features.gaia_crossmatch import (
    _build_gaia_cone_query,
    _build_gaia_batch_query,
    score_astrometric_anomalies,
    _empty_result,
    query_gaia_for_candidates,
)


def test_build_gaia_cone_query():
    q = _build_gaia_cone_query(180.0, 45.0, 1.5)
    assert "gaiadr3.gaia_source" in q
    assert "180.0" in q
    assert "45.0" in q


def test_build_gaia_batch_query():
    ra = np.array([180.0, 181.0])
    dec = np.array([45.0, 46.0])
    q = _build_gaia_batch_query(ra, dec, 1.5)
    assert "gaiadr3.gaia_source" in q
    assert "BETWEEN" in q


def test_score_astrometric_anomalies_empty():
    df = _empty_result()
    scored = score_astrometric_anomalies(df)
    assert "ruwe_flag" in scored.columns
    assert "excess_noise_flag" in scored.columns
    assert "astrometric_anomaly_score" in scored.columns
    assert len(scored) == 0


def test_score_astrometric_anomalies_normal():
    """Normal single star: RUWE ~1.0, low excess noise."""
    df = pd.DataFrame([{
        "source_id": 12345,
        "ra": 180.0,
        "dec": 45.0,
        "parallax": 5.0,
        "parallax_error": 0.05,
        "pmra": 10.0,
        "pmdec": -5.0,
        "phot_g_mean_mag": 12.0,
        "astrometric_excess_noise": 0.1,
        "astrometric_excess_noise_sig": 0.5,
        "ruwe": 1.0,
        "ipd_gof_harmonic_amplitude": 0.01,
        "visibility_periods_used": 15,
        "astrometric_params_solved": 31,
        "sdss_idx": 0,
        "separation_arcsec": 0.3,
    }])
    scored = score_astrometric_anomalies(df)
    assert not scored.iloc[0]["ruwe_flag"]
    assert not scored.iloc[0]["excess_noise_flag"]
    assert scored.iloc[0]["astrometric_anomaly_score"] < 1.0


def test_score_astrometric_anomalies_companion():
    """Star with unresolved companion: high RUWE and excess noise."""
    df = pd.DataFrame([{
        "source_id": 67890,
        "ra": 200.0,
        "dec": 30.0,
        "parallax": 2.0,
        "parallax_error": 0.1,
        "pmra": 5.0,
        "pmdec": -2.0,
        "phot_g_mean_mag": 14.0,
        "astrometric_excess_noise": 2.5,
        "astrometric_excess_noise_sig": 8.0,
        "ruwe": 3.5,
        "ipd_gof_harmonic_amplitude": 0.15,
        "visibility_periods_used": 12,
        "astrometric_params_solved": 31,
        "sdss_idx": 5,
        "separation_arcsec": 0.2,
    }])
    scored = score_astrometric_anomalies(df)
    assert scored.iloc[0]["ruwe_flag"]
    assert scored.iloc[0]["excess_noise_flag"]
    assert scored.iloc[0]["astrometric_anomaly_score"] > 2.0


def test_score_astrometric_anomalies_nan_handling():
    """NaN values should not crash or flag."""
    df = pd.DataFrame([{
        "source_id": 11111,
        "ra": 150.0,
        "dec": 20.0,
        "parallax": np.nan,
        "parallax_error": np.nan,
        "pmra": np.nan,
        "pmdec": np.nan,
        "phot_g_mean_mag": np.nan,
        "astrometric_excess_noise": np.nan,
        "astrometric_excess_noise_sig": np.nan,
        "ruwe": np.nan,
        "ipd_gof_harmonic_amplitude": np.nan,
        "visibility_periods_used": 5,
        "astrometric_params_solved": 31,
        "sdss_idx": 2,
        "separation_arcsec": 0.5,
    }])
    scored = score_astrometric_anomalies(df)
    assert not scored.iloc[0]["ruwe_flag"]
    assert not scored.iloc[0]["excess_noise_flag"]
    assert scored.iloc[0]["astrometric_anomaly_score"] == 0.0


def test_query_gaia_no_coords():
    """Metadata without RA/Dec should return empty."""
    meta = pd.DataFrame({"plate": [1, 2]})
    # Should fail gracefully since ra/dec columns are missing
    try:
        result = query_gaia_for_candidates(meta, candidate_indices=[0])
    except KeyError:
        # Expected if ra/dec columns missing
        pass


def test_score_multiple_sources():
    """Score multiple sources, verify relative ordering."""
    df = pd.DataFrame([
        {
            "source_id": 1, "ra": 180.0, "dec": 45.0,
            "parallax": 5.0, "parallax_error": 0.05,
            "pmra": 10.0, "pmdec": -5.0, "phot_g_mean_mag": 12.0,
            "astrometric_excess_noise": 0.1,
            "astrometric_excess_noise_sig": 0.5,
            "ruwe": 1.0, "ipd_gof_harmonic_amplitude": 0.01,
            "visibility_periods_used": 15, "astrometric_params_solved": 31,
            "sdss_idx": 0, "separation_arcsec": 0.3,
        },
        {
            "source_id": 2, "ra": 200.0, "dec": 30.0,
            "parallax": 2.0, "parallax_error": 0.1,
            "pmra": 5.0, "pmdec": -2.0, "phot_g_mean_mag": 14.0,
            "astrometric_excess_noise": 2.5,
            "astrometric_excess_noise_sig": 8.0,
            "ruwe": 3.5, "ipd_gof_harmonic_amplitude": 0.15,
            "visibility_periods_used": 12, "astrometric_params_solved": 31,
            "sdss_idx": 5, "separation_arcsec": 0.2,
        },
    ])
    scored = score_astrometric_anomalies(df)
    # Source 2 (high RUWE, high excess noise) should score higher
    assert scored.iloc[1]["astrometric_anomaly_score"] > scored.iloc[0]["astrometric_anomaly_score"]
