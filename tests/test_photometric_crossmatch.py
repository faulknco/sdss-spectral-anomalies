"""Tests for photometric cross-match module (offline, no SDSS queries)."""
import numpy as np
import pandas as pd
from src.features.photometric_crossmatch import (
    _parse_broad_class,
    _build_batch_photometry_query,
    compute_dereddened_colors,
    score_color_anomalies,
    _empty_photometry,
)


def test_parse_broad_class():
    assert _parse_broad_class("G2 IV") == "G"
    assert _parse_broad_class("K5") == "K"
    assert _parse_broad_class("WDhotter") == "unknown"
    assert _parse_broad_class("WDcooler") == "unknown"
    assert _parse_broad_class("") == "unknown"


def test_build_batch_query():
    q = _build_batch_photometry_query([1000], [55000], [100])
    assert "PhotoObj" in q
    assert "psfMag_u" in q
    assert "1000" in q


def test_compute_dereddened_colors():
    df = pd.DataFrame([{
        "plate": 1000, "mjd": 55000, "fiberid": 100, "sdss_idx": 0,
        "psfMag_u": 18.0, "psfMag_g": 16.5, "psfMag_r": 16.0,
        "psfMag_i": 15.8, "psfMag_z": 15.7,
        "psfMagErr_u": 0.02, "psfMagErr_g": 0.01, "psfMagErr_r": 0.01,
        "psfMagErr_i": 0.01, "psfMagErr_z": 0.02,
        "extinction_u": 0.2, "extinction_g": 0.15, "extinction_r": 0.10,
        "extinction_i": 0.08, "extinction_z": 0.05,
    }])
    result = compute_dereddened_colors(df)
    assert "u_g" in result.columns
    assert "g_r" in result.columns
    # u_g = (18.0 - 0.2) - (16.5 - 0.15) = 17.8 - 16.35 = 1.45
    assert abs(result.iloc[0]["u_g"] - 1.45) < 0.01
    # g_r = (16.5 - 0.15) - (16.0 - 0.10) = 16.35 - 15.90 = 0.45
    assert abs(result.iloc[0]["g_r"] - 0.45) < 0.01


def test_compute_dereddened_colors_empty():
    df = _empty_photometry()
    result = compute_dereddened_colors(df)
    assert "u_g" in result.columns
    assert len(result) == 0


def test_score_color_anomalies_normal_g_star():
    """G star with normal colors should have low anomaly score."""
    phot = pd.DataFrame([{
        "plate": 1000, "mjd": 55000, "fiberid": 100, "sdss_idx": 0,
        "psfMag_u": 17.3, "psfMag_g": 16.0, "psfMag_r": 15.5,
        "psfMag_i": 15.3, "psfMag_z": 15.2,
        "psfMagErr_u": 0.02, "psfMagErr_g": 0.01, "psfMagErr_r": 0.01,
        "psfMagErr_i": 0.01, "psfMagErr_z": 0.02,
        "extinction_u": 0.0, "extinction_g": 0.0, "extinction_r": 0.0,
        "extinction_i": 0.0, "extinction_z": 0.0,
    }])
    meta = pd.DataFrame([{"subclass": "G2"}])
    scored = score_color_anomalies(phot, meta)
    assert scored.iloc[0]["color_anomaly_score"] < 1.0
    assert not scored.iloc[0]["blue_excess_flag"]


def test_score_color_anomalies_blue_excess():
    """G star with anomalously blue u-g should be flagged."""
    phot = pd.DataFrame([{
        "plate": 1000, "mjd": 55000, "fiberid": 100, "sdss_idx": 0,
        # u-g = 14.5 - 16.0 = -1.5, expected for G ~1.3, excess = -2.8
        "psfMag_u": 14.5, "psfMag_g": 16.0, "psfMag_r": 15.5,
        "psfMag_i": 15.3, "psfMag_z": 15.2,
        "psfMagErr_u": 0.02, "psfMagErr_g": 0.01, "psfMagErr_r": 0.01,
        "psfMagErr_i": 0.01, "psfMagErr_z": 0.02,
        "extinction_u": 0.0, "extinction_g": 0.0, "extinction_r": 0.0,
        "extinction_i": 0.0, "extinction_z": 0.0,
    }])
    meta = pd.DataFrame([{"subclass": "G2"}])
    scored = score_color_anomalies(phot, meta)
    assert scored.iloc[0]["blue_excess_flag"]
    assert scored.iloc[0]["color_anomaly_score"] > 2.0


def test_score_color_anomalies_empty():
    scored = score_color_anomalies(_empty_photometry(), pd.DataFrame())
    assert "color_anomaly_score" in scored.columns
    assert len(scored) == 0


def test_score_multiple_stars():
    """Two stars: normal and anomalous, verify relative ordering."""
    phot = pd.DataFrame([
        {
            "plate": 1000, "mjd": 55000, "fiberid": 100, "sdss_idx": 0,
            "psfMag_u": 17.3, "psfMag_g": 16.0, "psfMag_r": 15.5,
            "psfMag_i": 15.3, "psfMag_z": 15.2,
            "psfMagErr_u": 0.02, "psfMagErr_g": 0.01, "psfMagErr_r": 0.01,
            "psfMagErr_i": 0.01, "psfMagErr_z": 0.02,
            "extinction_u": 0.0, "extinction_g": 0.0, "extinction_r": 0.0,
            "extinction_i": 0.0, "extinction_z": 0.0,
        },
        {
            "plate": 2000, "mjd": 55100, "fiberid": 200, "sdss_idx": 1,
            "psfMag_u": 14.0, "psfMag_g": 16.0, "psfMag_r": 15.5,
            "psfMag_i": 15.3, "psfMag_z": 15.2,
            "psfMagErr_u": 0.02, "psfMagErr_g": 0.01, "psfMagErr_r": 0.01,
            "psfMagErr_i": 0.01, "psfMagErr_z": 0.02,
            "extinction_u": 0.0, "extinction_g": 0.0, "extinction_r": 0.0,
            "extinction_i": 0.0, "extinction_z": 0.0,
        },
    ])
    meta = pd.DataFrame([{"subclass": "G2"}, {"subclass": "G2"}])
    scored = score_color_anomalies(phot, meta)
    # Second star (blue u) should have higher anomaly score
    assert scored.iloc[1]["color_anomaly_score"] > scored.iloc[0]["color_anomaly_score"]
