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
    loglam_arr = np.array([3.58, 3.59, 3.60])
    flux_arr = np.array([10.0, 12.0, 11.0])
    mock_data = MagicMock()
    mock_data.__getitem__ = lambda self, key: {"loglam": loglam_arr, "flux": flux_arr}[key]

    result = parse_spectrum_fits(mock_data)
    assert "wavelength" in result
    assert "flux" in result
    np.testing.assert_allclose(result["wavelength"], 10 ** np.array([3.58, 3.59, 3.60]))
    np.testing.assert_array_equal(result["flux"], np.array([10.0, 12.0, 11.0]))
