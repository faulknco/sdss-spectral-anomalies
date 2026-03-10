# tests/test_download.py
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.download import (
    MANIFEST_NAME,
    build_sdss_filename,
    build_sdss_sas_url,
    build_sdss_query,
    download_spectra,
    fetch_spectrum_file,
    normalize_run2d,
    parse_spectrum_fits,
    parse_sdss_filename,
    spectrum_identifiers_from_metadata,
)


def test_build_sdss_query_returns_sql_string():
    sql = build_sdss_query(limit=100, sn_min=10.0)
    assert "SELECT" in sql.upper()
    assert "specobjall" in sql.lower() or "specobj" in sql.lower()
    assert "run2d" in sql.lower()
    assert "100" in sql
    assert "sn_median" in sql.lower() or "snmedian" in sql.lower()


def test_build_and_parse_sdss_filename_round_trip():
    filename = build_sdss_filename(123, 45678, 9)
    assert filename == "spec-0123-45678-0009.fits"
    assert parse_sdss_filename(filename) == (123, 45678, 9)


def test_build_sdss_sas_url_uses_full_spectra_path_for_modern_run2d():
    url = build_sdss_sas_url(123, 45678, 9, "v6_1_3")
    assert "https://data.sdss.org/sas/dr17/sdss/spectro/redux/v6_1_3/spectra/full/0123/" in url
    assert url.endswith("spec-0123-45678-0009.fits")


def test_normalize_run2d_handles_bytes():
    assert normalize_run2d(b"v5_13_2") == "v5_13_2"


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


def test_download_spectra_retries_and_writes_manifest(tmp_path):
    metadata = pd.DataFrame([{"plate": 1, "mjd": 2, "fiberid": 3, "run2d": "v6_1_3"}])
    call_count = {"n": 0}

    class DummyResponse:
        def __enter__(self):
            from io import BytesIO
            return BytesIO(b"fits-bytes")

        def __exit__(self, exc_type, exc, tb):
            return False

    def mock_urlopen(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TimeoutError("temporary")
        return DummyResponse()

    with patch("src.data.download.urlopen", side_effect=mock_urlopen):
        downloaded = download_spectra(
            metadata,
            tmp_path,
            batch_size=1,
            timeout=1,
            max_retries=2,
            retry_delay=0,
            transport="https",
        )

    expected = tmp_path / "spec-0001-2-0003.fits"
    assert downloaded == [expected]
    assert expected.read_bytes() == b"fits-bytes"
    manifest = pd.read_parquet(tmp_path / MANIFEST_NAME)
    assert manifest.loc[0, "status"] == "downloaded"
    assert int(manifest.loc[0, "attempts"]) == 2


def test_download_spectra_skips_existing_file(tmp_path):
    metadata = pd.DataFrame([{"plate": 1, "mjd": 2, "fiberid": 3, "run2d": "v6_1_3"}])
    existing = tmp_path / "spec-0001-2-0003.fits"
    existing.touch()

    with patch("src.data.download.urlopen") as mock_urlopen:
        downloaded = download_spectra(
            metadata,
            tmp_path,
            batch_size=1,
            retry_delay=0,
            transport="https",
        )

    assert downloaded == [existing]
    mock_urlopen.assert_not_called()
    manifest = pd.read_parquet(tmp_path / MANIFEST_NAME)
    assert manifest.loc[0, "status"] == "exists"


def test_spectrum_identifiers_from_metadata_falls_back_to_filename():
    plate, mjd, fiberid, run2d = spectrum_identifiers_from_metadata(
        {"filename": "spec-0123-45678-0009.fits"}
    )
    assert (plate, mjd, fiberid) == (123, 45678, 9)
    assert run2d is None


def test_spectrum_identifiers_from_metadata_returns_run2d_when_present():
    plate, mjd, fiberid, run2d = spectrum_identifiers_from_metadata(
        {"plate": 123, "mjd": 45678, "fiberid": 9, "run2d": "v6_1_3"}
    )
    assert (plate, mjd, fiberid, run2d) == (123, 45678, 9, "v6_1_3")


def test_fetch_spectrum_file_returns_existing_path(tmp_path):
    existing = tmp_path / "spec-0001-2-0003.fits"
    existing.touch()
    with patch("src.data.download.urlopen") as mock_urlopen:
        result = fetch_spectrum_file(1, 2, 3, tmp_path, transport="https")
    assert result == existing
    mock_urlopen.assert_not_called()


def test_fetch_spectrum_file_falls_back_to_astroquery_without_run2d(tmp_path):
    hdu = MagicMock()
    with patch("src.data.download.SDSS.get_spectra", return_value=[hdu]) as mock_get_spectra:
        result = fetch_spectrum_file(
            1, 2, 3, tmp_path, run2d=None, max_retries=1, transport="astroquery"
        )
    expected = tmp_path / "spec-0001-2-0003.fits"
    assert result == expected
    mock_get_spectra.assert_called_once()
    hdu.writeto.assert_called_once_with(expected, overwrite=True)


def test_download_spectra_uses_rsync_transport(tmp_path):
    metadata = pd.DataFrame([{"plate": 1, "mjd": 2, "fiberid": 3, "run2d": "26"}])
    expected = tmp_path / "spec-0001-2-0003.fits"

    def mock_run(*args, **kwargs):
        expected.write_bytes(b"fits-bytes")
        return MagicMock()

    with patch("src.data.download.subprocess.run", side_effect=mock_run) as mock_run_cmd:
        downloaded = download_spectra(
            metadata,
            tmp_path,
            batch_size=1,
            timeout=1,
            max_retries=1,
            retry_delay=0,
            transport="rsync",
        )

    assert downloaded == [expected]
    mock_run_cmd.assert_called_once()
    manifest = pd.read_parquet(tmp_path / MANIFEST_NAME)
    assert manifest.loc[0, "status"] == "downloaded"
