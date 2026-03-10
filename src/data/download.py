"""Download stellar spectra from SDSS via astroquery."""
import logging
import shutil
import subprocess
from pathlib import Path
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from astroquery.sdss import conf
from astroquery.sdss import SDSS

logger = logging.getLogger(__name__)

SPECTRAL_CLASSES = ["O", "B", "A", "F", "G", "K", "M"]
DEFAULT_QUERY_TIMEOUT = 60
DEFAULT_DOWNLOAD_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
MANIFEST_NAME = "download_manifest.parquet"
DEFAULT_DATA_RELEASE = conf.default_release
DOWNLOAD_TRANSPORTS = {"auto", "rsync", "https", "astroquery"}


def build_sdss_query(limit: int = 10000, sn_min: float = 10.0) -> str:
    """Build SQL query for SDSS stellar spectra with S/N filtering."""
    return f"""
    SELECT TOP {limit}
        s.specobjid, s.run2d, s.plate, s.mjd, s.fiberid,
        s.ra, s.dec, s.class, s.subclass,
        s.z, s.zerr, s.snmedian,
        s.elodieTEff, s.elodieLogG, s.elodieFeH
    FROM SpecObj AS s
    WHERE s.class = 'STAR'
        AND s.snmedian > {sn_min}
        AND s.zwarning = 0
    ORDER BY NEWID()
    """


def build_sdss_filename(plate: int, mjd: int, fiberid: int) -> str:
    return f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"


def normalize_run2d(run2d: Any) -> str:
    if isinstance(run2d, bytes):
        return run2d.decode()
    if isinstance(run2d, (np.integer, int)):
        return str(run2d)
    return str(run2d)


def build_sdss_sas_url(
    plate: int,
    mjd: int,
    fiberid: int,
    run2d: Any,
    data_release: int = DEFAULT_DATA_RELEASE,
) -> str:
    run2d_str = normalize_run2d(run2d)
    spectra_path = "spectra"
    if data_release > 15 and run2d_str not in ("26", "103", "104"):
        spectra_path = "spectra/full"
    return (
        f"{conf.sas_baseurl}/dr{data_release}/sdss/spectro/redux/"
        f"{run2d_str}/{spectra_path}/{plate:04d}/"
        f"{build_sdss_filename(plate, mjd, fiberid)}"
    )


def parse_sdss_filename(filename: str) -> tuple[int, int, int]:
    stem = Path(filename).name
    if not stem.startswith("spec-") or not stem.endswith(".fits"):
        raise ValueError(f"Unrecognized SDSS spectrum filename: {filename}")

    parts = stem.removesuffix(".fits").split("-")
    if len(parts) != 4:
        raise ValueError(f"Unrecognized SDSS spectrum filename: {filename}")

    _, plate, mjd, fiberid = parts
    return int(plate), int(mjd), int(fiberid)


def query_stellar_metadata(
    limit: int = 10000,
    sn_min: float = 10.0,
    timeout: int = DEFAULT_QUERY_TIMEOUT,
) -> pd.DataFrame:
    """Query SDSS for stellar spectra metadata."""
    sql = build_sdss_query(limit=limit, sn_min=sn_min)
    logger.info(f"Querying SDSS for {limit} stellar spectra (S/N > {sn_min})...")
    previous_timeout = getattr(SDSS, "TIMEOUT", None)
    SDSS.TIMEOUT = timeout
    try:
        result = SDSS.query_sql(sql)
    finally:
        if previous_timeout is not None:
            SDSS.TIMEOUT = previous_timeout
    if result is None:
        raise RuntimeError("SDSS query returned no results")
    df = result.to_pandas()
    logger.info(f"Retrieved metadata for {len(df)} spectra")
    return df


def parse_spectrum_fits(coadd_data) -> dict:
    """Extract wavelength and flux arrays from SDSS FITS COADD extension data."""
    loglam = np.array(coadd_data["loglam"], dtype=np.float64)
    flux = np.array(coadd_data["flux"], dtype=np.float64)
    return {
        "wavelength": 10**loglam,
        "flux": flux,
    }


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / MANIFEST_NAME


def _load_manifest(output_dir: Path) -> pd.DataFrame:
    path = _manifest_path(output_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_manifest(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    current = pd.DataFrame(rows)
    existing = _load_manifest(output_dir)
    combined = pd.concat([existing, current], ignore_index=True) if not existing.empty else current
    combined = combined.drop_duplicates(subset=["filename"], keep="last")
    combined.to_parquet(_manifest_path(output_dir), index=False)


def _has_rsync() -> bool:
    return shutil.which("rsync") is not None


def _download_via_rsync(
    source_url: str,
    output_dir: Path,
    timeout: int,
) -> tuple[bool, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["rsync", "--no-motd", "-av", source_url, f"{output_dir}/"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True, None
    except subprocess.TimeoutExpired as e:
        return False, f"rsync timeout after {timeout}s"
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        if "No such file or directory" in stderr:
            return False, "404 Not Found"
        return False, stderr or str(e)


def _download_spectrum_file(
    plate: int,
    mjd: int,
    fiberid: int,
    output_dir: Path,
    run2d: Any | None,
    timeout: int,
    max_retries: int,
    retry_delay: float,
    cache: bool,
    show_progress: bool,
    transport: str,
) -> tuple[Path | None, str, int, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = build_sdss_filename(plate, mjd, fiberid)
    fpath = output_dir / fname

    if fpath.exists():
        return fpath, "exists", 0, None

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            effective_transport = transport
            if effective_transport == "auto":
                if run2d is not None and pd.notna(run2d) and _has_rsync():
                    effective_transport = "rsync"
                elif run2d is not None and pd.notna(run2d):
                    effective_transport = "https"
                else:
                    effective_transport = "astroquery"

            if effective_transport not in DOWNLOAD_TRANSPORTS:
                raise ValueError(f"Unsupported transport: {transport}")

            if effective_transport == "rsync" and run2d is not None and pd.notna(run2d):
                sas_url = build_sdss_sas_url(plate, mjd, fiberid, run2d)
                rsync_url = sas_url.replace("https://data.sdss.org/sas", "rsync://dtn.sdss.org")
                ok, error = _download_via_rsync(rsync_url, output_dir, timeout=timeout)
                if ok:
                    return fpath, "downloaded", attempt, None
                last_error = error
                if error == "404 Not Found":
                    return None, "not_found", attempt, error
            elif effective_transport == "https" and run2d is not None and pd.notna(run2d):
                url = build_sdss_sas_url(plate, mjd, fiberid, run2d)
                request = Request(url, headers={"User-Agent": "sdss-spectral-anomalies/0.1"})
                with urlopen(request, timeout=timeout) as response, open(fpath, "wb") as fh:
                    shutil.copyfileobj(response, fh)
                return fpath, "downloaded", attempt, None
            else:
                sp = SDSS.get_spectra(
                    plate=plate,
                    mjd=mjd,
                    fiberID=fiberid,
                    timeout=timeout,
                    cache=cache,
                    show_progress=show_progress,
                )
                if sp and len(sp) > 0:
                    sp[0].writeto(fpath, overwrite=True)
                    return fpath, "downloaded", attempt, None
                return None, "not_found", attempt, None
        except HTTPError as e:
            if e.code == 404:
                return None, "not_found", attempt, "404 Not Found"
            last_error = str(e)
        except (URLError, TimeoutError) as e:
            last_error = str(e)
        except Exception as e:
            last_error = str(e)
            if attempt == max_retries:
                return None, "failed", attempt, last_error
        if fpath.exists():
            fpath.unlink(missing_ok=True)
        if attempt < max_retries:
            sleep(retry_delay)

    return None, "failed", max_retries, last_error


def fetch_spectrum_file(
    plate: int,
    mjd: int,
    fiberid: int,
    output_dir: Path,
    run2d: Any | None = None,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    cache: bool = True,
    show_progress: bool = False,
    transport: str = "auto",
) -> Path | None:
    """Fetch a single SDSS spectrum FITS file into ``output_dir`` if needed."""
    path, status, attempts, error = _download_spectrum_file(
        plate=plate,
        mjd=mjd,
        fiberid=fiberid,
        output_dir=output_dir,
        run2d=run2d,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        cache=cache,
        show_progress=show_progress,
        transport=transport,
    )
    _write_manifest(output_dir, [{
        "filename": build_sdss_filename(plate, mjd, fiberid),
        "plate": plate,
        "mjd": mjd,
        "fiberid": fiberid,
        "run2d": None if run2d is None or pd.isna(run2d) else normalize_run2d(run2d),
        "status": status,
        "attempts": attempts,
        "error": error,
    }])
    if status == "failed":
        logger.warning(
            "Failed to download %s after %s attempts: %s",
            build_sdss_filename(plate, mjd, fiberid),
            attempts,
            error,
        )
    return path


def spectrum_identifiers_from_metadata(row: pd.Series | dict[str, Any]) -> tuple[int, int, int, Any | None]:
    """Extract ``plate``, ``mjd``, ``fiberid``, and optional ``run2d``."""
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    if all(key in data and pd.notna(data[key]) for key in ("plate", "mjd", "fiberid")):
        return (
            int(data["plate"]),
            int(data["mjd"]),
            int(data["fiberid"]),
            data.get("run2d"),
        )

    filename = data.get("filename")
    if filename:
        plate, mjd, fiberid = parse_sdss_filename(str(filename))
        return plate, mjd, fiberid, data.get("run2d")

    raise KeyError("Metadata row does not include plate/mjd/fiberid or filename")


def download_spectra(
    metadata: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 50,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    cache: bool = True,
    show_progress: bool = False,
    transport: str = "auto",
) -> list[Path]:
    """Download SDSS spectra FITS files for given metadata rows.

    Existing files are reused. A manifest is written to
    ``output_dir / download_manifest.parquet`` so runs are easier to audit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    manifest_rows = []
    if transport not in DOWNLOAD_TRANSPORTS:
        raise ValueError(f"Unsupported transport: {transport}")

    for start in range(0, len(metadata), batch_size):
        batch = metadata.iloc[start : start + batch_size]
        logger.info(
            f"Downloading batch {start // batch_size + 1} "
            f"({len(batch)} spectra)..."
        )

        for _, row in batch.iterrows():
            plate, mjd, fiberid, run2d = spectrum_identifiers_from_metadata(row)
            fname = build_sdss_filename(plate, mjd, fiberid)
            path, status, attempts, error = _download_spectrum_file(
                plate=plate,
                mjd=mjd,
                fiberid=fiberid,
                output_dir=output_dir,
                run2d=run2d,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                cache=cache,
                show_progress=show_progress,
                transport=transport,
            )
            if status == "failed":
                logger.warning(f"Failed to download {fname} after {attempts} attempts: {error}")
            elif status == "not_found":
                logger.warning(f"Spectrum not found for {fname}")
            if path is not None:
                downloaded.append(path)
            manifest_rows.append({
                "filename": fname,
                "plate": plate,
                "mjd": mjd,
                "fiberid": fiberid,
                "run2d": None if run2d is None or pd.isna(run2d) else normalize_run2d(run2d),
                "status": status,
                "attempts": attempts,
                "error": error,
            })

        _write_manifest(output_dir, manifest_rows)
        manifest_rows.clear()

    logger.info(f"Downloaded {len(downloaded)} spectra to {output_dir}")
    return downloaded
