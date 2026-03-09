"""Download stellar spectra from SDSS via astroquery."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astroquery.sdss import SDSS

logger = logging.getLogger(__name__)

SPECTRAL_CLASSES = ["O", "B", "A", "F", "G", "K", "M"]


def build_sdss_query(limit: int = 10000, sn_min: float = 10.0) -> str:
    """Build SQL query for SDSS stellar spectra with S/N filtering."""
    return f"""
    SELECT TOP {limit}
        s.specobjid, s.plate, s.mjd, s.fiberid,
        s.ra, s.dec, s.class, s.subclass,
        s.z, s.zerr, s.snmedian,
        s.elodieTEff, s.elodieLogG, s.elodieFeH
    FROM SpecObj AS s
    WHERE s.class = 'STAR'
        AND s.snmedian > {sn_min}
        AND s.zwarning = 0
    ORDER BY NEWID()
    """


def query_stellar_metadata(limit: int = 10000, sn_min: float = 10.0) -> pd.DataFrame:
    """Query SDSS for stellar spectra metadata."""
    sql = build_sdss_query(limit=limit, sn_min=sn_min)
    logger.info(f"Querying SDSS for {limit} stellar spectra (S/N > {sn_min})...")
    result = SDSS.query_sql(sql)
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


def download_spectra(
    metadata: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 50,
) -> list[Path]:
    """Download SDSS spectra FITS files for given metadata rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for start in range(0, len(metadata), batch_size):
        batch = metadata.iloc[start : start + batch_size]
        logger.info(
            f"Downloading batch {start // batch_size + 1} "
            f"({len(batch)} spectra)..."
        )

        for _, row in batch.iterrows():
            plate = int(row["plate"])
            mjd = int(row["mjd"])
            fiberid = int(row["fiberid"])
            fname = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
            fpath = output_dir / fname

            if fpath.exists():
                downloaded.append(fpath)
                continue

            try:
                sp = SDSS.get_spectra(
                    plate=plate, mjd=mjd, fiberID=fiberid
                )
                if sp and len(sp) > 0:
                    sp[0].writeto(fpath, overwrite=True)
                    downloaded.append(fpath)
            except Exception as e:
                logger.warning(f"Failed to download {fname}: {e}")

    logger.info(f"Downloaded {len(downloaded)} spectra to {output_dir}")
    return downloaded
