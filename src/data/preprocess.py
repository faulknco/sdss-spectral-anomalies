"""Preprocess SDSS spectra: resample, normalize, clean."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

DEFAULT_GRID = np.linspace(3800, 9200, 3500)


def _make_metadata_dict(
    filename: str,
    plate: int,
    mjd: int,
    fiberid: int,
    ra: float,
    dec: float,
    subclass: str,
    sn_median: float,
    teff: float,
    logg: float,
    feh: float,
) -> dict:
    return {
        "filename": filename,
        "plate": plate,
        "mjd": mjd,
        "fiberid": fiberid,
        "ra": ra,
        "dec": dec,
        "subclass": subclass,
        "sn_median": sn_median,
        "elodie_teff": teff,
        "elodie_logg": logg,
        "elodie_feh": feh,
    }


def resample_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    target_grid: np.ndarray,
) -> np.ndarray:
    """Resample a spectrum onto a common wavelength grid via linear interpolation."""
    mask = np.isfinite(flux) & np.isfinite(wavelength)
    if mask.sum() < 10:
        return np.full(len(target_grid), np.nan)

    f = interp1d(
        wavelength[mask],
        flux[mask],
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return f(target_grid)


def normalize_spectrum(flux: np.ndarray) -> np.ndarray:
    """Normalize flux by dividing by the median. Handles zero median."""
    median = np.median(flux)
    if median == 0 or not np.isfinite(median):
        return np.zeros_like(flux)
    return flux / median


def preprocess_spectra(
    wavelengths: list[np.ndarray],
    fluxes: list[np.ndarray],
    target_grid: np.ndarray = DEFAULT_GRID,
) -> np.ndarray:
    """Resample and normalize a list of spectra to a common grid."""
    processed = []
    for wl, fl in zip(wavelengths, fluxes):
        resampled = resample_spectrum(wl, fl, target_grid)
        normalized = normalize_spectrum(resampled)
        processed.append(normalized)
    return np.array(processed)


def load_and_preprocess(
    fits_dir: Path,
    target_grid: np.ndarray = DEFAULT_GRID,
) -> tuple[np.ndarray, list[dict]]:
    """Load FITS files from a directory and return preprocessed spectra + metadata."""
    from astropy.io import fits as astro_fits
    from src.data.download import parse_sdss_filename, parse_spectrum_fits

    fits_files = sorted(fits_dir.glob("spec-*.fits"))
    logger.info(f"Loading {len(fits_files)} FITS files from {fits_dir}")

    wavelengths = []
    fluxes = []
    metadata = []

    def _safe_float(arr, name, fallback=np.nan):
        return float(arr[name][0]) if name in arr.dtype.names else fallback

    for fpath in fits_files:
        try:
            with astro_fits.open(fpath) as hdul:
                parsed = parse_spectrum_fits(hdul["COADD"].data)
                wavelengths.append(parsed["wavelength"])
                fluxes.append(parsed["flux"])
                plate, mjd, fiberid = parse_sdss_filename(fpath.name)

                # SDSS-II uses SPECOBJ, BOSS uses SPALL
                if "SPECOBJ" in hdul:
                    meta_ext = hdul["SPECOBJ"].data
                elif "SPALL" in hdul:
                    meta_ext = hdul["SPALL"].data
                else:
                    raise KeyError("No SPECOBJ or SPALL extension found")

                # RA/DEC: try RA first, fall back to PLUG_RA
                ra = _safe_float(meta_ext, "RA",
                     _safe_float(meta_ext, "PLUG_RA", np.nan))
                dec = _safe_float(meta_ext, "DEC",
                      _safe_float(meta_ext, "PLUG_DEC", np.nan))

                metadata.append(_make_metadata_dict(
                    filename=fpath.name,
                    plate=plate,
                    mjd=mjd,
                    fiberid=fiberid,
                    ra=ra,
                    dec=dec,
                    subclass=str(meta_ext["SUBCLASS"][0]).strip(),
                    sn_median=_safe_float(meta_ext, "SN_MEDIAN_ALL", 0.0),
                    teff=_safe_float(meta_ext, "ELODIE_TEFF"),
                    logg=_safe_float(meta_ext, "ELODIE_LOGG"),
                    feh=_safe_float(meta_ext, "ELODIE_FEH"),
                ))
        except Exception as e:
            logger.warning(f"Failed to load {fpath.name}: {e}")

    spectra = preprocess_spectra(wavelengths, fluxes, target_grid)
    logger.info(f"Preprocessed {spectra.shape[0]} spectra to shape {spectra.shape}")
    return spectra, metadata


def load_or_fetch_processed_spectrum(
    metadata_row,
    raw_dir: Path,
    target_grid: np.ndarray = DEFAULT_GRID,
) -> np.ndarray:
    """Load a processed spectrum from local FITS, fetching the FITS on demand if needed."""
    from astropy.io import fits as astro_fits
    from src.data.download import (
        build_sdss_filename,
        fetch_spectrum_file,
        parse_spectrum_fits,
        spectrum_identifiers_from_metadata,
    )

    plate, mjd, fiberid, run2d = spectrum_identifiers_from_metadata(metadata_row)
    filename = build_sdss_filename(plate, mjd, fiberid)
    fpath = raw_dir / filename
    if not fpath.exists():
        fetched = fetch_spectrum_file(plate, mjd, fiberid, raw_dir, run2d=run2d)
        if fetched is None:
            raise FileNotFoundError(f"Could not fetch {filename} from SDSS")
        fpath = fetched

    with astro_fits.open(fpath) as hdul:
        parsed = parse_spectrum_fits(hdul["COADD"].data)
    return normalize_spectrum(
        resample_spectrum(parsed["wavelength"], parsed["flux"], target_grid)
    )


METADATA_FEATURE_COLS = ["elodie_teff", "elodie_logg", "elodie_feh", "sn_median"]


def build_metadata_features(metadata_df, stats: dict | None = None, return_stats: bool = False):
    """Extract and standardize stellar metadata into a float32 feature matrix.

    Missing values are filled with column median. Columns are z-score standardized.
    Returns array of shape (n_spectra, len(METADATA_FEATURE_COLS)).
    """
    df = metadata_df[METADATA_FEATURE_COLS].copy().astype(np.float64)
    if stats is None:
        medians = df.median()
        medians = medians.where(np.isfinite(medians), 0.0)
        filled = df.fillna(medians)
        mean = filled.mean()
        std = filled.std()
        std = std.where(np.isfinite(std) & (std != 0), 1.0)
        stats = {
            "medians": medians.to_dict(),
            "mean": mean.to_dict(),
            "std": std.to_dict(),
        }
    else:
        medians = np.array([stats["medians"][col] for col in METADATA_FEATURE_COLS], dtype=np.float64)
        mean = np.array([stats["mean"][col] for col in METADATA_FEATURE_COLS], dtype=np.float64)
        std = np.array([stats["std"][col] for col in METADATA_FEATURE_COLS], dtype=np.float64)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std != 0), std, 1.0)
        filled = df.fillna(pd.Series(medians, index=METADATA_FEATURE_COLS))
        features = (filled.values - mean) / std
        features = features.astype(np.float32)
        if return_stats:
            return features, stats
        return features

    features = ((filled - mean) / std).values.astype(np.float32)
    if return_stats:
        return features, stats
    return features
