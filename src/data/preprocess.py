"""Preprocess SDSS spectra: resample, normalize, clean."""
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

DEFAULT_GRID = np.linspace(3800, 9200, 3500)


def _make_metadata_dict(
    filename: str,
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
    from src.data.download import parse_spectrum_fits

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

                specobj = hdul["SPECOBJ"].data

                metadata.append(_make_metadata_dict(
                    filename=fpath.name,
                    ra=float(specobj["RA"][0]),
                    dec=float(specobj["DEC"][0]),
                    subclass=str(specobj["SUBCLASS"][0]).strip(),
                    sn_median=_safe_float(specobj, "SN_MEDIAN_ALL", 0.0),
                    teff=_safe_float(specobj, "ELODIE_TEFF"),
                    logg=_safe_float(specobj, "ELODIE_LOGG"),
                    feh=_safe_float(specobj, "ELODIE_FEH"),
                ))
        except Exception as e:
            logger.warning(f"Failed to load {fpath.name}: {e}")

    spectra = preprocess_spectra(wavelengths, fluxes, target_grid)
    logger.info(f"Preprocessed {spectra.shape[0]} spectra to shape {spectra.shape}")
    return spectra, metadata


METADATA_FEATURE_COLS = ["elodie_teff", "elodie_logg", "elodie_feh", "sn_median"]


def build_metadata_features(metadata_df) -> np.ndarray:
    """Extract and standardize stellar metadata into a float32 feature matrix.

    Missing values are filled with column median. Columns are z-score standardized.
    Returns array of shape (n_spectra, len(METADATA_FEATURE_COLS)).
    """
    df = metadata_df[METADATA_FEATURE_COLS].copy().astype(np.float64)
    for col in df.columns:
        median = df[col].median()
        df[col] = df[col].fillna(median if np.isfinite(median) else 0.0)
    mean = df.mean()
    std = df.std().replace(0, 1)
    df = (df - mean) / std
    return df.values.astype(np.float32)
