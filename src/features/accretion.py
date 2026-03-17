"""Accretion signature detection in stellar spectra.

PBH accretion onto nearby material can produce: unexpected emission lines,
blue/UV excess beyond the stellar photosphere, and power-law continuum
components.  This module scores each spectrum for these signatures.

CAVEAT: High accretion scores do NOT confirm PBHs -- chromospheric activity,
binaries with accretion disks, misclassified subclasses, and poor sky
subtraction all produce similar features.  Single-epoch median-normalized
spectra cannot measure absolute flux levels.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.features.spectral_lines import STELLAR_LINES

# Rough Teff lookup by broad spectral class (main-sequence midpoints).
_SUBCLASS_TEFF = {
    "O": 35000.0,
    "B": 18000.0,
    "A": 8500.0,
    "F": 6500.0,
    "G": 5600.0,
    "K": 4500.0,
    "M": 3200.0,
}


def _parse_broad_class(subclass: str) -> str:
    s = str(subclass).strip().upper()
    for char in s:
        if char in "OBAFGKM":
            return char
    return "unknown"


def _teff_for_row(metadata_row: pd.Series) -> float | None:
    """Return effective temperature from metadata or subclass lookup."""
    teff = metadata_row.get("elodie_teff", np.nan)
    if np.isfinite(teff) and teff > 0:
        return float(teff)
    subclass = str(metadata_row.get("subclass", ""))
    cls = _parse_broad_class(subclass)
    return _SUBCLASS_TEFF.get(cls)


def _planck_shape(wavelength_aa: np.ndarray, teff: float) -> np.ndarray:
    """Planck function shape (normalized to median=1) for a given Teff.

    Wavelength in Angstroms, converted internally to meters.
    """
    h = 6.626e-34
    c = 3.0e8
    k = 1.381e-23
    lam_m = wavelength_aa * 1e-10
    with np.errstate(over="ignore", divide="ignore"):
        exponent = h * c / (lam_m * k * teff)
        # Clip to avoid overflow
        exponent = np.clip(exponent, 0, 500)
        bb = 1.0 / (lam_m**5 * (np.exp(exponent) - 1.0))
    bb = np.where(np.isfinite(bb) & (bb > 0), bb, 0.0)
    med = np.median(bb[bb > 0]) if np.any(bb > 0) else 1.0
    if med == 0:
        med = 1.0
    return bb / med


def detect_unexpected_emission(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
    subclass: str,
) -> list[dict]:
    """Check for emission where absorption is expected for the spectral type.

    Returns list of dicts with line name, wavelength, and emission strength
    for each line showing unexpected emission.
    """
    broad_class = _parse_broad_class(subclass)
    detections = []

    for name, info in STELLAR_LINES.items():
        # Only check lines expected in absorption for this spectral class
        if broad_class not in info["classes"]:
            continue
        center = info["center"]
        window = 10.0
        cont_width = 50.0

        line_mask = (wavelength_grid >= center - window) & (wavelength_grid <= center + window)
        blue_mask = (wavelength_grid >= center - cont_width) & (wavelength_grid < center - window)
        red_mask = (wavelength_grid > center + window) & (wavelength_grid <= center + cont_width)
        continuum_mask = blue_mask | red_mask

        if line_mask.sum() == 0 or continuum_mask.sum() == 0:
            continue

        line_flux = np.mean(spectrum[line_mask])
        cont_flux = np.mean(spectrum[continuum_mask])

        # Skip regions where the continuum is too faint to measure reliably.
        # Spectra are median-normalized to ~1.0, so cont_flux < 0.1 means
        # essentially no signal (e.g. blue end of M-dwarf spectra).
        if cont_flux < 0.1 or not np.isfinite(cont_flux):
            continue

        # Emission: line flux exceeds continuum
        if line_flux > cont_flux:
            strength = float((line_flux - cont_flux) / cont_flux)
            detections.append({
                "line_name": name,
                "wavelength": center,
                "emission_strength": strength,
            })

    return detections


def measure_blue_excess(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
    teff: float | None,
) -> float:
    """Compare observed blue flux to Planck function shape for given Teff.

    Returns the fractional excess of observed blue flux (3800-4500 A) over
    the expected Planck shape.  Positive = bluer than expected.
    """
    if teff is None or teff <= 0:
        return 0.0

    blue_mask = (wavelength_grid >= 3800) & (wavelength_grid <= 4500)
    if blue_mask.sum() == 0:
        return 0.0

    bb = _planck_shape(wavelength_grid, teff)
    bb_blue = np.mean(bb[blue_mask])
    obs_blue = np.mean(spectrum[blue_mask])

    if bb_blue <= 0 or not np.isfinite(bb_blue):
        return 0.0

    # Scale the Planck shape to match the overall spectrum median
    full_mask = np.isfinite(spectrum) & (spectrum > 0)
    if full_mask.sum() == 0:
        return 0.0
    scale = np.median(spectrum[full_mask]) / np.median(bb[full_mask]) if np.median(bb[full_mask]) > 0 else 1.0
    expected_blue = bb_blue * scale

    if expected_blue <= 0:
        return 0.0

    return float((obs_blue - expected_blue) / expected_blue)


def fit_powerlaw_residual(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
    teff: float | None,
) -> dict:
    """Fit spectrum = blackbody(Teff) + A * lambda^(-alpha) and return fit params.

    Returns dict with amplitude, spectral_index, significance.
    """
    result = {"amplitude": 0.0, "spectral_index": 0.0, "significance": 0.0}

    if teff is None or teff <= 0:
        return result

    bb = _planck_shape(wavelength_grid, teff)
    # Scale blackbody to match spectrum
    valid = np.isfinite(spectrum) & (spectrum > 0) & (bb > 0)
    if valid.sum() < 10:
        return result

    scale = np.median(spectrum[valid]) / np.median(bb[valid])
    residual = spectrum - bb * scale

    # Fit power law: A * lambda^(-alpha) to the residual
    # Use log-space fit on positive residuals
    pos = valid & (residual > 0)
    if pos.sum() < 5:
        return result

    log_lam = np.log(wavelength_grid[pos])
    log_res = np.log(residual[pos])

    try:
        # Linear fit in log-log space: log(residual) = log(A) - alpha * log(lambda)
        coeffs = np.polyfit(log_lam, log_res, 1)
        alpha = -coeffs[0]
        log_a = coeffs[1]
        amplitude = np.exp(log_a)

        # Significance: ratio of power-law component to residual std
        pl_component = amplitude * wavelength_grid[valid] ** (-alpha)
        sig = float(np.mean(np.abs(pl_component)) / (np.std(residual[valid]) + 1e-10))

        result = {
            "amplitude": float(amplitude),
            "spectral_index": float(alpha),
            "significance": float(sig),
        }
    except (np.linalg.LinAlgError, ValueError):
        pass

    return result


def accretion_score(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    metadata_df: pd.DataFrame,
) -> np.ndarray:
    """Compute accretion candidate score for each spectrum.

    Weighted combination of:
    - Unexpected emission line count and total strength
    - Blue excess relative to Teff expectation
    - Power-law residual significance

    Returns array of shape (n_spectra,) with non-negative scores.
    """
    n = len(spectra)
    scores = np.zeros(n)

    subclasses = metadata_df["subclass"].values if "subclass" in metadata_df.columns else [""] * n

    for i in range(n):
        row = metadata_df.iloc[i] if len(metadata_df) > i else pd.Series()
        teff = _teff_for_row(row)

        # 1. Unexpected emission
        emissions = detect_unexpected_emission(spectra[i], wavelength_grid, str(subclasses[i]))
        emission_count = len(emissions)
        emission_strength = sum(e["emission_strength"] for e in emissions) if emissions else 0.0

        # 2. Blue excess
        blue_ex = measure_blue_excess(spectra[i], wavelength_grid, teff)
        blue_ex = max(blue_ex, 0.0)  # Only care about excess, not deficit

        # 3. Power-law residual
        pl = fit_powerlaw_residual(spectra[i], wavelength_grid, teff)
        pl_sig = max(pl["significance"], 0.0)

        # Weighted combination
        scores[i] = (
            0.3 * emission_count
            + 0.2 * emission_strength
            + 0.25 * blue_ex
            + 0.25 * pl_sig
        )

    scores = np.maximum(scores, 0.0)
    return scores
