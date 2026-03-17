"""Spectral line measurement utilities for stellar spectra."""
import numpy as np

# Line center wavelengths (Angstroms), expected type, and applicable spectral classes.
# 'absorption' = expected in absorption for most stellar types listed;
# 'emission' would be unexpected for normal stars of those classes.
STELLAR_LINES = {
    "Ca_K": {"center": 3933.7, "type": "absorption", "classes": {"F", "G", "K", "M"}},
    "Ca_H": {"center": 3968.5, "type": "absorption", "classes": {"F", "G", "K", "M"}},
    "H_gamma": {"center": 4340.5, "type": "absorption", "classes": {"A", "B", "F", "G"}},
    "H_beta": {"center": 4861.3, "type": "absorption", "classes": {"A", "B", "F", "G"}},
    "Mg_b": {"center": 5175.0, "type": "absorption", "classes": {"F", "G", "K"}},
    "Na_D": {"center": 5892.0, "type": "absorption", "classes": {"G", "K", "M"}},
    "H_alpha": {"center": 6562.8, "type": "absorption", "classes": {"A", "B", "F", "G", "K", "M"}},
}


def measure_line_flux(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
    line_center: float,
    window_half_width: float = 10.0,
    continuum_width: float = 50.0,
) -> dict:
    """Measure line and continuum flux around a spectral line.

    The continuum is estimated from two flanking regions outside the line window.
    Equivalent width is computed as the integral of (1 - flux/continuum) over the
    line window, in wavelength units (Angstroms).

    Returns dict with line_flux, continuum_flux, ratio, equivalent_width.
    """
    # Line window
    line_mask = (wavelength_grid >= line_center - window_half_width) & (
        wavelength_grid <= line_center + window_half_width
    )
    # Continuum flanking regions
    blue_mask = (wavelength_grid >= line_center - continuum_width) & (
        wavelength_grid < line_center - window_half_width
    )
    red_mask = (wavelength_grid > line_center + window_half_width) & (
        wavelength_grid <= line_center + continuum_width
    )
    continuum_mask = blue_mask | red_mask

    if line_mask.sum() == 0 or continuum_mask.sum() == 0:
        return {
            "line_flux": np.nan,
            "continuum_flux": np.nan,
            "ratio": np.nan,
            "equivalent_width": np.nan,
        }

    line_flux = np.mean(spectrum[line_mask])
    continuum_flux = np.mean(spectrum[continuum_mask])

    if continuum_flux == 0 or not np.isfinite(continuum_flux):
        ratio = np.nan
        equivalent_width = np.nan
    else:
        ratio = line_flux / continuum_flux
        # EW = integral of (1 - F_line/F_cont) * dlambda over line window
        dlambda = np.mean(np.diff(wavelength_grid[line_mask])) if line_mask.sum() > 1 else 1.0
        equivalent_width = float(
            np.sum((1.0 - spectrum[line_mask] / continuum_flux)) * dlambda
        )

    return {
        "line_flux": float(line_flux),
        "continuum_flux": float(continuum_flux),
        "ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "equivalent_width": float(equivalent_width) if np.isfinite(equivalent_width) else np.nan,
    }


def measure_line_profile(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
    line_center: float,
    window_half_width: float = 10.0,
    continuum_width: float = 50.0,
) -> dict:
    """Measure asymmetry and shape of a spectral line profile.

    Splits the line window at the center and compares the blue and red
    halves.  For a symmetric line, skewness ~ 0 and blue_red_ratio ~ 1.

    Returns dict with skewness, kurtosis, blue_red_ratio, blue_ew, red_ew.
    """
    line_mask = (wavelength_grid >= line_center - window_half_width) & (
        wavelength_grid <= line_center + window_half_width
    )
    blue_cont = (wavelength_grid >= line_center - continuum_width) & (
        wavelength_grid < line_center - window_half_width
    )
    red_cont = (wavelength_grid > line_center + window_half_width) & (
        wavelength_grid <= line_center + continuum_width
    )
    continuum_mask = blue_cont | red_cont

    nan_result = {
        "skewness": np.nan, "kurtosis": np.nan,
        "blue_red_ratio": np.nan, "blue_ew": np.nan, "red_ew": np.nan,
    }

    if line_mask.sum() < 4 or continuum_mask.sum() == 0:
        return nan_result

    cont_flux = np.mean(spectrum[continuum_mask])
    if cont_flux < 0.1 or not np.isfinite(cont_flux):
        return nan_result

    # Continuum-subtracted profile (absorption depth)
    wl_line = wavelength_grid[line_mask]
    profile = 1.0 - spectrum[line_mask] / cont_flux  # positive = absorption

    # Skewness and kurtosis of the profile shape
    mean_p = np.mean(profile)
    std_p = np.std(profile)
    if std_p == 0:
        return nan_result

    centered = profile - mean_p
    skewness = float(np.mean(centered**3) / std_p**3)
    kurtosis = float(np.mean(centered**4) / std_p**4 - 3.0)

    # Blue/red half comparison
    blue_half = profile[wl_line < line_center]
    red_half = profile[wl_line >= line_center]

    dlambda = np.mean(np.diff(wl_line)) if len(wl_line) > 1 else 1.0
    blue_ew = float(np.sum(blue_half) * dlambda) if len(blue_half) > 0 else 0.0
    red_ew = float(np.sum(red_half) * dlambda) if len(red_half) > 0 else 0.0

    if red_ew == 0:
        blue_red_ratio = np.nan
    else:
        blue_red_ratio = blue_ew / red_ew

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "blue_red_ratio": float(blue_red_ratio) if np.isfinite(blue_red_ratio) else np.nan,
        "blue_ew": blue_ew,
        "red_ew": red_ew,
    }


def measure_all_lines(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
) -> dict[str, dict]:
    """Measure flux properties for all lines in STELLAR_LINES."""
    results = {}
    for name, info in STELLAR_LINES.items():
        results[name] = measure_line_flux(spectrum, wavelength_grid, info["center"])
    return results


def measure_all_line_profiles(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
) -> dict[str, dict]:
    """Measure profile shape (asymmetry, kurtosis) for all STELLAR_LINES."""
    results = {}
    for name, info in STELLAR_LINES.items():
        results[name] = measure_line_profile(spectrum, wavelength_grid, info["center"])
    return results
