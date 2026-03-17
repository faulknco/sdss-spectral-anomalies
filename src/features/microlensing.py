"""Microlensing signature detection in stellar spectra.

Chromatic microlensing by a compact object (e.g. PBH) magnifies the compact
continuum-emitting region more than extended line-forming regions, producing
anomalous continuum-to-line flux ratios.  This module scores each spectrum by
how far its line ratios deviate from the population of the same spectral class.

CAVEAT: High scores do NOT confirm PBH microlensing -- many mundane
astrophysical phenomena (binaries, chromospheric activity, misclassified
subclass, poor sky subtraction) produce similar spectral features.
"""
import numpy as np
import pandas as pd

from src.features.spectral_lines import STELLAR_LINES, measure_line_flux


def _parse_broad_class(subclass: str) -> str:
    """Extract the broad spectral class letter from an SDSS subclass string.

    Examples: 'G2 IV' -> 'G', 'K5' -> 'K', 'dM4' -> 'M', '' -> 'unknown'.
    """
    s = str(subclass).strip().upper()
    for char in s:
        if char in "OBAFGKM":
            return char
    return "unknown"


def compute_continuum_line_ratios(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute continuum-to-line flux ratio for every line for every spectrum.

    Returns a DataFrame with one row per spectrum and columns:
      <line_name>_ratio, <line_name>_ew, broad_class
    """
    rows = []
    subclasses = metadata_df["subclass"].values if "subclass" in metadata_df.columns else [""] * len(spectra)
    for i in range(len(spectra)):
        row = {"broad_class": _parse_broad_class(subclasses[i])}
        for name, info in STELLAR_LINES.items():
            m = measure_line_flux(spectra[i], wavelength_grid, info["center"])
            row[f"{name}_ratio"] = m["ratio"]
            row[f"{name}_ew"] = m["equivalent_width"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_expected_ratios(
    ratio_df: pd.DataFrame,
    groupby_col: str = "broad_class",
    min_count: int = 20,
) -> pd.DataFrame:
    """Compute per-subclass median and std of each line ratio.

    For groups with fewer than ``min_count`` members, fall back to the global
    population statistics.
    """
    ratio_cols = [c for c in ratio_df.columns if c.endswith("_ratio")]
    grouped = ratio_df.groupby(groupby_col)

    global_median = ratio_df[ratio_cols].median()
    global_std = ratio_df[ratio_cols].std()
    global_std = global_std.where(global_std > 0, 1.0)

    records = []
    for cls, grp in grouped:
        if len(grp) >= min_count:
            med = grp[ratio_cols].median()
            std = grp[ratio_cols].std()
            std = std.where(std > 0, 1.0)
        else:
            med = global_median
            std = global_std
        record = {groupby_col: cls}
        for col in ratio_cols:
            record[f"{col}_median"] = med[col]
            record[f"{col}_std"] = std[col]
        records.append(record)

    return pd.DataFrame(records)


def achromatic_residual_score(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
    poly_order: int = 3,
) -> float:
    """Score how achromatic (wavelength-independent) the residuals are.

    Fits a low-order polynomial continuum and checks whether residuals show
    flat structure (achromatic magnification) vs. a reddening-like slope.
    A higher score means more achromatic residuals, consistent with lensing.
    """
    mask = np.isfinite(spectrum)
    if mask.sum() < poly_order + 1:
        return 0.0

    coeffs = np.polyfit(wavelength_grid[mask], spectrum[mask], poly_order)
    fit = np.polyval(coeffs, wavelength_grid)
    residuals = spectrum - fit

    valid_residuals = residuals[mask]
    if len(valid_residuals) < 2:
        return 0.0

    # Flatness: ratio of mean(|residual|) to std(residual).
    # Achromatic offset -> high mean, low std -> high ratio.
    mean_abs = np.mean(np.abs(valid_residuals))
    std_res = np.std(valid_residuals)
    if std_res == 0:
        return 0.0

    return float(mean_abs / std_res)


def microlensing_score(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    metadata_df: pd.DataFrame,
) -> np.ndarray:
    """Compute a microlensing candidate score for each spectrum.

    Combines:
    1. Aggregate z-score of line ratio deviations from spectral-class expectations
    2. Achromatic residual score (flat continuum offset)

    Returns array of shape (n_spectra,) with non-negative scores.
    """
    ratio_df = compute_continuum_line_ratios(spectra, wavelength_grid, metadata_df)
    expected = build_expected_ratios(ratio_df)
    ratio_cols = [c for c in ratio_df.columns if c.endswith("_ratio")]

    # Build lookup: broad_class -> {col_median, col_std}
    expected_lookup = {}
    for _, row in expected.iterrows():
        expected_lookup[row["broad_class"]] = row

    scores = np.zeros(len(spectra))
    for i in range(len(spectra)):
        cls = ratio_df.iloc[i]["broad_class"]
        exp = expected_lookup.get(cls)
        if exp is None:
            z_agg = 0.0
        else:
            z_scores = []
            for col in ratio_cols:
                val = ratio_df.iloc[i][col]
                med = exp[f"{col}_median"]
                std = exp[f"{col}_std"]
                if np.isfinite(val) and np.isfinite(med) and std > 0:
                    z_scores.append(abs((val - med) / std))
            z_agg = float(np.mean(z_scores)) if z_scores else 0.0

        achrom = achromatic_residual_score(spectra[i], wavelength_grid)
        # Weighted combination: ratio deviations + achromatic score
        scores[i] = 0.7 * z_agg + 0.3 * achrom

    # Ensure non-negative
    scores = np.maximum(scores, 0.0)
    return scores
