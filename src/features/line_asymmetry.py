"""Line profile asymmetry scoring for microlensing detection.

Gravitational microlensing of a rotating star differentially magnifies
the approaching and receding photospheric limbs, producing asymmetric
distortions in absorption line profiles.  This module measures the
asymmetry of spectral lines and compares each spectrum to the population
of the same spectral class to identify outliers.

CAVEAT: Line asymmetry has many mundane astrophysical causes — binary
orbital motion, stellar pulsation, convective blueshift variations,
spectral misclassification, and instrument artifacts.  High asymmetry
scores are exploratory flags, not confirmations.
"""
import logging

import numpy as np
import pandas as pd

from src.features.spectral_lines import STELLAR_LINES, measure_line_profile

logger = logging.getLogger(__name__)


def _parse_broad_class(subclass: str) -> str:
    s = str(subclass).strip().upper()
    for char in s:
        if char in "OBAFGKM":
            return char
    return "unknown"


def compute_line_asymmetry_features(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Compute line profile asymmetry features for every spectrum.

    Returns a DataFrame with one row per spectrum and columns:
        broad_class, <line>_skewness, <line>_kurtosis, <line>_blue_red_ratio
    for each line in STELLAR_LINES.
    """
    subclasses = metadata["subclass"].values if "subclass" in metadata.columns else [""] * len(spectra)
    rows = []
    for i in range(len(spectra)):
        row = {"broad_class": _parse_broad_class(subclasses[i])}
        for name, info in STELLAR_LINES.items():
            prof = measure_line_profile(spectra[i], wavelength_grid, info["center"])
            row[f"{name}_skewness"] = prof["skewness"]
            row[f"{name}_kurtosis"] = prof["kurtosis"]
            row[f"{name}_blue_red_ratio"] = prof["blue_red_ratio"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_population_asymmetry_stats(
    asym_df: pd.DataFrame,
    groupby_col: str = "broad_class",
    min_count: int = 20,
) -> pd.DataFrame:
    """Compute per-class median and std of asymmetry features.

    Groups with fewer than ``min_count`` members fall back to global stats.
    """
    feature_cols = [c for c in asym_df.columns if c.endswith(("_skewness", "_kurtosis", "_blue_red_ratio"))]
    grouped = asym_df.groupby(groupby_col)

    global_median = asym_df[feature_cols].median()
    global_std = asym_df[feature_cols].std()
    global_std = global_std.where(global_std > 0, 1.0)

    records = []
    for cls, grp in grouped:
        if len(grp) >= min_count:
            med = grp[feature_cols].median()
            std = grp[feature_cols].std()
            std = std.where(std > 0, 1.0)
        else:
            med = global_median
            std = global_std
        record = {groupby_col: cls}
        for col in feature_cols:
            record[f"{col}_median"] = med[col]
            record[f"{col}_std"] = std[col]
        records.append(record)

    return pd.DataFrame(records)


def line_asymmetry_score(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    metadata: pd.DataFrame,
) -> np.ndarray:
    """Score each spectrum for anomalous line profile asymmetry.

    For each spectrum, computes z-scores of skewness and blue/red EW ratio
    relative to the spectral-class population, then aggregates across lines.

    Higher scores indicate more asymmetric line profiles than expected.

    Returns array of shape (n_spectra,) with non-negative scores.
    """
    asym_df = compute_line_asymmetry_features(spectra, wavelength_grid, metadata)
    pop_stats = build_population_asymmetry_stats(asym_df)

    # Build lookup
    lookup = {}
    for _, row in pop_stats.iterrows():
        lookup[row["broad_class"]] = row

    skew_cols = [c for c in asym_df.columns if c.endswith("_skewness")]
    ratio_cols = [c for c in asym_df.columns if c.endswith("_blue_red_ratio")]

    scores = np.zeros(len(spectra))
    for i in range(len(spectra)):
        cls = asym_df.iloc[i]["broad_class"]
        stats = lookup.get(cls)
        if stats is None:
            continue

        z_scores = []
        for col in skew_cols + ratio_cols:
            val = asym_df.iloc[i][col]
            med = stats.get(f"{col}_median", np.nan)
            std = stats.get(f"{col}_std", np.nan)
            if np.isfinite(val) and np.isfinite(med) and np.isfinite(std) and std > 0:
                z_scores.append(abs((val - med) / std))

        if z_scores:
            scores[i] = float(np.mean(z_scores))

    return np.maximum(scores, 0.0)
