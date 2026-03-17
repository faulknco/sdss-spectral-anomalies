"""Multi-epoch variability search for transient microlensing events.

SDSS observed some sky regions on multiple plates/MJDs.  Stars with repeat
observations can be checked for epoch-to-epoch spectral changes consistent
with achromatic (wavelength-independent) brightening — the hallmark of
gravitational microlensing by a compact object such as a PBH.

This module works entirely from local data: it groups already-downloaded
spectra by sky position to find repeat observations, then scores the
variability pattern of each multi-epoch group.
"""
import logging

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.data.preprocess import DEFAULT_GRID

logger = logging.getLogger(__name__)

# Two SDSS pointings within this radius (arcsec) are treated as the same star.
_MATCH_RADIUS_ARCSEC = 2.0


def find_repeat_observations(
    metadata: pd.DataFrame,
    match_radius_arcsec: float = _MATCH_RADIUS_ARCSEC,
) -> list[list[int]]:
    """Group spectra by sky position to find repeat observations.

    Returns a list of groups, where each group is a list of row indices
    into *metadata* that share the same sky position (within
    ``match_radius_arcsec``).  Only groups with >= 2 members are returned.
    """
    if "ra" not in metadata.columns or "dec" not in metadata.columns:
        return []

    ra = metadata["ra"].values.astype(np.float64)
    dec = metadata["dec"].values.astype(np.float64)

    # Quick angular separation in arcsec (small-angle approx OK for <10")
    match_radius_deg = match_radius_arcsec / 3600.0
    assigned = np.full(len(metadata), -1, dtype=int)
    groups: list[list[int]] = []

    for i in range(len(metadata)):
        if not np.isfinite(ra[i]) or not np.isfinite(dec[i]):
            continue
        if assigned[i] >= 0:
            continue

        # Find all unassigned neighbours within radius
        cos_dec = np.cos(np.radians(dec[i]))
        dra = (ra - ra[i]) * cos_dec
        ddec = dec - dec[i]
        sep = np.sqrt(dra**2 + ddec**2)

        members = np.where((sep < match_radius_deg) & (assigned < 0))[0]
        if len(members) < 2:
            # Mark as assigned to avoid re-checking, but don't form a group
            assigned[i] = len(groups)
            continue

        group_id = len(groups)
        for idx in members:
            assigned[idx] = group_id
        groups.append(members.tolist())

    logger.info("Found %d multi-epoch groups from %d spectra", len(groups), len(metadata))
    return groups


def _spectral_ratio(spectrum_a: np.ndarray, spectrum_b: np.ndarray) -> np.ndarray:
    """Element-wise ratio spectrum_a / spectrum_b, masking zeros."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = spectrum_a / spectrum_b
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def epoch_variability_features(
    spectra_group: np.ndarray,
    wavelength_grid: np.ndarray,
) -> dict:
    """Compute variability features for a group of repeat observations.

    Parameters
    ----------
    spectra_group : array of shape (n_epochs, n_wavelengths)
        Median-normalized spectra for the same star at different epochs.
    wavelength_grid : array of shape (n_wavelengths,)

    Returns
    -------
    dict with keys:
        - max_achromatic_shift: largest epoch-pair mean flux ratio deviation from 1
        - chromaticity: std of the ratio spectrum across wavelength
          (low = achromatic = consistent with lensing)
        - n_epochs: number of epochs
        - variability_score: combined score (higher = more microlensing-like)
    """
    n_epochs = spectra_group.shape[0]
    if n_epochs < 2:
        return {
            "max_achromatic_shift": 0.0,
            "chromaticity": np.nan,
            "n_epochs": n_epochs,
            "variability_score": 0.0,
        }

    max_shift = 0.0
    min_chromaticity = np.inf

    for i in range(n_epochs):
        for j in range(i + 1, n_epochs):
            ratio = _spectral_ratio(spectra_group[i], spectra_group[j])
            valid = np.isfinite(ratio)
            if valid.sum() < 10:
                continue

            mean_ratio = float(np.nanmean(ratio[valid]))
            shift = abs(mean_ratio - 1.0)
            chrom = float(np.nanstd(ratio[valid]))

            if shift > max_shift:
                max_shift = shift
            if chrom < min_chromaticity:
                min_chromaticity = chrom

    if not np.isfinite(min_chromaticity):
        min_chromaticity = np.nan

    # Score: large achromatic shift + low chromaticity = lensing-like
    # Use shift / (chromaticity + epsilon) so perfectly achromatic scores highest
    epsilon = 1e-6
    if np.isfinite(min_chromaticity):
        variability_score = max_shift / (min_chromaticity + epsilon)
    else:
        variability_score = max_shift

    return {
        "max_achromatic_shift": float(max_shift),
        "chromaticity": float(min_chromaticity) if np.isfinite(min_chromaticity) else np.nan,
        "n_epochs": n_epochs,
        "variability_score": float(variability_score),
    }


def multiepoch_variability_scores(
    spectra: np.ndarray,
    metadata: pd.DataFrame,
    wavelength_grid: np.ndarray = DEFAULT_GRID,
    match_radius_arcsec: float = _MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Score all multi-epoch groups for transient microlensing variability.

    Returns a DataFrame with one row per multi-epoch group, columns:
        group_id, member_indices, n_epochs, max_achromatic_shift,
        chromaticity, variability_score, representative_idx
    Sorted by variability_score descending.
    """
    groups = find_repeat_observations(metadata, match_radius_arcsec)

    rows = []
    for gid, members in enumerate(groups):
        group_spectra = spectra[members]
        features = epoch_variability_features(group_spectra, wavelength_grid)
        rows.append({
            "group_id": gid,
            "member_indices": members,
            "n_epochs": features["n_epochs"],
            "max_achromatic_shift": features["max_achromatic_shift"],
            "chromaticity": features["chromaticity"],
            "variability_score": features["variability_score"],
            "representative_idx": members[0],
        })

    if not rows:
        return pd.DataFrame(columns=[
            "group_id", "member_indices", "n_epochs", "max_achromatic_shift",
            "chromaticity", "variability_score", "representative_idx",
        ])

    df = pd.DataFrame(rows)
    return df.sort_values("variability_score", ascending=False).reset_index(drop=True)


def assign_multiepoch_scores(
    variability_df: pd.DataFrame,
    n_spectra: int,
) -> np.ndarray:
    """Map group-level variability scores back to per-spectrum scores.

    Spectra without repeat observations get score 0.
    """
    scores = np.zeros(n_spectra)
    for _, row in variability_df.iterrows():
        for idx in row["member_indices"]:
            scores[idx] = row["variability_score"]
    return scores
