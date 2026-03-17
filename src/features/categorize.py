"""Categorize anomalies by type based on spectral properties."""
import numpy as np


def categorize_anomalies(
    spectra: np.ndarray,
    pca_errors: np.ndarray,
    noise_threshold_percentile: float = 90,
    continuum_threshold_percentile: float = 90,
    line_threshold_percentile: float = 90,
) -> list[str]:
    """Classify each spectrum into an anomaly category.

    Categories:
        - noise: high overall variance across wavelength
        - continuum: unusual overall shape (high PCA reconstruction error)
        - line: unusual local features (high local variance in residuals)
        - normal: none of the above
    """
    n = len(spectra)

    spectral_variance = np.var(spectra, axis=1)
    noise_thresh = np.percentile(spectral_variance, noise_threshold_percentile)

    continuum_thresh = np.percentile(pca_errors, continuum_threshold_percentile)

    diffs = np.diff(spectra, axis=1)
    local_variance = np.var(diffs, axis=1)
    line_thresh = np.percentile(local_variance, line_threshold_percentile)

    categories = []
    for i in range(n):
        if spectral_variance[i] > noise_thresh:
            categories.append("noise")
        elif pca_errors[i] > continuum_thresh:
            categories.append("continuum")
        elif local_variance[i] > line_thresh:
            categories.append("line")
        else:
            categories.append("normal")

    return categories


def categorize_pbh_candidates(
    microlensing_scores: np.ndarray,
    accretion_scores: np.ndarray,
    microlensing_threshold_pct: float = 95,
    accretion_threshold_pct: float = 95,
) -> list[str]:
    """Classify each spectrum as a PBH candidate based on physics-motivated scores.

    Categories:
        - microlensing_candidate: high microlensing score only
        - accretion_candidate: high accretion score only
        - both_candidate: high in both scores
        - none: below threshold in both
    """
    ml_thresh = np.percentile(microlensing_scores, microlensing_threshold_pct)
    acc_thresh = np.percentile(accretion_scores, accretion_threshold_pct)

    categories = []
    for ml, acc in zip(microlensing_scores, accretion_scores):
        ml_flag = ml > ml_thresh
        acc_flag = acc > acc_thresh
        if ml_flag and acc_flag:
            categories.append("both_candidate")
        elif ml_flag:
            categories.append("microlensing_candidate")
        elif acc_flag:
            categories.append("accretion_candidate")
        else:
            categories.append("none")

    return categories
