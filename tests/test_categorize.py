"""Tests for anomaly type categorization."""
import numpy as np
from src.features.categorize import categorize_anomalies, categorize_pbh_candidates


def test_categorize_returns_labels():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 500))
    pca_errors = rng.random(50)
    categories = categorize_anomalies(spectra, pca_errors)
    assert len(categories) == 50
    valid = {"continuum", "line", "noise", "normal"}
    assert all(c in valid for c in categories)


def test_categorize_noise_detection():
    rng = np.random.default_rng(42)
    spectra = np.zeros((10, 500))
    spectra[0] = rng.normal(0, 10, 500)  # very noisy
    spectra[1:] = rng.normal(0, 0.1, (9, 500))  # quiet
    pca_errors = np.zeros(10)
    pca_errors[0] = 0.5
    categories = categorize_anomalies(spectra, pca_errors)
    assert categories[0] == "noise"


def test_categorize_continuum_detection():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 0.1, (10, 500))
    pca_errors = np.zeros(10)
    pca_errors[0] = 100.0  # extreme PCA error -> continuum anomaly
    categories = categorize_anomalies(spectra, pca_errors)
    assert categories[0] == "continuum"


def test_categorize_pbh_candidates_basic():
    rng = np.random.default_rng(42)
    n = 100
    ml_scores = rng.random(n)
    acc_scores = rng.random(n)
    categories = categorize_pbh_candidates(ml_scores, acc_scores)
    assert len(categories) == n
    valid = {"microlensing_candidate", "accretion_candidate", "both_candidate", "none"}
    assert all(c in valid for c in categories)
    # At 95th percentile, ~5% should be flagged per score
    non_none = [c for c in categories if c != "none"]
    assert len(non_none) > 0


def test_categorize_pbh_candidates_both():
    """Spectrum with extreme scores in both should be 'both_candidate'."""
    n = 100
    ml_scores = np.zeros(n)
    acc_scores = np.zeros(n)
    # Make first spectrum extreme in both
    ml_scores[0] = 100.0
    acc_scores[0] = 100.0
    categories = categorize_pbh_candidates(ml_scores, acc_scores)
    assert categories[0] == "both_candidate"


def test_categorize_pbh_candidates_microlensing_only():
    n = 100
    ml_scores = np.zeros(n)
    acc_scores = np.zeros(n)
    ml_scores[0] = 100.0
    categories = categorize_pbh_candidates(ml_scores, acc_scores)
    assert categories[0] == "microlensing_candidate"


def test_categorize_pbh_candidates_accretion_only():
    n = 100
    ml_scores = np.zeros(n)
    acc_scores = np.zeros(n)
    acc_scores[0] = 100.0
    categories = categorize_pbh_candidates(ml_scores, acc_scores)
    assert categories[0] == "accretion_candidate"
