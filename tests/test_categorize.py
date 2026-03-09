"""Tests for anomaly type categorization."""
import numpy as np
from src.features.categorize import categorize_anomalies


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
