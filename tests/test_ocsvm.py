"""Tests for OC-SVM anomaly detector."""
import numpy as np
from src.models.ocsvm import OCSVMDetector


def test_ocsvm_fit_score():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (100, 500))
    detector = OCSVMDetector(n_components=20)
    detector.fit(spectra)
    scores = detector.score(spectra)
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))


def test_ocsvm_anomalies_score_higher():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (90, 500))
    anomalous = rng.normal(0, 1, (10, 500)) + 10
    spectra = np.vstack([normal, anomalous])
    detector = OCSVMDetector(n_components=20)
    detector.fit(spectra)
    scores = detector.score(spectra)
    assert np.mean(scores[90:]) > np.mean(scores[:90])


def test_ocsvm_param_count():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 500))
    detector = OCSVMDetector(n_components=20)
    detector.fit(spectra)
    count = detector.param_count()
    assert isinstance(count, int)
    assert count > 0


def test_ocsvm_subsamples_large_data():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (200, 500))
    detector = OCSVMDetector(n_components=20, max_train_samples=50)
    detector.fit(spectra)
    scores = detector.score(spectra)
    assert scores.shape == (200,)
    assert np.all(np.isfinite(scores))
    assert detector.svm.support_vectors_.shape[0] <= 50
