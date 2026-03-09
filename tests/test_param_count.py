"""Tests for parameter counting across all models."""
import numpy as np
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import SpectralAutoencoder


def test_classical_param_count():
    detector = ClassicalAnomalyDetector(n_components=20)
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 500))
    detector.fit(spectra)
    count = detector.param_count()
    assert isinstance(count, int)
    assert count > 0


def test_autoencoder_param_count():
    model = SpectralAutoencoder(input_dim=500, bottleneck_dim=16)
    count = model.param_count()
    assert isinstance(count, int)
    assert count > 0
