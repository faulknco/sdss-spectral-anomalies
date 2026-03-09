"""Tests for DAGMM anomaly detector."""
import numpy as np
from src.models.dagmm import DAGMM, train_dagmm


def test_dagmm_forward_shape():
    import torch
    model = DAGMM(input_dim=500, latent_dim=16, n_gmm=4)
    x = torch.randn(32, 500)
    x_hat, z, gamma = model(x)
    assert x_hat.shape == (32, 500)
    assert gamma.shape[0] == 32
    assert gamma.shape[1] == 4


def test_dagmm_train_and_score():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (100, 500)).astype(np.float32)
    model = train_dagmm(spectra, latent_dim=8, n_gmm=3, epochs=3, batch_size=32)
    scores = model.anomaly_score(spectra)
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))


def test_dagmm_anomalies_score_higher():
    import torch
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (90, 500)).astype(np.float32)
    anomalous = (rng.normal(0, 1, (10, 500)) + 10).astype(np.float32)
    spectra = np.vstack([normal, anomalous])
    model = train_dagmm(spectra, latent_dim=8, n_gmm=3, epochs=30, batch_size=32)
    scores = model.anomaly_score(spectra)
    assert np.mean(scores[90:]) > np.mean(scores[:90])


def test_dagmm_param_count():
    model = DAGMM(input_dim=500, latent_dim=16, n_gmm=4)
    count = model.param_count()
    assert isinstance(count, int)
    assert count > 0
