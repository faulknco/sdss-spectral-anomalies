"""Tests for Conditional VAE."""
import numpy as np
import torch
from src.models.cvae import ConditionalVAE, train_cvae


def make_meta(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_forward_output_shape():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=32)
    x = torch.randn(8, 1, 500)
    meta = torch.randn(8, 4)
    recon, mu, log_var = model(x, meta)
    assert recon.shape == (8, 1, 500)
    assert mu.shape == (8, 32)
    assert log_var.shape == (8, 32)


def test_encode_returns_mu_logvar():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=32)
    x = torch.randn(8, 1, 500)
    meta = torch.randn(8, 4)
    mu, log_var = model.encode(x, meta)
    assert mu.shape == (8, 32)
    assert log_var.shape == (8, 32)


def test_param_count_positive():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=32)
    assert model.param_count() > 0


def test_anomaly_score_shape():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=32)
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((16, 500)).astype(np.float32)
    meta = make_meta(16)
    scores = model.anomaly_score(spectra, meta)
    assert scores.shape == (16,)
    assert np.all(np.isfinite(scores))


def test_anomaly_score_nonnegative():
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=16)
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((16, 500)).astype(np.float32)
    meta = make_meta(16)
    scores = model.anomaly_score(spectra, meta)
    assert np.all(scores >= -0.1)


def test_train_reduces_loss():
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((64, 500)).astype(np.float32)
    meta = make_meta(64)
    model, losses = train_cvae(
        spectra, meta, bottleneck_dim=16, epochs=5, batch_size=16, lr=1e-3,
        beta_warmup_epochs=0,
    )
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_handles_nan_metadata():
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((8, 500)).astype(np.float32)
    meta = make_meta(8)
    meta[0, 0] = np.nan
    model = ConditionalVAE(input_dim=500, meta_dim=4, bottleneck_dim=16)
    scores = model.anomaly_score(spectra, meta)
    assert scores.shape == (8,)
    assert np.all(np.isfinite(scores))
