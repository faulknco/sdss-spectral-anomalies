import numpy as np
import torch
import pytest
from src.models.conditional_autoencoder import (
    ConditionalSpectralAutoencoder,
    train_conditional_autoencoder,
)


def make_meta(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 3)).astype(np.float32)


def test_forward_output_shape():
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=32)
    x = torch.randn(8, 1, 500)
    meta = torch.randn(8, 3)
    out = model(x, meta)
    assert out.shape == (8, 1, 500)


def test_encode_shape():
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=32)
    x = torch.randn(8, 1, 500)
    meta = torch.randn(8, 3)
    z = model.encode(x, meta)
    assert z.shape == (8, 32)


def test_param_count_positive():
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=32)
    assert model.param_count() > 0


def test_reconstruction_error_shape():
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=32)
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((16, 500)).astype(np.float32)
    meta = make_meta(16)
    errors = model.reconstruction_error(spectra, meta)
    assert errors.shape == (16,)
    assert np.all(errors >= 0)


def test_train_reduces_loss():
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((64, 500)).astype(np.float32)
    meta = make_meta(64)
    model, losses = train_conditional_autoencoder(
        spectra, meta, bottleneck_dim=16, epochs=5, batch_size=16, lr=1e-3
    )
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_handles_nan_metadata():
    rng = np.random.default_rng(0)
    spectra = rng.standard_normal((8, 500)).astype(np.float32)
    meta = make_meta(8)
    meta[0, 0] = np.nan
    model = ConditionalSpectralAutoencoder(input_dim=500, meta_dim=3, bottleneck_dim=16)
    errors = model.reconstruction_error(spectra, meta)
    assert errors.shape == (8,)
    assert np.all(np.isfinite(errors))
