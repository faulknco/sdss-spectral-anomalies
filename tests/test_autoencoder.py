import numpy as np
import torch
from src.models.autoencoder import SpectralAutoencoder, train_autoencoder


def test_autoencoder_forward_shape():
    model = SpectralAutoencoder(input_dim=3500, bottleneck_dim=64)
    x = torch.randn(8, 1, 3500)
    reconstructed = model(x)
    assert reconstructed.shape == (8, 1, 3500)


def test_autoencoder_encode_shape():
    model = SpectralAutoencoder(input_dim=3500, bottleneck_dim=64)
    x = torch.randn(8, 1, 3500)
    latent = model.encode(x)
    assert latent.shape[0] == 8
    assert latent.shape[1] == 64


def test_train_autoencoder_reduces_loss():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (64, 3500)).astype(np.float32)

    model, losses = train_autoencoder(
        data, bottleneck_dim=32, epochs=5, batch_size=16, lr=1e-3
    )

    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_reconstruction_error():
    model = SpectralAutoencoder(input_dim=3500, bottleneck_dim=64)
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (16, 3500)).astype(np.float32)

    errors = model.reconstruction_error(data)
    assert errors.shape == (16,)
    assert np.all(errors >= 0)
