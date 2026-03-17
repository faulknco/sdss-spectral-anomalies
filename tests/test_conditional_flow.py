"""Tests for conditional normalizing flow (MAF)."""
import numpy as np
import torch
from src.models.conditional_flow import ConditionalMAF, train_conditional_flow


def make_pca(n: int, dim: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, dim)).astype(np.float32)


def make_meta(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_log_prob_shape():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    pca = make_pca(16)
    meta = make_meta(16)
    log_probs = model.log_prob(pca, meta)
    assert log_probs.shape == (16,)
    assert np.all(np.isfinite(log_probs))


def test_anomaly_score_shape():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    pca = make_pca(16)
    meta = make_meta(16)
    scores = model.anomaly_score(pca, meta)
    assert scores.shape == (16,)
    assert np.all(np.isfinite(scores))


def test_anomaly_score_is_negative_log_prob():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    pca = make_pca(16)
    meta = make_meta(16)
    lp = model.log_prob(pca, meta)
    scores = model.anomaly_score(pca, meta)
    np.testing.assert_allclose(scores, -lp, atol=1e-5)


def test_param_count_positive():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    assert model.param_count() > 0


def test_train_reduces_loss():
    pca = make_pca(100)
    meta = make_meta(100)
    model, losses = train_conditional_flow(
        pca, meta, n_blocks=4, hidden_dim=64, epochs=5, batch_size=32, lr=1e-3
    )
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_handles_nan_metadata():
    model = ConditionalMAF(input_dim=50, context_dim=4, n_blocks=4, hidden_dim=64)
    pca = make_pca(8)
    meta = make_meta(8)
    meta[0, 0] = np.nan
    scores = model.anomaly_score(pca, meta)
    assert scores.shape == (8,)
    assert np.all(np.isfinite(scores))
