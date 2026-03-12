# tests/test_normalizing_flow.py
"""Tests for conditional RealNVP normalizing flow."""
import numpy as np
import pytest
from src.models.normalizing_flow import train_normalizing_flow

FAST_CONFIG = dict(
    n_components=10,
    n_coupling=4,
    hidden_dim=32,
    meta_embed_dim=8,
    epochs=3,
    batch_size=32,
)


def _spectra(n=100, bins=500, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, bins)).astype(np.float32)


def _meta(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 4)).astype(np.float32)


def test_flow_nll_shape():
    spectra = _spectra()
    meta = _meta()
    model, losses = train_normalizing_flow(spectra, meta, **FAST_CONFIG)
    scores = model.nll_score(spectra, meta)
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))
    assert len(losses) == FAST_CONFIG["epochs"]


def test_flow_anomalies_score_higher():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (90, 500)).astype(np.float32)
    anomalous = rng.normal(0, 1, (10, 500)).astype(np.float32) + 10.0
    meta_normal = rng.normal(0, 1, (90, 4)).astype(np.float32)
    meta_anomalous = rng.normal(0, 1, (10, 4)).astype(np.float32)

    spectra = np.vstack([normal, anomalous])
    meta = np.vstack([meta_normal, meta_anomalous])

    model, _ = train_normalizing_flow(spectra, meta, **FAST_CONFIG)
    scores = model.nll_score(spectra, meta)
    assert np.mean(scores[90:]) > np.mean(scores[:90])


def test_flow_param_count():
    model, _ = train_normalizing_flow(_spectra(50), _meta(50), **FAST_CONFIG)
    count = model.param_count()
    assert isinstance(count, int)
    assert count > 0


def test_flow_handles_nan_metadata():
    spectra = _spectra()
    meta = _meta()
    meta[0, 0] = float("nan")
    meta[5, 2] = float("nan")
    model, _ = train_normalizing_flow(spectra, meta, **FAST_CONFIG)
    scores = model.nll_score(spectra, meta)
    assert np.all(np.isfinite(scores))
