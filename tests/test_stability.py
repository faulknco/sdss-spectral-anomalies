"""Tests for multi-seed stability runner."""
import numpy as np
from src.models.stability import stability_run


def _mock_train_fn(spectra, seed):
    rng = np.random.default_rng(seed)
    return rng.random(len(spectra))


def test_stability_run_shape():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 100))
    result = stability_run(_mock_train_fn, spectra, n_runs=5)
    assert result["mean_scores"].shape == (50,)
    assert result["std_scores"].shape == (50,)
    assert result["all_scores"].shape == (5, 50)


def test_stability_run_std_nonzero():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 100))
    result = stability_run(_mock_train_fn, spectra, n_runs=5)
    assert np.any(result["std_scores"] > 0)


def test_stability_run_single_run():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (50, 100))
    result = stability_run(_mock_train_fn, spectra, n_runs=1)
    assert result["all_scores"].shape == (1, 50)
    assert np.allclose(result["std_scores"], 0)
