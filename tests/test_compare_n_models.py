"""Tests for N-model comparison."""
import numpy as np
from src.models.compare import compare_n_models


def test_compare_n_models_basic():
    rng = np.random.default_rng(42)
    scores = {
        "if": rng.random(100),
        "ae": rng.random(100),
        "ocsvm": rng.random(100),
        "dagmm": rng.random(100),
    }
    result = compare_n_models(scores, top_n=10)
    assert len(result) == 100
    for name in scores:
        assert f"{name}_score" in result.columns
        assert f"{name}_rank" in result.columns
    assert "n_models_agreed" in result.columns
    assert "combined_rank" in result.columns


def test_compare_n_models_agreement():
    scores = {}
    for name in ["a", "b", "c"]:
        s = np.zeros(50)
        s[:5] = 10.0
        scores[name] = s
    result = compare_n_models(scores, top_n=10)
    assert result["n_models_agreed"].iloc[0] == 3
