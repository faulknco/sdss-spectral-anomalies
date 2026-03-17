"""Tests for retrieval evaluation metrics."""
import numpy as np
from src.features.evaluate_retrieval import evaluate_retrieval


def test_evaluate_retrieval_basic():
    labels = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.8, 0.3, 0.7, 0.05, 0.12, 0.11])
    log = [
        {"index": 3, "type": "emission_line", "params": {}},
        {"index": 4, "type": "continuum_tilt", "params": {}},
        {"index": 6, "type": "missing_band", "params": {}},
    ]
    result = evaluate_retrieval(labels, scores, log, top_k_list=[3, 5])
    assert "precision_at_k" in result
    assert "recall_at_k" in result
    assert "auroc" in result
    assert "auprc" in result
    assert "per_type_recall_at_k" in result
    assert 3 in result["precision_at_k"]
    assert 5 in result["recall_at_k"]


def test_evaluate_retrieval_perfect_scores():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([1.0, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    log = [
        {"index": 0, "type": "emission_line", "params": {}},
        {"index": 1, "type": "emission_line", "params": {}},
        {"index": 2, "type": "emission_line", "params": {}},
    ]
    result = evaluate_retrieval(labels, scores, log, top_k_list=[3])
    assert result["precision_at_k"][3] == 1.0
    assert result["recall_at_k"][3] == 1.0
    assert result["auroc"] > 0.99


def test_evaluate_retrieval_per_type():
    labels = np.zeros(20, dtype=int)
    scores = np.random.default_rng(42).random(20)
    labels[:4] = 1
    scores[:4] = [0.95, 0.90, 0.85, 0.80]
    log = [
        {"index": 0, "type": "emission_line", "params": {}},
        {"index": 1, "type": "emission_line", "params": {}},
        {"index": 2, "type": "continuum_tilt", "params": {}},
        {"index": 3, "type": "continuum_tilt", "params": {}},
    ]
    result = evaluate_retrieval(labels, scores, log, top_k_list=[5])
    assert "emission_line" in result["per_type_recall_at_k"]
    assert "continuum_tilt" in result["per_type_recall_at_k"]
