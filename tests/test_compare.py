import numpy as np
import pandas as pd
from src.models.compare import compare_anomaly_scores, rank_anomalies


def test_compare_anomaly_scores():
    if_scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
    ae_scores = np.array([0.2, 0.8, 0.4, 0.7, 0.1])

    result = compare_anomaly_scores(if_scores, ae_scores, top_n=2)
    assert "if_rank" in result.columns
    assert "ae_rank" in result.columns
    assert "agreed" in result.columns
    assert len(result) == 5


def test_rank_anomalies():
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    metadata = [{"filename": f"spec-{i}.fits"} for i in range(5)]

    ranked = rank_anomalies(scores, metadata, top_n=3)
    assert len(ranked) == 3
    assert ranked.iloc[0]["anomaly_score"] == 0.9
