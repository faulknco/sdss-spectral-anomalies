"""Compare anomaly scores from multiple models."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_anomalies(
    scores: np.ndarray,
    metadata: list[dict],
    top_n: int = 100,
) -> pd.DataFrame:
    """Rank spectra by anomaly score, return top N."""
    df = pd.DataFrame(metadata)
    df["anomaly_score"] = scores
    df = df.sort_values("anomaly_score", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def compare_anomaly_scores(
    if_scores: np.ndarray,
    ae_scores: np.ndarray,
    top_n: int = 100,
) -> pd.DataFrame:
    """Compare Isolation Forest and Autoencoder anomaly rankings."""
    if_ranks = rankdata(-if_scores, method="ordinal")
    ae_ranks = rankdata(-ae_scores, method="ordinal")

    df = pd.DataFrame({
        "if_score": if_scores,
        "ae_score": ae_scores,
        "if_rank": if_ranks,
        "ae_rank": ae_ranks,
    })

    df["agreed"] = (df["if_rank"] <= top_n) & (df["ae_rank"] <= top_n)
    df["combined_rank"] = (df["if_rank"] + df["ae_rank"]) / 2

    return df
