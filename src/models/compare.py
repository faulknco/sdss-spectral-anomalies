"""Compare anomaly scores from multiple models."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def adaptive_top_n(
    n_samples: int,
    min_top_n: int = 25,
    max_top_n: int = 100,
    fraction: float = 0.1,
) -> int:
    """Choose a comparison cutoff that scales with dataset size."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    scaled = int(np.ceil(n_samples * fraction))
    return min(max_top_n, max(min_top_n, scaled, 1), n_samples)


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


def compare_n_models(
    scores_dict: dict[str, np.ndarray],
    top_n: int = 100,
) -> pd.DataFrame:
    """Compare anomaly rankings across N models."""
    df = pd.DataFrame()
    rank_cols = []

    for name, scores in scores_dict.items():
        df[f"{name}_score"] = scores
        ranks = rankdata(-scores, method="ordinal")
        df[f"{name}_rank"] = ranks
        rank_cols.append(f"{name}_rank")

    df["n_models_agreed"] = sum(
        (df[col] <= top_n).astype(int) for col in rank_cols
    )
    df["combined_rank"] = df[rank_cols].mean(axis=1)

    return df
