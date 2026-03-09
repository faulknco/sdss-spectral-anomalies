"""Multi-seed stability runner for anomaly detection models."""
import numpy as np
from typing import Callable


def stability_run(
    train_and_score_fn: Callable[[np.ndarray, int], np.ndarray],
    spectra: np.ndarray,
    n_runs: int = 5,
    base_seed: int = 42,
) -> dict:
    all_scores = []
    for i in range(n_runs):
        seed = base_seed + i
        scores = train_and_score_fn(spectra, seed)
        all_scores.append(scores)

    all_scores = np.array(all_scores)
    return {
        "mean_scores": all_scores.mean(axis=0),
        "std_scores": all_scores.std(axis=0),
        "all_scores": all_scores,
    }
