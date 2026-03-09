"""Split conformal calibration for anomaly scores."""
import numpy as np


class SplitConformalCalibrator:
    """Convert raw anomaly scores to conformal p-values.

    Usage:
        cal = SplitConformalCalibrator()
        cal.fit(calibration_scores)      # held-out scores; higher = more anomalous
        pvals = cal.pvalues(test_scores)
        thresh = cal.threshold(alpha=0.05)
    """

    def fit(self, calibration_scores: np.ndarray) -> "SplitConformalCalibrator":
        self.calibration_scores_ = np.array(calibration_scores, dtype=np.float64)
        return self

    def pvalues(self, test_scores: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
        if not hasattr(self, "calibration_scores_"):
            raise RuntimeError("Call fit() before pvalues()")
        test = np.array(test_scores, dtype=np.float64)
        m = len(self.calibration_scores_)
        counts = np.empty(len(test), dtype=np.int64)
        for start in range(0, len(test), chunk_size):
            chunk = test[start : start + chunk_size]
            counts[start : start + chunk_size] = (
                self.calibration_scores_[:, None] >= chunk[None, :]
            ).sum(axis=0)
        return (counts + 1) / (m + 1)

    def threshold(self, alpha: float = 0.05) -> float:
        if not hasattr(self, "calibration_scores_"):
            raise RuntimeError("Call fit() before threshold()")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        m = len(self.calibration_scores_)
        min_pvalue = 1.0 / (m + 1)
        if alpha < min_pvalue:
            return float("inf")

        sorted_scores = np.sort(self.calibration_scores_)
        rank = int(np.ceil((1.0 - alpha) * (m + 1)))
        rank = min(max(rank, 1), m)
        return float(sorted_scores[rank - 1])
