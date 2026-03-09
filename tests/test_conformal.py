import numpy as np
import pytest
from src.models.conformal import SplitConformalCalibrator


def test_fit_stores_calibration_scores():
    cal = SplitConformalCalibrator()
    cal.fit(np.random.default_rng(0).standard_normal(100))
    assert len(cal.calibration_scores_) == 100


def test_pvalues_shape():
    rng = np.random.default_rng(0)
    cal = SplitConformalCalibrator()
    cal.fit(rng.standard_normal(200))
    pvals = cal.pvalues(rng.standard_normal(50))
    assert pvals.shape == (50,)


def test_pvalues_in_unit_interval():
    rng = np.random.default_rng(0)
    cal = SplitConformalCalibrator()
    cal.fit(rng.standard_normal(200))
    pvals = cal.pvalues(rng.standard_normal(50))
    assert np.all(pvals >= 0) and np.all(pvals <= 1)


def test_extreme_outlier_gets_low_pvalue():
    rng = np.random.default_rng(0)
    cal = SplitConformalCalibrator()
    cal.fit(rng.standard_normal(500))
    pvals = cal.pvalues(np.array([100.0]))
    assert pvals[0] < 0.01


def test_typical_inlier_gets_high_pvalue():
    rng = np.random.default_rng(0)
    cal_scores = rng.standard_normal(500)
    cal = SplitConformalCalibrator()
    cal.fit(cal_scores)
    pvals = cal.pvalues(np.array([float(np.median(cal_scores))]))
    assert pvals[0] > 0.3


def test_raises_if_not_fitted():
    cal = SplitConformalCalibrator()
    with pytest.raises(RuntimeError, match="fit"):
        cal.pvalues(np.array([1.0]))


def test_threshold_raises_if_not_fitted():
    cal = SplitConformalCalibrator()
    with pytest.raises(RuntimeError, match="fit"):
        cal.threshold()


def test_threshold_returns_float():
    rng = np.random.default_rng(0)
    cal = SplitConformalCalibrator()
    cal.fit(rng.standard_normal(500))
    assert isinstance(cal.threshold(alpha=0.05), float)


def test_threshold_matches_pvalue_decision_boundary():
    cal = SplitConformalCalibrator().fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    threshold = cal.threshold(alpha=0.2)
    assert threshold == 5.0
    assert cal.pvalues(np.array([threshold]))[0] > 0.2
    assert cal.pvalues(np.array([threshold + 1e-6]))[0] <= 0.2


def test_threshold_can_be_infinite_when_alpha_too_small():
    cal = SplitConformalCalibrator().fit(np.array([1.0, 2.0, 3.0]))
    assert np.isinf(cal.threshold(alpha=0.01))
