import numpy as np
from src.models.classical import ClassicalAnomalyDetector


def test_fit_and_score():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (100, 200))
    anomalous = rng.normal(10, 5, (5, 200))
    data = np.vstack([normal, anomalous])

    detector = ClassicalAnomalyDetector(n_components=20, contamination=0.05)
    detector.fit(data)
    scores = detector.score(data)

    assert scores.shape == (105,)
    normal_mean = np.mean(scores[:100])
    anomalous_mean = np.mean(scores[100:])
    assert anomalous_mean > normal_mean


def test_pca_reconstruction_error():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (50, 200))

    detector = ClassicalAnomalyDetector(n_components=20)
    detector.fit(data)
    errors = detector.reconstruction_error(data)

    assert errors.shape == (50,)
    assert np.all(errors >= 0)


def test_get_pca_components():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (50, 200))

    detector = ClassicalAnomalyDetector(n_components=20)
    detector.fit(data)
    components = detector.transform(data)

    assert components.shape == (50, 20)
