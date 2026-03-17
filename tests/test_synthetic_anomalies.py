"""Tests for semi-synthetic anomaly injection."""
import numpy as np
from src.features.synthetic_anomalies import inject_anomalies, ANOMALY_TYPES


def make_spectra(n=100, n_wl=500):
    rng = np.random.default_rng(42)
    spectra = rng.normal(1.0, 0.05, (n, n_wl))
    wl = np.linspace(3800, 9200, n_wl)
    return spectra, wl


def test_inject_anomalies_output_shapes():
    spectra, wl = make_spectra()
    modified, labels, log = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    assert modified.shape == spectra.shape
    assert labels.shape == (100,)
    assert set(labels).issubset({0, 1})
    assert sum(labels) == 10


def test_inject_anomalies_modifies_only_labeled():
    spectra, wl = make_spectra()
    modified, labels, log = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    clean_mask = labels == 0
    np.testing.assert_array_equal(modified[clean_mask], spectra[clean_mask])
    injected_mask = labels == 1
    assert not np.allclose(modified[injected_mask], spectra[injected_mask])


def test_inject_anomalies_log_has_all_types():
    spectra, wl = make_spectra(n=500)
    _, _, log = inject_anomalies(spectra, wl, fraction=0.2, seed=42)
    types_seen = {entry["type"] for entry in log}
    assert types_seen == set(ANOMALY_TYPES)


def test_inject_anomalies_log_structure():
    spectra, wl = make_spectra()
    _, _, log = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    assert len(log) == 10
    for entry in log:
        assert "index" in entry
        assert "type" in entry
        assert "params" in entry
        assert entry["type"] in ANOMALY_TYPES


def test_inject_anomalies_deterministic():
    spectra, wl = make_spectra()
    m1, l1, _ = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    m2, l2, _ = inject_anomalies(spectra, wl, fraction=0.1, seed=42)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(l1, l2)


def test_inject_anomalies_zero_fraction():
    spectra, wl = make_spectra()
    modified, labels, log = inject_anomalies(spectra, wl, fraction=0.0, seed=42)
    np.testing.assert_array_equal(modified, spectra)
    assert sum(labels) == 0
    assert len(log) == 0
