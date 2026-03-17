"""Tests for multi-epoch variability search."""
import numpy as np
import pandas as pd
from src.features.multiepoch import (
    find_repeat_observations,
    epoch_variability_features,
    multiepoch_variability_scores,
    assign_multiepoch_scores,
)


def _make_metadata_with_repeats():
    """Create metadata where some stars have repeat observations."""
    return pd.DataFrame({
        "ra": [180.0, 180.0, 180.0, 200.0, 200.0, 250.0],
        "dec": [45.0, 45.0, 45.0, 30.0, 30.0, 10.0],
        "plate": [1000, 1001, 1002, 2000, 2001, 3000],
        "mjd": [55000, 55100, 55200, 55000, 55100, 55000],
        "fiberid": [100, 200, 300, 100, 200, 100],
        "subclass": ["G2", "G2", "G2", "K5", "K5", "F0"],
    })


def test_find_repeat_observations():
    meta = _make_metadata_with_repeats()
    groups = find_repeat_observations(meta)
    # Should find 2 groups: one with 3 members, one with 2
    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups)
    assert sizes == [2, 3]


def test_find_repeat_observations_no_repeats():
    meta = pd.DataFrame({
        "ra": [10.0, 20.0, 30.0],
        "dec": [10.0, 20.0, 30.0],
    })
    groups = find_repeat_observations(meta)
    assert len(groups) == 0


def test_find_repeat_observations_missing_coords():
    meta = pd.DataFrame({"plate": [1, 2, 3]})
    groups = find_repeat_observations(meta)
    assert len(groups) == 0


def test_epoch_variability_features_single_epoch():
    wl = np.linspace(3800, 9200, 500)
    spectra = np.ones((1, 500))
    features = epoch_variability_features(spectra, wl)
    assert features["n_epochs"] == 1
    assert features["variability_score"] == 0.0


def test_epoch_variability_features_identical_epochs():
    wl = np.linspace(3800, 9200, 500)
    spectra = np.ones((3, 500))
    features = epoch_variability_features(spectra, wl)
    assert features["max_achromatic_shift"] == 0.0
    assert features["variability_score"] == 0.0


def test_epoch_variability_features_achromatic_brightening():
    """Achromatic brightening should show high shift, low chromaticity."""
    wl = np.linspace(3800, 9200, 500)
    spectra = np.ones((2, 500))
    # Second epoch: uniform 20% brightening (achromatic)
    spectra[1] = 1.2

    features = epoch_variability_features(spectra, wl)
    assert features["max_achromatic_shift"] > 0.1
    # Chromaticity should be low (flat ratio)
    assert features["chromaticity"] < 0.01
    assert features["variability_score"] > 0


def test_epoch_variability_features_chromatic_change():
    """Chromatic change (reddening) should have high chromaticity."""
    wl = np.linspace(3800, 9200, 500)
    spectra = np.ones((2, 500))
    # Second epoch: wavelength-dependent change (reddening)
    spectra[1] = 1.0 + 0.3 * (wl - 3800) / (9200 - 3800)

    features = epoch_variability_features(spectra, wl)
    assert features["chromaticity"] > 0.05  # high chromaticity = not achromatic

    # Achromatic case with the SAME mean shift magnitude for fair comparison
    mean_ratio = float(np.mean(spectra[1]))  # ~1.15
    spectra_achrom = np.ones((2, 500))
    spectra_achrom[1] = mean_ratio  # same mean shift, but flat
    features_achrom = epoch_variability_features(spectra_achrom, wl)
    # Achromatic has same shift but lower chromaticity -> higher score
    assert features_achrom["chromaticity"] < features["chromaticity"]
    assert features_achrom["variability_score"] > features["variability_score"]


def test_multiepoch_variability_scores():
    meta = _make_metadata_with_repeats()
    n = len(meta)
    rng = np.random.default_rng(42)
    spectra = np.ones((n, 500)) + rng.normal(0, 0.01, (n, 500))

    df = multiepoch_variability_scores(spectra, meta)
    assert len(df) >= 1
    assert "variability_score" in df.columns
    assert "member_indices" in df.columns


def test_multiepoch_variability_scores_empty():
    meta = pd.DataFrame({"ra": [10.0], "dec": [10.0]})
    spectra = np.ones((1, 500))
    df = multiepoch_variability_scores(spectra, meta)
    assert len(df) == 0


def test_assign_multiepoch_scores():
    meta = _make_metadata_with_repeats()
    n = len(meta)
    spectra = np.ones((n, 500))
    # Make one group's spectra vary
    spectra[0] = 1.2  # different flux for first epoch of first group

    df = multiepoch_variability_scores(spectra, meta)
    scores = assign_multiepoch_scores(df, n)
    assert scores.shape == (n,)
    # The loner (index 5) should have score 0
    assert scores[5] == 0.0
    # Group members should share the same score
    if len(df) > 0:
        first_group = df.iloc[0]["member_indices"]
        group_scores = scores[first_group]
        assert len(set(group_scores)) == 1  # all same
