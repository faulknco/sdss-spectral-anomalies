"""Tests for review label I/O."""
import numpy as np
import pandas as pd
from pathlib import Path
from src.features.review_labels import (
    LABEL_CODES,
    LABEL_DISPLAY_NAMES,
    load_labels,
    save_labels,
)


def test_label_codes_and_display_names():
    assert len(LABEL_CODES) == 5
    assert set(LABEL_CODES) == {"artifact", "low_sn", "plausible_oddity", "known_rare", "unclear"}
    assert set(LABEL_DISPLAY_NAMES.keys()) == set(LABEL_CODES)
    for code in LABEL_CODES:
        assert isinstance(LABEL_DISPLAY_NAMES[code], str)


def test_load_labels_empty(tmp_path):
    df = load_labels(tmp_path / "nonexistent.parquet")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["filename", "label", "notes", "timestamp"]
    assert len(df) == 0


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "labels.parquet"
    df = pd.DataFrame({
        "filename": ["spec-0001-50000-0001.fits", "spec-0002-50000-0002.fits"],
        "label": ["artifact", "plausible_oddity"],
        "notes": ["bad pixel", ""],
        "timestamp": ["2026-03-17T12:00:00", "2026-03-17T12:01:00"],
    })
    save_labels(df, path)
    loaded = load_labels(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_save_labels_overwrites(tmp_path):
    path = tmp_path / "labels.parquet"
    df1 = pd.DataFrame({
        "filename": ["a.fits"],
        "label": ["artifact"],
        "notes": [""],
        "timestamp": ["2026-03-17T12:00:00"],
    })
    df2 = pd.DataFrame({
        "filename": ["b.fits"],
        "label": ["unclear"],
        "notes": ["check later"],
        "timestamp": ["2026-03-17T13:00:00"],
    })
    save_labels(df1, path)
    save_labels(df2, path)
    loaded = load_labels(path)
    pd.testing.assert_frame_equal(loaded, df2)


def test_load_labels_preserves_types(tmp_path):
    path = tmp_path / "labels.parquet"
    df = pd.DataFrame({
        "filename": ["spec.fits"],
        "label": ["low_sn"],
        "notes": ["noisy"],
        "timestamp": ["2026-03-17T12:00:00"],
    })
    save_labels(df, path)
    loaded = load_labels(path)
    assert loaded["label"].dtype == object
    assert loaded["notes"].dtype == object
    assert loaded["timestamp"].dtype == object
