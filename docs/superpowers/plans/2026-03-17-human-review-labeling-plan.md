# Human Review Labeling Workflow -- Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a labeling workflow to the Focused Review tab so reviewers can classify top anomaly candidates and persist those labels in a parquet file.

**Architecture:** New `src/features/review_labels.py` module owns label constants, load/save I/O. Dashboard Tab 6 (Focused Review) gets a label selectbox, notes input, and sidebar save button using Streamlit session state. Pipeline merges existing labels into focused_review output on re-run.

**Tech Stack:** Python, pandas, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-03-17-human-review-labeling-design.md`

---

## Chunk 1: Label Storage Module and Dashboard Integration

### Task 1: Review Labels Module

**Files:**
- Create: `src/features/review_labels.py`
- Create: `tests/test_review_labels.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_review_labels.py
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
    assert loaded["label"].dtype == object  # string
    assert loaded["notes"].dtype == object
    assert loaded["timestamp"].dtype == object
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_review_labels.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement review labels module**

```python
# src/features/review_labels.py
"""Load and save human review labels for anomaly candidates."""
from pathlib import Path

import pandas as pd

LABEL_CODES: list[str] = [
    "artifact",
    "low_sn",
    "plausible_oddity",
    "known_rare",
    "unclear",
]

LABEL_DISPLAY_NAMES: dict[str, str] = {
    "artifact": "Instrument/Reduction Artifact",
    "low_sn": "Low S/N Nuisance",
    "plausible_oddity": "Astrophysically Plausible Oddity",
    "known_rare": "Known Rare Subtype",
    "unclear": "Unclear",
}

_SCHEMA_COLUMNS = ["filename", "label", "notes", "timestamp"]


def load_labels(path: Path) -> pd.DataFrame:
    """Read review labels from parquet, or return an empty DataFrame if the file does not exist."""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_SCHEMA_COLUMNS)


def save_labels(df: pd.DataFrame, path: Path) -> None:
    """Write review labels to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_review_labels.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/features/review_labels.py tests/test_review_labels.py
git commit -m "feat: add review labels module with load/save and label constants"
```

---

### Task 2: Dashboard Labeling UI

**Files:**
- Modify: `src/dashboard/app.py`

- [ ] **Step 1: Read current dashboard**

Read `src/dashboard/app.py` to understand the Tab 6 (Focused Review) structure.

- [ ] **Step 2: Add import and session state initialization**

Add import after existing imports:

```python
from src.features.review_labels import LABEL_CODES, LABEL_DISPLAY_NAMES, load_labels, save_labels
```

In `main()`, after the `load_data()` call and before the tabs, add label state initialization:

```python
    # Load review labels into session state (uncached -- labels change during session)
    labels_path = RESULTS_DIR / "review_labels.parquet"
    if "review_labels" not in st.session_state:
        labels_df = load_labels(labels_path)
        st.session_state["review_labels"] = {
            row["filename"]: {"label": row["label"], "notes": row["notes"], "timestamp": row["timestamp"]}
            for _, row in labels_df.iterrows()
        }
```

- [ ] **Step 3: Add Save All Labels button to sidebar**

After the existing sidebar param counts block, add:

```python
    st.sidebar.divider()
    st.sidebar.header("Review Labels")
    if st.sidebar.button("Save All Labels"):
        rows = [
            {"filename": fn, **data}
            for fn, data in st.session_state.get("review_labels", {}).items()
        ]
        if rows:
            save_labels(pd.DataFrame(rows), labels_path)
            st.sidebar.success(f"Saved {len(rows)} labels")
        else:
            st.sidebar.info("No labels to save")
```

- [ ] **Step 4: Add labeling progress summary to Tab 6**

At the top of the `with tab6:` block, before the existing subheader, add:

```python
        # Labeling progress
        review_labels = st.session_state.get("review_labels", {})
        if not focused_review.empty:
            n_candidates = len(focused_review)
            n_labeled = sum(1 for fn in focused_review["filename"] if fn in review_labels)
            st.progress(n_labeled / max(n_candidates, 1))
            label_counts = {}
            for fn in focused_review["filename"]:
                if fn in review_labels:
                    lbl = review_labels[fn]["label"]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            summary_parts = [f"{n_labeled}/{n_candidates} labeled"]
            for code in LABEL_CODES:
                if code in label_counts:
                    summary_parts.append(f"{label_counts[code]} {LABEL_DISPLAY_NAMES[code]}")
            st.caption(" | ".join(summary_parts))
```

- [ ] **Step 5: Add label selectbox and notes input to candidate inspector**

Inside Tab 6, after the existing neighbor spectra display and before the RA/Dec markdown, add:

```python
            # --- Labeling controls ---
            st.divider()
            st.subheader("Label This Candidate")
            candidate_filename = candidate["filename"]
            current = review_labels.get(candidate_filename, {})
            current_label = current.get("label", "")

            display_options = ["(unlabeled)"] + [LABEL_DISPLAY_NAMES[c] for c in LABEL_CODES]
            code_for_display = {v: k for k, v in LABEL_DISPLAY_NAMES.items()}
            current_display = LABEL_DISPLAY_NAMES.get(current_label, "(unlabeled)")
            current_index = display_options.index(current_display) if current_display in display_options else 0

            selected_display = st.selectbox(
                "Classification",
                display_options,
                index=current_index,
                key=f"label_{candidate_filename}",
            )

            notes = st.text_input(
                "Notes",
                value=current.get("notes", ""),
                key=f"notes_{candidate_filename}",
            )

            from datetime import datetime, timezone
            if selected_display == "(unlabeled)":
                if candidate_filename in review_labels:
                    del st.session_state["review_labels"][candidate_filename]
            else:
                selected_code = code_for_display[selected_display]
                st.session_state["review_labels"][candidate_filename] = {
                    "label": selected_code,
                    "notes": notes,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
```

- [ ] **Step 6: Run all tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/ -v --ignore=tests/test_integration.py -x`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat: add labeling UI to Focused Review tab with session state and sidebar save"
```

---

### Task 3: Pipeline Label Merge

**Files:**
- Modify: `src/run_pipeline.py`

- [ ] **Step 1: Read the current pipeline**

Read `src/run_pipeline.py`, find the `focused_review.to_parquet(...)` line.

- [ ] **Step 2: Add label merge after focused_review is written**

After the line `focused_review.to_parquet(RESULTS_DIR / "focused_review.parquet")`, add:

```python
    # Merge existing review labels if available
    labels_path = RESULTS_DIR / "review_labels.parquet"
    if labels_path.exists():
        labels_df = pd.read_parquet(labels_path)
        for col in labels_df.columns:
            if col != "filename" and col in focused_review.columns:
                focused_review = focused_review.drop(columns=[col])
        focused_review = focused_review.merge(labels_df, on="filename", how="left")
        focused_review.to_parquet(RESULTS_DIR / "focused_review.parquet")
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/ -v --ignore=tests/test_integration.py -x`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/run_pipeline.py
git commit -m "feat: merge existing review labels into focused_review on pipeline re-run"
```
