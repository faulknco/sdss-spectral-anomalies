# Spec B: Human Review Labeling Workflow

**Date:** 2026-03-17
**Status:** Approved design, pending implementation plan

## Goal

Add a labeling workflow to the Focused Review tab so reviewers can classify top anomaly candidates and persist those labels for analysis.

## Approach

Integrate label controls into the existing Focused Review tab (Tab 6). Labels are stored in a parquet file and held in Streamlit session state during review. A sidebar save button persists all labels at once.

## 1. Label Storage

**File:** `data/results/review_labels.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Primary key -- matches spectra_metadata filename |
| `label` | str | One of: `artifact`, `low_sn`, `plausible_oddity`, `known_rare`, `unclear` |
| `notes` | str | Free-text reviewer notes (empty string if none) |
| `timestamp` | str | ISO 8601 timestamp of last label update |

Short codes for programmatic use. The dashboard maps these to display names:

| Code | Display Name |
|------|-------------|
| `artifact` | Instrument/Reduction Artifact |
| `low_sn` | Low S/N Nuisance |
| `plausible_oddity` | Astrophysically Plausible Oddity |
| `known_rare` | Known Rare Subtype |
| `unclear` | Unclear |

**New file:** `src/features/review_labels.py`

Constants:
- `LABEL_CODES` -- list of valid label codes: `["artifact", "low_sn", "plausible_oddity", "known_rare", "unclear"]`
- `LABEL_DISPLAY_NAMES` -- dict mapping codes to display names (single authoritative source)

Functions:
- `load_labels(path) -> pd.DataFrame` -- reads parquet if it exists, returns empty DataFrame with the correct schema if not.
- `save_labels(df, path)` -- writes DataFrame to parquet.

This keeps I/O logic and label vocabulary out of the dashboard and makes it testable.

## 2. Dashboard UI Changes

**Modified file:** `src/dashboard/app.py` -- changes only in Tab 6 (Focused Review).

### Label State Management

On dashboard load (outside `load_data()`, uncached), `load_labels()` reads existing labels into `st.session_state["review_labels"]` (a dict mapping `filename -> {label, notes, timestamp}`). This is called separately from the cached `load_data()` because labels change during a session and must not be cached.

### UI Additions to Candidate Inspector

After the existing neighbor spectra and metadata display:

1. **Label selectbox** -- options built from `LABEL_DISPLAY_NAMES` with `"(unlabeled)"` prepended. Pre-selects the current label if one exists. Selecting `"(unlabeled)"` removes the label from session state (the candidate is treated as not yet reviewed).

2. **Notes text input** -- single-line `st.text_input` pre-filled with existing notes.

3. **Save All Labels button** -- in the sidebar, not per-candidate. Writes the full session state to `review_labels.parquet` via `save_labels()`. Shows a success toast.

4. **Labeling progress summary** -- at the top of Focused Review tab, before the candidate table. Shows: "Labeled: X / Y candidates" with a breakdown by category.

### Interaction Flow

Reviewer selects a candidate from the focus_rank selectbox (already exists), sees the full profile, picks a label, optionally types a note. The label is stored in session state immediately. When done reviewing, clicks "Save All Labels" in the sidebar to persist.

## 3. Pipeline Integration

Minimal. The pipeline does not use labels during training.

One addition to `src/run_pipeline.py`: after `focused_review.parquet` is written, if `review_labels.parquet` exists, merge labels into the focused review output:

```python
labels_path = RESULTS_DIR / "review_labels.parquet"
if labels_path.exists():
    labels_df = pd.read_parquet(labels_path)
    # Drop any existing label columns to avoid _x/_y suffixes on re-runs
    for col in labels_df.columns:
        if col != "filename" and col in focused_review.columns:
            focused_review = focused_review.drop(columns=[col])
    focused_review = focused_review.merge(labels_df, on="filename", how="left")
```

Labels are keyed by filename, so they survive pipeline reruns as long as the same spectra are included.

## 4. Testing

**New file:** `tests/test_review_labels.py`

- `test_load_labels_empty` -- returns empty DataFrame with correct schema when file does not exist
- `test_save_and_load_roundtrip` -- save labels, load back, verify match
- `test_save_labels_overwrites` -- saving twice overwrites cleanly
- `test_load_labels_preserves_types` -- label and notes are strings, timestamp is string

No Streamlit UI tests. No integration test changes (labels are human-in-the-loop, do not affect model computation).

## New Files

| File | Purpose |
|------|---------|
| `src/features/review_labels.py` | Load/save review labels parquet |
| `tests/test_review_labels.py` | Tests for label I/O |

## Modified Files

| File | Change |
|------|--------|
| `src/dashboard/app.py` | Label selectbox, notes input, save button, progress summary in Tab 6 |
| `src/run_pipeline.py` | Merge existing labels into focused_review output |

## New Outputs

| File | Location |
|------|----------|
| `review_labels.parquet` | `data/results/` |

## Notes

- **Version control:** `review_labels.parquet` contains human-generated labels that cannot be regenerated. It should NOT be in `.gitignore`. Consider committing it to preserve review work.
- **`load_labels` is uncached:** Called separately from `@st.cache_data`-decorated `load_data()` to ensure label changes within a session are visible.
