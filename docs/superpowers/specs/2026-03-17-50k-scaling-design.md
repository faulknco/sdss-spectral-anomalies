# Spec C: 50K Spectra Scaling with Stream-and-Discard Pipeline

**Date:** 2026-03-17
**Status:** Approved design, pending implementation plan

## Goal

Scale the pipeline from 599 to 50,000 spectra while keeping disk usage under 2 GB and peak RAM under 3 GB. Downloads, preprocesses, and discards raw FITS in batches so the full 25-50 GB of raw data never hits disk.

## Approach

Stream-and-discard: download FITS in batches of 500 with 8 parallel workers, preprocess each batch immediately, append to a memory-mapped numpy array, delete the raw FITS. After scoring, re-download the top 200 anomaly FITS for dashboard inspection.

## Resource Profile

| Resource | Current (599) | At 50K |
|----------|--------------|--------|
| Disk (persistent) | ~700 MB raw + ~2 MB processed | ~1.5 GB processed only |
| Disk (temporary) | 0 | ~500 MB FITS buffer |
| RAM (peak) | ~200 MB | ~2-3 GB |
| Download time | ~30 min | ~2-4 hours (8 parallel) |
| Training time | ~2.5 min | ~30-45 min |

## 1. Streaming Download + Preprocess

### New function in `src/data/download.py`

`stream_and_preprocess(metadata_df, processed_dir, batch_size=500, n_workers=8, target_grid=DEFAULT_GRID)`

Flow:
1. Pre-allocate a memory-mapped numpy file at `(n_total, grid_size)` filled with zeros. `n_total` is `len(metadata_df)`. Rows for failed downloads remain zero and are compacted out at the end.
2. Split metadata into batches of `batch_size`
3. For each batch: download FITS files in parallel using a new `ThreadPoolExecutor` (written from scratch -- the existing `download_spectra` is serial). Each worker calls the existing `fetch_spectrum_file` for a single FITS, preprocesses it, returns the result. The manifest is NOT updated per-thread -- instead, a single manifest write happens per batch after all workers complete, avoiding thread-safety issues with `_write_manifest`.
4. Write preprocessed rows to the memmap at the current offset, delete the batch's FITS files
5. After all batches: truncate the memmap to actual successful count (copy to a new file of the right size, delete the oversized original). Save final metadata parquet.
6. Return total count of successfully processed spectra

**Failure handling:** Failed downloads are logged and skipped. If >10% of downloads fail within a single batch, log a warning. If >20% of total downloads fail, abort the pipeline with an error. Zero-rows from failed downloads are compacted out during the final truncation step, so no poisoned rows reach the models.

**Checkpoint/resume:** Each batch writes a small `streaming_checkpoint.json` with `{batch_index, offset, n_success, n_fail}`. If the pipeline crashes and restarts, it reads the checkpoint and resumes from the last completed batch. The memmap file already has the earlier batches written.

Temporary FITS files live in `data/raw_tmp/`, cleaned after each batch. The permanent `data/raw/` directory (existing 599 files) is untouched.

### New function in `src/data/preprocess.py`

`preprocess_to_memmap(wavelengths, fluxes, target_grid, output_path, offset, total_rows)`

Writes preprocessed spectra to a pre-allocated memory-mapped numpy file, starting at row `offset`. On first call (`offset=0`), creates the file with shape `(total_rows, len(target_grid))`. On subsequent calls, opens in `r+` mode and writes at the given offset. Does not resize the file -- caller is responsible for final truncation if needed.

### Backward Compatibility

The existing `load_and_preprocess` stays for the current small-file workflow and tests. The streaming path is used when `--n-spectra` exceeds 1000 or `--streaming` is passed.

## 2. OC-SVM Subsampling

**Modified file:** `src/models/ocsvm.py`

Add `max_train_samples` parameter to `OCSVMDetector.__init__` (default 5000).

During `fit()`:
- PCA and scaler are fit on the full data (fast, linear in N)
- If `len(components) > max_train_samples`, randomly subsample before fitting the SVM
- Only the SVM kernel computation is reduced: O(5K^2) instead of O(50K^2)

`score()` unchanged -- transforms and scores all spectra. Interface stays identical.

## 3. Pipeline Changes

**Modified file:** `src/run_pipeline.py`

### Streaming Mode

Auto-enabled when `--n-spectra > 1000` or forced with `--streaming` flag.

Steps 1+2 merge into a single streaming step:
1. Query SDSS metadata for N spectra
2. Call `stream_and_preprocess()` -- downloads in batches, preprocesses, appends to memmap, deletes raw FITS
3. Result: `data/processed/spectra.npy` (memmap) + `data/processed/spectra_metadata.parquet`

Steps 3-8 unchanged. They work on `spectra` (numpy array) regardless of source. The only difference: `np.load(path, mmap_mode='r')` for read-only memory mapping when file is large.

### New Step 9: Keep Top Anomaly FITS

After scoring and comparison:
1. Take the top 200 spectra by combined rank
2. Re-download their FITS files to `data/raw_kept/`
3. Dashboard checks `raw_kept/` before attempting remote fetch

### New CLI Arguments

- `--streaming` -- force streaming mode (auto-enabled for n-spectra > 1000)
- `--batch-size` -- FITS download batch size (default 500)
- `--download-workers` -- parallel download workers (default 8)
- `--keep-top-n` -- number of top anomaly FITS to keep (default 200)

## 4. Dashboard Changes

Minimal. Two changes:

1. **`load_or_fetch_processed_spectrum`** in `src/data/preprocess.py` -- add an optional `kept_dir` parameter (default `None`). When provided, check `kept_dir` before `raw_dir` before attempting remote fetch. The caller (dashboard) passes both directories. This preserves the existing function signature for backward compatibility.

2. **Spectra loading** in `src/dashboard/app.py` -- move the spectra load out of the `@st.cache_data`-decorated `load_data()` function. Instead, use a separate `@st.cache_resource` function that returns the memmap by reference (avoiding serialization which would fully materialize the array into RAM). When `spectra.npy` is small (<10K rows), load normally. When large, load with `mmap_mode='r'`.

Everything else works as-is. Score arrays, metadata, and comparison data are all small.

## 5. Testing

### Modified Files

**`tests/test_download.py`:**
- `test_stream_and_preprocess_basic` -- mock FITS downloads, verify spectra array and metadata returned
- `test_stream_and_preprocess_cleans_temp_files` -- verify temp FITS are deleted after each batch

**`tests/test_ocsvm.py`:**
- `test_ocsvm_subsamples_large_data` -- pass 200 spectra with `max_train_samples=50`, verify fits and scores without error, verify support vectors count is bounded

**`tests/test_preprocess.py`:**
- `test_preprocess_to_memmap_shape` -- verify memmap output has correct shape and values match regular preprocess

No integration test changes -- the integration test uses 100 synthetic spectra below the streaming threshold.

## New/Modified Files

| File | Change |
|------|--------|
| `src/data/download.py` | Add `stream_and_preprocess` |
| `src/data/preprocess.py` | Add `preprocess_to_memmap`, update `load_or_fetch_processed_spectrum` to check `raw_kept/` |
| `src/models/ocsvm.py` | Add `max_train_samples` subsampling |
| `src/run_pipeline.py` | Streaming mode, Step 9 (keep top FITS), new CLI args |
| `src/dashboard/app.py` | Memmap loading for large spectra arrays |
| `tests/test_download.py` | Stream-and-preprocess tests |
| `tests/test_ocsvm.py` | Subsampling test |
| `tests/test_preprocess.py` | Memmap test |

## Notes

- Streaming mode does not affect the existing 599-file workflow. Running without `--streaming` and with `--download-mode skip` behaves exactly as before.
- Both `data/raw_tmp/` and `data/raw_kept/` must be added to `.gitignore`.
- At 50K spectra, the conditional flow (MAF) trains on PCA components (50K x 50 = 20 MB), well within RAM.
- The DAGMM and CVAE use DataLoaders with batching and scale linearly.
- OC-SVM scoring time at 50K: `decision_function` is O(n_test * n_sv * n_features). With ~1.5K support vectors and 50 PCA features, scoring 50K spectra takes ~1-2 minutes. Included in the 30-45 min training time estimate.
- Spectra are stored as float32 (not float64) in the memmap, halving disk usage to ~650 MB for 50K spectra with negligible precision loss for normalized flux values.
- Step 9 re-download of top 200 FITS reuses the existing `download_spectra()` function with a filtered metadata DataFrame.
- The SDSS query uses `ORDER BY NEWID()` which is non-deterministic. Two runs at 50K will produce different spectra sets. This is acceptable for anomaly detection but should be noted if exact reproducibility is needed.
