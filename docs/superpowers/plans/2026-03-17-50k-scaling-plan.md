# 50K Spectra Scaling -- Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale the pipeline from 599 to 50,000 spectra with a stream-and-discard architecture that keeps disk usage under 2 GB and RAM under 3 GB.

**Architecture:** Download FITS in batches of 500 with 8 parallel workers, preprocess each batch immediately into a pre-allocated memory-mapped numpy array (float32), delete the raw FITS. OC-SVM trains on a 5K subsample. After scoring, re-download top 200 anomaly FITS for dashboard inspection. Checkpoint/resume supports crash recovery.

**Tech Stack:** Python, concurrent.futures, numpy (memmap), pandas, scikit-learn, pytest

**Spec:** `docs/superpowers/specs/2026-03-17-50k-scaling-design.md`

---

## Chunk 1: Memmap Preprocessing and OC-SVM Subsampling

### Task 1: create_memmap and write_to_memmap helpers

**Files:**
- Modify: `src/data/preprocess.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_preprocess.py`:

```python
def test_create_and_write_memmap(tmp_path):
    from src.data.preprocess import create_memmap, write_to_memmap, preprocess_spectra
    rng = np.random.default_rng(42)
    n, n_wl = 20, 500
    wavelengths = [np.linspace(3800, 9200, n_wl) for _ in range(n)]
    fluxes = [rng.normal(10, 1, n_wl) for _ in range(n)]
    target_grid = np.linspace(3800, 9200, n_wl)

    output_path = tmp_path / "spectra.npy"
    # Create the file
    create_memmap(output_path, total_rows=n, n_cols=n_wl)
    # Write first batch (rows 0-9)
    batch1 = preprocess_spectra(wavelengths[:10], fluxes[:10], target_grid).astype(np.float32)
    write_to_memmap(output_path, batch1, offset=0, total_rows=n, n_cols=n_wl)
    # Write second batch (rows 10-19)
    batch2 = preprocess_spectra(wavelengths[10:], fluxes[10:], target_grid).astype(np.float32)
    write_to_memmap(output_path, batch2, offset=10, total_rows=n, n_cols=n_wl)

    result = np.load(output_path, mmap_mode="r")
    assert result.shape == (20, 500)
    assert result.dtype == np.float32

    expected = preprocess_spectra(wavelengths, fluxes, target_grid).astype(np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_preprocess.py::test_create_and_write_memmap -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement helpers**

Append to `src/data/preprocess.py`:

```python
def create_memmap(output_path: Path, total_rows: int, n_cols: int) -> None:
    """Create a .npy file pre-filled with zeros, compatible with np.load(mmap_mode='r+')."""
    arr = np.zeros((total_rows, n_cols), dtype=np.float32)
    np.save(output_path, arr)


def write_to_memmap(
    output_path: Path, data: np.ndarray, offset: int, total_rows: int, n_cols: int
) -> None:
    """Write rows into an existing .npy memmap file at the given offset."""
    fp = np.load(output_path, mmap_mode="r+")
    fp[offset : offset + len(data)] = data.astype(np.float32)
    fp.flush()
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_preprocess.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocess.py tests/test_preprocess.py
git commit -m "feat: add create_memmap and write_to_memmap for incremental .npy writing"
```

---

### Task 2: OC-SVM Subsampling

**Files:**
- Modify: `src/models/ocsvm.py`
- Modify: `tests/test_ocsvm.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_ocsvm.py`:

```python
def test_ocsvm_subsamples_large_data():
    rng = np.random.default_rng(42)
    spectra = rng.normal(0, 1, (200, 500))
    detector = OCSVMDetector(n_components=20, max_train_samples=50)
    detector.fit(spectra)
    scores = detector.score(spectra)
    assert scores.shape == (200,)
    assert np.all(np.isfinite(scores))
    # Support vectors should come from the 50-sample subset, not all 200
    assert detector.svm.support_vectors_.shape[0] <= 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_ocsvm.py::test_ocsvm_subsamples_large_data -v`
Expected: FAIL with TypeError (unexpected keyword `max_train_samples`)

- [ ] **Step 3: Add max_train_samples to OCSVMDetector**

Replace the current `src/models/ocsvm.py`:

```python
"""One-Class SVM anomaly detector with PCA preprocessing."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class OCSVMDetector:
    def __init__(
        self,
        n_components: int = 50,
        kernel: str = "rbf",
        nu: float = 0.05,
        random_state: int = 42,
        max_train_samples: int = 5000,
    ):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.svm = OneClassSVM(kernel=kernel, nu=nu)
        self.max_train_samples = max_train_samples
        self._rng = np.random.default_rng(random_state)

    def fit(self, spectra: np.ndarray) -> "OCSVMDetector":
        scaled = self.scaler.fit_transform(spectra)
        components = self.pca.fit_transform(scaled)
        if len(components) > self.max_train_samples:
            idx = self._rng.choice(len(components), self.max_train_samples, replace=False)
            self.svm.fit(components[idx])
        else:
            self.svm.fit(components)
        return self

    def score(self, spectra: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(spectra)
        components = self.pca.transform(scaled)
        return -self.svm.decision_function(components)

    def param_count(self) -> int:
        n_sv = self.svm.support_vectors_.shape[0]
        n_features = self.svm.support_vectors_.shape[1]
        pca_params = self.pca.components_.size + self.pca.mean_.size
        return n_sv * n_features + pca_params
```

- [ ] **Step 4: Run all ocsvm tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_ocsvm.py -v`
Expected: All PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/models/ocsvm.py tests/test_ocsvm.py
git commit -m "feat: add max_train_samples subsampling to OC-SVM for large datasets"
```

---

## Chunk 2: Streaming Download and Pipeline Integration

### Task 3: Streaming Download + Preprocess

**Files:**
- Modify: `src/data/download.py`
- Modify: `tests/test_download.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_download.py`:

```python
from unittest.mock import MagicMock, patch
from src.data.download import stream_and_preprocess


def test_stream_and_preprocess_basic(tmp_path):
    """Test streaming pipeline with mocked FITS downloads."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    raw_tmp = tmp_path / "raw_tmp"

    metadata = pd.DataFrame({
        "plate": [1, 2, 3],
        "mjd": [50000, 50000, 50000],
        "fiberid": [1, 2, 3],
        "run2d": ["v5_13_2", "v5_13_2", "v5_13_2"],
    })

    # Mock fetch_spectrum_file to create fake FITS with COADD data
    def mock_fetch(plate, mjd, fiberid, output_dir, **kwargs):
        from astropy.io import fits as astro_fits
        fname = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
        path = output_dir / fname
        n_pix = 100
        col1 = astro_fits.Column(name="loglam", format="D", array=np.linspace(3.58, 3.96, n_pix))
        col2 = astro_fits.Column(name="flux", format="D", array=np.ones(n_pix))
        coadd = astro_fits.BinTableHDU.from_columns([col1, col2], name="COADD")
        # Minimal SPECOBJ
        col_ra = astro_fits.Column(name="RA", format="D", array=[180.0])
        col_dec = astro_fits.Column(name="DEC", format="D", array=[45.0])
        col_sub = astro_fits.Column(name="SUBCLASS", format="10A", array=["G5"])
        col_sn = astro_fits.Column(name="SN_MEDIAN_ALL", format="D", array=[25.0])
        col_teff = astro_fits.Column(name="ELODIE_TEFF", format="D", array=[5500.0])
        col_logg = astro_fits.Column(name="ELODIE_LOGG", format="D", array=[4.4])
        col_feh = astro_fits.Column(name="ELODIE_FEH", format="D", array=[-0.1])
        specobj = astro_fits.BinTableHDU.from_columns(
            [col_ra, col_dec, col_sub, col_sn, col_teff, col_logg, col_feh], name="SPECOBJ"
        )
        hdul = astro_fits.HDUList([astro_fits.PrimaryHDU(), coadd, specobj])
        hdul.writeto(path, overwrite=True)
        return path

    target_grid = np.linspace(3800, 9200, 200)
    with patch("src.data.download._download_spectrum_file", side_effect=lambda plate, mjd, fiberid, output_dir, **kw: (mock_fetch(plate, mjd, fiberid, output_dir), "ok", 1, None)):
        n_success = stream_and_preprocess(
            metadata, processed_dir, raw_tmp_dir=raw_tmp,
            batch_size=2, n_workers=1, target_grid=target_grid,
        )

    assert n_success == 3
    spectra = np.load(processed_dir / "spectra.npy", mmap_mode="r")
    assert spectra.shape == (3, 200)
    assert spectra.dtype == np.float32
    meta_df = pd.read_parquet(processed_dir / "spectra_metadata.parquet")
    assert len(meta_df) == 3


def test_stream_and_preprocess_cleans_temp_files(tmp_path):
    """Verify temp FITS files are deleted after each batch."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    raw_tmp = tmp_path / "raw_tmp"

    metadata = pd.DataFrame({
        "plate": [1],
        "mjd": [50000],
        "fiberid": [1],
        "run2d": ["v5_13_2"],
    })

    def mock_fetch(plate, mjd, fiberid, output_dir, **kwargs):
        from astropy.io import fits as astro_fits
        fname = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
        path = output_dir / fname
        col1 = astro_fits.Column(name="loglam", format="D", array=np.linspace(3.58, 3.96, 50))
        col2 = astro_fits.Column(name="flux", format="D", array=np.ones(50))
        coadd = astro_fits.BinTableHDU.from_columns([col1, col2], name="COADD")
        col_ra = astro_fits.Column(name="RA", format="D", array=[180.0])
        col_dec = astro_fits.Column(name="DEC", format="D", array=[45.0])
        col_sub = astro_fits.Column(name="SUBCLASS", format="10A", array=["G5"])
        specobj = astro_fits.BinTableHDU.from_columns([col_ra, col_dec, col_sub], name="SPECOBJ")
        hdul = astro_fits.HDUList([astro_fits.PrimaryHDU(), coadd, specobj])
        hdul.writeto(path, overwrite=True)
        return path

    target_grid = np.linspace(3800, 9200, 100)
    with patch("src.data.download._download_spectrum_file", side_effect=lambda plate, mjd, fiberid, output_dir, **kw: (mock_fetch(plate, mjd, fiberid, output_dir), "ok", 1, None)):
        stream_and_preprocess(
            metadata, processed_dir, raw_tmp_dir=raw_tmp,
            batch_size=1, n_workers=1, target_grid=target_grid,
        )

    # Temp directory should be empty or not exist
    if raw_tmp.exists():
        assert len(list(raw_tmp.glob("*.fits"))) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_download.py::test_stream_and_preprocess_basic -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement stream_and_preprocess**

Add to `src/data/download.py`:
- At top of file, add: `import json` and `from concurrent.futures import ThreadPoolExecutor, as_completed`
- Then append these functions:

```python
def _fetch_and_preprocess_one(
    row: dict,
    raw_tmp_dir: Path,
    target_grid: np.ndarray,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    transport: str = "auto",
) -> tuple[dict | None, np.ndarray | None]:
    """Download one FITS, preprocess it, return (metadata_dict, flux_array) or (None, None).

    Uses _download_spectrum_file directly (not fetch_spectrum_file) to avoid
    thread-unsafe manifest writes.
    """
    from src.data.preprocess import resample_spectrum, normalize_spectrum, _make_metadata_dict

    plate, mjd, fiberid = int(row["plate"]), int(row["mjd"]), int(row["fiberid"])
    run2d = row.get("run2d")

    # Use _download_spectrum_file directly to avoid manifest thread-safety issues
    path, status, attempts, error = _download_spectrum_file(
        plate=plate, mjd=mjd, fiberid=fiberid,
        output_dir=raw_tmp_dir,
        run2d=run2d, timeout=timeout, max_retries=max_retries,
        cache=False, transport=transport,
    )
    if status == "failed" or path is None:
        return None, None

    try:
        from astropy.io import fits as astro_fits
        with astro_fits.open(path) as hdul:
            parsed = parse_spectrum_fits(hdul["COADD"].data)
            flux = normalize_spectrum(resample_spectrum(
                parsed["wavelength"], parsed["flux"], target_grid
            ))

            if "SPECOBJ" in hdul:
                meta_ext = hdul["SPECOBJ"].data
            elif "SPALL" in hdul:
                meta_ext = hdul["SPALL"].data
            else:
                meta_ext = None

            def _sf(arr, name, fallback=np.nan):
                return float(arr[name][0]) if arr is not None and name in arr.dtype.names else fallback

            meta = _make_metadata_dict(
                filename=build_sdss_filename(plate, mjd, fiberid),
                plate=plate, mjd=mjd, fiberid=fiberid,
                ra=_sf(meta_ext, "RA", _sf(meta_ext, "PLUG_RA", np.nan)),
                dec=_sf(meta_ext, "DEC", _sf(meta_ext, "PLUG_DEC", np.nan)),
                subclass=str(meta_ext["SUBCLASS"][0]).strip() if meta_ext is not None else "",
                sn_median=_sf(meta_ext, "SN_MEDIAN_ALL", 0.0),
                teff=_sf(meta_ext, "ELODIE_TEFF"),
                logg=_sf(meta_ext, "ELODIE_LOGG"),
                feh=_sf(meta_ext, "ELODIE_FEH"),
            )
        path.unlink(missing_ok=True)
        return meta, flux
    except Exception as e:
        logger.warning("Failed to process %s: %s", build_sdss_filename(plate, mjd, fiberid), e)
        if path is not None and path.exists():
            path.unlink(missing_ok=True)
        return None, None


def stream_and_preprocess(
    metadata_df: pd.DataFrame,
    processed_dir: Path,
    raw_tmp_dir: Path | None = None,
    batch_size: int = 500,
    n_workers: int = 8,
    target_grid: np.ndarray | None = None,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    transport: str = "auto",
) -> int:
    """Stream-and-discard: download FITS in parallel batches, preprocess, delete.

    Pre-allocates a float32 memmap at (n_total, grid_size). Failed downloads are
    compacted out at the end. Writes checkpoint after each batch for crash recovery.

    Returns the number of successfully processed spectra.
    """
    from src.data.preprocess import DEFAULT_GRID, create_memmap, write_to_memmap

    if target_grid is None:
        target_grid = DEFAULT_GRID
    if raw_tmp_dir is None:
        raw_tmp_dir = processed_dir.parent / "raw_tmp"

    raw_tmp_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(metadata_df)
    grid_size = len(target_grid)
    memmap_path = processed_dir / "spectra.npy"
    checkpoint_path = processed_dir / "streaming_checkpoint.json"

    # Check for existing checkpoint
    start_batch = 0
    offset = 0
    total_success = 0
    total_fail = 0
    all_metadata = []

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_batch = ckpt["batch_index"] + 1
        offset = ckpt["offset"]
        total_success = ckpt["n_success"]
        total_fail = ckpt["n_fail"]
        # Load previously saved metadata
        meta_path = processed_dir / "spectra_metadata_partial.parquet"
        if meta_path.exists():
            all_metadata = pd.read_parquet(meta_path).to_dict("records")
        logger.info("Resuming from batch %d (offset=%d, success=%d, fail=%d)",
                     start_batch, offset, total_success, total_fail)

    # Pre-allocate .npy file on first run (always recreate when starting fresh)
    if start_batch == 0:
        create_memmap(memmap_path, total_rows=n_total, n_cols=grid_size)

    batches = [
        metadata_df.iloc[i : i + batch_size]
        for i in range(0, n_total, batch_size)
    ]

    for batch_idx in range(start_batch, len(batches)):
        batch = batches[batch_idx]
        batch_rows = batch.to_dict("records")
        batch_success = []
        batch_fail = 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _fetch_and_preprocess_one, row, raw_tmp_dir, target_grid,
                    timeout, max_retries, transport,
                ): row
                for row in batch_rows
            }
            for future in as_completed(futures):
                meta, flux = future.result()
                if meta is not None and flux is not None:
                    batch_success.append((meta, flux))
                else:
                    batch_fail += 1

        # Write successful spectra to memmap
        if batch_success:
            fluxes_arr = np.array([flux for _, flux in batch_success], dtype=np.float32)
            write_to_memmap(memmap_path, fluxes_arr, offset=offset, total_rows=n_total, n_cols=grid_size)
            all_metadata.extend([meta for meta, _ in batch_success])
            offset += len(batch_success)

        total_success += len(batch_success)
        total_fail += batch_fail

        # Clean temp files
        for f in raw_tmp_dir.glob("*.fits"):
            f.unlink(missing_ok=True)

        # Write checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump({
                "batch_index": batch_idx,
                "offset": offset,
                "n_success": total_success,
                "n_fail": total_fail,
            }, f)

        # Save partial metadata for resume
        pd.DataFrame(all_metadata).to_parquet(
            processed_dir / "spectra_metadata_partial.parquet", index=False
        )

        # Failure threshold checks
        batch_fail_rate = batch_fail / max(len(batch_rows), 1)
        if batch_fail_rate > 0.1:
            logger.warning("Batch %d had %.0f%% failure rate", batch_idx, batch_fail_rate * 100)

        total_fail_rate = total_fail / max(total_success + total_fail, 1)
        if total_fail_rate > 0.2:
            raise RuntimeError(
                f"Aborting: {total_fail_rate:.0%} total download failure rate "
                f"({total_fail} failures out of {total_success + total_fail})"
            )

        logger.info("Batch %d/%d: %d success, %d fail (total: %d/%d)",
                     batch_idx + 1, len(batches), len(batch_success), batch_fail,
                     total_success, n_total)

    # Compact: truncate to actual success count
    if total_success < n_total:
        full = np.load(memmap_path, mmap_mode="r")
        compacted = np.array(full[:total_success])
        del full
        np.save(memmap_path, compacted)

    # Save final metadata
    pd.DataFrame(all_metadata).to_parquet(
        processed_dir / "spectra_metadata.parquet", index=False
    )

    # Cleanup
    partial_meta = processed_dir / "spectra_metadata_partial.parquet"
    if partial_meta.exists():
        partial_meta.unlink()
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logger.info("Streaming complete: %d spectra processed (%d failed)", total_success, total_fail)
    return total_success
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_download.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/download.py tests/test_download.py
git commit -m "feat: add stream_and_preprocess with parallel downloads and checkpoint/resume"
```

---

### Task 4: Update load_or_fetch_processed_spectrum

**Files:**
- Modify: `src/data/preprocess.py`

- [ ] **Step 1: Read current function**

Read `src/data/preprocess.py` lines 145-172.

- [ ] **Step 2: Add kept_dir parameter**

Change the function signature and add the kept_dir check:

```python
def load_or_fetch_processed_spectrum(
    metadata_row,
    raw_dir: Path,
    target_grid: np.ndarray = DEFAULT_GRID,
    kept_dir: Path | None = None,
) -> np.ndarray:
    """Load a processed spectrum from local FITS, fetching the FITS on demand if needed."""
    from astropy.io import fits as astro_fits
    from src.data.download import (
        build_sdss_filename,
        fetch_spectrum_file,
        parse_spectrum_fits,
        spectrum_identifiers_from_metadata,
    )

    plate, mjd, fiberid, run2d = spectrum_identifiers_from_metadata(metadata_row)
    filename = build_sdss_filename(plate, mjd, fiberid)

    # Check kept_dir first (top anomaly FITS), then raw_dir
    fpath = None
    for search_dir in [kept_dir, raw_dir]:
        if search_dir is not None and (search_dir / filename).exists():
            fpath = search_dir / filename
            break

    if fpath is None:
        fetched = fetch_spectrum_file(plate, mjd, fiberid, raw_dir, run2d=run2d)
        if fetched is None:
            raise FileNotFoundError(f"Could not fetch {filename} from SDSS")
        fpath = fetched

    with astro_fits.open(fpath) as hdul:
        parsed = parse_spectrum_fits(hdul["COADD"].data)
    return normalize_spectrum(
        resample_spectrum(parsed["wavelength"], parsed["flux"], target_grid)
    )
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/test_preprocess.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/data/preprocess.py
git commit -m "feat: add kept_dir parameter to load_or_fetch_processed_spectrum"
```

---

### Task 5: Pipeline Streaming Mode and Step 9

**Files:**
- Modify: `src/run_pipeline.py`
- Modify: `.gitignore`

- [ ] **Step 1: Read current pipeline**

Read `src/run_pipeline.py` to understand the structure.

- [ ] **Step 2: Add .gitignore entries**

Append to `.gitignore`:

```
data/raw_tmp/
data/raw_kept/
```

- [ ] **Step 3: Add imports and CLI args**

Add import at top of `src/run_pipeline.py`:

```python
from src.data.download import stream_and_preprocess
```

Add new CLI arguments to `build_arg_parser()`:

```python
    parser.add_argument("--streaming", action="store_true",
                        help="Force streaming download mode (auto-enabled for n-spectra > 1000).")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="FITS download batch size for streaming mode.")
    parser.add_argument("--download-workers", type=int, default=8,
                        help="Parallel download workers for streaming mode.")
    parser.add_argument("--keep-top-n", type=int, default=200,
                        help="Number of top anomaly FITS to keep for dashboard inspection.")
```

Pass these through `main()` to `run()`.

- [ ] **Step 4: Add streaming mode to run()**

Add `streaming`, `batch_size`, `download_workers`, `keep_top_n` parameters to `run()`.

At the top of `run()`, after the existing download_mode logic, add a streaming branch:

```python
    use_streaming = streaming or (n_spectra > 1000 and download_mode != "skip")

    if use_streaming and download_mode != "skip":
        # Streaming mode: download + preprocess in batches
        logger.info("=== Steps 1+2: Streaming download and preprocess ===")
        metadata_df = query_stellar_metadata(limit=n_spectra, sn_min=sn_min, timeout=query_timeout)
        metadata_df.to_parquet(PROCESSED_DIR / "metadata.parquet")
        n_success = stream_and_preprocess(
            metadata_df, PROCESSED_DIR,
            batch_size=batch_size, n_workers=download_workers,
            timeout=download_timeout, max_retries=max_retries,
            transport=download_transport,
        )
        if n_success == 0:
            raise RuntimeError("No spectra were successfully downloaded")
        spectra = np.load(PROCESSED_DIR / "spectra.npy", mmap_mode="r")
        meta_list = pd.read_parquet(PROCESSED_DIR / "spectra_metadata.parquet").to_dict("records")
    else:
        # Original path (existing code for Steps 1+2)
        ...existing code...
```

- [ ] **Step 5: Add Step 9 (keep top anomaly FITS) at end of run()**

After Step 8 (semi-synthetic evaluation), before the final logger.info:

```python
    # Step 9: Keep top anomaly FITS for dashboard inspection
    if keep_top_n > 0:
        logger.info("=== Step 9: Keeping top %d anomaly FITS ===", keep_top_n)
        kept_dir = PROJECT_ROOT / "data" / "raw_kept"
        kept_dir.mkdir(parents=True, exist_ok=True)
        top_filenames = (
            comparison_with_meta
            .sort_values("combined_rank")
            .head(keep_top_n)["filename"]
            .tolist()
        )
        # Filter to filenames that need downloading (not already in raw/ or raw_kept/)
        to_download = []
        for fn in top_filenames:
            if not (RAW_DIR / fn).exists() and not (kept_dir / fn).exists():
                match = meta_df[meta_df["filename"] == fn] if "filename" in meta_df.columns else pd.DataFrame()
                if not match.empty:
                    to_download.append(match.iloc[0])
        if to_download:
            download_spectra(
                pd.DataFrame(to_download), kept_dir,
                timeout=download_timeout, max_retries=max_retries, transport=download_transport,
            )
        # Copy any that exist in raw/ to raw_kept/
        import shutil
        for fn in top_filenames:
            src = RAW_DIR / fn
            dst = kept_dir / fn
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        logger.info("Kept %d FITS files in %s", len(list(kept_dir.glob("*.fits"))), kept_dir)
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/ -v --ignore=tests/test_integration.py -x`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/run_pipeline.py .gitignore
git commit -m "feat: add streaming pipeline mode with Step 9 top-anomaly FITS retention"
```

---

### Task 6: Dashboard Memmap Loading

**Files:**
- Modify: `src/dashboard/app.py`

- [ ] **Step 1: Read current dashboard load_data()**

Read the `load_data()` function in `src/dashboard/app.py`.

- [ ] **Step 2: Extract spectra loading into a separate cached function**

Add a new function before `load_data()`:

```python
@st.cache_resource
def _load_spectra(path: str):
    """Load spectra array, using memmap for large files."""
    p = Path(path)
    if not p.exists():
        return None
    arr = np.load(p, mmap_mode="r")
    if len(arr) > 10000:
        return arr  # keep as memmap reference
    return np.array(arr)  # small enough to copy into RAM
```

In `load_data()`, replace the spectra loading line:

```python
    spectra = np.load(spectra_path) if spectra_path.exists() else None
```

With:

```python
    spectra = None  # loaded separately via _load_spectra to avoid serialization
```

In `main()`, after `load_data()` is called, add:

```python
    spectra = _load_spectra(str(PROCESSED_DIR / "spectra.npy"))
```

Update `load_spectrum_for_row` to pass `kept_dir`:

```python
KEPT_DIR = PROJECT_ROOT / "data" / "raw_kept"

@st.cache_data(show_spinner=False)
def load_spectrum_for_row(metadata_row: dict) -> np.ndarray:
    return load_or_fetch_processed_spectrum(metadata_row, RAW_DIR, kept_dir=KEPT_DIR)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/faulknco/Projects/sdss-spectral-anomalies && uv run pytest tests/ -v --ignore=tests/test_integration.py -x`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat: use st.cache_resource for memmap spectra loading and add kept_dir lookup"
```
