# src/run_pipeline.py
"""Main pipeline: download, preprocess, train models, compare, save results."""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import query_stellar_metadata, download_spectra, stream_and_preprocess
from src.data.preprocess import load_and_preprocess, DEFAULT_GRID
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import train_autoencoder
from src.models.ocsvm import OCSVMDetector
from src.models.dagmm import train_dagmm
from src.models.stability import stability_run
from src.features.categorize import categorize_anomalies
from src.models.compare import adaptive_top_n, compare_anomaly_scores, compare_n_models
import json
from src.models.conditional_autoencoder import train_conditional_autoencoder
from src.models.conformal import SplitConformalCalibrator
from src.data.preprocess import build_metadata_features, METADATA_FEATURE_COLS
from src.data.download import (
    DEFAULT_DOWNLOAD_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_QUERY_TIMEOUT,
    DOWNLOAD_TRANSPORTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def _effective_n_components(spectra: np.ndarray, requested: int = 50) -> int:
    max_components = min(spectra.shape[0], spectra.shape[1])
    if max_components < 1:
        raise ValueError("spectra array must contain at least one sample and one feature")
    return min(requested, max_components)


def run(
    n_spectra: int = 5000,
    sn_min: float = 10.0,
    metadata_only: bool = False,
    download_mode: str = "missing",
    query_timeout: int = DEFAULT_QUERY_TIMEOUT,
    download_timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    download_transport: str = "auto",
    streaming: bool = False,
    batch_size: int = 500,
    download_workers: int = 8,
    keep_top_n: int = 200,
):
    """Run the full anomaly detection pipeline.

    ``download_mode``:
    - ``"missing"``: query SDSS metadata and download any missing local FITS
    - ``"skip"``: skip remote FITS downloads and use only existing local FITS
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if download_mode not in {"missing", "skip"}:
        raise ValueError("download_mode must be 'missing' or 'skip'")

    use_streaming = streaming or (n_spectra > 1000 and download_mode != "skip")

    if use_streaming and download_mode != "skip":
        logger.info("=== Steps 1+2: Streaming download and preprocess ===")
        metadata_df = query_stellar_metadata(limit=n_spectra, sn_min=sn_min, timeout=query_timeout)
        metadata_df.to_parquet(PROCESSED_DIR / "metadata.parquet")
        if metadata_only:
            logger.info("Metadata-only mode; skipping downloads and training.")
            return metadata_df
        n_success = stream_and_preprocess(
            metadata_df, PROCESSED_DIR,
            batch_size=batch_size, n_workers=download_workers,
            timeout=download_timeout, max_retries=max_retries,
            transport=download_transport,
        )
        if n_success == 0:
            raise RuntimeError("No spectra were successfully downloaded")
        spectra = np.array(np.load(PROCESSED_DIR / "spectra.npy", mmap_mode="r"))
        meta_list = pd.read_parquet(PROCESSED_DIR / "spectra_metadata.parquet").to_dict("records")
        n_components = _effective_n_components(spectra, requested=50)
    elif download_mode == "skip":
        logger.info("=== Step 1: Skipping remote metadata/download ===")
        metadata_path = PROCESSED_DIR / "metadata.parquet"
        if metadata_path.exists():
            metadata_df = pd.read_parquet(metadata_path)
        else:
            metadata_df = pd.DataFrame()

        # Check for existing processed spectra (from a previous streaming run)
        spectra_path = PROCESSED_DIR / "spectra.npy"
        meta_parquet_path = PROCESSED_DIR / "spectra_metadata.parquet"
        if spectra_path.exists() and meta_parquet_path.exists():
            logger.info("=== Step 2: Loading existing processed spectra ===")
            spectra = np.array(np.load(spectra_path))
            meta_list = pd.read_parquet(meta_parquet_path).to_dict("records")
            logger.info("Loaded %d spectra from %s", len(spectra), spectra_path)
        else:
            logger.info("=== Step 2: Preprocessing spectra from local FITS ===")
            spectra, meta_list = load_and_preprocess(RAW_DIR, DEFAULT_GRID)
            if len(spectra) == 0:
                raise RuntimeError("No local FITS spectra available to preprocess")
            np.save(spectra_path, spectra)
            pd.DataFrame(meta_list).to_parquet(meta_parquet_path)
        n_components = _effective_n_components(spectra, requested=50)
    else:
        # Step 1: Download
        logger.info("=== Step 1: Downloading spectra ===")
        metadata_df = query_stellar_metadata(limit=n_spectra, sn_min=sn_min, timeout=query_timeout)
        metadata_df.to_parquet(PROCESSED_DIR / "metadata.parquet")
        if metadata_only:
            logger.info("Metadata-only mode enabled; skipping FITS download and model training.")
            return metadata_df
        download_spectra(
            metadata_df,
            RAW_DIR,
            timeout=download_timeout,
            max_retries=max_retries,
            transport=download_transport,
        )

        # Step 2: Preprocess
        logger.info("=== Step 2: Preprocessing spectra ===")
        spectra, meta_list = load_and_preprocess(RAW_DIR, DEFAULT_GRID)
        if len(spectra) == 0:
            raise RuntimeError("No local FITS spectra available to preprocess")
        np.save(PROCESSED_DIR / "spectra.npy", spectra)
        pd.DataFrame(meta_list).to_parquet(PROCESSED_DIR / "spectra_metadata.parquet")
        n_components = _effective_n_components(spectra, requested=50)

    # Step 3: Classical model
    logger.info("=== Step 3: Training PCA + Isolation Forest ===")
    classical = ClassicalAnomalyDetector(n_components=n_components, contamination=0.05)
    classical.fit(spectra)
    if_scores = classical.score(spectra)
    pca_errors = classical.reconstruction_error(spectra)
    pca_components = classical.transform(spectra)
    np.save(RESULTS_DIR / "if_scores.npy", if_scores)
    np.save(RESULTS_DIR / "pca_errors.npy", pca_errors)
    np.save(RESULTS_DIR / "pca_components.npy", pca_components)

    # Step 3b: OC-SVM
    logger.info("=== Step 3b: Training OC-SVM ===")
    ocsvm = OCSVMDetector(n_components=n_components)
    ocsvm.fit(spectra)
    ocsvm_scores = ocsvm.score(spectra)
    np.save(RESULTS_DIR / "ocsvm_scores.npy", ocsvm_scores)

    # Step 4: Autoencoder
    logger.info("=== Step 4: Training Autoencoder ===")
    model, losses = train_autoencoder(spectra, bottleneck_dim=64, epochs=50)
    ae_scores = model.reconstruction_error(spectra)
    np.save(RESULTS_DIR / "ae_scores.npy", ae_scores)
    np.save(RESULTS_DIR / "ae_losses.npy", np.array(losses))

    # Step 4b: DAGMM
    logger.info("=== Step 4b: Training DAGMM ===")
    dagmm_model = train_dagmm(spectra.astype(np.float32), latent_dim=16, n_gmm=4, epochs=50)
    dagmm_scores = dagmm_model.anomaly_score(spectra.astype(np.float32))
    np.save(RESULTS_DIR / "dagmm_scores.npy", dagmm_scores)

    # Step 5: Conditional Autoencoder + Conformal Calibration
    logger.info("=== Step 5: Training Conditional Autoencoder ===")
    meta_df = pd.read_parquet(PROCESSED_DIR / "spectra_metadata.parquet")
    for col in METADATA_FEATURE_COLS:
        if col not in meta_df.columns:
            meta_df[col] = float("nan")
    assert len(meta_df) == len(spectra), (
        f"metadata rows ({len(meta_df)}) != spectra rows ({len(spectra)}); "
        "spectra_metadata.parquet may be stale from a previous run"
    )

    n = len(spectra)
    cal_size = max(1, int(0.2 * n))
    rng = np.random.default_rng(42)
    permutation = rng.permutation(n)
    cal_idx = np.sort(permutation[:cal_size])
    train_idx = np.sort(permutation[cal_size:])

    _, meta_stats = build_metadata_features(meta_df.iloc[train_idx], return_stats=True)
    meta_features = build_metadata_features(meta_df, stats=meta_stats)

    train_mask = np.zeros(n, dtype=bool)
    calibration_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    calibration_mask[cal_idx] = True
    np.save(RESULTS_DIR / "conditional_ae_train_mask.npy", train_mask)
    np.save(RESULTS_DIR / "conditional_ae_calibration_mask.npy", calibration_mask)

    cond_ae_model, cond_ae_losses = train_conditional_autoencoder(
        spectra[train_idx].astype(np.float32), meta_features[train_idx], bottleneck_dim=64, epochs=50
    )
    np.save(RESULTS_DIR / "cond_ae_losses.npy", np.array(cond_ae_losses))

    cond_ae_scores = cond_ae_model.reconstruction_error(spectra.astype(np.float32), meta_features)
    np.save(RESULTS_DIR / "conditional_ae_scores.npy", cond_ae_scores)

    logger.info("=== Step 5c: Conformal calibration ===")
    conformal = SplitConformalCalibrator()
    conformal.fit(cond_ae_scores[cal_idx])
    cond_ae_pvalues = conformal.pvalues(cond_ae_scores)
    np.save(RESULTS_DIR / "conditional_ae_pvalues.npy", cond_ae_pvalues)
    threshold = conformal.threshold(alpha=0.05)
    with open(RESULTS_DIR / "conformal_threshold.json", "w") as f:
        json.dump({
            "alpha": 0.05,
            "threshold": None if not np.isfinite(threshold) else threshold,
            "comparison": ">",
            "valid_for": "calibration_or_new_data",
        }, f, indent=2)

    # Step 5b: Stability runs
    logger.info("=== Step 5b: Multi-seed stability runs ===")
    def if_train_fn(spectra, seed):
        det = ClassicalAnomalyDetector(
            n_components=_effective_n_components(spectra, requested=50),
            contamination=0.05,
            random_state=seed,
        )
        det.fit(spectra)
        return det.score(spectra)

    if_stability = stability_run(if_train_fn, spectra, n_runs=5)
    np.save(RESULTS_DIR / "if_stability_mean.npy", if_stability["mean_scores"])
    np.save(RESULTS_DIR / "if_stability_std.npy", if_stability["std_scores"])

    # Step 6: Categorization
    logger.info("=== Step 6: Categorizing anomalies ===")
    categories = categorize_anomalies(spectra, pca_errors)
    np.save(RESULTS_DIR / "categories.npy", np.array(categories))

    # Parameter counts
    param_counts = {
        "classical_if": classical.param_count(),
        "autoencoder": model.param_count(),
        "ocsvm": ocsvm.param_count(),
        "dagmm": dagmm_model.param_count(),
        "conditional_ae": cond_ae_model.param_count(),
    }
    with open(RESULTS_DIR / "param_counts.json", "w") as f:
        json.dump(param_counts, f, indent=2)

    # Step 7: Compare all models
    logger.info("=== Step 7: Comparing all models ===")
    all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores, "cond_ae": cond_ae_scores}
    comparison_top_n = adaptive_top_n(len(spectra))
    comparison = compare_n_models(all_scores, top_n=comparison_top_n)
    comparison_with_meta = pd.concat([comparison, pd.DataFrame(meta_list)], axis=1)
    comparison_with_meta.to_parquet(RESULTS_DIR / "comparison.parquet")
    with open(RESULTS_DIR / "comparison_config.json", "w") as f:
        json.dump({
            "top_n": comparison_top_n,
            "fraction": 0.1,
            "min_top_n": 25,
            "max_top_n": 100,
            "n_spectra": len(spectra),
        }, f, indent=2)

    top_agreed = comparison_with_meta[comparison_with_meta["n_models_agreed"] >= 3].sort_values("combined_rank")
    top_agreed.to_parquet(RESULTS_DIR / "top_anomalies_agreed.parquet")
    focused_review = (
        comparison_with_meta
        .sort_values(["n_models_agreed", "combined_rank"], ascending=[False, True])
        .head(min(10, len(comparison_with_meta)))
        .copy()
    )
    focused_review.insert(0, "focus_rank", np.arange(1, len(focused_review) + 1))
    focused_review["agreement_fraction"] = focused_review["n_models_agreed"] / len(all_scores)
    focused_review.to_parquet(RESULTS_DIR / "focused_review.parquet")

    # Step 9: Keep top anomaly FITS for dashboard inspection
    if keep_top_n > 0:
        logger.info("=== Step 9: Keeping top %d anomaly FITS ===", keep_top_n)
        import shutil
        kept_dir = PROJECT_ROOT / "data" / "raw_kept"
        kept_dir.mkdir(parents=True, exist_ok=True)
        top_filenames = (
            comparison_with_meta
            .sort_values("combined_rank")
            .head(keep_top_n)["filename"]
            .tolist()
        )
        for fn in top_filenames:
            src = RAW_DIR / fn
            dst = kept_dir / fn
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        logger.info("Kept %d FITS files in %s", len(list(kept_dir.glob("*.fits"))), kept_dir)

    logger.info(
        "Pipeline complete. %s anomalies agreed by 3+ models within top-%s per-model ranks.",
        len(top_agreed),
        comparison_top_n,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SDSS spectral anomaly pipeline.")
    parser.add_argument("--n-spectra", type=int, default=5000, help="Number of spectra to query from SDSS.")
    parser.add_argument("--sn-min", type=float, default=10.0, help="Minimum S/N threshold for SDSS query.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Query and persist metadata only; skip FITS downloads and model training.",
    )
    parser.add_argument(
        "--download-mode",
        choices=["missing", "skip"],
        default="missing",
        help="Download missing FITS from SDSS or use only existing local FITS.",
    )
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=DEFAULT_QUERY_TIMEOUT,
        help="Timeout in seconds for the SDSS metadata query.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=DEFAULT_DOWNLOAD_TIMEOUT,
        help="Timeout in seconds for each SDSS FITS download request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per SDSS FITS download.",
    )
    parser.add_argument(
        "--download-transport",
        choices=sorted(DOWNLOAD_TRANSPORTS),
        default="auto",
        help="Transport for FITS downloads: auto, rsync, https, or astroquery.",
    )
    parser.add_argument("--streaming", action="store_true",
                        help="Force streaming download mode (auto-enabled for n-spectra > 1000).")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="FITS download batch size for streaming mode.")
    parser.add_argument("--download-workers", type=int, default=8,
                        help="Parallel download workers for streaming mode.")
    parser.add_argument("--keep-top-n", type=int, default=200,
                        help="Number of top anomaly FITS to keep for dashboard inspection.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(
        n_spectra=args.n_spectra,
        sn_min=args.sn_min,
        metadata_only=args.metadata_only,
        download_mode=args.download_mode,
        query_timeout=args.query_timeout,
        download_timeout=args.download_timeout,
        max_retries=args.max_retries,
        download_transport=args.download_transport,
        streaming=args.streaming,
        batch_size=args.batch_size,
        download_workers=args.download_workers,
        keep_top_n=args.keep_top_n,
    )


if __name__ == "__main__":
    main()
