# src/run_pipeline.py
"""Main pipeline: download, preprocess, train models, compare, save results."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import query_stellar_metadata, download_spectra
from src.data.preprocess import load_and_preprocess, DEFAULT_GRID
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import train_autoencoder
from src.models.ocsvm import OCSVMDetector
from src.models.dagmm import train_dagmm
from src.models.stability import stability_run
from src.features.categorize import categorize_anomalies
from src.models.compare import compare_anomaly_scores, compare_n_models
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def run(n_spectra: int = 5000, sn_min: float = 10.0):
    """Run the full anomaly detection pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    logger.info("=== Step 1: Downloading spectra ===")
    metadata_df = query_stellar_metadata(limit=n_spectra, sn_min=sn_min)
    metadata_df.to_parquet(PROCESSED_DIR / "metadata.parquet")
    download_spectra(metadata_df, RAW_DIR)

    # Step 2: Preprocess
    logger.info("=== Step 2: Preprocessing spectra ===")
    spectra, meta_list = load_and_preprocess(RAW_DIR, DEFAULT_GRID)
    np.save(PROCESSED_DIR / "spectra.npy", spectra)
    pd.DataFrame(meta_list).to_parquet(PROCESSED_DIR / "spectra_metadata.parquet")

    # Step 3: Classical model
    logger.info("=== Step 3: Training PCA + Isolation Forest ===")
    classical = ClassicalAnomalyDetector(n_components=50, contamination=0.05)
    classical.fit(spectra)
    if_scores = classical.score(spectra)
    pca_errors = classical.reconstruction_error(spectra)
    pca_components = classical.transform(spectra)
    np.save(RESULTS_DIR / "if_scores.npy", if_scores)
    np.save(RESULTS_DIR / "pca_errors.npy", pca_errors)
    np.save(RESULTS_DIR / "pca_components.npy", pca_components)

    # Step 3b: OC-SVM
    logger.info("=== Step 3b: Training OC-SVM ===")
    ocsvm = OCSVMDetector(n_components=50)
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

    # Step 5b: Stability runs
    logger.info("=== Step 5b: Multi-seed stability runs ===")
    def if_train_fn(spectra, seed):
        det = ClassicalAnomalyDetector(n_components=50, contamination=0.05, random_state=seed)
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
    }
    with open(RESULTS_DIR / "param_counts.json", "w") as f:
        json.dump(param_counts, f, indent=2)

    # Step 7: Compare all models
    logger.info("=== Step 7: Comparing all models ===")
    all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores}
    comparison = compare_n_models(all_scores, top_n=100)
    comparison_with_meta = pd.concat([comparison, pd.DataFrame(meta_list)], axis=1)
    comparison_with_meta.to_parquet(RESULTS_DIR / "comparison.parquet")

    top_agreed = comparison_with_meta[comparison_with_meta["n_models_agreed"] >= 3].sort_values("combined_rank")
    top_agreed.to_parquet(RESULTS_DIR / "top_anomalies_agreed.parquet")

    logger.info(f"Pipeline complete. {len(top_agreed)} anomalies agreed by 3+ models.")


if __name__ == "__main__":
    run()
