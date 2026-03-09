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
from src.models.compare import compare_anomaly_scores

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

    # Step 4: Autoencoder
    logger.info("=== Step 4: Training Autoencoder ===")
    model, losses = train_autoencoder(spectra, bottleneck_dim=64, epochs=50)
    ae_scores = model.reconstruction_error(spectra)
    np.save(RESULTS_DIR / "ae_scores.npy", ae_scores)
    np.save(RESULTS_DIR / "ae_losses.npy", np.array(losses))

    # Step 5: Compare
    logger.info("=== Step 5: Comparing models ===")
    comparison = compare_anomaly_scores(if_scores, ae_scores, top_n=100)
    comparison_with_meta = pd.concat(
        [comparison, pd.DataFrame(meta_list)], axis=1
    )
    comparison_with_meta.to_parquet(RESULTS_DIR / "comparison.parquet")

    top_agreed = comparison_with_meta[comparison_with_meta["agreed"]].sort_values(
        "combined_rank"
    )
    top_agreed.to_parquet(RESULTS_DIR / "top_anomalies_agreed.parquet")

    logger.info(f"Pipeline complete. {len(top_agreed)} agreed anomalies found.")
    logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
