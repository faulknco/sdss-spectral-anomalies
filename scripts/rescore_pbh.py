#!/usr/bin/env python3
"""Rescore PBH features on existing spectra without rerunning the full pipeline.

Use this after a full pipeline run to apply updated PBH scoring code
(e.g. accretion continuum floor fix) without retraining unsupervised models.

Usage:
    uv run python scripts/rescore_pbh.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import DEFAULT_GRID
from src.features.microlensing import microlensing_score
from src.features.accretion import accretion_score
from src.features.line_asymmetry import line_asymmetry_score
from src.features.categorize import categorize_pbh_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def main():
    spectra_path = PROCESSED_DIR / "spectra.npy"
    meta_path = PROCESSED_DIR / "spectra_metadata.parquet"

    if not spectra_path.exists() or not meta_path.exists():
        logger.error("No processed spectra found. Run the full pipeline first.")
        sys.exit(1)

    spectra = np.load(spectra_path)
    meta = pd.read_parquet(meta_path)
    wl = DEFAULT_GRID
    logger.info("Loaded %d spectra", len(spectra))

    # Rescore
    logger.info("Computing microlensing scores...")
    ml = microlensing_score(spectra, wl, meta)
    np.save(RESULTS_DIR / "microlensing_scores.npy", ml)

    logger.info("Computing accretion scores...")
    acc = accretion_score(spectra, wl, meta)
    np.save(RESULTS_DIR / "accretion_scores.npy", acc)

    logger.info("Computing line asymmetry scores...")
    asym = line_asymmetry_score(spectra, wl, meta)
    np.save(RESULTS_DIR / "line_asymmetry_scores.npy", asym)

    logger.info("Categorizing PBH candidates...")
    pbh = categorize_pbh_candidates(ml, acc)
    np.save(RESULTS_DIR / "pbh_categories.npy", np.array(pbh))

    # Update comparison.parquet
    comp_path = RESULTS_DIR / "comparison.parquet"
    if comp_path.exists():
        comp = pd.read_parquet(comp_path)
        comp["microlensing_score"] = ml
        comp["accretion_score"] = acc
        comp["line_asymmetry_score"] = asym
        comp["pbh_category"] = pbh
        comp.to_parquet(comp_path)
        logger.info("Updated comparison.parquet")

    # Summary
    from collections import Counter
    cats = Counter(pbh)
    logger.info(
        "Done. %d spectra rescored: %d microlensing, %d accretion, %d both, %d none",
        len(spectra), cats["microlensing_candidate"], cats["accretion_candidate"],
        cats["both_candidate"], cats["none"],
    )
    logger.info("Score ranges: ML=[%.3f, %.3f], ACC=[%.3f, %.3f], ASYM=[%.3f, %.3f]",
                ml.min(), ml.max(), acc.min(), acc.max(), asym.min(), asym.max())


if __name__ == "__main__":
    main()
