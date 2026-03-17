# tests/test_integration.py
"""Integration test using synthetic data to verify full pipeline works."""
import numpy as np
from src.data.preprocess import preprocess_spectra
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import train_autoencoder
from src.models.ocsvm import OCSVMDetector
from src.models.dagmm import train_dagmm
from src.models.compare import compare_anomaly_scores, compare_n_models
from src.models.stability import stability_run
from src.features.categorize import categorize_anomalies, categorize_pbh_candidates
from src.models.conditional_autoencoder import train_conditional_autoencoder
from src.models.conformal import SplitConformalCalibrator
from src.data.preprocess import build_metadata_features
from src.features.microlensing import microlensing_score
from src.features.accretion import accretion_score
from src.features.line_asymmetry import line_asymmetry_score
from src.features.multiepoch import multiepoch_variability_scores, assign_multiepoch_scores
from src.features.gaia_crossmatch import score_astrometric_anomalies, _empty_result
from src.features.photometric_crossmatch import compute_dereddened_colors, score_color_anomalies, _empty_photometry
from src.features.line_windows import compute_derivative_spectra, extract_line_features
from src.models.cvae import train_cvae
from src.models.conditional_flow import train_conditional_flow
from src.features.synthetic_anomalies import inject_anomalies
from src.features.evaluate_retrieval import evaluate_retrieval
import pandas as pd


def test_full_pipeline_synthetic():
    """Run the full pipeline on synthetic spectra."""
    rng = np.random.default_rng(42)
    n_spectra = 100
    n_wavelengths = 500

    wavelengths = [np.linspace(3800, 9200, n_wavelengths) for _ in range(n_spectra)]
    normal_flux = [rng.normal(10, 1, n_wavelengths) for _ in range(90)]
    anomalous_flux = [
        rng.normal(10, 1, n_wavelengths) + 5 * np.sin(np.linspace(0, 10, n_wavelengths))
        for _ in range(10)
    ]
    fluxes = normal_flux + anomalous_flux

    target_grid = np.linspace(3800, 9200, n_wavelengths)

    # Preprocess
    spectra = preprocess_spectra(wavelengths, fluxes, target_grid)
    assert spectra.shape == (100, n_wavelengths)

    # Classical model
    classical = ClassicalAnomalyDetector(n_components=20, contamination=0.1)
    classical.fit(spectra)
    if_scores = classical.score(spectra)
    pca_errors = classical.reconstruction_error(spectra)
    assert if_scores.shape == (100,)

    # Autoencoder
    model, losses = train_autoencoder(
        spectra.astype(np.float32),
        bottleneck_dim=16, epochs=3, batch_size=32,
    )
    ae_scores = model.reconstruction_error(spectra.astype(np.float32))
    assert ae_scores.shape == (100,)

    # OC-SVM
    ocsvm = OCSVMDetector(n_components=20)
    ocsvm.fit(spectra)
    ocsvm_scores = ocsvm.score(spectra)
    assert ocsvm_scores.shape == (100,)

    # DAGMM
    dagmm_model = train_dagmm(
        spectra.astype(np.float32),
        latent_dim=8, n_gmm=3, epochs=3, batch_size=32,
    )
    dagmm_scores = dagmm_model.anomaly_score(spectra.astype(np.float32))
    assert dagmm_scores.shape == (100,)

    # Compare (original 2-model, keep backward compat)
    comparison = compare_anomaly_scores(if_scores, ae_scores, top_n=10)
    assert len(comparison) == 100
    assert "agreed" in comparison.columns

    # Compare (N-model)
    all_scores = {"if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores, "dagmm": dagmm_scores}
    n_comparison = compare_n_models(all_scores, top_n=10)
    assert "n_models_agreed" in n_comparison.columns
    assert "combined_rank" in n_comparison.columns

    # Param counts
    assert classical.param_count() > 0
    assert model.param_count() > 0
    assert ocsvm.param_count() > 0
    assert dagmm_model.param_count() > 0

    # Stability
    def mock_train(s, seed):
        rng2 = np.random.default_rng(seed)
        return rng2.random(len(s))

    stability = stability_run(mock_train, spectra, n_runs=3)
    assert stability["mean_scores"].shape == (100,)

    # Categorization
    categories = categorize_anomalies(spectra, pca_errors)
    assert len(categories) == 100
    valid = {"continuum", "line", "noise", "normal"}
    assert all(c in valid for c in categories)

    # Conditional Autoencoder
    meta_df = pd.DataFrame({
        "elodie_teff": rng.uniform(4000, 8000, n_spectra),
        "elodie_logg": rng.uniform(2.0, 5.0, n_spectra),
        "elodie_feh": rng.uniform(-1.5, 0.5, n_spectra),
        "sn_median": rng.uniform(10, 100, n_spectra),
    })
    meta_features = build_metadata_features(meta_df)
    assert meta_features.shape == (n_spectra, 4)

    cal_size = 20
    train_idx = np.arange(n_spectra - cal_size)
    cal_idx = np.arange(n_spectra - cal_size, n_spectra)

    cond_ae_model, cond_ae_losses = train_conditional_autoencoder(
        spectra[train_idx].astype(np.float32),
        meta_features[train_idx],
        bottleneck_dim=16, epochs=3, batch_size=16,
    )
    assert len(cond_ae_losses) == 3
    assert cond_ae_model.param_count() > 0

    cond_ae_scores = cond_ae_model.reconstruction_error(spectra.astype(np.float32), meta_features)
    assert cond_ae_scores.shape == (n_spectra,)
    assert np.all(cond_ae_scores >= 0)

    conformal = SplitConformalCalibrator()
    conformal.fit(cond_ae_scores[cal_idx])
    pvals = conformal.pvalues(cond_ae_scores)
    assert pvals.shape == (n_spectra,)
    assert np.all(pvals >= 0) and np.all(pvals <= 1)
    assert isinstance(conformal.threshold(alpha=0.05), float)

    # PBH Feature Extraction
    wl_grid = np.linspace(3800, 9200, n_wavelengths)
    ml_scores = microlensing_score(spectra, wl_grid, meta_df)
    assert ml_scores.shape == (n_spectra,)
    assert np.all(ml_scores >= 0)

    acc_scores = accretion_score(spectra, wl_grid, meta_df)
    assert acc_scores.shape == (n_spectra,)
    assert np.all(acc_scores >= 0)

    asym_scores = line_asymmetry_score(spectra, wl_grid, meta_df)
    assert asym_scores.shape == (n_spectra,)
    assert np.all(asym_scores >= 0)

    pbh_cats = categorize_pbh_candidates(ml_scores, acc_scores)
    assert len(pbh_cats) == n_spectra
    valid_pbh = {"microlensing_candidate", "accretion_candidate", "both_candidate", "none"}
    assert all(c in valid_pbh for c in pbh_cats)

    # Multi-epoch variability (synthetic data has no repeats, but verify empty case)
    meta_df_with_coords = meta_df.copy()
    meta_df_with_coords["ra"] = rng.uniform(0, 360, n_spectra)
    meta_df_with_coords["dec"] = rng.uniform(-90, 90, n_spectra)
    variability_df = multiepoch_variability_scores(spectra, meta_df_with_coords, wl_grid)
    assert isinstance(variability_df, pd.DataFrame)
    me_scores = assign_multiepoch_scores(variability_df, n_spectra)
    assert me_scores.shape == (n_spectra,)

    # Gaia cross-match scoring (offline, just verify scoring logic)
    gaia_scored = score_astrometric_anomalies(_empty_result())
    assert "astrometric_anomaly_score" in gaia_scored.columns

    # Photometric cross-match scoring (offline, just verify scoring logic)
    phot_scored = score_color_anomalies(_empty_photometry(), meta_df)
    assert "color_anomaly_score" in phot_scored.columns

    # Line-window preprocessing
    derivative = compute_derivative_spectra(spectra, target_grid)
    assert derivative.shape == (n_spectra, n_wavelengths - 1)
    line_feats = extract_line_features(spectra, target_grid)
    assert line_feats.shape == (n_spectra, 44)

    # CVAE
    cvae_model, cvae_losses = train_cvae(
        spectra[train_idx].astype(np.float32),
        meta_features[train_idx],
        bottleneck_dim=16, epochs=3, batch_size=16,
    )
    assert len(cvae_losses) == 3
    assert cvae_model.param_count() > 0
    cvae_scores = cvae_model.anomaly_score(spectra.astype(np.float32), meta_features)
    assert cvae_scores.shape == (n_spectra,)
    assert np.all(np.isfinite(cvae_scores))

    cvae_conformal = SplitConformalCalibrator()
    cvae_conformal.fit(cvae_scores[cal_idx])
    cvae_pvals = cvae_conformal.pvalues(cvae_scores)
    assert cvae_pvals.shape == (n_spectra,)
    assert np.all(cvae_pvals >= 0) and np.all(cvae_pvals <= 1)

    # Conditional flow
    pca_components = classical.transform(spectra)
    flow_model, flow_losses = train_conditional_flow(
        pca_components[train_idx].astype(np.float32),
        meta_features[train_idx],
        n_blocks=4, hidden_dim=64, epochs=5, batch_size=32,
    )
    assert len(flow_losses) == 5
    assert flow_model.param_count() > 0
    flow_scores = flow_model.anomaly_score(
        pca_components.astype(np.float32), meta_features
    )
    assert flow_scores.shape == (n_spectra,)
    assert np.all(np.isfinite(flow_scores))

    # N-model comparison with 7 models
    all_scores_7 = {
        "if": if_scores, "ae": ae_scores, "ocsvm": ocsvm_scores,
        "dagmm": dagmm_scores, "cond_ae": cond_ae_scores,
        "cvae": cvae_scores, "flow": flow_scores,
    }
    comp_7 = compare_n_models(all_scores_7, top_n=10)
    assert "n_models_agreed" in comp_7.columns
    assert len(comp_7) == n_spectra

    # Semi-synthetic evaluation (smoke test: verifies interface, not detection quality,
    # since if_scores are from original spectra not modified ones)
    modified, labels, log = inject_anomalies(spectra, target_grid, fraction=0.1, seed=42)
    assert modified.shape == spectra.shape
    assert sum(labels) == 10

    retrieval_metrics = evaluate_retrieval(labels, if_scores, log, top_k_list=[5, 10])
    assert "auroc" in retrieval_metrics
    assert "precision_at_k" in retrieval_metrics

    # Verify flow can score PCA-projected modified spectra (critical Step 8 path)
    modified_pca = classical.transform(modified)
    flow_modified_scores = flow_model.anomaly_score(
        modified_pca.astype(np.float32), meta_features
    )
    assert flow_modified_scores.shape == (n_spectra,)
    assert np.all(np.isfinite(flow_modified_scores))
