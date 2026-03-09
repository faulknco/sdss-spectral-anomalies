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
from src.features.categorize import categorize_anomalies


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
