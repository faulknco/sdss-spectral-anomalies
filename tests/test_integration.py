# tests/test_integration.py
"""Integration test using synthetic data to verify full pipeline works."""
import numpy as np
from src.data.preprocess import preprocess_spectra
from src.models.classical import ClassicalAnomalyDetector
from src.models.autoencoder import train_autoencoder
from src.models.compare import compare_anomaly_scores


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
    assert if_scores.shape == (100,)

    # Autoencoder
    model, losses = train_autoencoder(
        spectra.astype(np.float32),
        bottleneck_dim=16, epochs=3, batch_size=32,
    )
    ae_scores = model.reconstruction_error(spectra.astype(np.float32))
    assert ae_scores.shape == (100,)

    # Compare
    comparison = compare_anomaly_scores(if_scores, ae_scores, top_n=10)
    assert len(comparison) == 100
    assert "agreed" in comparison.columns
