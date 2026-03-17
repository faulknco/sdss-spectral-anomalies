"""Tests for spectral line measurement utilities."""
import numpy as np
from src.features.spectral_lines import measure_line_flux, measure_all_lines, STELLAR_LINES


def test_measure_line_flux_gaussian_absorption():
    """Synthetic spectrum with a known Gaussian absorption line."""
    wavelength_grid = np.linspace(3800, 9200, 3500)
    # Flat continuum at 1.0 with a Gaussian absorption at H-alpha (6562.8)
    spectrum = np.ones_like(wavelength_grid)
    line_center = 6562.8
    depth = 0.5
    sigma = 5.0  # Angstroms
    spectrum -= depth * np.exp(-0.5 * ((wavelength_grid - line_center) / sigma) ** 2)

    result = measure_line_flux(spectrum, wavelength_grid, line_center)

    # Continuum should be ~1.0
    assert abs(result["continuum_flux"] - 1.0) < 0.05
    # Line flux should be below continuum
    assert result["line_flux"] < result["continuum_flux"]
    # Ratio should be < 1 (absorption)
    assert result["ratio"] < 1.0
    # EW should be positive for absorption
    assert result["equivalent_width"] > 0
    # Analytical EW of a Gaussian: depth * sigma * sqrt(2*pi) ~ 0.5 * 5 * 2.507 ~ 6.27 A
    expected_ew = depth * sigma * np.sqrt(2 * np.pi)
    assert abs(result["equivalent_width"] - expected_ew) < 1.0  # within 1 Angstrom


def test_measure_line_flux_flat_spectrum():
    """Flat spectrum should have ratio ~1 and EW ~0."""
    wavelength_grid = np.linspace(3800, 9200, 3500)
    spectrum = np.ones_like(wavelength_grid)

    result = measure_line_flux(spectrum, wavelength_grid, 5000.0)
    assert abs(result["ratio"] - 1.0) < 1e-6
    assert abs(result["equivalent_width"]) < 1e-6


def test_measure_line_flux_emission():
    """Emission line should have ratio > 1 and negative EW."""
    wavelength_grid = np.linspace(3800, 9200, 3500)
    spectrum = np.ones_like(wavelength_grid)
    line_center = 6562.8
    # Add emission bump
    spectrum += 0.5 * np.exp(-0.5 * ((wavelength_grid - line_center) / 5.0) ** 2)

    result = measure_line_flux(spectrum, wavelength_grid, line_center)
    assert result["ratio"] > 1.0
    assert result["equivalent_width"] < 0  # emission -> negative EW


def test_measure_all_lines():
    """Verify measure_all_lines returns entries for all STELLAR_LINES."""
    wavelength_grid = np.linspace(3800, 9200, 3500)
    spectrum = np.ones_like(wavelength_grid)

    results = measure_all_lines(spectrum, wavelength_grid)
    assert set(results.keys()) == set(STELLAR_LINES.keys())
    for name, measurement in results.items():
        assert "line_flux" in measurement
        assert "continuum_flux" in measurement
        assert "ratio" in measurement
        assert "equivalent_width" in measurement


def test_measure_line_flux_edge_of_grid():
    """Line at edge of wavelength grid should return nan gracefully."""
    wavelength_grid = np.linspace(3800, 9200, 3500)
    spectrum = np.ones_like(wavelength_grid)

    # Line center way outside the grid
    result = measure_line_flux(spectrum, wavelength_grid, 2000.0)
    assert np.isnan(result["line_flux"])
