"""Tests for spectrum-to-RGB color conversion."""

import numpy as np
import pytest

from src.features.color import rgb_to_hex, spectra_to_rgb, spectrum_to_rgb


def _blackbody_spectrum(temperature_k, wavelength_aa):
    """Generate a Planck blackbody spectrum at given wavelengths (Angstroms)."""
    # Convert Angstroms to meters.
    wl_m = wavelength_aa * 1e-10
    h = 6.626e-34
    c = 3.0e8
    k = 1.381e-23
    exponent = np.clip(h * c / (wl_m * k * temperature_k), 0, 500)
    return 2 * h * c**2 / (wl_m**5) / (np.exp(exponent) - 1)


@pytest.fixture
def wavelength_grid():
    return np.linspace(3800, 9200, 3500)


class TestSpectrumToRgb:
    def test_hot_star_is_bluish(self, wavelength_grid):
        """A 10000K star should appear blue-white (B > R)."""
        flux = _blackbody_spectrum(10000, wavelength_grid)
        r, g, b = spectrum_to_rgb(wavelength_grid, flux)
        assert b > r, f"Expected blue > red for 10000K star, got R={r} G={g} B={b}"

    def test_sun_like_star_is_warm(self, wavelength_grid):
        """A ~5800K star (solar) should be near white with slight warmth (R >= B)."""
        flux = _blackbody_spectrum(5800, wavelength_grid)
        r, g, b = spectrum_to_rgb(wavelength_grid, flux)
        assert r >= b, f"Expected red >= blue for 5800K star, got R={r} G={g} B={b}"

    def test_cool_star_is_reddish(self, wavelength_grid):
        """A 3000K star should be clearly reddish (R > B by a wide margin)."""
        flux = _blackbody_spectrum(3000, wavelength_grid)
        r, g, b = spectrum_to_rgb(wavelength_grid, flux)
        assert r > b + 30, f"Expected clearly red for 3000K star, got R={r} G={g} B={b}"

    def test_rgb_values_in_range(self, wavelength_grid):
        """RGB values should be in 0-255."""
        flux = _blackbody_spectrum(6000, wavelength_grid)
        r, g, b = spectrum_to_rgb(wavelength_grid, flux)
        for val in (r, g, b):
            assert 0 <= val <= 255

    def test_flat_spectrum_near_white(self, wavelength_grid):
        """A flat spectrum (equal energy) should produce near-white."""
        flux = np.ones_like(wavelength_grid)
        r, g, b = spectrum_to_rgb(wavelength_grid, flux)
        # All channels should be reasonably bright and close to each other.
        assert r > 150 and g > 150 and b > 150, f"Expected near-white, got R={r} G={g} B={b}"

    def test_empty_visible_range_returns_gray(self):
        """Spectrum entirely outside visible range returns neutral gray."""
        wl = np.linspace(8000, 9200, 100)
        flux = np.ones(100)
        r, g, b = spectrum_to_rgb(wl, flux)
        assert (r, g, b) == (128, 128, 128)


class TestSpectraToRgb:
    def test_batch_shape(self, wavelength_grid):
        """Batch conversion should return (N, 3) array."""
        spectra = np.array([
            _blackbody_spectrum(3000, wavelength_grid),
            _blackbody_spectrum(6000, wavelength_grid),
            _blackbody_spectrum(10000, wavelength_grid),
        ])
        colors = spectra_to_rgb(wavelength_grid, spectra)
        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8

    def test_batch_ordering(self, wavelength_grid):
        """Hotter stars should have higher blue-to-red ratio."""
        spectra = np.array([
            _blackbody_spectrum(3000, wavelength_grid),
            _blackbody_spectrum(10000, wavelength_grid),
        ])
        colors = spectra_to_rgb(wavelength_grid, spectra)
        cool_ratio = colors[0, 2] / max(colors[0, 0], 1)
        hot_ratio = colors[1, 2] / max(colors[1, 0], 1)
        assert hot_ratio > cool_ratio


class TestRgbToHex:
    def test_black(self):
        assert rgb_to_hex(0, 0, 0) == "#000000"

    def test_white(self):
        assert rgb_to_hex(255, 255, 255) == "#ffffff"

    def test_red(self):
        assert rgb_to_hex(255, 0, 0) == "#ff0000"
