"""Semi-synthetic anomaly injection for evaluation."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from src.features.line_windows import SPECTRAL_LINES

ANOMALY_TYPES = [
    "emission_line",
    "line_broadening",
    "continuum_tilt",
    "wavelength_shift",
    "missing_band",
]


def _inject_emission_line(spectrum, wl, rng):
    line_idx = rng.integers(len(SPECTRAL_LINES))
    _, center = SPECTRAL_LINES[line_idx]
    fwhm = rng.uniform(3.0, 15.0)
    sigma = fwhm / 2.355
    local_idx = np.argmin(np.abs(wl - center))
    amplitude = rng.uniform(2.0, 10.0) * max(abs(spectrum[local_idx]), 0.01)
    gaussian = amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2)
    return spectrum + gaussian, {"center": center, "fwhm": fwhm, "amplitude": amplitude}


def _inject_line_broadening(spectrum, wl, rng):
    line_idx = rng.integers(len(SPECTRAL_LINES))
    _, center = SPECTRAL_LINES[line_idx]
    sigma_a = rng.uniform(5.0, 20.0)
    dlambda = np.median(np.diff(wl))
    sigma_pix = sigma_a / dlambda
    half_window = 50.0
    mask = (wl >= center - half_window) & (wl <= center + half_window)
    result = spectrum.copy()
    if mask.sum() > 3:
        result[mask] = gaussian_filter1d(spectrum[mask], sigma=sigma_pix)
    return result, {"center": center, "sigma_angstrom": sigma_a}


def _inject_continuum_tilt(spectrum, wl, rng):
    slope = rng.uniform(-0.0005, 0.0005)
    mid = 0.5 * (wl[0] + wl[-1])
    ramp = 1.0 + slope * (wl - mid)
    return spectrum * ramp, {"slope": slope, "midpoint": mid}


def _inject_wavelength_shift(spectrum, wl, rng):
    shift = rng.uniform(5.0, 50.0) * rng.choice([-1, 1])
    f = interp1d(wl + shift, spectrum, kind="linear", bounds_error=False, fill_value=0.0)
    return f(wl), {"shift_angstrom": shift}


def _inject_missing_band(spectrum, wl, rng):
    band_width = rng.uniform(100.0, 500.0)
    max_start = wl[-1] - band_width
    start = rng.uniform(wl[0], max(wl[0], max_start))
    end = start + band_width
    mask = (wl >= start) & (wl <= end)
    result = spectrum.copy()
    result[mask] = 0.0
    return result, {"start": start, "end": end}


_INJECTORS = {
    "emission_line": _inject_emission_line,
    "line_broadening": _inject_line_broadening,
    "continuum_tilt": _inject_continuum_tilt,
    "wavelength_shift": _inject_wavelength_shift,
    "missing_band": _inject_missing_band,
}


def inject_anomalies(spectra, wavelength_grid, fraction=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n = len(spectra)
    n_inject = int(round(n * fraction))
    modified = spectra.copy()
    labels = np.zeros(n, dtype=int)
    injection_log = []
    if n_inject == 0:
        return modified, labels, injection_log
    inject_indices = rng.choice(n, size=n_inject, replace=False)
    inject_indices.sort()
    for idx in inject_indices:
        anomaly_type = rng.choice(ANOMALY_TYPES)
        injector = _INJECTORS[anomaly_type]
        modified[idx], params = injector(spectra[idx].copy(), wavelength_grid, rng)
        labels[idx] = 1
        injection_log.append({"index": int(idx), "type": anomaly_type, "params": params})
    return modified, labels, injection_log
