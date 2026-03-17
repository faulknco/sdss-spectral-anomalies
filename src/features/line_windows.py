"""Line-window preprocessing: derivative spectra, line-window features, and shared line catalog."""
import numpy as np

SPECTRAL_LINES: list[tuple[str, float]] = [
    ("Ca K", 3933.7),
    ("Ca H", 3968.5),
    ("H-gamma", 4340.5),
    ("H-beta", 4861.3),
    ("MgH", 5210.0),
    ("Na D", 5892.0),
    ("H-alpha", 6562.8),
    ("TiO", 7050.0),
    ("Ca II a", 8498.0),
    ("Ca II b", 8542.0),
    ("Ca II c", 8662.0),
]

# DISPLAY_LINES uses the same ASCII names as SPECTRAL_LINES so name-membership
# checks pass.  A separate mapping (DISPLAY_LABELS) provides Unicode labels for
# plot annotations.
DISPLAY_LINES: list[tuple[str, float]] = [
    ("Ca K", 3933.7),
    ("Ca H", 3968.5),
    ("H-gamma", 4340.5),
    ("H-beta", 4861.3),
    ("Na D", 5892.0),
    ("H-alpha", 6562.8),
]

# Unicode display labels keyed by the ASCII names above.
DISPLAY_LABELS: dict[str, str] = {
    "H-gamma": "H\u03b3",
    "H-beta": "H\u03b2",
    "H-alpha": "H\u03b1",
}


def compute_derivative_spectra(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
) -> np.ndarray:
    dlambda = np.diff(wavelength_grid)
    dflux = np.diff(spectra, axis=1)
    derivatives = dflux / dlambda[np.newaxis, :]
    med_abs = np.median(np.abs(derivatives), axis=1, keepdims=True)
    med_abs = np.where(med_abs > 0, med_abs, 1.0)
    return derivatives / med_abs
