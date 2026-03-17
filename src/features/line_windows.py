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


def _extract_window(
    spectrum: np.ndarray,
    wavelength_grid: np.ndarray,
    center: float,
    half_width: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (wavelength_grid >= center - half_width) & (wavelength_grid <= center + half_width)
    return spectrum[mask], wavelength_grid[mask]


def _window_features(flux: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    if len(flux) < 3:
        return np.zeros(4)
    continuum = 0.5 * (flux[0] + flux[-1])
    if continuum == 0:
        continuum = 1.0
    normalized = flux / continuum
    dlambda = np.median(np.diff(wavelength)) if len(wavelength) > 1 else 1.0
    ew_proxy = float(np.sum(1.0 - normalized) * dlambda)
    depth = float(np.max(np.abs(1.0 - normalized)))
    mid = len(flux) // 2
    left_mean = float(np.mean(flux[:mid])) if mid > 0 else 0.0
    right_mean = float(np.mean(flux[mid:])) if mid < len(flux) else 0.0
    asymmetry = left_mean - right_mean
    dflux = np.diff(flux)
    deriv_var = float(np.var(dflux)) if len(dflux) > 0 else 0.0
    return np.array([ew_proxy, depth, asymmetry, deriv_var])


def extract_line_features(
    spectra: np.ndarray,
    wavelength_grid: np.ndarray,
    half_width: float = 40.0,
) -> np.ndarray:
    n = len(spectra)
    n_lines = len(SPECTRAL_LINES)
    features = np.zeros((n, n_lines * 4))
    for i in range(n):
        for j, (_, center) in enumerate(SPECTRAL_LINES):
            flux_win, wl_win = _extract_window(spectra[i], wavelength_grid, center, half_width)
            features[i, j * 4 : (j + 1) * 4] = _window_features(flux_win, wl_win)
    return features
