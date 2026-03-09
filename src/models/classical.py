"""PCA + Isolation Forest anomaly detection pipeline."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class ClassicalAnomalyDetector:
    def __init__(
        self,
        n_components: int = 50,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.iforest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )

    def fit(self, spectra: np.ndarray) -> "ClassicalAnomalyDetector":
        """Fit scaler, PCA, and Isolation Forest on spectra."""
        scaled = self.scaler.fit_transform(spectra)
        components = self.pca.fit_transform(scaled)
        self.iforest.fit(components)
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """Project spectra into PCA space."""
        scaled = self.scaler.transform(spectra)
        return self.pca.transform(scaled)

    def score(self, spectra: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        components = self.transform(spectra)
        return -self.iforest.decision_function(components)

    def reconstruction_error(self, spectra: np.ndarray) -> np.ndarray:
        """Compute PCA reconstruction error per spectrum."""
        scaled = self.scaler.transform(spectra)
        components = self.pca.transform(scaled)
        reconstructed = self.pca.inverse_transform(components)
        return np.mean((scaled - reconstructed) ** 2, axis=1)
