"""One-Class SVM anomaly detector with PCA preprocessing."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class OCSVMDetector:
    def __init__(
        self,
        n_components: int = 50,
        kernel: str = "rbf",
        nu: float = 0.05,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.svm = OneClassSVM(kernel=kernel, nu=nu)

    def fit(self, spectra: np.ndarray) -> "OCSVMDetector":
        scaled = self.scaler.fit_transform(spectra)
        components = self.pca.fit_transform(scaled)
        self.svm.fit(components)
        return self

    def score(self, spectra: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(spectra)
        components = self.pca.transform(scaled)
        return -self.svm.decision_function(components)

    def param_count(self) -> int:
        n_sv = self.svm.support_vectors_.shape[0]
        n_features = self.svm.support_vectors_.shape[1]
        pca_params = self.pca.components_.size + self.pca.mean_.size
        return n_sv * n_features + pca_params
