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
        max_train_samples: int = 5000,
    ):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.svm = OneClassSVM(kernel=kernel, nu=nu)
        self.max_train_samples = max_train_samples
        self._rng = np.random.default_rng(random_state)

    def fit(self, spectra: np.ndarray) -> "OCSVMDetector":
        scaled = self.scaler.fit_transform(spectra)
        components = self.pca.fit_transform(scaled)
        if len(components) > self.max_train_samples:
            idx = self._rng.choice(len(components), self.max_train_samples, replace=False)
            self.svm.fit(components[idx])
        else:
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
