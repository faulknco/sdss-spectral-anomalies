"""Deep Autoencoding Gaussian Mixture Model for anomaly detection."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DAGMM(nn.Module):
    def __init__(self, input_dim: int = 3500, latent_dim: int = 16, n_gmm: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_gmm = n_gmm

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

        # Estimation network: latent_dim + 2 (recon error features) -> n_gmm
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, n_gmm),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)

        recon_mse = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)
        recon_cos = 1 - nn.functional.cosine_similarity(x, x_hat, dim=1).unsqueeze(1)
        z = torch.cat([z_c, recon_mse, recon_cos], dim=1)

        gamma = self.estimation(z)
        return x_hat, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = z.size(0)
        gamma_sum = gamma.sum(dim=0)
        phi = gamma_sum / N

        mu = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(dim=0) / gamma_sum.unsqueeze(1)

        z_centered = z.unsqueeze(1) - mu.unsqueeze(0)
        cov = torch.zeros(self.n_gmm, z.size(1), z.size(1), device=z.device)
        for k in range(self.n_gmm):
            diff = z_centered[:, k]
            weighted = gamma[:, k].unsqueeze(1) * diff
            cov[k] = (weighted.T @ diff) / gamma_sum[k]
            cov[k] += 1e-3 * torch.eye(z.size(1), device=z.device)

        return phi, mu, cov

    def compute_energy(self, z, phi, mu, cov):
        K, D = mu.shape
        # Use log-space to avoid overflow with (2*pi)^D for large D
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=z.device))
        log_terms = []

        for k in range(K):
            diff = z - mu[k]
            cov_inv = torch.linalg.inv(cov[k])
            log_cov_det = torch.linalg.slogdet(cov[k])[1]
            mahal = (diff @ cov_inv * diff).sum(dim=1)
            log_prob = -0.5 * (D * log_2pi + log_cov_det + mahal)
            log_component = torch.log(phi[k].clamp(min=1e-12)) + log_prob
            log_terms.append(log_component)

        log_energy = torch.logsumexp(torch.stack(log_terms, dim=0), dim=0)
        return -log_energy

    @torch.no_grad()
    def anomaly_score(self, spectra: np.ndarray) -> np.ndarray:
        self.train(False)
        x = torch.tensor(spectra, dtype=torch.float32)
        _, z, gamma = self.forward(x)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        energy = self.compute_energy(z, phi, mu, cov)
        return energy.numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_dagmm(
    spectra: np.ndarray,
    latent_dim: int = 16,
    n_gmm: int = 4,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_energy: float = 0.1,
    lambda_cov: float = 0.005,
) -> DAGMM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(spectra, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DAGMM(
        input_dim=spectra.shape[1], latent_dim=latent_dim, n_gmm=n_gmm
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, z, gamma = model(batch)

            recon_loss = ((batch - x_hat) ** 2).mean()
            phi, mu, cov = model.compute_gmm_params(z, gamma)
            energy = model.compute_energy(z, phi, mu, cov).mean()
            cov_diag = sum(1.0 / cov[k].diag().sum() for k in range(model.n_gmm))

            loss = recon_loss + lambda_energy * energy + lambda_cov * cov_diag
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    model = model.cpu()
    return model
