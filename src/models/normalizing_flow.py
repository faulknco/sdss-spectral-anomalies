# src/models/normalizing_flow.py
"""Conditional RealNVP normalizing flow for spectral anomaly detection."""
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class CouplingBlock(nn.Module):
    """Affine coupling layer with FiLM conditioning from a metadata embedding.

    The coupling MLP maps z_a (pass-through dims) → hidden activations.
    FiLM (Feature-wise Linear Modulation) applies a per-channel scale+shift
    to those activations using a linear projection of the metadata embedding.
    The modulated hidden state is then projected to (s_raw, t), which define
    the affine transform applied to z_b (transformed dims).
    """

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 128,
        meta_embed_dim: int = 16,
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        in_dim = int(mask.sum().item())
        out_dim = dim - in_dim

        # Maps z_a → hidden activations
        self.net_hidden = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        # FiLM: meta_embedding → (γ, β), each of size hidden_dim
        self.film = nn.Linear(meta_embed_dim, hidden_dim * 2)
        # Maps modulated hidden → (s_raw, t), each of size out_dim
        self.net_out = nn.Linear(hidden_dim, out_dim * 2)
        self._out_dim = out_dim

    def forward(
        self, z: torch.Tensor, meta_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (z_out, log_det_per_sample)."""
        z_a = z[:, self.mask]    # pass-through dims
        z_b = z[:, ~self.mask]   # dims to transform

        h = self.net_hidden(z_a)

        # FiLM modulation
        gamma, beta = self.film(meta_emb).chunk(2, dim=1)
        h = h * gamma + beta

        s_raw, t = self.net_out(h).chunk(2, dim=1)
        s = torch.tanh(s_raw)

        z_out = z.clone()
        z_out[:, ~self.mask] = z_b * torch.exp(s) + t
        log_det = s.sum(dim=1)  # log |det J| for this block
        return z_out, log_det


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP flow: PCA compression + affine coupling blocks with FiLM conditioning.

    Anomaly score = NLL under the Gaussian base distribution.
    The log_det is accumulated in the forward (data→latent) direction, so the
    negative sign in the NLL formula is correct:
        NLL = 0.5*||z||² + 0.5*D*log(2π) - sum(log_det)
    """

    def __init__(
        self,
        n_components: int = 50,
        n_coupling: int = 8,
        hidden_dim: int = 128,
        meta_dim: int = 4,
        meta_embed_dim: int = 16,
    ):
        super().__init__()
        self.n_components = n_components
        self.meta_dim = meta_dim

        self.meta_embedder = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, meta_embed_dim),
            nn.ReLU(),
        )

        self.coupling_blocks = nn.ModuleList()
        for i in range(n_coupling):
            mask = torch.zeros(n_components, dtype=torch.bool)
            if i % 2 == 0:
                mask[::2] = True   # even indices pass through
            else:
                mask[1::2] = True  # odd indices pass through
            self.coupling_blocks.append(
                CouplingBlock(n_components, mask, hidden_dim=hidden_dim, meta_embed_dim=meta_embed_dim)
            )

        # Set by train_normalizing_flow; not nn parameters
        self.pca: PCA | None = None
        self.scaler: StandardScaler | None = None

    def _embed_meta(self, meta: torch.Tensor) -> torch.Tensor:
        meta = meta.clone()
        meta[~torch.isfinite(meta)] = 0.0
        return self.meta_embedder(meta)

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow forward pass. Returns (z_final, total_log_det)."""
        meta_emb = self._embed_meta(meta)
        total_log_det = torch.zeros(z.size(0), device=z.device)
        for block in self.coupling_blocks:
            z, log_det = block(z, meta_emb)
            total_log_det += log_det
        return z, total_log_det

    def nll(self, z_out: torch.Tensor, total_log_det: torch.Tensor) -> torch.Tensor:
        D = z_out.size(1)
        return 0.5 * (z_out ** 2).sum(dim=1) + 0.5 * D * math.log(2 * math.pi) - total_log_det

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pca_transform(self, spectra: np.ndarray) -> np.ndarray:
        if self.pca is None or self.scaler is None:
            raise RuntimeError("Model has no fitted PCA — call train_normalizing_flow first")
        return self.pca.transform(self.scaler.transform(spectra))

    @torch.no_grad()
    def nll_score(self, spectra: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        """Per-spectrum NLL anomaly score. Higher = more anomalous given metadata."""
        was_training = self.training
        self.eval()
        z_np = self._pca_transform(spectra.astype(np.float32))
        z = torch.tensor(z_np, dtype=torch.float32)
        meta = torch.tensor(metadata.copy(), dtype=torch.float32)
        if metadata.shape[1] != self.meta_dim:
            raise ValueError(
                f"metadata has {metadata.shape[1]} columns but model expects {self.meta_dim}"
            )
        z_out, log_det = self.forward(z, meta)
        scores = self.nll(z_out, log_det).numpy()
        if was_training:
            self.train()
        return scores


def train_normalizing_flow(
    spectra: np.ndarray,
    metadata: np.ndarray,
    n_components: int = 50,
    n_coupling: int = 8,
    hidden_dim: int = 128,
    meta_embed_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple["ConditionalRealNVP", list[float]]:
    """Fit PCA on spectra, then train the conditional RealNVP. Returns (model, epoch_losses).

    n_components is clamped to min(n_components, min(spectra.shape)) so small
    datasets and test runs don't raise a PCA error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clamp to valid PCA range
    n_components = min(n_components, min(spectra.shape[0], spectra.shape[1]))

    # Fit PCA on training spectra
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    z_np = pca.fit_transform(scaler.fit_transform(spectra))  # (n, n_components)

    # Zero-fill NaN metadata for training
    meta_clean = metadata.copy().astype(np.float32)
    meta_clean[~np.isfinite(meta_clean)] = 0.0

    model = ConditionalRealNVP(
        n_components=n_components,
        n_coupling=n_coupling,
        hidden_dim=hidden_dim,
        meta_dim=metadata.shape[1],
        meta_embed_dim=meta_embed_dim,
    ).to(device)
    model.pca = pca
    model.scaler = scaler

    z_tensor = torch.tensor(z_np, dtype=torch.float32)
    meta_tensor = torch.tensor(meta_clean, dtype=torch.float32)
    loader = DataLoader(TensorDataset(z_tensor, meta_tensor), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for z_batch, meta_batch in loader:
            z_batch, meta_batch = z_batch.to(device), meta_batch.to(device)
            z_out, log_det = model(z_batch, meta_batch)
            loss = model.nll(z_out, log_det).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        epoch_losses.append(total / n_batches)

    return model.cpu(), epoch_losses
