"""Metadata-conditioned autoencoder for spectral anomaly detection."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ConditionalSpectralAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3500,
        meta_dim: int = 3,
        bottleneck_dim: int = 64,
        meta_embed_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.meta_dim = meta_dim
        self.bottleneck_dim = bottleneck_dim
        self.meta_embed_dim = meta_embed_dim

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )  # output: (B, 128)

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, meta_embed_dim),
            nn.ReLU(),
        )  # output: (B, meta_embed_dim)

        self.encoder_fc = nn.Linear(128 + meta_embed_dim, bottleneck_dim)

        self._conv_out_dim = (input_dim + 7) // 8
        self.decoder_fc = nn.Linear(bottleneck_dim + meta_embed_dim, 128 * self._conv_out_dim)

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _embed_meta(self, meta: torch.Tensor) -> torch.Tensor:
        meta = meta.clone()
        meta[~torch.isfinite(meta)] = 0.0
        return self.meta_mlp(meta)

    def encode(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_encoder(x)
        meta_emb = self._embed_meta(meta)
        return self.encoder_fc(torch.cat([conv_out, meta_emb], dim=1))

    def decode(self, z: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        meta_emb = self._embed_meta(meta)
        x = self.decoder_fc(torch.cat([z, meta_emb], dim=1))
        x = x.view(x.size(0), 128, self._conv_out_dim)
        x = self.conv_decoder(x)
        if x.size(2) > self.input_dim:
            x = x[:, :, : self.input_dim]
        elif x.size(2) < self.input_dim:
            x = nn.functional.pad(x, (0, self.input_dim - x.size(2)))
        return x

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x, meta), meta)

    @torch.no_grad()
    def reconstruction_error(self, spectra: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        self.train(False)
        x = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        meta = torch.tensor(metadata, dtype=torch.float32)
        recon = self.forward(x, meta)
        return ((x - recon) ** 2).mean(dim=(1, 2)).numpy()


def train_conditional_autoencoder(
    spectra: np.ndarray,
    metadata: np.ndarray,
    bottleneck_dim: int = 64,
    meta_embed_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple["ConditionalSpectralAutoencoder", list[float]]:
    """Train the conditional autoencoder. metadata shape: (n, meta_dim). NaNs -> 0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
    meta = torch.tensor(metadata, dtype=torch.float32)
    meta[~torch.isfinite(meta)] = 0.0

    loader = DataLoader(TensorDataset(x, meta), batch_size=batch_size, shuffle=True)

    model = ConditionalSpectralAutoencoder(
        input_dim=spectra.shape[1],
        meta_dim=metadata.shape[1],
        bottleneck_dim=bottleneck_dim,
        meta_embed_dim=meta_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for x_batch, meta_batch in loader:
            x_batch, meta_batch = x_batch.to(device), meta_batch.to(device)
            loss = criterion(model(x_batch, meta_batch), x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        epoch_losses.append(total / n_batches)

    return model.cpu(), epoch_losses
