"""1D Convolutional Autoencoder for spectral anomaly detection."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SpectralAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 3500, bottleneck_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, bottleneck_dim),
        )

        self._conv_out_dim = (input_dim + 7) // 8

        self.decoder_fc = nn.Linear(bottleneck_dim, 128 * self._conv_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 128, self._conv_out_dim)
        x = self.decoder(x)
        if x.size(2) > self.input_dim:
            x = x[:, :, : self.input_dim]
        elif x.size(2) < self.input_dim:
            pad = self.input_dim - x.size(2)
            x = nn.functional.pad(x, (0, pad))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    @torch.no_grad()
    def reconstruction_error(self, spectra: np.ndarray) -> np.ndarray:
        """Compute per-spectrum MSE reconstruction error."""
        self.train(False)
        x = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        reconstructed = self.forward(x)
        mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
        return mse.numpy()


def train_autoencoder(
    spectra: np.ndarray,
    bottleneck_dim: int = 64,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple["SpectralAutoencoder", list[float]]:
    """Train autoencoder on spectra array of shape (n, wavelength_bins)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpectralAutoencoder(
        input_dim=spectra.shape[1], bottleneck_dim=bottleneck_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)

    model = model.cpu()
    return model, epoch_losses
