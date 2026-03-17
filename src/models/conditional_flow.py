"""Conditional Masked Autoregressive Flow for spectral anomaly detection."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return nn.functional.linear(x, self.linear.weight * self.mask, self.linear.bias)


class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim, reverse=False):
        super().__init__()
        self.input_dim = input_dim
        if reverse:
            input_order = np.arange(input_dim - 1, -1, -1)
        else:
            input_order = np.arange(input_dim)
        hidden_order = np.arange(hidden_dim) % input_dim
        mask1 = torch.tensor((hidden_order[:, None] >= input_order[None, :]).astype(np.float32))
        mask2 = torch.tensor((input_order[:, None] > hidden_order[None, :]).astype(np.float32))
        self.context_fc = nn.Linear(context_dim, hidden_dim)
        self.masked1 = MaskedLinear(input_dim, hidden_dim, mask1)
        self.masked2_mu = MaskedLinear(hidden_dim, input_dim, mask2)
        self.masked2_log_s = MaskedLinear(hidden_dim, input_dim, mask2)

    def forward(self, x, context):
        h = torch.relu(self.masked1(x) + self.context_fc(context))
        return self.masked2_mu(h), self.masked2_log_s(h)


class ConditionalMAF(nn.Module):
    def __init__(self, input_dim=50, context_dim=4, n_blocks=8, hidden_dim=128, meta_embed_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.meta_mlp = nn.Sequential(
            nn.Linear(context_dim, 32), nn.ReLU(),
            nn.Linear(32, meta_embed_dim), nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            MADE(input_dim, hidden_dim, meta_embed_dim, reverse=(i % 2 == 1))
            for i in range(n_blocks)
        ])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(n_blocks)])

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def _embed_meta(self, meta):
        meta = meta.clone()
        meta[~torch.isfinite(meta)] = 0.0
        return self.meta_mlp(meta)

    def forward(self, x, meta):
        context = self._embed_meta(meta)
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        z = x
        for block, bn in zip(self.blocks, self.batch_norms):
            mu, log_s = block(z, context)
            z = (z - mu) * torch.exp(-log_s)
            log_det_sum -= log_s.sum(dim=1)
            z = bn(z)
        return z, log_det_sum

    def _log_prob_tensor(self, x, meta):
        z, log_det = self.forward(x, meta)
        log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(dim=1)
        return log_pz + log_det

    @torch.no_grad()
    def log_prob(self, pca_components, metadata):
        was_training = self.training
        self.eval()
        x = torch.tensor(pca_components, dtype=torch.float32)
        meta = torch.tensor(metadata, dtype=torch.float32)
        result = self._log_prob_tensor(x, meta).numpy()
        if was_training:
            self.train()
        return result

    @torch.no_grad()
    def anomaly_score(self, pca_components, metadata):
        return -self.log_prob(pca_components, metadata)


def train_conditional_flow(
    pca_components, metadata, n_blocks=8, hidden_dim=128, meta_embed_dim=16,
    epochs=100, batch_size=64, lr=1e-4, patience=10, val_fraction=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(pca_components)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train = torch.tensor(pca_components[train_idx], dtype=torch.float32)
    meta_train = torch.tensor(metadata[train_idx], dtype=torch.float32)
    meta_train[~torch.isfinite(meta_train)] = 0.0
    x_val = torch.tensor(pca_components[val_idx], dtype=torch.float32)
    meta_val = torch.tensor(metadata[val_idx], dtype=torch.float32)
    meta_val[~torch.isfinite(meta_val)] = 0.0

    loader = DataLoader(TensorDataset(x_train, meta_train), batch_size=batch_size, shuffle=True, drop_last=True)

    model = ConditionalMAF(
        input_dim=pca_components.shape[1], context_dim=metadata.shape[1],
        n_blocks=n_blocks, hidden_dim=hidden_dim, meta_embed_dim=meta_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for x_batch, meta_batch in loader:
            x_batch, meta_batch = x_batch.to(device), meta_batch.to(device)
            loss = -model._log_prob_tensor(x_batch, meta_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        epoch_losses.append(total / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_loss = -model._log_prob_tensor(x_val.to(device), meta_val.to(device)).mean().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), epoch_losses
