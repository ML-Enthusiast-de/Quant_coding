# src/models_lstm.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        output, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        out = self.fc(last_hidden)  # (batch, 1)
        return out.squeeze(-1)      # (batch,)


def train_lstm_regressor(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    num_layers: int = 2,
) -> LSTMRegressor:
    """
    Train a simple LSTM regressor on sequence data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.from_numpy(X_train_seq)  # (N, seq_len, input_dim)
    y_tensor = torch.from_numpy(y_train_seq)  # (N,)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train_seq.shape[-1]
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(dataset)
        # you can comment this out if it’s too chatty
        print(f"Epoch {epoch+1}/{num_epochs} - train MSE: {epoch_loss:.6f}")

    return model


def predict_lstm(
    model: LSTMRegressor,
    X_seq: np.ndarray,
) -> np.ndarray:
    """
    Run inference with a trained LSTMRegressor on sequence data.
    """
    device = next(model.parameters()).device
    model.eval()

    X_tensor = torch.from_numpy(X_seq).to(device)
    preds = []

    with torch.no_grad():
        for xb in torch.split(X_tensor, 256, dim=0):
            pb = model(xb)
            preds.append(pb.cpu().numpy())

    return np.concatenate(preds, axis=0)
