# src/models_transformer.py
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding.
    Input/Output: (batch, seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerRegressor(nn.Module):
    """
    Transformer-based regressor for sequence data.

    Input:  (batch, seq_len, input_dim)
    Output: (batch,) scalar next-day return prediction
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)               # (batch, seq_len, d_model)
        x = self.pos_encoder(x)              # add positional info
        h = self.encoder(x)                  # (batch, seq_len, d_model)

        # Use representation of the last time step (like LSTM last hidden)
        last = h[:, -1, :]                   # (batch, d_model)
        out = self.fc(last)                  # (batch, 1)
        return out.squeeze(-1)               # (batch,)


def train_transformer_regressor(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    d_model: int = 32,
    nhead: int = 2,
    num_layers: int = 2,
    dim_feedforward: int = 64,
    dropout: float = 0.1,
) -> TransformerRegressor:
    """
    Train a TransformerRegressor on sequence data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.from_numpy(X_train_seq)  # (N, seq_len, input_dim)
    y_tensor = torch.from_numpy(y_train_seq)  # (N,)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train_seq.shape[-1]
    model = TransformerRegressor(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
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
        print(f"Epoch {epoch+1}/{num_epochs} - Transformer train MSE: {epoch_loss:.6f}")

    return model


def predict_transformer(
    model: TransformerRegressor,
    X_seq: np.ndarray,
) -> np.ndarray:
    """
    Run inference with a trained TransformerRegressor on sequence data.
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
