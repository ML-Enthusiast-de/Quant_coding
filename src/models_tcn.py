# src/models_tcn.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class CausalConv1d(nn.Module):
    """
    1D convolution that is causal: output at time t only depends on <= t.
    We do this by padding on the left and using no padding in Conv1d itself.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        x = F.pad(x, (self.pad, 0))  # pad left side
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    Basic TCN residual block with two causal conv layers.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Match channels for residual connection if needed
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCNRegressor(nn.Module):
    """
    Temporal Convolutional Network regressor for sequence data.

    Input:  (batch, seq_len, input_dim)
    Output: (batch,) scalar next-day return prediction
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            out_channels = hidden_dim
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)          # -> (batch, input_dim, seq_len)
        y = self.tcn(x)                # -> (batch, hidden_dim, seq_len)
        last = y[:, :, -1]             # last time step
        out = self.fc(last)            # (batch, 1)
        return out.squeeze(-1)         # (batch,)


def train_tcn_regressor(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    num_layers: int = 2,
    kernel_size: int = 3,
    dropout: float = 0.1,
) -> TCNRegressor:
    """
    Train a TCNRegressor on sequence data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.from_numpy(X_train_seq)  # (N, seq_len, input_dim)
    y_tensor = torch.from_numpy(y_train_seq)  # (N,)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train_seq.shape[-1]
    model = TCNRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
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
        print(f"Epoch {epoch+1}/{num_epochs} - TCN train MSE: {epoch_loss:.6f}")

    return model


def predict_tcn(
    model: TCNRegressor,
    X_seq: np.ndarray,
) -> np.ndarray:
    """
    Run inference with a trained TCNRegressor on sequence data.
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
