# src/models_lstm_class.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)  # logit for "up" class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # (batch, hidden_dim)
        logit = self.fc(last_hidden)   # (batch, 1)
        return logit.squeeze(-1)       # (batch,)


def train_lstm_classifier(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    num_layers: int = 2,
) -> LSTMClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.from_numpy(X_train_seq)   # (N, seq_len, input_dim)
    y_tensor = torch.from_numpy(y_train_seq)   # (N,)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train_seq.shape[-1]
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    # Class imbalance: more "up" than "down".
    # Heuristic: weight positive/negative differently if you like.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        total_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - train BCE: {total_loss:.4f}")

    return model


def predict_lstm_proba(
    model: LSTMClassifier,
    X_seq: np.ndarray,
) -> np.ndarray:
    """Return p(up) for each sequence."""
    device = next(model.parameters()).device
    model.eval()

    X_tensor = torch.from_numpy(X_seq).to(device)
    probs = []

    with torch.no_grad():
        for xb in torch.split(X_tensor, 256, dim=0):
            logits = model(xb)
            p_up = torch.sigmoid(logits)
            probs.append(p_up.cpu().numpy())

    return np.concatenate(probs, axis=0)
