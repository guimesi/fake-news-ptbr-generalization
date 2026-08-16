"""Arquiteturas neurais sobre embeddings BERT: TextCNN, TextLSTM, TextConvLSTM (cell 31)."""

from __future__ import annotations


import torch
import torch.nn as nn




# -- Modelos ------------------------------------------------------------------
class TextCNN(nn.Module):
    """CNN 1D em embeddings: Conv1d(64) → Conv1d(128) → AdaptiveMaxPool → MLP."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.do = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.do(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(self.do(x))


class TextLSTM(nn.Module):
    """LSTM bidirecional sobre embeddings, com dropout entre camadas."""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.do = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.do(h)
        h = torch.relu(self.fc1(h))
        return self.fc2(self.do(h))


class TextConvLSTM(nn.Module):
    """Híbrido CNN+LSTM: convoluções extraem features locais, LSTM modela sequência."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_filters: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, num_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            num_filters, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.do = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = self.do(h)
        h = torch.relu(self.fc1(h))
        return self.fc2(self.do(h))


__all__ = [
    "TextCNN",
    "TextLSTM",
    "TextConvLSTM",
]
