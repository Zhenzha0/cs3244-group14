"""
model.py — GRU-based siamese network for duplicate question detection.

Architecture:
  Each question embedding (2560-dim) is reshaped into a sequence of chunks,
  then fed through a shared GRU. The final hidden states of both questions
  are combined (concatenation, absolute difference, element-wise product)
  and passed through a classification head.

Why reshape into chunks?
  RNNs operate on sequences. A single 2560-dim vector isn't a sequence,
  but we can split it into e.g. 10 chunks of 256 — this is analogous to
  the matryoshka idea (information at different prefix scales).
"""

import torch
import torch.nn as nn


class SiameseGRU(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2560,
        chunk_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.seq_len = embedding_dim // chunk_size  # 2560 / 256 = 10 steps

        assert embedding_dim % chunk_size == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by chunk_size ({chunk_size})"
        )

        # Shared GRU encoder
        self.gru = nn.GRU(
            input_size=chunk_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Classification head
        # Input: [h1; h2; |h1-h2|; h1*h2] = 4 * (2 * hidden_size)
        clf_input = 4 * (2 * hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of embeddings into fixed-size hidden states.

        Args:
            x: (batch, embedding_dim)

        Returns:
            h: (batch, 2 * hidden_size)  — final hidden from bidirectional GRU
        """
        # Reshape: (batch, seq_len, chunk_size)
        x = x.view(-1, self.seq_len, self.chunk_size)

        # GRU output: (batch, seq_len, 2*hidden), h_n: (2*num_layers, batch, hidden)
        _, h_n = self.gru(x)

        # Take last layer's forward and backward hidden states
        # h_n shape: (num_layers * 2, batch, hidden)
        h_forward = h_n[-2]   # last layer, forward
        h_backward = h_n[-1]  # last layer, backward
        h = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden)
        return h

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1: (batch, embedding_dim) — question 1 embeddings
            emb2: (batch, embedding_dim) — question 2 embeddings

        Returns:
            logits: (batch, 1)
        """
        h1 = self.encode(emb1)
        h2 = self.encode(emb2)

        # Combine representations
        combined = torch.cat([
            h1,
            h2,
            torch.abs(h1 - h2),
            h1 * h2,
        ], dim=1)

        return self.classifier(combined)
