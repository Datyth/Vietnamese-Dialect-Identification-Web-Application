"""PhoWhisper classifier heads for Phase 9 setup experiments."""

from __future__ import annotations

import torch
from torch import nn


class PhoWhisperEmbeddingClassifier(nn.Module):
    """Small MLP for offline PhoWhisper encoder embeddings."""

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dim = max(64, embedding_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(dim=1)
        return self.classifier(embeddings)
