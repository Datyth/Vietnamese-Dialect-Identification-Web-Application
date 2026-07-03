"""wav2vec2 classifier heads for Phase 9 E3."""

from __future__ import annotations

import torch
from torch import nn


class Wav2Vec2EmbeddingClassifier(nn.Module):
    """Classifier head trained on frozen wav2vec2 utterance embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


class Wav2Vec2TinyClassifier(nn.Module):
    """Tiny waveform encoder plus classifier head.

    This is an offline setup/smoke-test stand-in. Full pretrained wav2vec2
    fine-tuning can reuse the same Phase 9 output contract.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(1, 32, kernel_size=9, stride=4),
            conv_block(32, 64, kernel_size=7, stride=4),
            conv_block(64, 96, kernel_size=5, stride=2),
            conv_block(96, 128, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
        return self.classifier(self.encoder(waveforms))


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
    )
