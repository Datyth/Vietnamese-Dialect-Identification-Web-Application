"""ViP-VL / ChunkFormer-style classifier for Phase 9 E5."""

from __future__ import annotations

import torch
from torch import nn


class VipvlChunkFormerTinyClassifier(nn.Module):
    """Small chunked waveform encoder for dialect classification.

    The real ViP-VL/ChunkFormer checkpoint is dependency-heavy, so this model
    keeps the same chunked waveform-classification idea with plain PyTorch. The
    strided frontend reduces a 16 second waveform to roughly 500 frames before
    self-attention, which keeps batch size 4 practical on 16 GB machines.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            conv_block(1, 32, kernel_size=11, stride=4),
            conv_block(32, 64, kernel_size=7, stride=4),
            conv_block(64, 96, kernel_size=5, stride=4),
            conv_block(96, 128, kernel_size=5, stride=4),
            conv_block(128, 128, kernel_size=3, stride=2),
        )
        self.chunk_mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
        features = self.frontend(waveforms).transpose(1, 2)
        encoded = self.chunk_mixer(features)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


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
