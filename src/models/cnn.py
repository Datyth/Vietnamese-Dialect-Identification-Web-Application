"""Lightweight CNN for log-Mel dialect classification."""

from __future__ import annotations

import torch
from torch import nn


class LightweightCNN(nn.Module):
    """Small convolutional classifier for 3-class log-Mel inputs."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            nn.MaxPool2d(kernel_size=2),
            conv_block(16, 32),
            nn.MaxPool2d(kernel_size=2),
            conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
