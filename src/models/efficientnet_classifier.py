"""EfficientNet-style classifier for log-Mel spectrogram experiments."""

from __future__ import annotations

import torch
from torch import nn


class EfficientNetB0Classifier(nn.Module):
    """Small EfficientNet-B0-inspired model for Phase 9 E2.

    The block layout keeps the MBConv/depthwise flavor without requiring
    torchvision or pretrained ImageNet weights.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_silu(input_channels, 24, stride=2),
            MBConv(24, 24, expansion=1, stride=1),
            MBConv(24, 32, expansion=4, stride=2),
            MBConv(32, 32, expansion=4, stride=1),
            MBConv(32, 56, expansion=4, stride=2),
            MBConv(56, 80, expansion=4, stride=1),
            conv_bn_silu(80, 128, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class MBConv(nn.Module):
    """Minimal MBConv block with squeeze-and-excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        stride: int,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.append(conv_bn_silu(in_channels, hidden_channels, kernel_size=1))
        layers.extend(
            [
                conv_bn_silu(
                    hidden_channels,
                    hidden_channels,
                    stride=stride,
                    groups=hidden_channels,
                ),
                SqueezeExcite(hidden_channels),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.block(inputs)
        if self.use_residual:
            return inputs + output
        return output


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, squeeze_ratio: int = 4) -> None:
        super().__init__()
        squeezed = max(8, channels // squeeze_ratio)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeezed, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeezed, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


def conv_bn_silu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )
