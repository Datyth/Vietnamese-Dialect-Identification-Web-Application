"""MobileNetV3-style classifier for log-Mel spectrogram experiments."""

from __future__ import annotations

import torch
from torch import nn


class MobileNetV3SmallClassifier(nn.Module):
    """Small MobileNetV3-inspired model for Phase 9 E1.

    This intentionally avoids a torchvision dependency. It is a lightweight
    experiment scaffold rather than a byte-for-byte MobileNetV3 implementation.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_activation(input_channels, 16, stride=2, activation=nn.Hardswish),
            InvertedResidual(16, 16, hidden_channels=16, stride=1, use_se=True),
            InvertedResidual(16, 24, hidden_channels=64, stride=2),
            InvertedResidual(24, 24, hidden_channels=72, stride=1),
            InvertedResidual(24, 40, hidden_channels=72, stride=2, use_se=True),
            InvertedResidual(40, 48, hidden_channels=120, stride=1, use_se=True),
            conv_bn_activation(48, 96, kernel_size=1, activation=nn.Hardswish),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class InvertedResidual(nn.Module):
    """Minimal MobileNet inverted residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        stride: int,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            conv_bn_activation(
                in_channels,
                hidden_channels,
                kernel_size=1,
                activation=nn.Hardswish,
            ),
            conv_bn_activation(
                hidden_channels,
                hidden_channels,
                stride=stride,
                groups=hidden_channels,
                activation=nn.Hardswish,
            ),
            SqueezeExcite(hidden_channels) if use_se else nn.Identity(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.block(inputs)
        if self.use_residual:
            return inputs + output
        return output


class SqueezeExcite(nn.Module):
    """Small squeeze-and-excitation block."""

    def __init__(self, channels: int, squeeze_ratio: int = 4) -> None:
        super().__init__()
        squeezed = max(8, channels // squeeze_ratio)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeezed, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


def conv_bn_activation(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        activation(inplace=True),
    )
