"""Hybrid frozen-Whisper and CNN fusion classifier."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LocalSpectrogramEncoder(nn.Module):
    """Small CNN that turns log-Mel features into a local embedding."""

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            nn.MaxPool2d(kernel_size=2),
            conv_block(16, 32),
            nn.MaxPool2d(kernel_size=2),
            conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            dropout_layer(dropout),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(logmel))


class WhisperCnnFusionClassifier(nn.Module):
    """Fuse frozen Whisper encoder states with frozen local CNN features."""

    def __init__(
        self,
        whisper_encoder: nn.Module,
        whisper_hidden_size: int,
        num_classes: int = 3,
        local_encoder: nn.Module | None = None,
        local_embedding_dim: int = 256,
        fusion_dim: int = 256,
        classifier_hidden_dim: int = 256,
        fusion_type: str = "concat",
        dropout: float = 0.0,
        freeze_local_encoder: bool = True,
    ) -> None:
        super().__init__()
        if fusion_type not in {"concat", "gated"}:
            raise ValueError("fusion_type must be one of: concat, gated.")
        if fusion_dim != whisper_hidden_size:
            raise ValueError(
                "fusion_dim must match whisper_hidden_size because the global "
                "PhoWhisper branch is not projected."
            )
        self.whisper_encoder = whisper_encoder
        self.freeze_local = freeze_local_encoder
        self.fusion_type = fusion_type
        self.local_encoder = local_encoder or LocalSpectrogramEncoder(
            embedding_dim=local_embedding_dim,
            dropout=dropout,
        )
        self.global_norm = nn.LayerNorm(whisper_hidden_size)
        self.local_projection = nn.Sequential(
            nn.LayerNorm(local_embedding_dim),
            nn.Linear(local_embedding_dim, fusion_dim),
            nn.ReLU(inplace=True),
            dropout_layer(dropout),
        )
        if fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid(),
            )
            classifier_input_dim = fusion_dim
        else:
            self.gate = None
            classifier_input_dim = fusion_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            dropout_layer(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        self.freeze_whisper_encoder()
        if self.freeze_local:
            self.freeze_local_encoder()

    def freeze_whisper_encoder(self) -> None:
        self.whisper_encoder.eval()
        for parameter in self.whisper_encoder.parameters():
            parameter.requires_grad = False

    def freeze_local_encoder(self) -> None:
        self.local_encoder.eval()
        for parameter in self.local_encoder.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "WhisperCnnFusionClassifier":
        super().train(mode)
        self.whisper_encoder.eval()
        if self.freeze_local:
            self.local_encoder.eval()
        return self

    def forward(
        self,
        whisper_input_features: torch.Tensor,
        logmel: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            encoder_outputs = self.whisper_encoder(
                input_features=whisper_input_features
            )
            hidden_states = encoder_outputs.last_hidden_state
        global_embedding = hidden_states.mean(dim=1)
        if self.freeze_local:
            with torch.no_grad():
                local_embedding = self.local_encoder(logmel)
        else:
            local_embedding = self.local_encoder(logmel)
        if local_embedding.ndim > 2:
            local_embedding = torch.flatten(local_embedding, start_dim=1)
        projected_global = self.global_norm(global_embedding)
        projected_local = self.local_projection(local_embedding)

        if self.fusion_type == "gated":
            if self.gate is None:
                raise RuntimeError("Gated fusion was not initialized.")
            alpha = self.gate(torch.cat([projected_global, projected_local], dim=-1))
            fused = alpha * projected_global + (1.0 - alpha) * projected_local
        else:
            fused = torch.cat([projected_global, projected_local], dim=-1)
        return self.classifier(fused)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def dropout_layer(dropout: float) -> nn.Module:
    if dropout == 0.0:
        return nn.Identity()
    return nn.Dropout(p=dropout)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    whisper_total = sum(parameter.numel() for parameter in model.whisper_encoder.parameters())
    whisper_trainable = sum(
        parameter.numel()
        for parameter in model.whisper_encoder.parameters()
        if parameter.requires_grad
    )
    local_total = sum(parameter.numel() for parameter in model.local_encoder.parameters())
    local_trainable = sum(
        parameter.numel()
        for parameter in model.local_encoder.parameters()
        if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "whisper_encoder_total": whisper_total,
        "whisper_encoder_trainable": whisper_trainable,
        "local_encoder_total": local_total,
        "local_encoder_trainable": local_trainable,
    }


def infer_whisper_hidden_size(encoder: Any) -> int:
    config = getattr(encoder, "config", None)
    for name in ("d_model", "hidden_size"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer Whisper encoder hidden size from config.")
