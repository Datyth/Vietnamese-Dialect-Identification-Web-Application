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
    """Fuse frozen Whisper encoder states with local CNN features."""

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
        classifier_head_type: str = "mlp",
        beta_init: float = 0.1,
        dropout: float = 0.0,
        freeze_local_encoder: bool = True,
    ) -> None:
        super().__init__()
        if fusion_type not in {"concat", "gated", "residual_gated"}:
            raise ValueError("fusion_type must be one of: concat, gated, residual_gated.")
        if classifier_head_type not in {"mlp", "phowhisper_linear"}:
            raise ValueError("classifier_head_type must be one of: mlp, phowhisper_linear.")
        if classifier_head_type == "phowhisper_linear" and fusion_type != "residual_gated":
            raise ValueError("phowhisper_linear classifier head is only supported for residual_gated fusion.")
        if fusion_dim != whisper_hidden_size:
            raise ValueError(
                "fusion_dim must match whisper_hidden_size because the global "
                "PhoWhisper branch is not projected."
            )
        self.whisper_encoder = whisper_encoder
        self.freeze_local = freeze_local_encoder
        self.local_trainable_child_names: set[str] | None = None
        self.fusion_type = fusion_type
        self.classifier_head_type = classifier_head_type
        self.beta_init = float(beta_init)
        self.local_encoder = local_encoder or LocalSpectrogramEncoder(
            embedding_dim=local_embedding_dim,
            dropout=dropout,
        )
        self.global_norm = (
            nn.Identity()
            if fusion_type == "residual_gated"
            else nn.LayerNorm(whisper_hidden_size)
        )
        if fusion_type == "residual_gated":
            self.local_projection = nn.Sequential(
                nn.LayerNorm(local_embedding_dim),
                nn.Linear(local_embedding_dim, fusion_dim),
            )
        else:
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
            self.residual_gate = None
            self.beta = None
            classifier_input_dim = fusion_dim
        elif fusion_type == "residual_gated":
            self.gate = None
            self.residual_gate = nn.Linear(fusion_dim * 2, fusion_dim)
            self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
            classifier_input_dim = fusion_dim
        else:
            self.gate = None
            self.residual_gate = None
            self.beta = None
            classifier_input_dim = fusion_dim * 2
        if classifier_head_type == "phowhisper_linear":
            self.projector = nn.Linear(classifier_input_dim, classifier_hidden_dim)
            self.classifier = nn.Linear(classifier_hidden_dim, num_classes)
        else:
            self.projector = None
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
        self.freeze_local = True
        self.local_trainable_child_names = None
        self.local_encoder.eval()
        for parameter in self.local_encoder.parameters():
            parameter.requires_grad = False

    def enable_local_encoder_finetuning(self, trainable_layers: int) -> list[str]:
        if trainable_layers < 0:
            raise ValueError("trainable_layers must be non-negative.")
        if trainable_layers == 0:
            self.freeze_local_encoder()
            return []

        candidates = [
            (name, module)
            for name, module in self.local_encoder.named_children()
            if any(True for _parameter in module.parameters(recurse=True))
        ]
        if not candidates:
            raise ValueError("local_encoder has no parameterized child modules to fine-tune.")
        if trainable_layers > len(candidates):
            raise ValueError(
                "trainable_layers cannot exceed the number of parameterized "
                f"local encoder child modules ({len(candidates)})."
            )

        for parameter in self.local_encoder.parameters():
            parameter.requires_grad = False

        selected = candidates[-trainable_layers:]
        selected_names = [name for name, _module in selected]
        for _name, module in selected:
            for parameter in module.parameters():
                parameter.requires_grad = True

        self.freeze_local = False
        self.local_trainable_child_names = set(selected_names)
        self.train(self.training)
        return selected_names

    def train(self, mode: bool = True) -> "WhisperCnnFusionClassifier":
        super().train(mode)
        self.whisper_encoder.eval()
        if self.freeze_local:
            self.local_encoder.eval()
        elif mode and self.local_trainable_child_names is not None:
            for name, module in self.local_encoder.named_children():
                if name not in self.local_trainable_child_names:
                    module.eval()
        return self

    def forward(
        self,
        whisper_input_features: torch.Tensor,
        logmel: torch.Tensor,
    ) -> torch.Tensor:
        logits, _diagnostics = self.forward_with_diagnostics(
            whisper_input_features,
            logmel,
        )
        return logits

    def forward_with_diagnostics(
        self,
        whisper_input_features: torch.Tensor,
        logmel: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        diagnostics = {
            "global_embedding": projected_global,
            "projected_local": projected_local,
        }

        if self.fusion_type == "gated":
            if self.gate is None:
                raise RuntimeError("Gated fusion was not initialized.")
            alpha = self.gate(torch.cat([projected_global, projected_local], dim=-1))
            fused = alpha * projected_global + (1.0 - alpha) * projected_local
            diagnostics["gate"] = alpha
        elif self.fusion_type == "residual_gated":
            if self.residual_gate is None or self.beta is None:
                raise RuntimeError("Residual-gated fusion was not initialized.")
            gate_input = torch.cat([projected_global, projected_local], dim=-1)
            residual_gate = torch.sigmoid(self.residual_gate(gate_input))
            fused = projected_global + self.beta * residual_gate * projected_local
            diagnostics["residual_gate"] = residual_gate
        else:
            fused = torch.cat([projected_global, projected_local], dim=-1)
        diagnostics["fused"] = fused
        if self.classifier_head_type == "phowhisper_linear":
            if self.projector is None:
                raise RuntimeError("PhoWhisper linear head projector was not initialized.")
            logits = self.classifier(self.projector(fused))
        else:
            logits = self.classifier(fused)
        return logits, diagnostics


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
