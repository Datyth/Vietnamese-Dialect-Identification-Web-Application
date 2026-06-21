"""PyTorch inference for the trained lightweight CNN.

Softmax confidence is uncalibrated and should not be interpreted as a verified
probability that a speaker belongs to a region.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.features.logmel import DEFAULT_N_MELS, log_mel_spectrogram
from src.models.cnn import LightweightCNN
from src.training.train_cnn import LABELS
from src.utils.audio import (
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
    load_audio,
    preprocess_file,
)


DEFAULT_CHECKPOINT_PATH = Path("outputs/models/lightweight_cnn_logmel.pt")


@dataclass(frozen=True)
class InferenceState:
    model: LightweightCNN
    device: torch.device
    labels: tuple[str, ...]
    checkpoint_path: Path


_STATE: InferenceState | None = None


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, mps, cpu.")


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: str | torch.device = "auto",
) -> LightweightCNN:
    """Rebuild the training model, load its state dict, and set eval mode."""
    global _STATE

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {path}. Run Phase 5 training first or "
            "set CNN_CHECKPOINT_PATH."
        )
    resolved_device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unsupported checkpoint format in {path}; expected model_state_dict."
        )

    labels = tuple(checkpoint.get("label_order", ()))
    expected_labels = tuple(LABELS)
    contracts = {
        "model": (checkpoint.get("model"), "LightweightCNN"),
        "feature": (checkpoint.get("feature"), "log_mel_spectrogram"),
        "sample_rate": (checkpoint.get("sample_rate"), TARGET_SAMPLE_RATE),
        "target_samples": (checkpoint.get("target_samples"), TARGET_SAMPLES),
        "n_mels": (checkpoint.get("n_mels"), DEFAULT_N_MELS),
        "label_order": (labels, expected_labels),
    }
    mismatches = {
        name: {"checkpoint": actual, "code": expected}
        for name, (actual, expected) in contracts.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint/training contract mismatch: {mismatches}")

    model = LightweightCNN(num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()
    _STATE = InferenceState(
        model=model,
        device=resolved_device,
        labels=labels,
        checkpoint_path=path,
    )
    return model


def predict(audio_path: str | Path) -> dict[str, Any]:
    """Predict one audio file with shared preprocessing and log-Mel extraction."""
    if _STATE is None:
        raise RuntimeError("CNN model is not loaded. Call load_model() first.")

    path = Path(audio_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        with tempfile.TemporaryDirectory(prefix="dialect-preprocessed-") as temporary:
            preprocessed_path = Path(temporary) / "audio.wav"
            preprocess_file(path, preprocessed_path)
            waveform, sample_rate = load_audio(preprocessed_path)
            feature = log_mel_spectrogram(
                waveform,
                sample_rate=sample_rate,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read or preprocess audio file {path}: {exc}") from exc

    if feature.shape[0] != DEFAULT_N_MELS:
        raise ValueError(
            f"Unexpected log-Mel shape {feature.shape}; expected "
            f"{DEFAULT_N_MELS} Mel bins."
        )
    features = np.asarray(feature[None, None, :, :], dtype=np.float32)
    tensor = torch.from_numpy(features).to(_STATE.device)
    with torch.no_grad():
        logits = _STATE.model(tensor)
        softmax = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    predicted_index = int(np.argmax(softmax))
    probabilities = {
        label: float(softmax[index])
        for index, label in enumerate(_STATE.labels)
    }
    return {
        "predicted_label": _STATE.labels[predicted_index],
        "confidence": float(softmax[predicted_index]),
        "probabilities": probabilities,
    }


def loaded_device() -> str:
    """Return the active inference device for health reporting."""
    if _STATE is None:
        raise RuntimeError("CNN model is not loaded. Call load_model() first.")
    return str(_STATE.device)
