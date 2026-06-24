"""Inference helpers for the trained dialect classifiers.

Scores are intended for a local demo. Softmax confidence and SVM-derived scores
are uncalibrated and should not be interpreted as verified probabilities that a
speaker belongs to a region.
"""

from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.features.logmel import DEFAULT_N_MELS, log_mel_spectrogram
from src.features.mfcc import DEFAULT_N_MFCC, mfcc_mean_std
from src.models.cnn import LightweightCNN
from src.training.train_cnn import LABELS
from src.training.train_phowhisper import DEFAULT_MODEL_ID as DEFAULT_PHOWHISPER_MODEL_ID
from src.utils.audio import (
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
    load_audio,
    preprocess_file,
)


DEFAULT_CNN_CHECKPOINT_PATH = Path("outputs/models/lightweight_cnn_logmel.pt")
DEFAULT_CHECKPOINT_PATH = DEFAULT_CNN_CHECKPOINT_PATH
DEFAULT_SVM_MODEL_PATH = Path("outputs/models/svm_mfcc.pkl")
DEFAULT_PHOWHISPER_CHECKPOINT_PATH = Path(
    "outputs/models/phowhisper_pretrained_frozen_encoder.pt"
)
DEFAULT_PHOWHISPER_CACHE_DIR = Path("outputs/models/hf_cache")
SUPPORTED_MODELS = ("cnn", "svm", "phowhisper")
MODEL_ALIASES = {
    "cnn": "cnn",
    "lightweight_cnn": "cnn",
    "lightweight-cnn": "cnn",
    "svm": "svm",
    "support_vector_machine": "svm",
    "support-vector-machine": "svm",
    "phowhisper": "phowhisper",
    "phowishper": "phowhisper",
    "pho_whisper": "phowhisper",
    "pho-whisper": "phowhisper",
    "phowhisper_fine_tuned": "phowhisper",
    "phowhisper-fine-tuned": "phowhisper",
}


@dataclass(frozen=True)
class CnnInferenceState:
    model: LightweightCNN
    device: torch.device
    labels: tuple[str, ...]
    checkpoint_path: Path


@dataclass(frozen=True)
class SklearnInferenceState:
    model: Any
    labels: tuple[str, ...]
    model_path: Path


@dataclass(frozen=True)
class PhoWhisperInferenceState:
    model: Any
    feature_extractor: Any
    device: torch.device
    labels: tuple[str, ...]
    checkpoint_path: Path
    model_id: str
    cache_dir: Path


_CNN_STATE: CnnInferenceState | None = None
_SVM_STATE: SklearnInferenceState | None = None
_PHOWHISPER_STATE: PhoWhisperInferenceState | None = None


def normalize_model_name(model_name: str = "cnn") -> str:
    """Normalize model names accepted from the browser or API clients."""
    normalized = MODEL_ALIASES.get(model_name.strip().lower())
    if normalized is None:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model {model_name!r}; choose one of: {supported}.")
    return normalized


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
    checkpoint_path: str | Path = DEFAULT_CNN_CHECKPOINT_PATH,
    device: str | torch.device = "auto",
) -> LightweightCNN:
    """Rebuild the training model, load its state dict, and set eval mode."""
    global _CNN_STATE

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
    _CNN_STATE = CnnInferenceState(
        model=model,
        device=resolved_device,
        labels=labels,
        checkpoint_path=path,
    )
    return model


def load_svm_model(model_path: str | Path = DEFAULT_SVM_MODEL_PATH) -> Any:
    """Load the saved Phase 4 SVM MFCC baseline."""
    global _SVM_STATE

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SVM model not found: {path}. Run Phase 4 training first or set "
            "SVM_MODEL_PATH."
        )
    with path.open("rb") as input_file:
        payload = pickle.load(input_file)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unsupported SVM model format in {path}; expected payload.")

    labels = tuple(payload.get("label_order", ()))
    contracts = {
        "feature": (payload.get("feature"), "mfcc_mean_std"),
        "sample_rate": (payload.get("sample_rate"), TARGET_SAMPLE_RATE),
        "target_samples": (payload.get("target_samples"), TARGET_SAMPLES),
        "n_mfcc": (payload.get("n_mfcc"), DEFAULT_N_MFCC),
        "label_order": (labels, tuple(LABELS)),
    }
    mismatches = {
        name: {"model": actual, "code": expected}
        for name, (actual, expected) in contracts.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"SVM/training contract mismatch: {mismatches}")

    _SVM_STATE = SklearnInferenceState(
        model=payload["model"],
        labels=labels,
        model_path=path,
    )
    return payload["model"]


def require_phowhisper_dependencies() -> tuple[Any, Any]:
    try:
        from transformers import AutoFeatureExtractor, WhisperForAudioClassification
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for PhoWhisper inference. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc
    return AutoFeatureExtractor, WhisperForAudioClassification


def load_phowhisper_model(
    checkpoint_path: str | Path = DEFAULT_PHOWHISPER_CHECKPOINT_PATH,
    device: str | torch.device = "auto",
    cache_dir: str | Path = DEFAULT_PHOWHISPER_CACHE_DIR,
    local_files_only: bool = True,
) -> Any:
    """Load the trained PhoWhisper audio-classification checkpoint."""
    global _PHOWHISPER_STATE

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PhoWhisper checkpoint not found: {path}. Run Phase 6 training first "
            "or set PHOWHISPER_CHECKPOINT_PATH."
        )
    resolved_device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unsupported PhoWhisper checkpoint format in {path}; expected "
            "model_state_dict."
        )

    labels = tuple(checkpoint.get("label_order", ()))
    model_id = str(checkpoint.get("model_id") or DEFAULT_PHOWHISPER_MODEL_ID)
    contracts = {
        "model": (checkpoint.get("model"), "WhisperForAudioClassification"),
        "feature": (checkpoint.get("feature"), "PhoWhisper/Whisper input_features"),
        "sample_rate": (checkpoint.get("sample_rate"), TARGET_SAMPLE_RATE),
        "target_samples": (checkpoint.get("target_samples"), TARGET_SAMPLES),
        "label_order": (labels, tuple(LABELS)),
    }
    mismatches = {
        name: {"checkpoint": actual, "code": expected}
        for name, (actual, expected) in contracts.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"PhoWhisper/training contract mismatch: {mismatches}")

    AutoFeatureExtractor, WhisperForAudioClassification = (
        require_phowhisper_dependencies()
    )
    cache_path = Path(cache_dir)
    label_to_index = {label: index for index, label in enumerate(labels)}
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id,
            cache_dir=cache_path,
            local_files_only=local_files_only,
        )
        model = WhisperForAudioClassification.from_pretrained(
            model_id,
            num_labels=len(labels),
            label2id=label_to_index,
            id2label={index: label for label, index in label_to_index.items()},
            ignore_mismatched_sizes=True,
            cache_dir=cache_path,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise FileNotFoundError(
            f"Could not load PhoWhisper base files for {model_id!r} from "
            f"{cache_path}. Keep the Hugging Face cache available locally or set "
            "PHOWHISPER_LOCAL_FILES_ONLY=0 in an environment with network access."
        ) from exc

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()
    _PHOWHISPER_STATE = PhoWhisperInferenceState(
        model=model,
        feature_extractor=feature_extractor,
        device=resolved_device,
        labels=labels,
        checkpoint_path=path,
        model_id=model_id,
        cache_dir=cache_path,
    )
    return model


def predict(audio_path: str | Path, model_name: str = "cnn") -> dict[str, Any]:
    """Predict one audio file with the selected trained model."""
    selected_model = normalize_model_name(model_name)
    if selected_model == "cnn":
        return predict_cnn(audio_path)
    if selected_model == "svm":
        return predict_svm(audio_path)
    if selected_model == "phowhisper":
        return predict_phowhisper(audio_path)
    raise AssertionError(f"Unhandled model: {selected_model}")


def predict_cnn(audio_path: str | Path) -> dict[str, Any]:
    """Predict one audio file with shared preprocessing and log-Mel extraction."""
    if _CNN_STATE is None:
        raise RuntimeError("CNN model is not loaded. Call load_model() first.")

    waveform, sample_rate = preprocess_for_inference(audio_path)
    feature = log_mel_spectrogram(
        waveform,
        sample_rate=sample_rate,
    )

    if feature.shape[0] != DEFAULT_N_MELS:
        raise ValueError(
            f"Unexpected log-Mel shape {feature.shape}; expected "
            f"{DEFAULT_N_MELS} Mel bins."
        )
    features = np.asarray(feature[None, None, :, :], dtype=np.float32)
    tensor = torch.from_numpy(features).to(_CNN_STATE.device)
    with torch.no_grad():
        logits = _CNN_STATE.model(tensor)
        softmax = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    predicted_index = int(np.argmax(softmax))
    probabilities = {
        label: float(softmax[index])
        for index, label in enumerate(_CNN_STATE.labels)
    }
    return {
        "model": "cnn",
        "predicted_label": _CNN_STATE.labels[predicted_index],
        "confidence": float(softmax[predicted_index]),
        "probabilities": probabilities,
        "score_type": "softmax_probability",
    }


def predict_svm(audio_path: str | Path) -> dict[str, Any]:
    """Predict one audio file with the saved SVM MFCC baseline."""
    if _SVM_STATE is None:
        raise RuntimeError("SVM model is not loaded. Call load_svm_model() first.")

    waveform, sample_rate = preprocess_for_inference(audio_path)
    feature = mfcc_mean_std(waveform, sample_rate=sample_rate).reshape(1, -1)
    predicted_label = str(_SVM_STATE.model.predict(feature)[0])
    probabilities, score_type = sklearn_scores(
        _SVM_STATE.model,
        feature,
        _SVM_STATE.labels,
    )
    return {
        "model": "svm",
        "predicted_label": predicted_label,
        "confidence": float(probabilities.get(predicted_label, 0.0)),
        "probabilities": probabilities,
        "score_type": score_type,
    }


def predict_phowhisper(audio_path: str | Path) -> dict[str, Any]:
    """Predict one audio file with the trained PhoWhisper classifier."""
    if _PHOWHISPER_STATE is None:
        raise RuntimeError(
            "PhoWhisper model is not loaded. Call load_phowhisper_model() first."
        )

    waveform, sample_rate = preprocess_for_inference(audio_path)
    encoded = _PHOWHISPER_STATE.feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = encoded["input_features"].to(_PHOWHISPER_STATE.device)
    with torch.no_grad():
        output = _PHOWHISPER_STATE.model(input_features=input_features)
        softmax = torch.softmax(output.logits, dim=1)[0].detach().cpu().numpy()

    predicted_index = int(np.argmax(softmax))
    probabilities = {
        label: float(softmax[index])
        for index, label in enumerate(_PHOWHISPER_STATE.labels)
    }
    return {
        "model": "phowhisper",
        "predicted_label": _PHOWHISPER_STATE.labels[predicted_index],
        "confidence": float(softmax[predicted_index]),
        "probabilities": probabilities,
        "score_type": "softmax_probability",
    }


def preprocess_for_inference(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Apply the shared Phase 2 preprocessing and return a fixed waveform."""
    path = Path(audio_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        with tempfile.TemporaryDirectory(prefix="dialect-preprocessed-") as temporary:
            preprocessed_path = Path(temporary) / "audio.wav"
            preprocess_file(path, preprocessed_path)
            waveform, sample_rate = load_audio(preprocessed_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read or preprocess audio file {path}: {exc}") from exc

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Unexpected preprocessed sample rate: {sample_rate}.")
    if waveform.shape != (TARGET_SAMPLES,):
        raise ValueError(f"Unexpected preprocessed waveform shape: {waveform.shape}.")
    return waveform, sample_rate


def sklearn_scores(
    model: Any,
    feature: np.ndarray,
    labels: tuple[str, ...],
) -> tuple[dict[str, float], str]:
    """Return label-ordered scores for sklearn models.

    The trained SVM does not use calibrated probabilities, so its decision
    margins are softmax-normalized only to keep the UI comparable.
    """
    class_order = tuple(str(label) for label in getattr(model, "classes_", labels))

    try:
        raw_probabilities = np.asarray(model.predict_proba(feature)[0], dtype=np.float64)
    except (AttributeError, ValueError):
        raw_probabilities = None
    if raw_probabilities is not None:
        return align_scores(raw_probabilities, class_order, labels), "predict_proba"

    try:
        decision_scores = np.asarray(
            model.decision_function(feature),
            dtype=np.float64,
        ).reshape(-1)
    except (AttributeError, ValueError):
        decision_scores = np.zeros(len(class_order), dtype=np.float64)
    if decision_scores.size != len(class_order):
        predicted_label = str(model.predict(feature)[0])
        values = {label: 0.0 for label in labels}
        values[predicted_label] = 1.0
        return values, "predicted_class_only"

    shifted = decision_scores - float(np.max(decision_scores))
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / max(float(exp_scores.sum()), 1e-12)
    return (
        align_scores(probabilities, class_order, labels),
        "decision_function_softmax_uncalibrated",
    )


def align_scores(
    raw_scores: np.ndarray,
    class_order: tuple[str, ...],
    labels: tuple[str, ...],
) -> dict[str, float]:
    score_by_class = {
        label: float(raw_scores[index])
        for index, label in enumerate(class_order)
        if index < raw_scores.size
    }
    return {label: float(score_by_class.get(label, 0.0)) for label in labels}


def is_model_loaded(model_name: str = "cnn") -> bool:
    selected_model = normalize_model_name(model_name)
    if selected_model == "cnn":
        return _CNN_STATE is not None
    if selected_model == "svm":
        return _SVM_STATE is not None
    if selected_model == "phowhisper":
        return _PHOWHISPER_STATE is not None
    raise AssertionError(f"Unhandled model: {selected_model}")


def loaded_device(model_name: str = "cnn") -> str:
    """Return the active inference device for health reporting."""
    selected_model = normalize_model_name(model_name)
    if selected_model == "cnn":
        if _CNN_STATE is None:
            raise RuntimeError("CNN model is not loaded. Call load_model() first.")
        return str(_CNN_STATE.device)
    if selected_model == "svm":
        if _SVM_STATE is None:
            raise RuntimeError("SVM model is not loaded. Call load_svm_model() first.")
        return "cpu"
    if selected_model == "phowhisper":
        if _PHOWHISPER_STATE is None:
            raise RuntimeError(
                "PhoWhisper model is not loaded. Call load_phowhisper_model() first."
            )
        return str(_PHOWHISPER_STATE.device)
    raise AssertionError(f"Unhandled model: {selected_model}")
