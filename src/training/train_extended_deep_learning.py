"""Phase 9 extended deep-learning experiment runner.

The default is a smoke run: E1/E2 train on a tiny balanced subset for one epoch,
E4 reuses existing Phase 6 PhoWhisper results, and dependency-heavy E3/E5 write
honest skipped/setup artifacts without blocking the demo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.features.logmel import DEFAULT_N_MELS
from src.training.train_cnn import (
    build_loader,
    extract_logmel_features,
    read_preprocessed_metadata,
    require_sklearn_metrics,
    resolve_device,
    set_seed,
    split_rows,
    train_one_epoch,
    write_training_log,
)
from src.utils.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES
from src.utils.audio import load_audio


LABELS = ("Northern", "Central", "Southern")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
SPLITS = ("train", "valid", "test")
PHASE = "phase9_extended_deep_learning_experiments"
PHASE10 = "phase10_whisper_cnn_fusion"
PHASE10_METRIC_PATH = Path("outputs/metrics/e7_whisper_cnn_fusion_results.json")
DEFAULT_SEED = 42
DEFAULT_SMOKE_LIMIT_PER_SPLIT = 9
DEFAULT_SMOKE_EPOCHS = 1
DEFAULT_FULL_EPOCHS = 20
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_LEARNING_RATE_HEAD = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_WAV2VEC2_MODEL_ID = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
DEFAULT_DROPOUT = 0.3

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "vietnamese_dialect_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "vietnamese_dialect_xdg_cache"),
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    aliases: tuple[str, ...]
    model_name: str
    input_type: str
    pretrained: str
    trainable_setting: str
    metric_path: Path
    confusion_figure_path: Path
    checkpoint_path: Path
    training_log_path: Path | None
    notes: str


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "e1_mobilenetv3": ExperimentSpec(
        experiment_id="e1_mobilenetv3",
        aliases=("e1",),
        model_name="MobileNetV3-Small-style log-Mel classifier",
        input_type="log_mel_spectrogram",
        pretrained="none",
        trainable_setting="all_parameters_trainable",
        metric_path=Path("outputs/metrics/e1_mobilenetv3_results.json"),
        confusion_figure_path=Path(
            "outputs/figures/e1_mobilenetv3_confusion_matrix.png"
        ),
        checkpoint_path=Path("outputs/models/e1_mobilenetv3_logmel.pt"),
        training_log_path=Path("outputs/metrics/e1_mobilenetv3_training_log.csv"),
        notes="PyTorch-only MobileNetV3-inspired scaffold; no torchvision dependency.",
    ),
    "e2_efficientnetb0": ExperimentSpec(
        experiment_id="e2_efficientnetb0",
        aliases=("e2", "e2_efficientnet"),
        model_name="EfficientNet-B0-style log-Mel classifier",
        input_type="log_mel_spectrogram",
        pretrained="none",
        trainable_setting="all_parameters_trainable",
        metric_path=Path("outputs/metrics/e2_efficientnetb0_results.json"),
        confusion_figure_path=Path(
            "outputs/figures/e2_efficientnetb0_confusion_matrix.png"
        ),
        checkpoint_path=Path("outputs/models/e2_efficientnetb0_logmel.pt"),
        training_log_path=Path("outputs/metrics/e2_efficientnetb0_training_log.csv"),
        notes="PyTorch-only EfficientNet-inspired scaffold; ImageNet weights deferred.",
    ),
    "e3_wav2vec2": ExperimentSpec(
        experiment_id="e3_wav2vec2",
        aliases=("e3",),
        model_name="wav2vec2 Vietnamese encoder + classifier",
        input_type="waveform_16khz",
        pretrained=DEFAULT_WAV2VEC2_MODEL_ID,
        trainable_setting="frozen_encoder_offline_embeddings_classifier_head",
        metric_path=Path("outputs/metrics/e3_wav2vec2_results.json"),
        confusion_figure_path=Path(
            "outputs/figures/e3_wav2vec2_confusion_matrix.png"
        ),
        checkpoint_path=Path("outputs/models/e3_wav2vec2_classifier.pt"),
        training_log_path=Path("outputs/metrics/e3_wav2vec2_training_log.csv"),
        notes=(
            "Frozen wav2vec2 Vietnamese encoder with a trainable classifier head; "
            "uses offline embedding extraction for lower memory."
        ),
    ),
    "e4_phowhisper": ExperimentSpec(
        experiment_id="e4_phowhisper",
        aliases=("e4",),
        model_name="PhoWhisper-base encoder + classifier",
        input_type="phowhisper_input_features",
        pretrained="vinai/PhoWhisper-base",
        trainable_setting="reused_phase6_frozen_encoder_by_default",
        metric_path=Path("outputs/metrics/e4_phowhisper_results.json"),
        confusion_figure_path=Path(
            "outputs/figures/e4_phowhisper_confusion_matrix.png"
        ),
        checkpoint_path=Path("outputs/models/phowhisper_pretrained_frozen_encoder.pt"),
        training_log_path=None,
        notes="Reuses existing Phase 6 PhoWhisper metrics/checkpoint by default.",
    ),
    "e5_vipvl_chunkformer": ExperimentSpec(
        experiment_id="e5_vipvl_chunkformer",
        aliases=("e5",),
        model_name="ViP-VL / ChunkFormer encoder + classifier",
        input_type="waveform_16khz",
        pretrained="not_configured",
        trainable_setting="planned_frozen_encoder_then_partial_finetune",
        metric_path=Path("outputs/metrics/e5_vipvl_chunkformer_results.json"),
        confusion_figure_path=Path(
            "outputs/figures/e5_vipvl_chunkformer_confusion_matrix.png"
        ),
        checkpoint_path=Path("outputs/models/e5_vipvl_chunkformer_classifier.pt"),
        training_log_path=Path("outputs/metrics/e5_vipvl_chunkformer_training_log.csv"),
        notes=(
            "Plain PyTorch ChunkFormer-style waveform classifier; official "
            "ViP-VL/ChunkFormer dependency integration is recorded as a limitation."
        ),
    ),
    "e6_whisper_base": ExperimentSpec(
        experiment_id="e6_whisper_base",
        aliases=("e6", "e6_whisper"),
        model_name="Whisper-base original encoder + classifier",
        input_type="whisper_input_features",
        pretrained="openai/whisper-base",
        trainable_setting="frozen_encoder",
        metric_path=Path("outputs/metrics/e6_whisper_base_results.json"),
        confusion_figure_path=Path(
            "outputs/figures/e6_whisper_base_confusion_matrix.png"
        ),
        checkpoint_path=Path("outputs/models/e6_whisper_base_frozen_encoder.pt"),
        training_log_path=Path("outputs/metrics/e6_whisper_base_training_log.csv"),
        notes=(
            "Original OpenAI Whisper-base checkpoint with frozen encoder; "
            "same base-size family as PhoWhisper-base."
        ),
    ),
}

EXPERIMENT_ALIASES = {
    alias: spec.experiment_id
    for spec in EXPERIMENTS.values()
    for alias in (spec.experiment_id, *spec.aliases)
}

COMPARISON_FIELDS = [
    "experiment_id",
    "model_name",
    "input_type",
    "pretrained",
    "trainable_setting",
    "accuracy",
    "macro_f1",
    "northern_f1",
    "central_f1",
    "southern_f1",
    "model_size_mb",
    "inference_latency_ms",
    "training_time_minutes",
    "device",
    "status",
    "metrics_path",
    "notes",
]
METHOD_COMPARISON_FIELDS = [
    "method_id",
    "group",
    "input_type",
    "status",
    "valid_accuracy",
    "valid_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "model_size_mb",
    "latency_ms_per_sample",
    "device",
    "metrics_path",
    "notes",
]


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Phase 9 experiments. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc
    return torch


def canonical_experiment_id(experiment: str) -> str:
    if experiment == "all":
        return "all"
    if experiment not in EXPERIMENT_ALIASES:
        supported = ", ".join(("all", *sorted(EXPERIMENT_ALIASES)))
        raise ValueError(f"Unsupported experiment {experiment!r}; choose one of: {supported}.")
    return EXPERIMENT_ALIASES[experiment]


def selected_specs(experiment: str) -> list[ExperimentSpec]:
    canonical = canonical_experiment_id(experiment)
    if canonical == "all":
        return list(EXPERIMENTS.values())
    return [EXPERIMENTS[canonical]]


def build_model(spec: ExperimentSpec) -> Any:
    if spec.experiment_id == "e1_mobilenetv3":
        from src.models.mobilenetv3_classifier import MobileNetV3SmallClassifier

        return MobileNetV3SmallClassifier(num_classes=len(LABELS))
    if spec.experiment_id == "e2_efficientnetb0":
        from src.models.efficientnet_classifier import EfficientNetB0Classifier

        return EfficientNetB0Classifier(num_classes=len(LABELS))
    raise ValueError(f"Training model is not implemented for {spec.experiment_id}.")


def read_rows_by_split(
    metadata_path: Path,
    limit_per_split: int | None = None,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
    rows = read_preprocessed_metadata(metadata_path)
    all_rows_by_split = split_rows(rows)
    full_counts = split_label_counts(all_rows_by_split)
    if limit_per_split is None:
        return all_rows_by_split, full_counts
    return limit_rows_by_split(all_rows_by_split, limit_per_split), full_counts


def split_label_counts(
    rows_by_split: dict[str, list[dict[str, str]]],
) -> dict[str, dict[str, int]]:
    return {
        split: {
            label: sum(1 for row in rows if row["label"] == label)
            for label in LABELS
        }
        for split, rows in rows_by_split.items()
    }


def limit_rows_by_split(
    rows_by_split: dict[str, list[dict[str, str]]],
    limit_per_split: int,
) -> dict[str, list[dict[str, str]]]:
    if limit_per_split <= 0:
        raise ValueError("--limit-per-split must be positive.")
    limited: dict[str, list[dict[str, str]]] = {}
    for split, rows in rows_by_split.items():
        by_label = {label: [] for label in LABELS}
        for row in rows:
            by_label[row["label"]].append(row)

        quotas = balanced_label_quotas(limit_per_split)
        selected: list[dict[str, str]] = []
        for label in LABELS:
            selected.extend(by_label[label][: quotas[label]])

        if len(selected) < min(limit_per_split, len(rows)):
            selected_ids = {row["sample_id"] for row in selected}
            for row in rows:
                if row["sample_id"] not in selected_ids:
                    selected.append(row)
                if len(selected) >= limit_per_split:
                    break
        limited[split] = selected[:limit_per_split]
    return limited


def balanced_label_quotas(limit: int) -> dict[str, int]:
    base = limit // len(LABELS)
    remainder = limit % len(LABELS)
    return {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(LABELS)
    }


def train_logmel_experiment(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    device: Any,
    rows_by_split: dict[str, list[dict[str, str]]],
    full_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    torch = require_torch()
    set_seed(args.seed)

    started = time.perf_counter()
    features_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        features, labels = extract_logmel_features(rows_by_split[split])
        features_by_split[split] = features
        labels_by_split[split] = labels
        print(f"{spec.experiment_id}: extracted {split} {features.shape}", flush=True)

    loaders = {
        split: build_loader(
            features_by_split[split],
            labels_by_split[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            seed=args.seed,
        )
        for split in SPLITS
    }

    model = build_model(spec).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_epoch = 0
    best_valid_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    training_rows: list[dict[str, Any]] = []
    max_epochs = resolved_max_epochs(args)

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )
        valid_metrics, _valid_matrix, _true, _pred = evaluate_torch_classifier(
            model,
            loaders["valid"],
            criterion,
            device,
        )
        elapsed = time.perf_counter() - epoch_started
        training_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_accuracy": f"{train_metrics['accuracy']:.6f}",
                "valid_loss": f"{valid_metrics['loss']:.6f}",
                "valid_accuracy": f"{valid_metrics['accuracy']:.6f}",
                "valid_macro_f1": f"{valid_metrics['macro_f1']:.6f}",
                "epoch_seconds": f"{elapsed:.3f}",
            }
        )
        print(
            f"{spec.experiment_id}: epoch={epoch} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if valid_metrics["macro_f1"] > best_valid_macro_f1:
            best_valid_macro_f1 = float(valid_metrics["macro_f1"])
            best_epoch = epoch
            best_state = checkpoint_state(model, spec, epoch, args, device)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"{spec.experiment_id}: early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"{spec.experiment_id} finished without a best checkpoint.")
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    final_metrics: dict[str, Any] = {}
    final_matrices: dict[str, np.ndarray] = {}
    for split in SPLITS:
        metrics, matrix, _true, _pred = evaluate_torch_classifier(
            model,
            loaders[split],
            criterion,
            device,
        )
        final_metrics[split] = metrics
        final_matrices[split] = matrix

    spec.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, spec.checkpoint_path)
    if spec.training_log_path is not None:
        write_training_log(spec.training_log_path, training_rows)
    write_confusion_matrix_figure(
        spec.confusion_figure_path,
        final_matrices["test"],
        f"{spec.experiment_id} test confusion matrix",
    )

    latency = estimate_latency(
        model,
        features_by_split["valid"],
        device,
        sample_count=args.latency_samples,
    )
    training_time_minutes = (time.perf_counter() - started) / 60.0
    checkpoint_size_mb = file_size_mb(spec.checkpoint_path)

    result = {
        "phase": PHASE,
        "experiment_id": spec.experiment_id,
        "model_name": spec.model_name,
        "input_type": spec.input_type,
        "pretrained": spec.pretrained,
        "trainable_setting": spec.trainable_setting,
        "status": "trained",
        "run_mode": args.run_mode,
        "metadata_path": args.metadata_path.as_posix(),
        "label_order": list(LABELS),
        "device": str(device),
        "seed": args.seed,
        "split_counts_full_metadata": full_counts,
        "split_counts_used": split_label_counts(rows_by_split),
        "feature": {
            "name": "log_mel_spectrogram",
            "n_mels": DEFAULT_N_MELS,
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
            "input_shape": list(features_by_split["train"].shape[1:]),
            "standardized_per_sample": True,
        },
        "training": {
            "batch_size": args.batch_size,
            "max_epochs": max_epochs,
            "epochs_completed": len(training_rows),
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "training_time_minutes": training_time_minutes,
            "training_log_path": (
                spec.training_log_path.as_posix()
                if spec.training_log_path is not None
                else None
            ),
        },
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_macro_f1,
        "checkpoint_path": spec.checkpoint_path.as_posix(),
        "confusion_matrix_path": spec.confusion_figure_path.as_posix(),
        "model_size_mb": checkpoint_size_mb,
        "latency_estimate": latency,
        "metrics": {
            split: {
                **final_metrics[split],
                **(
                    {"confusion_matrix_path": spec.confusion_figure_path.as_posix()}
                    if split == "test"
                    else {}
                ),
            }
            for split in SPLITS
        },
        "notes": spec.notes,
    }
    write_json(spec.metric_path, result)
    return result


class WaveformRowsDataset:
    """Load fixed-length preprocessed waveforms on demand."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch = require_torch()
        row = self.rows[index]
        waveform = load_preprocessed_waveform(row)
        label = LABEL_TO_INDEX[row["label"]]
        return torch.from_numpy(waveform), torch.tensor(label, dtype=torch.long)


def load_preprocessed_waveform(row: dict[str, str]) -> np.ndarray:
    path = Path(row["preprocessed_audio_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed audio for {row['sample_id']} not found: {path}"
        )
    waveform, sample_rate = load_audio(path)
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Wrong sample rate for {path}: {sample_rate}")
    if waveform.shape != (TARGET_SAMPLES,):
        raise ValueError(f"Wrong waveform shape for {path}: {waveform.shape}")
    return waveform.astype(np.float32, copy=False)


def build_waveform_loader(
    rows: list[dict[str, str]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        WaveformRowsDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def train_waveform_experiment(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    device: Any,
    rows_by_split: dict[str, list[dict[str, str]]],
    full_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    torch = require_torch()
    from src.models.vipvl_chunkformer_classifier import VipvlChunkFormerTinyClassifier

    set_seed(args.seed)
    started = time.perf_counter()
    loaders = {
        split: build_waveform_loader(
            rows_by_split[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            seed=args.seed,
        )
        for split in SPLITS
    }

    model = VipvlChunkFormerTinyClassifier(
        num_classes=len(LABELS),
        dropout=args.dropout,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_epoch = 0
    best_valid_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    training_rows: list[dict[str, Any]] = []
    max_epochs = resolved_max_epochs(args)

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )
        valid_metrics, _valid_matrix, _true, _pred = evaluate_torch_classifier(
            model,
            loaders["valid"],
            criterion,
            device,
        )
        elapsed = time.perf_counter() - epoch_started
        training_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_accuracy": f"{train_metrics['accuracy']:.6f}",
                "valid_loss": f"{valid_metrics['loss']:.6f}",
                "valid_accuracy": f"{valid_metrics['accuracy']:.6f}",
                "valid_macro_f1": f"{valid_metrics['macro_f1']:.6f}",
                "epoch_seconds": f"{elapsed:.3f}",
            }
        )
        print(
            f"{spec.experiment_id}: epoch={epoch} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if valid_metrics["macro_f1"] > best_valid_macro_f1:
            best_valid_macro_f1 = float(valid_metrics["macro_f1"])
            best_epoch = epoch
            best_state = waveform_checkpoint_state(model, spec, epoch, args, device)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"{spec.experiment_id}: early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"{spec.experiment_id} finished without a best checkpoint.")
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    final_metrics: dict[str, Any] = {}
    final_matrices: dict[str, np.ndarray] = {}
    for split in SPLITS:
        metrics, matrix, _true, _pred = evaluate_torch_classifier(
            model,
            loaders[split],
            criterion,
            device,
        )
        final_metrics[split] = metrics
        final_matrices[split] = matrix

    spec.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, spec.checkpoint_path)
    if spec.training_log_path is not None:
        write_training_log(spec.training_log_path, training_rows)
    write_confusion_matrix_figure(
        spec.confusion_figure_path,
        final_matrices["test"],
        f"{spec.experiment_id} test confusion matrix",
    )

    latency = estimate_latency(
        model,
        sample_waveforms(rows_by_split["valid"], args.latency_samples),
        device,
        sample_count=args.latency_samples,
    )
    training_time_minutes = (time.perf_counter() - started) / 60.0
    checkpoint_size_mb = file_size_mb(spec.checkpoint_path)

    result = {
        "phase": PHASE,
        "experiment_id": spec.experiment_id,
        "model_name": spec.model_name,
        "input_type": spec.input_type,
        "pretrained": spec.pretrained,
        "trainable_setting": "all_parameters_trainable_from_scratch",
        "status": "trained",
        "run_mode": args.run_mode,
        "metadata_path": args.metadata_path.as_posix(),
        "label_order": list(LABELS),
        "device": str(device),
        "seed": args.seed,
        "split_counts_full_metadata": full_counts,
        "split_counts_used": split_label_counts(rows_by_split),
        "feature": {
            "name": "waveform_16khz",
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
            "duration_sec": TARGET_SAMPLES / TARGET_SAMPLE_RATE,
            "input_shape": [TARGET_SAMPLES],
        },
        "training": {
            "batch_size": args.batch_size,
            "max_epochs": max_epochs,
            "epochs_completed": len(training_rows),
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "training_time_minutes": training_time_minutes,
            "training_log_path": (
                spec.training_log_path.as_posix()
                if spec.training_log_path is not None
                else None
            ),
        },
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_macro_f1,
        "checkpoint_path": spec.checkpoint_path.as_posix(),
        "confusion_matrix_path": spec.confusion_figure_path.as_posix(),
        "model_size_mb": checkpoint_size_mb,
        "latency_estimate": latency,
        "metrics": {
            split: {
                **final_metrics[split],
                **(
                    {"confusion_matrix_path": spec.confusion_figure_path.as_posix()}
                    if split == "test"
                    else {}
                ),
            }
            for split in SPLITS
        },
        "notes": spec.notes,
    }
    write_json(spec.metric_path, result)
    return result


def train_wav2vec2_experiment(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    device: Any,
    rows_by_split: dict[str, list[dict[str, str]]],
    full_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    torch = require_torch()
    from src.models.wav2vec2_classifier import Wav2Vec2EmbeddingClassifier

    set_seed(args.seed)
    started = time.perf_counter()
    feature_extractor, encoder = load_wav2vec2_encoder(args, device)
    embedding_batches: dict[str, np.ndarray] = {}
    label_batches: dict[str, np.ndarray] = {}
    for split in SPLITS:
        embeddings, labels = extract_wav2vec2_embeddings(
            rows_by_split[split],
            feature_extractor,
            encoder,
            args,
            device,
            split,
        )
        embedding_batches[split] = embeddings
        label_batches[split] = labels
        print(f"{spec.experiment_id}: extracted {split} {embeddings.shape}", flush=True)

    loaders = {
        split: build_loader(
            embedding_batches[split],
            label_batches[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            seed=args.seed,
        )
        for split in SPLITS
    }

    embedding_dim = int(embedding_batches["train"].shape[1])
    model = Wav2Vec2EmbeddingClassifier(
        embedding_dim=embedding_dim,
        num_classes=len(LABELS),
        dropout=args.dropout,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate_head,
        weight_decay=args.weight_decay,
    )

    best_epoch = 0
    best_valid_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    training_rows: list[dict[str, Any]] = []
    max_epochs = resolved_max_epochs(args)

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )
        valid_metrics, _valid_matrix, _true, _pred = evaluate_torch_classifier(
            model,
            loaders["valid"],
            criterion,
            device,
        )
        elapsed = time.perf_counter() - epoch_started
        training_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_accuracy": f"{train_metrics['accuracy']:.6f}",
                "valid_loss": f"{valid_metrics['loss']:.6f}",
                "valid_accuracy": f"{valid_metrics['accuracy']:.6f}",
                "valid_macro_f1": f"{valid_metrics['macro_f1']:.6f}",
                "epoch_seconds": f"{elapsed:.3f}",
            }
        )
        print(
            f"{spec.experiment_id}: epoch={epoch} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if valid_metrics["macro_f1"] > best_valid_macro_f1:
            best_valid_macro_f1 = float(valid_metrics["macro_f1"])
            best_epoch = epoch
            best_state = wav2vec2_checkpoint_state(
                model,
                spec,
                epoch,
                args,
                device,
                embedding_dim,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"{spec.experiment_id}: early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"{spec.experiment_id} finished without a best checkpoint.")
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    final_metrics: dict[str, Any] = {}
    final_matrices: dict[str, np.ndarray] = {}
    for split in SPLITS:
        metrics, matrix, _true, _pred = evaluate_torch_classifier(
            model,
            loaders[split],
            criterion,
            device,
        )
        final_metrics[split] = metrics
        final_matrices[split] = matrix

    spec.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, spec.checkpoint_path)
    if spec.training_log_path is not None:
        write_training_log(spec.training_log_path, training_rows)
    write_confusion_matrix_figure(
        spec.confusion_figure_path,
        final_matrices["test"],
        f"{spec.experiment_id} test confusion matrix",
    )

    latency = estimate_wav2vec2_latency(
        feature_extractor,
        encoder,
        model,
        rows_by_split["valid"],
        args,
        device,
    )
    training_time_minutes = (time.perf_counter() - started) / 60.0
    classifier_size_mb = file_size_mb(spec.checkpoint_path)
    encoder_cache_size_mb = hf_cached_model_size_mb(args.cache_dir, args.wav2vec2_model_id)
    combined_size_mb = sum_optional_mb(classifier_size_mb, encoder_cache_size_mb)

    result = {
        "phase": PHASE,
        "experiment_id": spec.experiment_id,
        "model_name": spec.model_name,
        "input_type": spec.input_type,
        "pretrained": args.wav2vec2_model_id,
        "trainable_setting": spec.trainable_setting,
        "status": "trained",
        "run_mode": args.run_mode,
        "metadata_path": args.metadata_path.as_posix(),
        "label_order": list(LABELS),
        "device": str(device),
        "seed": args.seed,
        "split_counts_full_metadata": full_counts,
        "split_counts_used": split_label_counts(rows_by_split),
        "model_id": args.wav2vec2_model_id,
        "cache_dir": args.cache_dir.as_posix(),
        "feature": {
            "name": "wav2vec2_mean_pooled_embedding",
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
            "duration_sec": TARGET_SAMPLES / TARGET_SAMPLE_RATE,
            "embedding_dim": embedding_dim,
            "pooling": "mean",
        },
        "training": {
            "mode": "frozen_encoder_offline_embeddings",
            "batch_size": args.batch_size,
            "max_epochs": max_epochs,
            "epochs_completed": len(training_rows),
            "patience": args.patience,
            "learning_rate": args.learning_rate_head,
            "weight_decay": args.weight_decay,
            "training_time_minutes": training_time_minutes,
            "training_log_path": (
                spec.training_log_path.as_posix()
                if spec.training_log_path is not None
                else None
            ),
        },
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_macro_f1,
        "checkpoint_path": spec.checkpoint_path.as_posix(),
        "encoder_cache_size_mb": encoder_cache_size_mb,
        "classifier_checkpoint_size_mb": classifier_size_mb,
        "confusion_matrix_path": spec.confusion_figure_path.as_posix(),
        "model_size_mb": combined_size_mb,
        "latency_estimate": latency,
        "metrics": {
            split: {
                **final_metrics[split],
                **(
                    {"confusion_matrix_path": spec.confusion_figure_path.as_posix()}
                    if split == "test"
                    else {}
                ),
            }
            for split in SPLITS
        },
        "notes": spec.notes,
    }
    write_json(spec.metric_path, result)
    return result


def evaluate_torch_classifier(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
) -> tuple[dict[str, Any], np.ndarray, list[int], list[int]]:
    torch = require_torch()
    accuracy_score, classification_report, confusion_matrix, f1_score = (
        require_sklearn_metrics()
    )
    model.eval()
    total_loss = 0.0
    total_count = 0
    true_labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            predicted = torch.argmax(logits, dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size
            true_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
            predictions.extend(predicted.detach().cpu().numpy().astype(int).tolist())

    label_indexes = list(range(len(LABELS)))
    report = classification_report(
        true_labels,
        predictions,
        labels=label_indexes,
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predictions, labels=label_indexes)
    return (
        {
            "loss": total_loss / max(total_count, 1),
            "accuracy": float(accuracy_score(true_labels, predictions)),
            "macro_f1": float(
                f1_score(
                    true_labels,
                    predictions,
                    labels=label_indexes,
                    average="macro",
                )
            ),
            "per_class": {
                label: {
                    "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]),
                }
                for label in LABELS
            },
            "per_class_f1": {
                label: float(report[label]["f1-score"]) for label in LABELS
            },
        },
        matrix,
        true_labels,
        predictions,
    )


def checkpoint_state(
    model: Any,
    spec: ExperimentSpec,
    epoch: int,
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    return {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "epoch": epoch,
        "label_order": LABELS,
        "experiment_id": spec.experiment_id,
        "model": spec.model_name,
        "feature": "log_mel_spectrogram",
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "n_mels": DEFAULT_N_MELS,
        "device": str(device),
        "seed": args.seed,
        "run_mode": args.run_mode,
    }


def waveform_checkpoint_state(
    model: Any,
    spec: ExperimentSpec,
    epoch: int,
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    return {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "epoch": epoch,
        "label_order": LABELS,
        "experiment_id": spec.experiment_id,
        "model": spec.model_name,
        "feature": "waveform_16khz",
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "device": str(device),
        "seed": args.seed,
        "run_mode": args.run_mode,
    }


def wav2vec2_checkpoint_state(
    model: Any,
    spec: ExperimentSpec,
    epoch: int,
    args: argparse.Namespace,
    device: Any,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "epoch": epoch,
        "label_order": LABELS,
        "experiment_id": spec.experiment_id,
        "model": spec.model_name,
        "feature": "wav2vec2_mean_pooled_embedding",
        "model_id": args.wav2vec2_model_id,
        "embedding_dim": embedding_dim,
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "device": str(device),
        "seed": args.seed,
        "run_mode": args.run_mode,
    }


def load_wav2vec2_encoder(args: argparse.Namespace, device: Any) -> tuple[Any, Any]:
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for E3 wav2vec2 training. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc

    local_files_only = not args.allow_download
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.wav2vec2_model_id,
        cache_dir=args.cache_dir,
        local_files_only=local_files_only,
    )
    encoder = AutoModel.from_pretrained(
        args.wav2vec2_model_id,
        cache_dir=args.cache_dir,
        local_files_only=local_files_only,
    )
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return feature_extractor, encoder


def extract_wav2vec2_embeddings(
    rows: list[dict[str, str]],
    feature_extractor: Any,
    encoder: Any,
    args: argparse.Namespace,
    device: Any,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch = require_torch()
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    total_batches = math.ceil(len(rows) / args.batch_size)
    for batch_index, start in enumerate(range(0, len(rows), args.batch_size), start=1):
        batch_rows = rows[start : start + args.batch_size]
        waveforms = [load_preprocessed_waveform(row) for row in batch_rows]
        inputs = feature_extractor(
            waveforms,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }
        with torch.no_grad():
            outputs = encoder(**model_inputs)
            pooled = mean_pool_wav2vec2(
                encoder,
                outputs.last_hidden_state,
                model_inputs.get("attention_mask"),
            )
        embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
        labels.extend(LABEL_TO_INDEX[row["label"]] for row in batch_rows)
        if batch_index == 1 or batch_index == total_batches or batch_index % 50 == 0:
            print(
                f"e3_wav2vec2: extracting {split} "
                f"batch {batch_index}/{total_batches}",
                flush=True,
            )
    return np.concatenate(embeddings, axis=0), np.asarray(labels, dtype=np.int64)


def mean_pool_wav2vec2(
    encoder: Any,
    hidden_states: Any,
    attention_mask: Any | None,
) -> Any:
    if attention_mask is None or not hasattr(encoder, "_get_feature_vector_attention_mask"):
        return hidden_states.mean(dim=1)
    feature_attention_mask = encoder._get_feature_vector_attention_mask(
        hidden_states.shape[1],
        attention_mask,
    ).to(hidden_states.device)
    mask = feature_attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def estimate_wav2vec2_latency(
    feature_extractor: Any,
    encoder: Any,
    classifier: Any,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    torch = require_torch()
    count = min(max(args.latency_samples, 0), len(rows))
    if count == 0:
        return {"sample_count": 0, "mean_milliseconds_per_sample": None}
    elapsed_values: list[float] = []
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for row in rows[:count]:
            waveform = load_preprocessed_waveform(row)
            synchronize_device(torch, device)
            started = time.perf_counter()
            inputs = feature_extractor(
                [waveform],
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }
            outputs = encoder(**model_inputs)
            pooled = mean_pool_wav2vec2(
                encoder,
                outputs.last_hidden_state,
                model_inputs.get("attention_mask"),
            )
            _logits = classifier(pooled)
            synchronize_device(torch, device)
            elapsed_values.append((time.perf_counter() - started) * 1000.0)
    return {
        "sample_count": count,
        "mean_milliseconds_per_sample": float(np.mean(elapsed_values)),
    }


def sample_waveforms(rows: list[dict[str, str]], sample_count: int) -> np.ndarray:
    selected = rows[: min(max(sample_count, 0), len(rows))]
    if not selected:
        return np.zeros((0, TARGET_SAMPLES), dtype=np.float32)
    return np.stack([load_preprocessed_waveform(row) for row in selected]).astype(
        np.float32,
        copy=False,
    )


def hf_cached_model_size_mb(cache_dir: Path, model_id: str) -> float | None:
    model_cache = cache_dir / f"models--{model_id.replace('/', '--')}"
    return directory_size_mb(model_cache)


def directory_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024 * 1024)


def sum_optional_mb(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(sum(present))


def estimate_latency(
    model: Any,
    features: np.ndarray,
    device: Any,
    sample_count: int,
) -> dict[str, Any]:
    torch = require_torch()
    count = min(max(sample_count, 0), int(features.shape[0]))
    if count == 0:
        return {"sample_count": 0, "mean_milliseconds_per_sample": None}
    elapsed_values: list[float] = []
    model.eval()
    with torch.no_grad():
        for index in range(count):
            sample = torch.from_numpy(features[index : index + 1]).to(device)
            synchronize_device(torch, device)
            started = time.perf_counter()
            _logits = model(sample)
            synchronize_device(torch, device)
            elapsed_values.append((time.perf_counter() - started) * 1000.0)
    return {
        "sample_count": count,
        "mean_milliseconds_per_sample": float(np.mean(elapsed_values)),
    }


def synchronize_device(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def e3_wav2vec2_result(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    device: Any,
    full_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    available, message = local_transformers_config_available(
        args.wav2vec2_model_id,
        args.cache_dir,
    )
    if available:
        status = "setup_only"
        notes = (
            f"Local config for {args.wav2vec2_model_id} is available, but full "
            "wav2vec2 training is intentionally not run in Phase 9 smoke mode."
        )
    else:
        status = "skipped"
        notes = (
            f"Local wav2vec2 checkpoint is unavailable and network download is "
            f"disabled by default: {message}"
        )
    result = base_non_training_result(spec, args, device, full_counts, status, notes)
    result["model_id"] = args.wav2vec2_model_id
    result["cache_dir"] = args.cache_dir.as_posix()
    write_empty_confusion_figure(spec.confusion_figure_path, spec.experiment_id, status)
    write_json(spec.metric_path, result)
    return result


def e4_phowhisper_reuse_result(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    device: Any,
    full_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    source = read_json_or_none(args.phowhisper_source_metrics_path)
    if not source:
        result = base_non_training_result(
            spec,
            args,
            device,
            full_counts,
            "skipped",
            f"Source Phase 6 metrics not found: {args.phowhisper_source_metrics_path}",
        )
        write_empty_confusion_figure(
            spec.confusion_figure_path,
            spec.experiment_id,
            "skipped",
        )
        write_json(spec.metric_path, result)
        return result

    source_metrics = source.get("metrics", {})
    latency_seconds = source.get("latency_estimate", {}).get(
        "mean_seconds_per_sample"
    )
    result = {
        "phase": PHASE,
        "experiment_id": spec.experiment_id,
        "model_name": spec.model_name,
        "input_type": spec.input_type,
        "pretrained": spec.pretrained,
        "trainable_setting": source.get("training", {}).get(
            "mode",
            spec.trainable_setting,
        ),
        "status": "reused",
        "run_mode": args.run_mode,
        "metadata_path": source.get("metadata_path", args.metadata_path.as_posix()),
        "label_order": source.get("label_order", list(LABELS)),
        "device": source.get("device", str(device)),
        "split_counts_full_metadata": full_counts,
        "split_counts_used": source.get("split_counts", {}),
        "checkpoint_path": source.get(
            "checkpoint_path",
            args.phowhisper_source_checkpoint_path.as_posix(),
        ),
        "source_metrics_path": args.phowhisper_source_metrics_path.as_posix(),
        "confusion_matrix_path": spec.confusion_figure_path.as_posix(),
        "model_size_mb": source.get("model_size_mb"),
        "latency_estimate": {
            "sample_count": source.get("latency_estimate", {}).get("sample_count"),
            "mean_milliseconds_per_sample": (
                float(latency_seconds) * 1000.0
                if latency_seconds is not None
                else None
            ),
        },
        "training": {
            "mode": "reused_phase6_metrics",
            "training_time_minutes": None,
            "source_training": source.get("training", {}),
        },
        "metrics": source_metrics,
        "notes": spec.notes,
    }
    matrix = read_confusion_matrix_csv_from_metrics(source_metrics.get("test", {}))
    write_confusion_matrix_figure(
        spec.confusion_figure_path,
        matrix if matrix is not None else np.zeros((len(LABELS), len(LABELS))),
        f"{spec.experiment_id} reused Phase 6 test confusion matrix",
    )
    write_json(spec.metric_path, result)
    return result


def e5_skipped_result(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    device: Any,
    full_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    result = base_non_training_result(
        spec,
        args,
        device,
        full_counts,
        "skipped",
        "ViP-VL/ChunkFormer checkpoint and dependency integration are not configured.",
    )
    write_empty_confusion_figure(spec.confusion_figure_path, spec.experiment_id, "skipped")
    write_json(spec.metric_path, result)
    return result


def e6_whisper_reuse_result(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    device: Any,
    full_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    source = read_json_or_none(spec.metric_path)
    if not source:
        result = base_non_training_result(
            spec,
            args,
            device,
            full_counts,
            "skipped",
            "E6 Whisper-base metrics are not available yet. Run src.training.train_e6_whisper first.",
        )
        write_empty_confusion_figure(
            spec.confusion_figure_path,
            spec.experiment_id,
            "skipped",
        )
        write_json(spec.metric_path, result)
        return result

    source.setdefault("experiment_id", spec.experiment_id)
    source.setdefault("model_name", spec.model_name)
    source.setdefault("input_type", spec.input_type)
    source.setdefault("pretrained", spec.pretrained)
    source.setdefault("trainable_setting", spec.trainable_setting)
    source.setdefault("status", "trained")
    source.setdefault("notes", spec.notes)
    source.setdefault("split_counts_full_metadata", full_counts)
    matrix = read_confusion_matrix_csv_from_metrics(
        source.get("metrics", {}).get("test", {})
    )
    write_confusion_matrix_figure(
        spec.confusion_figure_path,
        matrix if matrix is not None else np.zeros((len(LABELS), len(LABELS))),
        f"{spec.experiment_id} test confusion matrix",
    )
    source["confusion_matrix_path"] = spec.confusion_figure_path.as_posix()
    write_json(spec.metric_path, source)
    return source


def base_non_training_result(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    device: Any,
    full_counts: dict[str, dict[str, int]],
    status: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "experiment_id": spec.experiment_id,
        "model_name": spec.model_name,
        "input_type": spec.input_type,
        "pretrained": spec.pretrained,
        "trainable_setting": spec.trainable_setting,
        "status": status,
        "run_mode": args.run_mode,
        "metadata_path": args.metadata_path.as_posix(),
        "label_order": list(LABELS),
        "device": str(device),
        "split_counts_full_metadata": full_counts,
        "split_counts_used": {},
        "checkpoint_path": spec.checkpoint_path.as_posix(),
        "confusion_matrix_path": spec.confusion_figure_path.as_posix(),
        "model_size_mb": file_size_mb(spec.checkpoint_path),
        "latency_estimate": {
            "sample_count": 0,
            "mean_milliseconds_per_sample": None,
        },
        "training": {
            "mode": status,
            "training_time_minutes": None,
        },
        "metrics": {"valid": empty_metrics(), "test": empty_metrics()},
        "notes": notes,
    }


def local_transformers_config_available(
    model_id: str,
    cache_dir: Path,
) -> tuple[bool, str]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        return False, f"transformers import failed: {exc}"
    try:
        AutoConfig.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception as exc:  # transformers raises several cache-specific errors.
        return False, f"{type(exc).__name__}: {exc}"
    return True, "local config found"


def empty_metrics() -> dict[str, Any]:
    return {
        "accuracy": None,
        "macro_f1": None,
        "per_class": {
            label: {"precision": None, "recall": None, "f1": None}
            for label in LABELS
        },
        "per_class_f1": {label: None for label in LABELS},
    }


def read_confusion_matrix_csv_from_metrics(metrics: dict[str, Any]) -> np.ndarray | None:
    path_value = metrics.get("confusion_matrix_path")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    rows: list[list[int]] = []
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file)
        _header = next(reader, None)
        for row in reader:
            rows.append([int(value) for value in row[1:]])
    if len(rows) != len(LABELS):
        return None
    return np.asarray(rows, dtype=np.int64)


def write_confusion_matrix_figure(path: Path, matrix: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(LABELS)), labels=LABELS, rotation=30, ha="right")
    axis.set_yticks(range(len(LABELS)), labels=LABELS)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(int(matrix[row_index, column_index])),
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_empty_confusion_figure(path: Path, experiment_id: str, status: str) -> None:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    write_confusion_matrix_figure(
        path,
        matrix,
        f"{experiment_id} confusion matrix ({status})",
    )


def comparison_row(result: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    valid_metrics = result.get("metrics", {}).get("valid", {})
    per_class_f1 = valid_metrics.get("per_class_f1") or {}
    training = result.get("training", {})
    return {
        "experiment_id": result["experiment_id"],
        "model_name": result["model_name"],
        "input_type": result["input_type"],
        "pretrained": result["pretrained"],
        "trainable_setting": result["trainable_setting"],
        "accuracy": valid_metrics.get("accuracy"),
        "macro_f1": valid_metrics.get("macro_f1"),
        "northern_f1": per_class_f1.get("Northern"),
        "central_f1": per_class_f1.get("Central"),
        "southern_f1": per_class_f1.get("Southern"),
        "model_size_mb": result.get("model_size_mb"),
        "inference_latency_ms": latency_ms_from_result(result),
        "training_time_minutes": training.get("training_time_minutes"),
        "device": result.get("device", ""),
        "status": result.get("status", ""),
        "metrics_path": metrics_path.as_posix(),
        "notes": clean_text(result.get("notes", "")),
    }


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: format_value(row.get(field)) for field in COMPARISON_FIELDS}
            )


def write_comparison_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["experiment_id"].replace("_", "\n") for row in rows]
    values = [float(row.get("macro_f1") or 0.0) for row in rows]
    colors = [
        "#4C78A8" if row.get("status") in {"trained", "reused"} else "#BAB0AC"
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values, color=colors)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Validation macro F1")
    axis.set_title("Phase 9 deep-learning comparison")
    for index, row in enumerate(rows):
        if row.get("status") not in {"trained", "reused"}:
            axis.text(index, 0.04, row.get("status", ""), ha="center", rotation=90)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 9 Extended Deep Learning Experiments",
        "",
        "This report is generated by the Phase 9 runner. E1/E2 train on "
        "log-Mel features, E3 trains a frozen Vietnamese wav2vec2 embedding "
        "classifier, E4 reuses Phase 6 PhoWhisper results, and E5 trains the "
        "local ChunkFormer-style waveform classifier.",
        "",
        "| Experiment | Status | Valid Accuracy | Valid Macro F1 | Latency (ms/sample) | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_id']} | {row['status']} | "
            f"{format_value(row.get('accuracy')) or 'N/A'} | "
            f"{format_value(row.get('macro_f1')) or 'N/A'} | "
            f"{format_value(row.get('inference_latency_ms')) or 'N/A'} | "
            f"{clean_text(row.get('notes', ''))} |"
        )
    lines.extend(
        [
            "",
            "Model selection should still use validation macro F1. Test metrics "
            "are reported only for final comparison after training.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_phase9_summary_from_available() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in EXPERIMENTS.values():
        result = read_json_or_none(spec.metric_path)
        if result is None:
            continue
        rows.append(comparison_row(result, spec.metric_path))
    if not rows:
        return []
    write_comparison_csv(Path("outputs/metrics/deep_learning_comparison.csv"), rows)
    write_comparison_figure(Path("outputs/figures/deep_learning_comparison.png"), rows)
    write_report(Path("outputs/reports/extended_deep_learning_experiments.md"), rows)
    return rows


def write_method_comparison_from_available() -> list[dict[str, Any]]:
    rows = collect_method_comparison_rows()
    if not rows:
        return []
    comparison_path = Path("outputs/metrics/model_method_comparison.csv")
    metrics_figure_path = Path("outputs/figures/model_method_comparison_metrics.png")
    tradeoffs_figure_path = Path("outputs/figures/model_method_comparison_tradeoffs.png")
    write_method_comparison_csv(comparison_path, rows)
    write_method_metric_figure(metrics_figure_path, rows)
    write_method_tradeoff_figure(tradeoffs_figure_path, rows)
    return rows


def collect_method_comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    append_baseline_comparison_rows(
        rows,
        Path("outputs/metrics/baseline_results.json"),
    )
    append_cnn_comparison_row(rows, Path("outputs/metrics/cnn_results.json"))
    for spec in EXPERIMENTS.values():
        append_phase9_comparison_row(rows, spec)
    append_phase10_comparison_row(rows, PHASE10_METRIC_PATH)
    return rows


def append_baseline_comparison_rows(rows: list[dict[str, Any]], path: Path) -> None:
    result = read_json_or_none(path)
    if result is None:
        return
    for model_name, model_result in result.get("models", {}).items():
        rows.append(
            {
                "method_id": model_name,
                "group": "phase4_mfcc_baseline",
                "input_type": "mfcc_mean_std",
                "status": "trained",
                "valid_accuracy": model_result.get("valid", {}).get("accuracy"),
                "valid_macro_f1": model_result.get("valid", {}).get("macro_f1"),
                "test_accuracy": model_result.get("test", {}).get("accuracy"),
                "test_macro_f1": model_result.get("test", {}).get("macro_f1"),
                "model_size_mb": file_size_mb(Path(model_result.get("model_path", ""))),
                "latency_ms_per_sample": None,
                "device": "cpu",
                "metrics_path": path.as_posix(),
                "notes": "MFCC mean/std traditional baseline.",
            }
        )


def append_cnn_comparison_row(rows: list[dict[str, Any]], path: Path) -> None:
    result = read_json_or_none(path)
    if result is None:
        return
    metrics = result.get("metrics", {})
    rows.append(
        {
            "method_id": "lightweight_cnn",
            "group": result.get("phase", "phase5_lightweight_cnn"),
            "input_type": "log_mel_spectrogram",
            "status": "trained",
            "valid_accuracy": metrics.get("valid", {}).get("accuracy"),
            "valid_macro_f1": metrics.get("valid", {}).get("macro_f1"),
            "test_accuracy": metrics.get("test", {}).get("accuracy"),
            "test_macro_f1": metrics.get("test", {}).get("macro_f1"),
            "model_size_mb": file_size_mb(Path(result.get("checkpoint_path", ""))),
            "latency_ms_per_sample": latency_ms_from_result(result),
            "device": result.get("device", ""),
            "metrics_path": path.as_posix(),
            "notes": "Main Phase 5 lightweight CNN.",
        }
    )


def append_phase9_comparison_row(rows: list[dict[str, Any]], spec: ExperimentSpec) -> None:
    result = read_json_or_none(spec.metric_path)
    if result is None:
        return
    metrics = result.get("metrics", {})
    rows.append(
        {
            "method_id": spec.experiment_id,
            "group": PHASE,
            "input_type": result.get("input_type", spec.input_type),
            "status": result.get("status", ""),
            "valid_accuracy": metrics.get("valid", {}).get("accuracy"),
            "valid_macro_f1": metrics.get("valid", {}).get("macro_f1"),
            "test_accuracy": metrics.get("test", {}).get("accuracy"),
            "test_macro_f1": metrics.get("test", {}).get("macro_f1"),
            "model_size_mb": result.get("model_size_mb"),
            "latency_ms_per_sample": latency_ms_from_result(result),
            "device": result.get("device", ""),
            "metrics_path": spec.metric_path.as_posix(),
            "notes": result.get("notes", spec.notes),
        }
    )


def append_phase10_comparison_row(rows: list[dict[str, Any]], path: Path) -> None:
    result = read_json_or_none(path)
    if result is None:
        return
    metrics = result.get("metrics", {})
    rows.append(
        {
            "method_id": result.get("experiment_id", "e7_whisper_cnn_fusion"),
            "group": result.get("phase", PHASE10),
            "input_type": result.get(
                "input_type",
                "waveform_16khz_to_whisper_features_and_log_mel",
            ),
            "status": result.get("status", ""),
            "valid_accuracy": metrics.get("valid", {}).get("accuracy"),
            "valid_macro_f1": metrics.get("valid", {}).get("macro_f1"),
            "test_accuracy": metrics.get("test", {}).get("accuracy"),
            "test_macro_f1": metrics.get("test", {}).get("macro_f1"),
            "model_size_mb": result.get("model_size_mb"),
            "latency_ms_per_sample": latency_ms_from_result(result),
            "device": result.get("device", ""),
            "metrics_path": path.as_posix(),
            "notes": result.get(
                "notes",
                "Hybrid frozen PhoWhisper encoder plus trainable log-Mel CNN branch.",
            ),
        }
    )


def latency_ms_from_result(result: dict[str, Any]) -> float | None:
    latency = result.get("latency_estimate", {})
    milliseconds = latency.get("mean_milliseconds_per_sample")
    if milliseconds is not None:
        return float(milliseconds)
    seconds = latency.get("mean_seconds_per_sample")
    if seconds is not None:
        return float(seconds) * 1000.0
    return None


def write_method_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METHOD_COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: format_value(row.get(field))
                    for field in METHOD_COMPARISON_FIELDS
                }
            )


def write_method_metric_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plottable = [row for row in rows if row.get("test_macro_f1") is not None]
    if not plottable:
        return
    labels = [row["method_id"].replace("_", "\n") for row in plottable]
    x = np.arange(len(plottable))
    width = 0.2
    series = [
        ("Valid accuracy", "valid_accuracy", "#4C78A8"),
        ("Valid macro F1", "valid_macro_f1", "#F58518"),
        ("Test accuracy", "test_accuracy", "#54A24B"),
        ("Test macro F1", "test_macro_f1", "#E45756"),
    ]
    figure, axis = plt.subplots(figsize=(12, 5))
    for offset, (label, key, color) in enumerate(series):
        values = [float(row.get(key) or 0.0) for row in plottable]
        axis.bar(
            x + (offset - 1.5) * width,
            values,
            width=width,
            label=label,
            color=color,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Vietnamese dialect classification methods")
    axis.set_xticks(x, labels=labels)
    axis.legend(ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_method_tradeoff_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plottable = [row for row in rows if row.get("test_macro_f1") is not None]
    if not plottable:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    size_rows = [
        row
        for row in plottable
        if row.get("model_size_mb") is not None and float(row["model_size_mb"]) > 0
    ]
    latency_rows = [
        row
        for row in plottable
        if row.get("latency_ms_per_sample") is not None
        and float(row["latency_ms_per_sample"]) > 0
    ]
    draw_tradeoff_panel(
        axes[0],
        size_rows,
        x_key="model_size_mb",
        x_label="Model size (MB, log scale)",
        use_log_x=True,
    )
    draw_tradeoff_panel(
        axes[1],
        latency_rows,
        x_key="latency_ms_per_sample",
        x_label="Latency (ms/sample, log scale)",
        use_log_x=True,
    )
    figure.suptitle("Performance vs deploy trade-offs")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def draw_tradeoff_panel(
    axis: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    x_label: str,
    use_log_x: bool,
) -> None:
    if not rows:
        axis.text(0.5, 0.5, "No measured values", ha="center", va="center")
        axis.set_axis_off()
        return
    x_values = [float(row[x_key]) for row in rows]
    y_values = [float(row["test_macro_f1"]) for row in rows]
    colors = [
        "#4C78A8" if row.get("status") in {"trained", "reused"} else "#BAB0AC"
        for row in rows
    ]
    axis.scatter(x_values, y_values, color=colors, s=60)
    for x_value, y_value, row in zip(x_values, y_values, rows):
        axis.annotate(
            row["method_id"],
            (x_value, y_value),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    if use_log_x:
        axis.set_xscale("log")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Test macro F1")
    axis.grid(True, alpha=0.25)


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing Phase 9 outputs: {formatted}. "
            "Pass --overwrite to regenerate them."
        )


def output_paths_for(specs: list[ExperimentSpec], include_summary: bool) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        paths.extend([spec.metric_path, spec.confusion_figure_path])
        if spec.experiment_id in {
            "e1_mobilenetv3",
            "e2_efficientnetb0",
            "e3_wav2vec2",
            "e5_vipvl_chunkformer",
            "e6_whisper_base",
        }:
            paths.append(spec.checkpoint_path)
        if spec.training_log_path is not None:
            paths.append(spec.training_log_path)
    if include_summary:
        paths.extend(
            [
                Path("outputs/metrics/deep_learning_comparison.csv"),
                Path("outputs/figures/deep_learning_comparison.png"),
                Path("outputs/reports/extended_deep_learning_experiments.md"),
            ]
        )
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def file_size_mb(path: Path) -> float | None:
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return clean_text(str(value))


def clean_text(value: str) -> str:
    return " ".join(str(value).split())


def resolved_max_epochs(args: argparse.Namespace) -> int:
    if args.max_epochs is not None:
        return args.max_epochs
    if args.run_mode == "full":
        return DEFAULT_FULL_EPOCHS
    return DEFAULT_SMOKE_EPOCHS


def resolved_limit_per_split(args: argparse.Namespace) -> int | None:
    if args.limit_per_split is not None:
        return args.limit_per_split
    if args.run_mode in {"smoke", "setup"}:
        return DEFAULT_SMOKE_LIMIT_PER_SPLIT
    return None


def parse_args(default_experiment: str = "all") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 9 extended deep-learning experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=("all", *sorted(EXPERIMENT_ALIASES)),
        default=default_experiment,
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/processed/preprocessed_metadata.csv"),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke", dest="run_mode", action="store_const", const="smoke")
    mode_group.add_argument("--full", dest="run_mode", action="store_const", const="full")
    parser.add_argument(
        "--mode",
        choices=("setup", "smoke", "full"),
        default=None,
        help="Backward-compatible mode selector; --smoke/--full are preferred.",
    )
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--learning-rate-head",
        type=float,
        default=DEFAULT_LEARNING_RATE_HEAD,
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--latency-samples", type=int, default=5)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/models/hf_cache"),
    )
    parser.add_argument("--wav2vec2-model-id", default=DEFAULT_WAV2VEC2_MODEL_ID)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads for pretrained E3 assets.",
    )
    parser.add_argument(
        "--phowhisper-source-metrics-path",
        type=Path,
        default=Path("outputs/metrics/phowhisper_pretrained_results.json"),
    )
    parser.add_argument(
        "--phowhisper-source-checkpoint-path",
        type=Path,
        default=Path("outputs/models/phowhisper_pretrained_frozen_encoder.pt"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode is not None and args.run_mode is not None and args.mode != args.run_mode:
        raise ValueError("Use only one of --mode, --smoke, or --full.")
    args.run_mode = args.mode or args.run_mode or "smoke"
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.max_epochs is not None and args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.learning_rate_head <= 0:
        raise ValueError("--learning-rate-head must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.latency_samples < 0:
        raise ValueError("--latency-samples cannot be negative.")
    return args


def main(default_experiment: str = "all") -> None:
    args = parse_args(default_experiment)
    specs = selected_specs(args.experiment)
    include_summary = canonical_experiment_id(args.experiment) == "all"
    output_paths = output_paths_for(specs, include_summary=include_summary)
    if not args.overwrite:
        ensure_outputs_absent(output_paths)

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    rows_by_split, full_counts = read_rows_by_split(
        args.metadata_path,
        limit_per_split=resolved_limit_per_split(args),
    )

    results: list[dict[str, Any]] = []
    for spec in specs:
        if spec.experiment_id in {"e1_mobilenetv3", "e2_efficientnetb0"}:
            result = train_logmel_experiment(spec, args, device, rows_by_split, full_counts)
        elif spec.experiment_id == "e3_wav2vec2":
            result = train_wav2vec2_experiment(
                spec,
                args,
                device,
                rows_by_split,
                full_counts,
            )
        elif spec.experiment_id == "e4_phowhisper":
            result = e4_phowhisper_reuse_result(spec, args, device, full_counts)
        elif spec.experiment_id == "e5_vipvl_chunkformer":
            result = train_waveform_experiment(
                spec,
                args,
                device,
                rows_by_split,
                full_counts,
            )
        elif spec.experiment_id == "e6_whisper_base":
            result = e6_whisper_reuse_result(spec, args, device, full_counts)
        else:
            raise ValueError(f"Unsupported experiment: {spec.experiment_id}")
        results.append(result)
        print(
            f"{spec.experiment_id}: status={result['status']} "
            f"metrics={spec.metric_path}",
            flush=True,
        )

    rows = [
        comparison_row(result, EXPERIMENTS[result["experiment_id"]].metric_path)
        for result in results
    ]
    if include_summary:
        comparison_path = Path("outputs/metrics/deep_learning_comparison.csv")
        comparison_figure_path = Path("outputs/figures/deep_learning_comparison.png")
        report_path = Path("outputs/reports/extended_deep_learning_experiments.md")
        write_comparison_csv(comparison_path, rows)
        write_comparison_figure(comparison_figure_path, rows)
        write_report(report_path, rows)
        print(f"Phase 9 comparison written to {comparison_path}", flush=True)

    phase9_rows = write_phase9_summary_from_available()
    method_rows = write_method_comparison_from_available()
    if phase9_rows:
        print("Updated Phase 9 summary artifacts.", flush=True)
    if method_rows:
        print("Updated full method comparison artifacts.", flush=True)


if __name__ == "__main__":
    main()
