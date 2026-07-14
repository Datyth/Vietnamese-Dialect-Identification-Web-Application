"""Train Phase 10 hybrid PhoWhisper + CNN fusion experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.features.logmel import DEFAULT_N_MELS, log_mel_spectrogram
from src.models.whisper_cnn_fusion import (
    WhisperCnnFusionClassifier,
    count_parameters,
    infer_whisper_hidden_size,
)
from src.training.train_baseline import write_confusion_matrix
from src.training.train_cnn import (
    require_sklearn_metrics,
    resolve_device,
    set_seed,
)
from src.training.train_extended_deep_learning import (
    LABELS,
    LABEL_TO_INDEX,
    SPLITS,
    file_size_mb,
    hf_cached_model_size_mb,
    limit_rows_by_split,
    read_preprocessed_metadata,
    split_label_counts,
    split_rows,
    write_method_comparison_from_available,
)
from src.utils.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES, load_audio


PHASE = "phase10_whisper_cnn_fusion"
EXPERIMENT_ID = "e7_whisper_cnn_fusion"
DEFAULT_MODEL_ID = "vinai/PhoWhisper-base"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_EPOCHS = 20
DEFAULT_PATIENCE = 5
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.0
DEFAULT_FUSION_TYPE = "gated"
DEFAULT_LOCAL_EMBEDDING_DIM = 128
DEFAULT_FUSION_DIM = 512
DEFAULT_CLASSIFIER_HIDDEN_DIM = 256
DEFAULT_CNN_TRAINABLE_LAYERS = 2
DEFAULT_CNN_LEARNING_RATE = 1e-5
DEFAULT_CNN_CHECKPOINT_PATH = Path("outputs/models/e2_efficientnetb0_logmel.pt")


class WhisperCnnRowsDataset:
    """Build Whisper input features and log-Mel features from one waveform."""

    def __init__(self, rows: list[dict[str, str]], feature_extractor: Any) -> None:
        self.rows = rows
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = require_torch()
        row = self.rows[index]
        waveform = load_preprocessed_waveform(row)
        encoded = self.feature_extractor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="np",
        )
        whisper_features = np.asarray(encoded["input_features"][0], dtype=np.float32)
        logmel = log_mel_spectrogram(waveform, sample_rate=TARGET_SAMPLE_RATE)
        label = LABEL_TO_INDEX[row["label"]]
        return {
            "whisper_input_features": torch.from_numpy(whisper_features),
            "logmel": torch.from_numpy(logmel[None, :, :]),
            "label": torch.tensor(label, dtype=torch.long),
        }


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Phase 10 fusion training. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc
    return torch


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoFeatureExtractor, WhisperForAudioClassification
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Phase 10 fusion training. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc
    return AutoFeatureExtractor, WhisperForAudioClassification


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


def build_loader(
    rows: list[dict[str, str]],
    feature_extractor: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = require_torch()
    dataset = WhisperCnnRowsDataset(rows, feature_extractor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def load_frozen_whisper_encoder(args: argparse.Namespace, device: Any) -> tuple[Any, Any]:
    AutoFeatureExtractor, WhisperForAudioClassification = require_transformers()
    local_files_only = not args.allow_download
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=local_files_only,
    )
    whisper_model = WhisperForAudioClassification.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=local_files_only,
    )
    encoder = whisper_model.encoder
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return feature_extractor, encoder


def load_efficientnet_encoder(args: argparse.Namespace, device: Any) -> tuple[Any, dict[str, Any]]:
    torch = require_torch()
    from src.models.efficientnet_classifier import EfficientNetB0Classifier

    path = args.cnn_checkpoint_path
    if not path.exists():
        raise FileNotFoundError(
            f"EfficientNetB0 checkpoint not found: {path}. Run E2 first with "
            "scripts/train_e3_e5_mps.sh or src.training.train_e2_efficientnet."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unsupported EfficientNet checkpoint format in {path}; expected model_state_dict."
        )

    labels = tuple(checkpoint.get("label_order", ()))
    contracts = {
        "experiment_id": (checkpoint.get("experiment_id"), "e2_efficientnetb0"),
        "feature": (checkpoint.get("feature"), "log_mel_spectrogram"),
        "sample_rate": (checkpoint.get("sample_rate"), TARGET_SAMPLE_RATE),
        "target_samples": (checkpoint.get("target_samples"), TARGET_SAMPLES),
        "n_mels": (checkpoint.get("n_mels"), DEFAULT_N_MELS),
        "label_order": (labels, tuple(LABELS)),
    }
    mismatches = {
        name: {"checkpoint": actual, "code": expected}
        for name, (actual, expected) in contracts.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"EfficientNet checkpoint/training contract mismatch: {mismatches}")

    model = EfficientNetB0Classifier(num_classes=len(LABELS))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    encoder = model.features
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, checkpoint


def optimizer_parameter_groups(
    model: Any,
    learning_rate: float,
    cnn_learning_rate: float,
) -> list[dict[str, Any]]:
    local_parameters = [
        parameter for parameter in model.local_encoder.parameters() if parameter.requires_grad
    ]
    local_parameter_ids = {id(parameter) for parameter in local_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in local_parameter_ids
    ]
    groups: list[dict[str, Any]] = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": learning_rate})
    if local_parameters:
        groups.append({"params": local_parameters, "lr": cnn_learning_rate})
    return groups


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    torch = require_torch()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        whisper_features = batch["whisper_input_features"].to(device)
        logmel = batch["logmel"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(whisper_features, logmel)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().cpu())
        total_count += batch_size
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def evaluate_model(
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
        for batch in loader:
            whisper_features = batch["whisper_input_features"].to(device)
            logmel = batch["logmel"].to(device)
            labels = batch["label"].to(device)
            logits = model(whisper_features, logmel)
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
    epoch: int,
    valid_metrics: dict[str, Any],
    args: argparse.Namespace,
    device: Any,
    parameter_counts: dict[str, int],
) -> dict[str, Any]:
    include_local_encoder = args.cnn_trainable_layers > 0
    trainable_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("whisper_encoder.")
        and (include_local_encoder or not key.startswith("local_encoder."))
    }
    local_trainable_child_names = sorted(model.local_trainable_child_names or [])
    return {
        "model_state_dict": trainable_state,
        "epoch": epoch,
        "valid_metrics": valid_metrics,
        "label_order": LABELS,
        "experiment_id": EXPERIMENT_ID,
        "model": "WhisperCnnFusionClassifier",
        "model_id": args.model_id,
        "whisper_encoder_frozen": True,
        "local_encoder": "e2_efficientnetb0_features",
        "local_encoder_checkpoint_path": args.cnn_checkpoint_path.as_posix(),
        "local_encoder_frozen": args.cnn_trainable_layers == 0,
        "local_encoder_trainable_layers": args.cnn_trainable_layers,
        "local_encoder_trainable_child_names": local_trainable_child_names,
        "fusion_type": args.fusion_type,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "fusion_dim": args.fusion_dim,
        "cnn_learning_rate": args.cnn_learning_rate,
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "n_mels": DEFAULT_N_MELS,
        "device": str(device),
        "seed": args.seed,
        "parameter_counts": parameter_counts,
    }


def write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "valid_loss",
        "valid_accuracy",
        "valid_macro_f1",
        "valid_central_recall",
        "valid_central_f1",
        "epoch_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def write_report(path: Path, results: dict[str, Any]) -> None:
    metrics = results["metrics"]
    central = results["central_error_analysis"]
    local_trainable = results["parameter_counts"]["local_encoder_trainable"]
    if local_trainable:
        branch_description = (
            "This experiment combines a frozen PhoWhisper encoder branch with a "
            "lightly fine-tuned trained E2 EfficientNetB0-style log-Mel branch. "
            "Only the selected CNN tail blocks plus the projection, fusion, and "
            "classification head are trained."
        )
    else:
        branch_description = (
            "This experiment combines a frozen PhoWhisper encoder branch with a "
            "frozen trained E2 EfficientNetB0-style log-Mel branch. Only the "
            "projection, fusion, and classification head are trained."
        )
    lines = [
        "# Phase 10 PhoWhisper + CNN Fusion Report",
        "",
        branch_description,
        "The decoder is not used, and no ASR transcript is generated.",
        "",
        "| Split | Accuracy | Macro F1 | Central Recall | Central F1 | Loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        split_metrics = metrics[split]
        central_metrics = split_metrics["per_class"]["Central"]
        lines.append(
            f"| {split} | {split_metrics['accuracy']:.4f} | "
            f"{split_metrics['macro_f1']:.4f} | "
            f"{central_metrics['recall']:.4f} | "
            f"{central_metrics['f1']:.4f} | "
            f"{split_metrics['loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Best epoch by validation macro F1: {results['best_epoch']}.",
            f"Training device: `{results['device']}`.",
            f"Fusion type: `{results['fusion']['type']}`.",
            f"PhoWhisper encoder trainable parameters: {results['parameter_counts']['whisper_encoder_trainable']}.",
            f"EfficientNet local encoder trainable parameters: {results['parameter_counts']['local_encoder_trainable']}.",
            f"EfficientNet trainable child modules: `{', '.join(results['training']['cnn_trainable_child_names']) or 'none'}`.",
            f"EfficientNet checkpoint: `{results['cnn_checkpoint_path']}`.",
            f"Checkpoint: `{results['checkpoint_path']}`.",
            "",
            "## Central Dialect Focus",
            "",
            f"- Test Central recall: {central['test_central_recall']:.4f}.",
            f"- Test Central F1: {central['test_central_f1']:.4f}.",
            f"- Central -> Northern errors: {central['central_to_northern_errors']}.",
            f"- Central -> Southern errors: {central['central_to_southern_errors']}.",
            "",
            "The model predicts only the three dataset-defined regional dialect "
            "labels. It does not infer hometown, identity, ethnicity, or personal "
            "background.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def estimate_latency(
    model: Any,
    loader: Any,
    device: Any,
    sample_count: int,
) -> dict[str, Any]:
    torch = require_torch()
    count = max(sample_count, 0)
    if count == 0:
        return {"sample_count": 0, "mean_milliseconds_per_sample": None}
    elapsed_values: list[float] = []
    measured = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["label"].shape[0])
            for index in range(batch_size):
                if measured >= count:
                    break
                whisper_features = batch["whisper_input_features"][index : index + 1].to(
                    device
                )
                logmel = batch["logmel"][index : index + 1].to(device)
                synchronize_device(torch, device)
                started = time.perf_counter()
                _logits = model(whisper_features, logmel)
                synchronize_device(torch, device)
                elapsed_values.append((time.perf_counter() - started) * 1000.0)
                measured += 1
            if measured >= count:
                break
    return {
        "sample_count": measured,
        "mean_milliseconds_per_sample": (
            float(np.mean(elapsed_values)) if elapsed_values else None
        ),
    }


def synchronize_device(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def central_error_analysis(matrix: np.ndarray, metrics: dict[str, Any]) -> dict[str, Any]:
    central_index = LABEL_TO_INDEX["Central"]
    northern_index = LABEL_TO_INDEX["Northern"]
    southern_index = LABEL_TO_INDEX["Southern"]
    central_metrics = metrics["per_class"]["Central"]
    return {
        "test_central_recall": central_metrics["recall"],
        "test_central_f1": central_metrics["f1"],
        "central_to_northern_errors": int(matrix[central_index, northern_index]),
        "central_to_southern_errors": int(matrix[central_index, southern_index]),
    }


def combined_model_size_mb(args: argparse.Namespace) -> float | None:
    checkpoint_size = file_size_mb(args.checkpoint_path)
    encoder_size = hf_cached_model_size_mb(args.cache_dir, args.model_id)
    cnn_size = file_size_mb(args.cnn_checkpoint_path)
    if checkpoint_size is None and encoder_size is None and cnn_size is None:
        return None
    return float((checkpoint_size or 0.0) + (encoder_size or 0.0) + (cnn_size or 0.0))


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite Phase 10 outputs: {formatted}. "
            "Pass --overwrite to regenerate them."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase 10 hybrid frozen PhoWhisper + CNN fusion model."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/processed/preprocessed_metadata.csv"),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/models/hf_cache"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("outputs/models/e7_whisper_cnn_fusion.pt"),
    )
    parser.add_argument(
        "--cnn-checkpoint-path",
        type=Path,
        default=DEFAULT_CNN_CHECKPOINT_PATH,
        help="Trained E2 EfficientNetB0-style checkpoint for the frozen local branch.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("outputs/metrics/e7_whisper_cnn_fusion_results.json"),
    )
    parser.add_argument(
        "--training-log-path",
        type=Path,
        default=Path("outputs/metrics/e7_whisper_cnn_fusion_training_log.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/phase10_whisper_cnn_fusion_report.md"),
    )
    parser.add_argument(
        "--valid-confusion-path",
        type=Path,
        default=Path("outputs/metrics/e7_whisper_cnn_fusion_valid_confusion_matrix.csv"),
    )
    parser.add_argument(
        "--test-confusion-path",
        type=Path,
        default=Path("outputs/metrics/e7_whisper_cnn_fusion_test_confusion_matrix.csv"),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        "--cnn-learning-rate",
        type=float,
        default=DEFAULT_CNN_LEARNING_RATE,
        help="Learning rate for trainable EfficientNetB0 local-branch layers.",
    )
    parser.add_argument(
        "--cnn-trainable-layers",
        type=int,
        default=DEFAULT_CNN_TRAINABLE_LAYERS,
        help=(
            "Number of parameterized local CNN child modules to fine-tune from "
            "the tail. Use 0 to keep the local branch frozen."
        ),
    )
    parser.add_argument("--local-embedding-dim", type=int, default=DEFAULT_LOCAL_EMBEDDING_DIM)
    parser.add_argument("--fusion-dim", type=int, default=DEFAULT_FUSION_DIM)
    parser.add_argument(
        "--classifier-hidden-dim",
        type=int,
        default=DEFAULT_CLASSIFIER_HIDDEN_DIM,
    )
    parser.add_argument(
        "--fusion-type",
        choices=("concat", "gated"),
        default=DEFAULT_FUSION_TYPE,
    )
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--latency-samples", type=int, default=5)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads for the Whisper checkpoint.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.cnn_learning_rate <= 0:
        raise ValueError("--cnn-learning-rate must be positive.")
    if args.cnn_trainable_layers < 0:
        raise ValueError("--cnn-trainable-layers cannot be negative.")
    if args.local_embedding_dim <= 0:
        raise ValueError("--local-embedding-dim must be positive.")
    if args.fusion_dim <= 0:
        raise ValueError("--fusion-dim must be positive.")
    if args.classifier_hidden_dim <= 0:
        raise ValueError("--classifier-hidden-dim must be positive.")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        raise ValueError("--limit-per-split must be positive.")
    if args.latency_samples < 0:
        raise ValueError("--latency-samples cannot be negative.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    return args


def main() -> None:
    args = parse_args()
    output_paths = [
        args.checkpoint_path,
        args.metrics_path,
        args.training_log_path,
        args.report_path,
        args.valid_confusion_path,
        args.test_confusion_path,
    ]
    if not args.overwrite:
        ensure_outputs_absent(output_paths)

    torch = require_torch()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(f"Loading frozen PhoWhisper/Whisper-family encoder: {args.model_id}", flush=True)
    feature_extractor, whisper_encoder = load_frozen_whisper_encoder(args, device)
    print(f"Loading EfficientNetB0 local branch: {args.cnn_checkpoint_path}", flush=True)
    local_encoder, cnn_checkpoint = load_efficientnet_encoder(args, device)

    rows = read_preprocessed_metadata(args.metadata_path)
    rows_by_split = split_rows(rows)
    full_counts = split_label_counts(rows_by_split)
    if args.limit_per_split is not None:
        rows_by_split = limit_rows_by_split(rows_by_split, args.limit_per_split)

    loaders = {
        split: build_loader(
            rows_by_split[split],
            feature_extractor,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            seed=args.seed,
        )
        for split in SPLITS
    }

    model = WhisperCnnFusionClassifier(
        whisper_encoder=whisper_encoder,
        whisper_hidden_size=infer_whisper_hidden_size(whisper_encoder),
        num_classes=len(LABELS),
        local_encoder=local_encoder,
        local_embedding_dim=args.local_embedding_dim,
        fusion_dim=args.fusion_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        fusion_type=args.fusion_type,
        dropout=args.dropout,
        freeze_local_encoder=True,
    ).to(device)
    local_trainable_child_names = model.enable_local_encoder_finetuning(
        args.cnn_trainable_layers
    )
    parameter_counts = count_parameters(model)
    print(
        "Trainable parameters: "
        f"{parameter_counts['trainable']:,}/{parameter_counts['total']:,}; "
        "Whisper encoder trainable: "
        f"{parameter_counts['whisper_encoder_trainable']:,}; "
        "EfficientNet local encoder trainable: "
        f"{parameter_counts['local_encoder_trainable']:,}",
        flush=True,
    )
    print(
        "EfficientNet fine-tune child modules: "
        f"{', '.join(local_trainable_child_names) or 'none'}; "
        f"cnn_learning_rate={args.cnn_learning_rate:g}",
        flush=True,
    )
    criterion = torch.nn.CrossEntropyLoss()
    trainable_parameter_groups = optimizer_parameter_groups(
        model,
        learning_rate=args.learning_rate,
        cnn_learning_rate=args.cnn_learning_rate,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameter_groups,
        weight_decay=args.weight_decay,
    )

    best_epoch = 0
    best_valid_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    training_rows: list[dict[str, Any]] = []
    started_training = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        valid_metrics, _valid_matrix, _true, _pred = evaluate_model(
            model,
            loaders["valid"],
            criterion,
            device,
        )
        elapsed = time.perf_counter() - epoch_started
        central_valid = valid_metrics["per_class"]["Central"]
        training_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_accuracy": f"{train_metrics['accuracy']:.6f}",
                "valid_loss": f"{valid_metrics['loss']:.6f}",
                "valid_accuracy": f"{valid_metrics['accuracy']:.6f}",
                "valid_macro_f1": f"{valid_metrics['macro_f1']:.6f}",
                "valid_central_recall": f"{central_valid['recall']:.6f}",
                "valid_central_f1": f"{central_valid['f1']:.6f}",
                "epoch_seconds": f"{elapsed:.3f}",
            }
        )
        print(
            f"epoch={epoch} valid_macro_f1={valid_metrics['macro_f1']:.4f} "
            f"central_recall={central_valid['recall']:.4f} "
            f"central_f1={central_valid['f1']:.4f}",
            flush=True,
        )
        if valid_metrics["macro_f1"] > best_valid_macro_f1:
            best_valid_macro_f1 = float(valid_metrics["macro_f1"])
            best_epoch = epoch
            best_state = checkpoint_state(
                model,
                epoch,
                valid_metrics,
                args,
                device,
                parameter_counts,
            )
            epochs_without_improvement = 0
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, args.checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")
    model.load_state_dict(best_state["model_state_dict"], strict=False)
    model.to(device)
    training_time_minutes = (time.perf_counter() - started_training) / 60.0

    final_metrics: dict[str, Any] = {}
    final_matrices: dict[str, np.ndarray] = {}
    for split in SPLITS:
        metrics, matrix, _true, _pred = evaluate_model(
            model,
            loaders[split],
            criterion,
            device,
        )
        final_metrics[split] = metrics
        final_matrices[split] = matrix

    write_confusion_matrix(args.valid_confusion_path, final_matrices["valid"])
    write_confusion_matrix(args.test_confusion_path, final_matrices["test"])
    write_training_log(args.training_log_path, training_rows)
    latency = estimate_latency(model, loaders["test"], device, args.latency_samples)
    central = central_error_analysis(final_matrices["test"], final_metrics["test"])

    results = {
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "model_name": "Hybrid frozen PhoWhisper-base encoder + log-Mel CNN fusion",
        "input_type": "waveform_16khz_to_whisper_features_and_log_mel",
        "pretrained": args.model_id,
        "trainable_setting": (
            "frozen_whisper_encoder_lightly_finetuned_efficientnetb0_trainable_fusion_head"
            if args.cnn_trainable_layers > 0
            else "frozen_whisper_encoder_frozen_efficientnetb0_trainable_fusion_head"
        ),
        "status": "trained",
        "metadata_path": args.metadata_path.as_posix(),
        "label_order": list(LABELS),
        "device": str(device),
        "requested_device": args.device,
        "seed": args.seed,
        "model_id": args.model_id,
        "cache_dir": args.cache_dir.as_posix(),
        "cnn_checkpoint_path": args.cnn_checkpoint_path.as_posix(),
        "cnn_checkpoint_experiment_id": cnn_checkpoint.get("experiment_id"),
        "split_counts_full_metadata": full_counts,
        "split_counts_used": split_label_counts(rows_by_split),
        "feature": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
            "duration_sec": TARGET_SAMPLES / TARGET_SAMPLE_RATE,
            "whisper_input": "PhoWhisper/Whisper input_features",
            "local_input": "standardized log_mel_spectrogram",
            "local_encoder": (
                "lightly_finetuned_e2_efficientnetb0_features"
                if args.cnn_trainable_layers > 0
                else "frozen_e2_efficientnetb0_features"
            ),
            "n_mels": DEFAULT_N_MELS,
        },
        "fusion": {
            "type": args.fusion_type,
            "local_embedding_dim": args.local_embedding_dim,
            "fusion_dim": args.fusion_dim,
            "global_embedding_dim": infer_whisper_hidden_size(whisper_encoder),
            "global_projection": "identity_layernorm",
            "classifier_hidden_dim": args.classifier_hidden_dim,
            "default": DEFAULT_FUSION_TYPE,
        },
        "training": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "epochs_completed": len(training_rows),
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "cnn_learning_rate": args.cnn_learning_rate,
            "cnn_trainable_layers": args.cnn_trainable_layers,
            "cnn_trainable_child_names": local_trainable_child_names,
            "weight_decay": args.weight_decay,
            "training_time_minutes": training_time_minutes,
            "training_log_path": args.training_log_path.as_posix(),
        },
        "parameter_counts": parameter_counts,
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_macro_f1,
        "checkpoint_path": args.checkpoint_path.as_posix(),
        "model_size_mb": combined_model_size_mb(args),
        "classifier_checkpoint_size_mb": file_size_mb(args.checkpoint_path),
        "encoder_cache_size_mb": hf_cached_model_size_mb(args.cache_dir, args.model_id),
        "cnn_checkpoint_size_mb": file_size_mb(args.cnn_checkpoint_path),
        "latency_estimate": latency,
        "central_error_analysis": central,
        "metrics": {
            split: {
                **final_metrics[split],
                **(
                    {
                        "confusion_matrix_path": (
                            args.valid_confusion_path.as_posix()
                            if split == "valid"
                            else args.test_confusion_path.as_posix()
                        )
                    }
                    if split in {"valid", "test"}
                    else {}
                ),
            }
            for split in SPLITS
        },
        "comparison_targets": [
            "e1_mobilenetv3",
            "e2_efficientnetb0",
            "e4_phowhisper",
            "e6_whisper_base",
            "e3_wav2vec2",
        ],
        "notes": (
            "Frozen PhoWhisper encoder plus "
            + (
                "lightly fine-tuned trained E2 EfficientNetB0-style log-Mel branch; "
                f"trainable CNN child modules: {', '.join(local_trainable_child_names)}. "
                if args.cnn_trainable_layers > 0
                else "frozen trained E2 EfficientNetB0-style log-Mel branch; "
            )
            + "Projection, fusion, and classification head are trained. "
            "No decoder, no ASR transcript, and no personal-background inference."
        ),
    }
    write_json(args.metrics_path, results)
    write_report(args.report_path, results)
    write_method_comparison_from_available()
    print(
        f"Phase 10 complete: best_epoch={best_epoch}, "
        f"valid_macro_f1={final_metrics['valid']['macro_f1']:.4f}, "
        f"test_macro_f1={final_metrics['test']['macro_f1']:.4f}, "
        f"test_central_f1={central['test_central_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
