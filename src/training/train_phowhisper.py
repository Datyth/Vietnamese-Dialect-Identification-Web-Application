"""Train frozen-encoder and fine-tuned PhoWhisper dialect classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.training.train_baseline import write_confusion_matrix
from src.training.train_cnn import is_mps_available
from src.utils.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES, load_audio


LABELS = ("Northern", "Central", "Southern")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
SPLITS = ("train", "valid", "test")
DEFAULT_MODEL_ID = "vinai/PhoWhisper-base"
TRAINING_MODES = ("frozen_encoder", "full_fine_tune")
DEFAULT_TRAINING_MODE = "full_fine_tune"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_EPOCHS = 8
DEFAULT_PATIENCE = 3
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.01
MODEL_PARAMETER_COUNT = 74_000_000
MODEL_REPOSITORY_SIZE_MB = 294
MODEL_WEIGHTS_SIZE_MB = 290


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for PhoWhisper training. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc
    return torch


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoFeatureExtractor, WhisperForAudioClassification
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for PhoWhisper training. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc
    return AutoFeatureExtractor, WhisperForAudioClassification


def require_sklearn_metrics() -> tuple[Any, Any, Any, Any]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for PhoWhisper metrics. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc
    return accuracy_score, classification_report, confusion_matrix, f1_score


def resolve_device(device_name: str) -> Any:
    torch = require_torch()
    requested = device_name.lower()
    if requested == "auto":
        if is_mps_available(torch):
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not is_mps_available(torch):
            raise ValueError(
                "Requested device 'mps' but PyTorch MPS is not available. "
                "Use --device auto/cpu, or install a PyTorch build with MPS support."
            )
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "Requested device 'cuda' but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --device auto/cpu."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, mps, cuda, cpu.")


def set_seed(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_preprocessed_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed metadata not found: {path}. "
            "Run python -m src.data.preprocess_audio first."
        )
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "sample_id",
            "source_split",
            "label",
            "preprocessed_audio_path",
            "preprocessing_status",
            "preprocessed_duration_seconds",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Preprocessed metadata missing fields: {missing}")
        rows = [
            row for row in reader if row["preprocessing_status"] == "preprocessed"
        ]
    if not rows:
        raise ValueError("No preprocessed rows available for PhoWhisper training.")
    return rows


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_split = {split: [] for split in SPLITS}
    for row in rows:
        split = row["source_split"]
        label = row["label"]
        if split not in by_split:
            raise ValueError(f"Unsupported split {split!r} in {row['sample_id']}.")
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"Unsupported label {label!r} in {row['sample_id']}.")
        by_split[split].append(row)
    missing = [split for split, split_rows_ in by_split.items() if not split_rows_]
    if missing:
        raise ValueError(f"Missing preprocessed rows for splits: {missing}")
    return by_split


def load_waveform(row: dict[str, str]) -> np.ndarray:
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
    return waveform


def extract_input_features(
    rows: list[dict[str, str]],
    feature_extractor: Any,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        waveform = load_waveform(row)
        encoded = feature_extractor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="np",
        )
        features.append(np.asarray(encoded["input_features"][0], dtype=np.float32))
        labels.append(LABEL_TO_INDEX[row["label"]])
    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)


def build_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = require_torch()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    freeze_encoder: bool = False,
) -> dict[str, float]:
    torch = require_torch()
    model.train()
    if freeze_encoder:
        model.encoder.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_features=features, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()

        logits = output.logits
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
    device: Any,
) -> tuple[dict[str, Any], np.ndarray, list[int], list[int], list[float]]:
    torch = require_torch()
    accuracy_score, classification_report, confusion_matrix, f1_score = (
        require_sklearn_metrics()
    )
    model.eval()
    total_loss = 0.0
    total_count = 0
    true_labels: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            output = model(input_features=features, labels=labels)
            probabilities = torch.softmax(output.logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

            batch_size = int(labels.shape[0])
            total_loss += float(output.loss.detach().cpu()) * batch_size
            total_count += batch_size
            true_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
            predictions.extend(predicted.detach().cpu().numpy().astype(int).tolist())
            confidences.extend(confidence.detach().cpu().numpy().astype(float).tolist())

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
    metrics = {
        "loss": total_loss / max(total_count, 1),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(
            f1_score(true_labels, predictions, labels=label_indexes, average="macro")
        ),
        "per_class_f1": {
            label: float(report[label]["f1-score"]) for label in LABELS
        },
    }
    return metrics, matrix, true_labels, predictions, confidences


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing Phase 6 outputs: {formatted}. "
            "Pass --overwrite to regenerate them."
        )


def checkpoint_state(
    model: Any,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    device: Any,
    feature_shape: list[int],
) -> dict[str, Any]:
    return {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "epoch": epoch,
        "valid_metrics": metrics,
        "label_order": LABELS,
        "model_id": args.model_id,
        "model": "WhisperForAudioClassification",
        "feature": "PhoWhisper/Whisper input_features",
        "input_shape": feature_shape,
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "device": str(device),
        "seed": args.seed,
        "training_mode": args.training_mode,
    }


def configure_trainable_parameters(model: Any, training_mode: str) -> dict[str, int]:
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"training_mode must be one of: {', '.join(TRAINING_MODES)}.")
    if training_mode == "frozen_encoder":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def apply_mode_output_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.training_mode == "frozen_encoder":
        defaults = {
            "checkpoint_path": Path(
                "outputs/models/phowhisper_pretrained_frozen_encoder.pt"
            ),
            "metrics_path": Path("outputs/metrics/phowhisper_pretrained_results.json"),
            "training_log_path": Path(
                "outputs/metrics/phowhisper_pretrained_training_log.csv"
            ),
            "predictions_path": Path(
                "outputs/metrics/phowhisper_pretrained_test_predictions.csv"
            ),
            "report_path": Path(
                "outputs/reports/phase6_phowhisper_pretrained_report.md"
            ),
            "valid_confusion_path": Path(
                "outputs/metrics/phowhisper_pretrained_valid_confusion_matrix.csv"
            ),
            "test_confusion_path": Path(
                "outputs/metrics/phowhisper_pretrained_test_confusion_matrix.csv"
            ),
        }
    else:
        defaults = {
            "checkpoint_path": Path("outputs/models/phowhisper_dialect.pt"),
            "metrics_path": Path("outputs/metrics/phowhisper_results.json"),
            "training_log_path": Path("outputs/metrics/phowhisper_training_log.csv"),
            "predictions_path": Path(
                "outputs/metrics/phowhisper_test_predictions.csv"
            ),
            "report_path": Path("outputs/reports/phase6_phowhisper_report.md"),
            "valid_confusion_path": Path(
                "outputs/metrics/phowhisper_valid_confusion_matrix.csv"
            ),
            "test_confusion_path": Path(
                "outputs/metrics/phowhisper_test_confusion_matrix.csv"
            ),
        }
    for name, path in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, path)
    return args


def write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "valid_loss",
        "valid_accuracy",
        "valid_macro_f1",
        "epoch_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    true_labels: list[int],
    predictions: list[int],
    confidences: list[float],
) -> None:
    fields = [
        "sample_id",
        "filepath",
        "true_label",
        "predicted_label",
        "confidence",
        "duration",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row, true_index, predicted_index, confidence in zip(
            rows, true_labels, predictions, confidences
        ):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "filepath": row["preprocessed_audio_path"],
                    "true_label": LABELS[true_index],
                    "predicted_label": LABELS[predicted_index],
                    "confidence": f"{confidence:.6f}",
                    "duration": row.get("preprocessed_duration_seconds", ""),
                    "notes": "softmax_probability",
                }
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def write_report(path: Path, results: dict[str, Any]) -> None:
    training_mode = results["training"]["mode"]
    if training_mode == "frozen_encoder":
        title = "# Phase 6 PhoWhisper-base Pretrained Encoder Report"
        summary = (
            "PhoWhisper-base was evaluated as a pretrained frozen-encoder "
            "baseline. Only the projector and 3-class classifier head were trained."
        )
    else:
        title = "# Phase 6 PhoWhisper-base Fine-Tuning Report"
        summary = (
            "PhoWhisper-base was fine-tuned end-to-end for 3-class dialect "
            "classification using preprocessed 16 kHz / 16 s audio."
        )
    lines = [
        title,
        "",
        summary,
        "",
        "| Split | Accuracy | Macro F1 | Loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        metrics = results["metrics"][split]
        lines.append(
            f"| {split} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {metrics['loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Best epoch by validation macro F1: {results['best_epoch']}.",
            f"Training device: `{results['device']}`.",
            f"Training mode: `{training_mode}`.",
            f"Trainable parameters: {results['parameter_counts']['trainable']:,} "
            f"of {results['parameter_counts']['total']:,}.",
            f"Checkpoint: `{results['checkpoint_path']}`.",
            "",
            "## Model Size",
            "",
            f"- Model ID: `{results['model_id']}`.",
            f"- Published parameter count estimate: {MODEL_PARAMETER_COUNT:,}.",
            f"- Hugging Face repository size estimate: {MODEL_REPOSITORY_SIZE_MB} MB.",
            f"- PyTorch weights size estimate: {MODEL_WEIGHTS_SIZE_MB} MB.",
            f"- Local checkpoint size: {results['model_size_mb']:.2f} MB.",
            "",
        ]
    )
    latency = results.get("latency_estimate", {})
    if latency:
        lines.extend(
            [
                "## Latency Estimate",
                "",
                f"- Samples measured: {latency['sample_count']}.",
                f"- Mean seconds per sample: {latency['mean_seconds_per_sample']:.4f}.",
                "",
            ]
        )
    lines.append("Confusion matrices and test predictions are saved under `outputs/metrics/`.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def estimate_latency(model: Any, features: np.ndarray, device: Any, sample_count: int) -> dict[str, Any]:
    torch = require_torch()
    count = min(sample_count, int(features.shape[0]))
    if count <= 0:
        return {"sample_count": 0, "mean_seconds_per_sample": 0.0}
    model.eval()
    elapsed_values: list[float] = []
    with torch.no_grad():
        for index in range(count):
            sample = torch.from_numpy(features[index : index + 1]).to(device)
            synchronize_device(torch, device)
            started = time.perf_counter()
            _output = model(input_features=sample)
            synchronize_device(torch, device)
            elapsed_values.append(time.perf_counter() - started)
    return {
        "sample_count": count,
        "mean_seconds_per_sample": float(np.mean(elapsed_values)),
    }


def synchronize_device(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PhoWhisper-base for dialect classification."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/processed/preprocessed_metadata.csv"),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--training-mode",
        choices=TRAINING_MODES,
        default=DEFAULT_TRAINING_MODE,
        help=(
            "frozen_encoder trains only the classification stack; "
            "full_fine_tune updates the encoder and classification stack."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/models/hf_cache"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--training-log-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--valid-confusion-path", type=Path, default=None)
    parser.add_argument("--test-confusion-path", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--latency-samples", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = apply_mode_output_defaults(parse_args())
    output_paths = [
        args.checkpoint_path,
        args.metrics_path,
        args.training_log_path,
        args.predictions_path,
        args.report_path,
        args.valid_confusion_path,
        args.test_confusion_path,
    ]
    if not args.overwrite:
        ensure_outputs_absent(output_paths)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")

    torch = require_torch()
    AutoFeatureExtractor, WhisperForAudioClassification = require_transformers()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(f"Loading {args.model_id}...", flush=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
    )
    model = WhisperForAudioClassification.from_pretrained(
        args.model_id,
        num_labels=len(LABELS),
        label2id=LABEL_TO_INDEX,
        id2label={index: label for label, index in LABEL_TO_INDEX.items()},
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    ).to(device)
    parameter_counts = configure_trainable_parameters(model, args.training_mode)
    print(
        f"Training mode: {args.training_mode}; trainable parameters: "
        f"{parameter_counts['trainable']:,}/{parameter_counts['total']:,}",
        flush=True,
    )

    rows = read_preprocessed_metadata(args.metadata_path)
    rows_by_split = split_rows(rows)
    features_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        features, labels = extract_input_features(rows_by_split[split], feature_extractor)
        features_by_split[split] = features
        labels_by_split[split] = labels
        print(f"Extracted {split}: {features.shape}", flush=True)

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

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_epoch = 0
    best_valid_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    training_rows: list[dict[str, Any]] = []
    feature_shape = list(features_by_split["train"].shape[1:])

    for epoch in range(1, args.max_epochs + 1):
        started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            freeze_encoder=args.training_mode == "frozen_encoder",
        )
        valid_metrics, _matrix, _true, _pred, _conf = evaluate_model(
            model,
            loaders["valid"],
            device,
        )
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_accuracy": f"{train_metrics['accuracy']:.6f}",
            "valid_loss": f"{valid_metrics['loss']:.6f}",
            "valid_accuracy": f"{valid_metrics['accuracy']:.6f}",
            "valid_macro_f1": f"{valid_metrics['macro_f1']:.6f}",
            "epoch_seconds": f"{elapsed:.3f}",
        }
        training_rows.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']} "
            f"valid_macro_f1={row['valid_macro_f1']} "
            f"valid_accuracy={row['valid_accuracy']}",
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
                feature_shape,
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
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    final_metrics: dict[str, Any] = {}
    final_matrices: dict[str, np.ndarray] = {}
    final_predictions: dict[str, tuple[list[int], list[int], list[float]]] = {}
    for split in SPLITS:
        metrics, matrix, true_labels, predictions, confidences = evaluate_model(
            model,
            loaders[split],
            device,
        )
        final_metrics[split] = metrics
        final_matrices[split] = matrix
        final_predictions[split] = (true_labels, predictions, confidences)

    valid_matrix_path = args.valid_confusion_path
    test_matrix_path = args.test_confusion_path
    write_confusion_matrix(valid_matrix_path, final_matrices["valid"])
    write_confusion_matrix(test_matrix_path, final_matrices["test"])
    write_training_log(args.training_log_path, training_rows)
    test_true, test_pred, test_conf = final_predictions["test"]
    write_predictions(
        args.predictions_path,
        rows_by_split["test"],
        test_true,
        test_pred,
        test_conf,
    )

    latency = estimate_latency(
        model,
        features_by_split["test"],
        device,
        args.latency_samples,
    )
    checkpoint_size_mb = (
        args.checkpoint_path.stat().st_size / (1024 * 1024)
        if args.checkpoint_path.exists()
        else 0.0
    )
    results = {
        "phase": (
            "phase6_phowhisper_pretrained_frozen_encoder"
            if args.training_mode == "frozen_encoder"
            else "phase6_phowhisper_fine_tuned"
        ),
        "metadata_path": args.metadata_path.as_posix(),
        "label_order": list(LABELS),
        "model_id": args.model_id,
        "published_parameter_count": MODEL_PARAMETER_COUNT,
        "published_repository_size_mb": MODEL_REPOSITORY_SIZE_MB,
        "published_weights_size_mb": MODEL_WEIGHTS_SIZE_MB,
        "model_size_mb": checkpoint_size_mb,
        "device": str(device),
        "requested_device": args.device,
        "seed": args.seed,
        "parameter_counts": parameter_counts,
        "feature": {
            "name": "PhoWhisper/Whisper input_features",
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
            "input_shape": feature_shape,
        },
        "training": {
            "mode": args.training_mode,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "epochs_completed": len(training_rows),
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "training_log_path": args.training_log_path.as_posix(),
        },
        "split_counts": {
            split: int(labels_by_split[split].shape[0]) for split in SPLITS
        },
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_macro_f1,
        "checkpoint_path": args.checkpoint_path.as_posix(),
        "latency_estimate": latency,
        "metrics": {
            split: {
                **final_metrics[split],
                **(
                    {
                        "confusion_matrix_path": (
                            valid_matrix_path.as_posix()
                            if split == "valid"
                            else test_matrix_path.as_posix()
                        )
                    }
                    if split in ("valid", "test")
                    else {}
                ),
                **(
                    {"predictions_path": args.predictions_path.as_posix()}
                    if split == "test"
                    else {}
                ),
            }
            for split in SPLITS
        },
    }
    write_json(args.metrics_path, results)
    write_report(args.report_path, results)
    print(
        f"Phase 6 complete: best_epoch={best_epoch}, "
        f"valid_macro_f1={final_metrics['valid']['macro_f1']:.4f}, "
        f"test_macro_f1={final_metrics['test']['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
