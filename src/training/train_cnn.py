"""Train a lightweight CNN from log-Mel spectrograms."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.features.logmel import (
    DEFAULT_N_MELS,
    log_mel_spectrogram,
)
from src.training.train_baseline import write_confusion_matrix
from src.utils.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES, load_audio


LABELS = ("Northern", "Central", "Southern")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
SPLITS = ("train", "valid", "test")
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Phase 5 CNN training. "
            "Install dependencies with: uv pip install --python .venv/bin/python "
            "-r requirements.txt"
        ) from exc
    return torch


def require_lightweight_cnn() -> Any:
    require_torch()
    from src.models.cnn import LightweightCNN

    return LightweightCNN


def is_mps_available(torch: Any) -> bool:
    return bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


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
            "scikit-learn is required for CNN metrics. "
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
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Preprocessed metadata missing fields: {missing}")
        rows = [
            row for row in reader if row["preprocessing_status"] == "preprocessed"
        ]
    if not rows:
        raise ValueError("No preprocessed rows available for CNN training.")
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

    missing = [split for split, split_items in by_split.items() if not split_items]
    if missing:
        raise ValueError(f"Missing preprocessed rows for splits: {missing}")
    return by_split


def extract_logmel_features(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
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
        features.append(log_mel_spectrogram(waveform, sample_rate=sample_rate))
        labels.append(LABEL_TO_INDEX[row["label"]])

    feature_array = np.stack(features).astype(np.float32)
    feature_array = feature_array[:, None, :, :]
    return feature_array, np.asarray(labels, dtype=np.int64)


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
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    torch = require_torch()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
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
) -> tuple[dict[str, Any], np.ndarray]:
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
    return metrics, matrix


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing Phase 5 outputs: {formatted}. "
            "Pass --overwrite to regenerate them."
        )


def checkpoint_state(
    model: Any,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    return {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "epoch": epoch,
        "valid_metrics": metrics,
        "label_order": LABELS,
        "feature": "log_mel_spectrogram",
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "n_mels": DEFAULT_N_MELS,
        "model": "LightweightCNN",
        "device": str(device),
        "seed": args.seed,
    }


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def read_baseline_valid_macro_f1(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as input_file:
        results = json.load(input_file)
    best_name = results.get("best_model_by_valid_macro_f1")
    if not best_name:
        return None
    model_results = results.get("models", {}).get(best_name, {})
    valid = model_results.get("valid", {})
    value = valid.get("macro_f1")
    return float(value) if value is not None else None


def write_report(path: Path, results: dict[str, Any]) -> None:
    baseline_f1 = results.get("baseline_best_valid_macro_f1")
    lines = [
        "# Phase 5 Lightweight CNN Report",
        "",
        "The CNN uses standardized log-Mel spectrograms from Phase 2 fixed-length "
        "audio and is trained from scratch.",
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
            f"Checkpoint: `{results['checkpoint_path']}`.",
            "",
        ]
    )
    if baseline_f1 is not None:
        lines.extend(
            [
                "## Baseline Comparison",
                "",
                f"- Best Phase 4 validation macro F1: {baseline_f1:.4f}.",
                (
                    "- Phase 5 CNN validation macro F1: "
                    f"{results['metrics']['valid']['macro_f1']:.4f}."
                ),
                "",
            ]
        )
    lines.append("Confusion matrices and the per-epoch training log are saved under `outputs/metrics/`.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight CNN from log-Mel spectrograms."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/processed/preprocessed_metadata.csv"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("outputs/models/lightweight_cnn_logmel.pt"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("outputs/metrics/cnn_results.json"),
    )
    parser.add_argument(
        "--training-log-path",
        type=Path,
        default=Path("outputs/metrics/cnn_training_log.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/phase5_cnn_report.md"),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = [
        args.checkpoint_path,
        args.metrics_path,
        args.training_log_path,
        args.report_path,
        Path("outputs/metrics/cnn_valid_confusion_matrix.csv"),
        Path("outputs/metrics/cnn_test_confusion_matrix.csv"),
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
    LightweightCNN = require_lightweight_cnn()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    rows = read_preprocessed_metadata(args.metadata_path)
    rows_by_split = split_rows(rows)
    features_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    for split in SPLITS:
        features, labels = extract_logmel_features(rows_by_split[split])
        features_by_split[split] = features
        labels_by_split[split] = labels
        print(f"Extracted {split}: {features.shape}")

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

    model = LightweightCNN(num_classes=len(LABELS)).to(device)
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

    for epoch in range(1, args.max_epochs + 1):
        started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )
        valid_metrics, _valid_matrix = evaluate_model(
            model,
            loaders["valid"],
            criterion,
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
            f"valid_accuracy={row['valid_accuracy']}"
        )

        if valid_metrics["macro_f1"] > best_valid_macro_f1:
            best_valid_macro_f1 = float(valid_metrics["macro_f1"])
            best_epoch = epoch
            best_state = checkpoint_state(model, epoch, valid_metrics, args, device)
            epochs_without_improvement = 0
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, args.checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)

    final_metrics: dict[str, Any] = {}
    final_matrices: dict[str, np.ndarray] = {}
    for split in SPLITS:
        metrics, matrix = evaluate_model(model, loaders[split], criterion, device)
        final_metrics[split] = metrics
        final_matrices[split] = matrix

    valid_matrix_path = Path("outputs/metrics/cnn_valid_confusion_matrix.csv")
    test_matrix_path = Path("outputs/metrics/cnn_test_confusion_matrix.csv")
    write_confusion_matrix(valid_matrix_path, final_matrices["valid"])
    write_confusion_matrix(test_matrix_path, final_matrices["test"])
    write_training_log(args.training_log_path, training_rows)

    results = {
        "phase": "phase5_lightweight_cnn",
        "metadata_path": args.metadata_path.as_posix(),
        "label_order": list(LABELS),
        "device": str(device),
        "requested_device": args.device,
        "seed": args.seed,
        "feature": {
            "name": "log_mel_spectrogram",
            "n_mels": DEFAULT_N_MELS,
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
            "input_shape": list(features_by_split["train"].shape[1:]),
            "standardized_per_sample": True,
        },
        "model": {
            "name": "LightweightCNN",
            "num_classes": len(LABELS),
            "dropout": 0.25,
        },
        "training": {
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
            }
            for split in SPLITS
        },
        "baseline_best_valid_macro_f1": read_baseline_valid_macro_f1(
            Path("outputs/metrics/baseline_results.json")
        ),
    }
    write_json(args.metrics_path, results)
    write_report(args.report_path, results)
    print(
        f"Phase 5 complete: best_epoch={best_epoch}, "
        f"valid_macro_f1={final_metrics['valid']['macro_f1']:.4f}, "
        f"test_macro_f1={final_metrics['test']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
