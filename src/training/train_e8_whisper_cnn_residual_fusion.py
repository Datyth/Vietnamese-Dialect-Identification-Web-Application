"""Train Phase 11 residual-gated PhoWhisper + CNN fusion experiment."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.features.logmel import DEFAULT_N_MELS
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
from src.training.train_e7_whisper_cnn_fusion import (
    build_loader,
    central_error_analysis,
    combined_model_size_mb,
    ensure_outputs_absent,
    estimate_latency,
    load_efficientnet_encoder,
    load_frozen_whisper_encoder,
    optimizer_parameter_groups,
    require_torch,
    train_one_epoch,
    write_json,
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
from src.utils.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES


PHASE = "phase11_whisper_cnn_residual_fusion"
EXPERIMENT_ID = "e8_whisper_cnn_residual_fusion"
DEFAULT_MODEL_ID = "vinai/PhoWhisper-base"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 14
DEFAULT_MAX_EPOCHS = 20
DEFAULT_PATIENCE = 10
DEFAULT_FUSION_LEARNING_RATE = 1e-4
DEFAULT_HEAD_LEARNING_RATE = 3e-5
DEFAULT_CNN_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.0
DEFAULT_FUSION_TYPE = "residual_gated"
DEFAULT_LOCAL_EMBEDDING_DIM = 128
DEFAULT_FUSION_DIM = 512
DEFAULT_CLASSIFIER_HIDDEN_DIM = 256
DEFAULT_CNN_TRAINABLE_LAYERS = 2
DEFAULT_BETA_INIT = 0.1
DEFAULT_BEST_SCORE_TYPE = "hybrid_macro_central"
DEFAULT_CNN_CHECKPOINT_PATH = Path("outputs/models/e2_efficientnetb0_logmel.pt")
DEFAULT_PHOWHISPER_HEAD_CHECKPOINT_PATH = Path(
    "outputs/models/phowhisper_pretrained_frozen_encoder.pt"
)


def load_phowhisper_head_weights(model: Any, checkpoint_path: Path, device: Any) -> dict[str, Any]:
    torch = require_torch()
    if model.classifier_head_type != "phowhisper_linear":
        return {"loaded": False, "reason": "classifier head is not phowhisper_linear"}
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PhoWhisper head checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unsupported PhoWhisper checkpoint format in {checkpoint_path}; expected model_state_dict."
        )
    labels = tuple(checkpoint.get("label_order", ()))
    if labels != tuple(LABELS):
        raise ValueError(
            "PhoWhisper checkpoint label order mismatch: "
            f"checkpoint={labels}, code={tuple(LABELS)}"
        )

    state = checkpoint["model_state_dict"]
    required_shapes = {
        "projector.weight": tuple(model.projector.weight.shape),
        "projector.bias": tuple(model.projector.bias.shape),
        "classifier.weight": tuple(model.classifier.weight.shape),
        "classifier.bias": tuple(model.classifier.bias.shape),
    }
    mismatches = {
        key: {"checkpoint": tuple(state[key].shape), "model": shape}
        for key, shape in required_shapes.items()
        if key not in state or tuple(state[key].shape) != shape
    }
    if mismatches:
        raise ValueError(f"PhoWhisper head checkpoint shape mismatch: {mismatches}")

    with torch.no_grad():
        model.projector.weight.copy_(state["projector.weight"])
        model.projector.bias.copy_(state["projector.bias"])
        model.classifier.weight.copy_(state["classifier.weight"])
        model.classifier.bias.copy_(state["classifier.bias"])
    return {
        "loaded": True,
        "checkpoint_path": checkpoint_path.as_posix(),
        "source_model": checkpoint.get("model"),
        "source_training_mode": checkpoint.get("training_mode"),
        "source_valid_macro_f1": checkpoint.get("valid_metrics", {}).get("macro_f1"),
    }


def evaluate_model(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
) -> tuple[dict[str, Any], np.ndarray, list[int], list[int], dict[str, Any] | None]:
    torch = require_torch()
    accuracy_score, classification_report, confusion_matrix, f1_score = (
        require_sklearn_metrics()
    )
    model.eval()
    total_loss = 0.0
    total_count = 0
    true_labels: list[int] = []
    predictions: list[int] = []
    gate_sum = 0.0
    gate_count = 0
    gate_sum_by_label = {index: 0.0 for index in range(len(LABELS))}
    gate_count_by_label = {index: 0 for index in range(len(LABELS))}

    with torch.no_grad():
        for batch in loader:
            whisper_features = batch["whisper_input_features"].to(device)
            logmel = batch["logmel"].to(device)
            labels = batch["label"].to(device)
            logits, diagnostics = model.forward_with_diagnostics(whisper_features, logmel)
            loss = criterion(logits, labels)
            predicted = torch.argmax(logits, dim=1)

            residual_gate = diagnostics.get("residual_gate")
            if residual_gate is not None:
                gate_sum += float(residual_gate.detach().sum().cpu())
                gate_count += int(residual_gate.numel())
                for label_index in range(len(LABELS)):
                    label_mask = labels == label_index
                    if bool(label_mask.any().detach().cpu()):
                        class_gate = residual_gate[label_mask]
                        gate_sum_by_label[label_index] += float(
                            class_gate.detach().sum().cpu()
                        )
                        gate_count_by_label[label_index] += int(class_gate.numel())

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
    gate_diagnostics = None
    if gate_count > 0:
        gate_diagnostics = {
            "overall_mean": gate_sum / gate_count,
            "mean_by_true_label": {
                LABELS[index]: (
                    gate_sum_by_label[index] / gate_count_by_label[index]
                    if gate_count_by_label[index]
                    else None
                )
                for index in range(len(LABELS))
            },
        }
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
        gate_diagnostics,
    )


def best_score(valid_metrics: dict[str, Any], score_type: str) -> float:
    if score_type == "macro_f1":
        return float(valid_metrics["macro_f1"])
    if score_type == "hybrid_macro_central":
        return float(
            0.7 * valid_metrics["macro_f1"]
            + 0.3 * valid_metrics["per_class"]["Central"]["f1"]
        )
    raise ValueError(f"Unsupported best score type: {score_type}")


def checkpoint_state(
    model: Any,
    epoch: int,
    valid_metrics: dict[str, Any],
    valid_gate_diagnostics: dict[str, Any] | None,
    args: argparse.Namespace,
    device: Any,
    parameter_counts: dict[str, int],
    head_warm_start: dict[str, Any],
    score: float,
) -> dict[str, Any]:
    include_local_encoder = args.cnn_trainable_layers > 0
    trainable_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("whisper_encoder.")
        and (include_local_encoder or not key.startswith("local_encoder."))
    }
    local_trainable_child_names = sorted(model.local_trainable_child_names or [])
    beta_value = float(model.beta.detach().cpu()) if model.beta is not None else None
    return {
        "model_state_dict": trainable_state,
        "epoch": epoch,
        "valid_metrics": valid_metrics,
        "valid_best_score": score,
        "valid_gate_diagnostics": valid_gate_diagnostics,
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
        "fusion_dim": args.fusion_dim,
        "local_embedding_dim": args.local_embedding_dim,
        "classifier_head_type": model.classifier_head_type,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "beta_init": args.beta_init,
        "beta_learned": beta_value,
        "head_warm_start": head_warm_start,
        "fusion_learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
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
        "valid_best_score",
        "valid_gate_mean",
        "epoch_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, results: dict[str, Any]) -> None:
    metrics = results["metrics"]
    central = results["central_error_analysis"]
    gate_test = results["gate_diagnostics"].get("test") or {}
    beta_learned = results["fusion"].get("beta_learned")
    beta_learned_text = (
        f"{beta_learned:.4f}" if beta_learned is not None else "N/A"
    )
    lines = [
        "# Phase 11 PhoWhisper + CNN Residual Fusion Report",
        "",
        "This experiment keeps the PhoWhisper encoder frozen and uses the trained "
        "E2 EfficientNetB0-style log-Mel branch as a residual acoustic correction. "
        "The residual-gated fusion is `z = g + beta * r(g,l) * P(l)`, where the "
        "PhoWhisper baseline head is warm-started from the frozen PhoWhisper checkpoint.",
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
            f"Best epoch by `{results['training']['best_score_type']}`: {results['best_epoch']}.",
            f"Best validation score: {results['best_valid_score']:.4f}.",
            f"Training device: `{results['device']}`.",
            f"Fusion type: `{results['fusion']['type']}`.",
            f"Beta init: {results['fusion']['beta_init']:.4f}.",
            f"Beta learned: {beta_learned_text}.",
            f"Test gate mean: {gate_test.get('overall_mean')}.",
            f"PhoWhisper encoder trainable parameters: {results['parameter_counts']['whisper_encoder_trainable']}.",
            f"EfficientNet local encoder trainable parameters: {results['parameter_counts']['local_encoder_trainable']}.",
            f"EfficientNet trainable child modules: `{', '.join(results['training']['cnn_trainable_child_names']) or 'none'}`.",
            f"PhoWhisper head warm-start: `{results['fusion']['head_warm_start'].get('loaded')}`.",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase 11 residual-gated PhoWhisper + CNN fusion model."
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
        default=Path("outputs/models/e8_whisper_cnn_residual_fusion.pt"),
    )
    parser.add_argument(
        "--cnn-checkpoint-path",
        type=Path,
        default=DEFAULT_CNN_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--phowhisper-head-checkpoint-path",
        type=Path,
        default=DEFAULT_PHOWHISPER_HEAD_CHECKPOINT_PATH,
    )
    parser.add_argument("--skip-phowhisper-head-warm-start", action="store_true")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("outputs/metrics/e8_whisper_cnn_residual_fusion_results.json"),
    )
    parser.add_argument(
        "--training-log-path",
        type=Path,
        default=Path("outputs/metrics/e8_whisper_cnn_residual_fusion_training_log.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/phase11_whisper_cnn_residual_fusion_report.md"),
    )
    parser.add_argument(
        "--valid-confusion-path",
        type=Path,
        default=Path(
            "outputs/metrics/e8_whisper_cnn_residual_fusion_valid_confusion_matrix.csv"
        ),
    )
    parser.add_argument(
        "--test-confusion-path",
        type=Path,
        default=Path(
            "outputs/metrics/e8_whisper_cnn_residual_fusion_test_confusion_matrix.csv"
        ),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_FUSION_LEARNING_RATE)
    parser.add_argument("--head-learning-rate", type=float, default=DEFAULT_HEAD_LEARNING_RATE)
    parser.add_argument("--cnn-learning-rate", type=float, default=DEFAULT_CNN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--beta-init", type=float, default=DEFAULT_BETA_INIT)
    parser.add_argument("--cnn-trainable-layers", type=int, default=DEFAULT_CNN_TRAINABLE_LAYERS)
    parser.add_argument("--local-embedding-dim", type=int, default=DEFAULT_LOCAL_EMBEDDING_DIM)
    parser.add_argument("--fusion-dim", type=int, default=DEFAULT_FUSION_DIM)
    parser.add_argument(
        "--classifier-hidden-dim",
        type=int,
        default=DEFAULT_CLASSIFIER_HIDDEN_DIM,
    )
    parser.add_argument(
        "--fusion-type",
        choices=("concat", "gated", "residual_gated"),
        default=DEFAULT_FUSION_TYPE,
    )
    parser.add_argument(
        "--best-score-type",
        choices=("macro_f1", "hybrid_macro_central"),
        default=DEFAULT_BEST_SCORE_TYPE,
    )
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--latency-samples", type=int, default=5)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    for name in ("learning_rate", "head_learning_rate", "cnn_learning_rate"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.cnn_trainable_layers < 0:
        raise ValueError("--cnn-trainable-layers cannot be negative.")
    if args.local_embedding_dim <= 0 or args.fusion_dim <= 0:
        raise ValueError("--local-embedding-dim and --fusion-dim must be positive.")
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

    classifier_head_type = (
        "phowhisper_linear" if args.fusion_type == "residual_gated" else "mlp"
    )
    model = WhisperCnnFusionClassifier(
        whisper_encoder=whisper_encoder,
        whisper_hidden_size=infer_whisper_hidden_size(whisper_encoder),
        num_classes=len(LABELS),
        local_encoder=local_encoder,
        local_embedding_dim=args.local_embedding_dim,
        fusion_dim=args.fusion_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        fusion_type=args.fusion_type,
        classifier_head_type=classifier_head_type,
        beta_init=args.beta_init,
        dropout=args.dropout,
        freeze_local_encoder=True,
    ).to(device)
    if args.fusion_type == "residual_gated" and not args.skip_phowhisper_head_warm_start:
        head_warm_start = load_phowhisper_head_weights(
            model,
            args.phowhisper_head_checkpoint_path,
            device,
        )
    else:
        head_warm_start = {"loaded": False, "reason": "warm-start disabled or unused"}

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
    print(f"PhoWhisper head warm-start: {head_warm_start}", flush=True)

    criterion = torch.nn.CrossEntropyLoss()
    trainable_parameter_groups = optimizer_parameter_groups(
        model,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        cnn_learning_rate=args.cnn_learning_rate,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameter_groups,
        weight_decay=args.weight_decay,
    )

    best_epoch = 0
    best_valid_score = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    training_rows: list[dict[str, Any]] = []
    started_training = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        valid_metrics, _valid_matrix, _true, _pred, valid_gate = evaluate_model(
            model,
            loaders["valid"],
            criterion,
            device,
        )
        elapsed = time.perf_counter() - epoch_started
        central_valid = valid_metrics["per_class"]["Central"]
        score = best_score(valid_metrics, args.best_score_type)
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
                "valid_best_score": f"{score:.6f}",
                "valid_gate_mean": (
                    f"{valid_gate['overall_mean']:.6f}" if valid_gate else ""
                ),
                "epoch_seconds": f"{elapsed:.3f}",
            }
        )
        print(
            f"epoch={epoch} valid_score={score:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f} "
            f"central_recall={central_valid['recall']:.4f} "
            f"central_f1={central_valid['f1']:.4f}",
            flush=True,
        )
        if score > best_valid_score:
            best_valid_score = score
            best_epoch = epoch
            best_state = checkpoint_state(
                model,
                epoch,
                valid_metrics,
                valid_gate,
                args,
                device,
                parameter_counts,
                head_warm_start,
                score,
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
    gate_diagnostics: dict[str, Any] = {}
    for split in SPLITS:
        metrics, matrix, _true, _pred, gate = evaluate_model(
            model,
            loaders[split],
            criterion,
            device,
        )
        final_metrics[split] = metrics
        final_matrices[split] = matrix
        gate_diagnostics[split] = gate

    write_confusion_matrix(args.valid_confusion_path, final_matrices["valid"])
    write_confusion_matrix(args.test_confusion_path, final_matrices["test"])
    write_training_log(args.training_log_path, training_rows)
    latency = estimate_latency(model, loaders["test"], device, args.latency_samples)
    central = central_error_analysis(final_matrices["test"], final_metrics["test"])
    beta_learned = float(model.beta.detach().cpu()) if model.beta is not None else None
    local_projection_description = (
        "LayerNorm(128)+Linear(128,512)"
        if args.fusion_type == "residual_gated"
        else "LayerNorm(128)+Linear(128,512)+ReLU"
    )

    results = {
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "model_name": "Residual-gated PhoWhisper-base encoder + log-Mel CNN fusion",
        "input_type": "waveform_16khz_to_whisper_features_and_log_mel",
        "pretrained": args.model_id,
        "trainable_setting": (
            "frozen_whisper_encoder_lightly_finetuned_efficientnetb0_residual_fusion"
            if args.cnn_trainable_layers > 0
            else "frozen_whisper_encoder_frozen_efficientnetb0_residual_fusion"
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
        "phowhisper_head_checkpoint_path": args.phowhisper_head_checkpoint_path.as_posix(),
        "split_counts_full_metadata": full_counts,
        "split_counts_used": split_label_counts(rows_by_split),
        "feature": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
            "duration_sec": TARGET_SAMPLES / TARGET_SAMPLE_RATE,
            "whisper_input": "PhoWhisper/Whisper input_features",
            "global_embedding": "mean_pool_hidden_state",
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
            "default": DEFAULT_FUSION_TYPE,
            "local_embedding_dim": args.local_embedding_dim,
            "fusion_dim": args.fusion_dim,
            "global_embedding_dim": infer_whisper_hidden_size(whisper_encoder),
            "global_projection": (
                "mean_pool_identity"
                if args.fusion_type == "residual_gated"
                else "LayerNorm(mean_pool_hidden_state)"
            ),
            "local_projection": local_projection_description,
            "residual_formula": (
                "z = g + beta * sigmoid(W[g;P(l)] + b) * P(l)"
                if args.fusion_type == "residual_gated"
                else None
            ),
            "beta_init": args.beta_init,
            "beta_learned": beta_learned,
            "classifier_head_type": classifier_head_type,
            "classifier_hidden_dim": args.classifier_hidden_dim,
            "head_warm_start": head_warm_start,
        },
        "gate_diagnostics": gate_diagnostics,
        "training": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "epochs_completed": len(training_rows),
            "patience": args.patience,
            "best_score_type": args.best_score_type,
            "fusion_learning_rate": args.learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "cnn_learning_rate": args.cnn_learning_rate,
            "cnn_trainable_layers": args.cnn_trainable_layers,
            "cnn_trainable_child_names": local_trainable_child_names,
            "weight_decay": args.weight_decay,
            "training_time_minutes": training_time_minutes,
            "training_log_path": args.training_log_path.as_posix(),
        },
        "parameter_counts": parameter_counts,
        "best_epoch": best_epoch,
        "best_valid_score": best_valid_score,
        "best_valid_macro_f1": best_state["valid_metrics"]["macro_f1"],
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
            "e4_phowhisper",
            "e7_whisper_cnn_fusion",
            "e2_efficientnetb0",
            "e6_whisper_base",
        ],
        "notes": (
            "Frozen PhoWhisper encoder plus residual-gated local EfficientNetB0 "
            "log-Mel correction. The PhoWhisper projector/classifier head is "
            "warm-started from the frozen PhoWhisper baseline checkpoint; no decoder, "
            "no ASR transcript, and no personal-background inference."
        ),
    }
    write_json(args.metrics_path, results)
    write_report(args.report_path, results)
    write_method_comparison_from_available()
    print(
        f"Phase 11 complete: best_epoch={best_epoch}, "
        f"valid_score={best_valid_score:.4f}, "
        f"valid_macro_f1={final_metrics['valid']['macro_f1']:.4f}, "
        f"test_macro_f1={final_metrics['test']['macro_f1']:.4f}, "
        f"test_central_f1={central['test_central_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
