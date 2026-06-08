"""Train traditional MFCC baselines for three-region dialect classification."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.features.mfcc import DEFAULT_N_MFCC, mfcc_mean_std
from src.utils.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES, load_audio


LABELS = ("Northern", "Central", "Southern")
SPLITS = ("train", "valid", "test")


def require_sklearn() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for Phase 4 baselines. "
            "Install dependencies with: uv pip install -r requirements.txt"
        ) from exc
    return (
        LogisticRegression,
        SVC,
        Pipeline,
        StandardScaler,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )


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
        raise ValueError("No preprocessed rows available for training.")
    return rows


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_split = {split: [] for split in SPLITS}
    for row in rows:
        split = row["source_split"]
        label = row["label"]
        if split not in by_split:
            raise ValueError(f"Unsupported split {split!r} in {row['sample_id']}.")
        if label not in LABELS:
            raise ValueError(f"Unsupported label {label!r} in {row['sample_id']}.")
        by_split[split].append(row)

    missing = [split for split, split_items in by_split.items() if not split_items]
    if missing:
        raise ValueError(f"Missing preprocessed rows for splits: {missing}")
    return by_split


def extract_features(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
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
        features.append(mfcc_mean_std(waveform, sample_rate=sample_rate))
        labels.append(row["label"])

    return np.vstack(features).astype(np.float32), np.asarray(labels)


def build_models(seed: int = 42) -> dict[str, Any]:
    (
        LogisticRegression,
        SVC,
        Pipeline,
        StandardScaler,
        *_,
    ) = require_sklearn()
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=seed,
                        solver="newton-cg",
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=10.0, gamma="scale")),
            ]
        ),
    }


def evaluate(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    (
        _LogisticRegression,
        _SVC,
        _Pipeline,
        _StandardScaler,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    ) = require_sklearn()
    predictions = model.predict(features)
    report = classification_report(
        labels,
        predictions,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(LABELS))
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=list(LABELS), average="macro")
        ),
        "per_class_f1": {
            label: float(report[label]["f1-score"]) for label in LABELS
        },
    }
    return metrics, matrix


def write_confusion_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["true_label", *[f"pred_{label}" for label in LABELS]])
        for label, row in zip(LABELS, matrix.tolist()):
            writer.writerow([label, *row])


def save_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output_file:
        pickle.dump(
            {
                "model": model,
                "label_order": LABELS,
                "feature": "mfcc_mean_std",
                "sample_rate": TARGET_SAMPLE_RATE,
                "target_samples": TARGET_SAMPLES,
                "n_mfcc": DEFAULT_N_MFCC,
            },
            output_file,
        )


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing Phase 4 outputs: {formatted}. "
            "Pass --overwrite to regenerate them."
        )


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Phase 4 MFCC Baseline Report",
        "",
        "Traditional baselines use MFCC mean/std features from Phase 2 "
        "fixed-length audio.",
        "",
        "| Model | Split | Accuracy | Macro F1 |",
        "| --- | --- | ---: | ---: |",
    ]
    for model_name, model_result in results["models"].items():
        for split in ("valid", "test"):
            metrics = model_result[split]
            lines.append(
                f"| {model_name} | {split} | {metrics['accuracy']:.4f} | "
                f"{metrics['macro_f1']:.4f} |"
            )
    lines.extend(
        [
            "",
            f"Best model by validation macro F1: "
            f"{results['best_model_by_valid_macro_f1']}.",
            "",
            "Confusion matrices are saved as CSV files under `outputs/metrics/`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression and SVM MFCC baselines."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/processed/preprocessed_metadata.csv"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("outputs/metrics/baseline_results.json"),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("outputs/models"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/phase4_baseline_report.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = [
        args.metrics_path,
        args.report_path,
        args.models_dir / "logistic_regression_mfcc.pkl",
        args.models_dir / "svm_mfcc.pkl",
        Path("outputs/metrics/logistic_regression_valid_confusion_matrix.csv"),
        Path("outputs/metrics/logistic_regression_test_confusion_matrix.csv"),
        Path("outputs/metrics/svm_valid_confusion_matrix.csv"),
        Path("outputs/metrics/svm_test_confusion_matrix.csv"),
    ]
    if not args.overwrite:
        ensure_outputs_absent(output_paths)

    rows = read_preprocessed_metadata(args.metadata_path)
    rows_by_split = split_rows(rows)
    features_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    for split, split_items in rows_by_split.items():
        features, labels = extract_features(split_items)
        features_by_split[split] = features
        labels_by_split[split] = labels

    models = build_models(args.seed)
    results: dict[str, Any] = {
        "phase": "phase4_mfcc_baseline",
        "metadata_path": args.metadata_path.as_posix(),
        "label_order": list(LABELS),
        "feature": {
            "name": "mfcc_mean_std",
            "dimension": int(features_by_split["train"].shape[1]),
            "n_mfcc": DEFAULT_N_MFCC,
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
        },
        "split_counts": {
            split: int(labels_by_split[split].shape[0]) for split in SPLITS
        },
        "models": {},
    }

    for model_name, model in models.items():
        model.fit(features_by_split["train"], labels_by_split["train"])
        model_path = args.models_dir / f"{model_name}_mfcc.pkl"
        save_model(model_path, model)
        model_result: dict[str, Any] = {"model_path": model_path.as_posix()}
        for split in ("valid", "test"):
            metrics, matrix = evaluate(
                model,
                features_by_split[split],
                labels_by_split[split],
            )
            matrix_path = (
                Path("outputs/metrics")
                / f"{model_name}_{split}_confusion_matrix.csv"
            )
            write_confusion_matrix(matrix_path, matrix)
            model_result[split] = {
                **metrics,
                "confusion_matrix_path": matrix_path.as_posix(),
            }
        results["models"][model_name] = model_result

    results["best_model_by_valid_macro_f1"] = max(
        results["models"],
        key=lambda name: results["models"][name]["valid"]["macro_f1"],
    )
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    write_report(args.report_path, results)
    print(
        "Phase 4 complete: "
        + ", ".join(
            f"{name} valid_macro_f1="
            f"{result['valid']['macro_f1']:.4f}"
            for name, result in results["models"].items()
        )
    )


if __name__ == "__main__":
    main()
