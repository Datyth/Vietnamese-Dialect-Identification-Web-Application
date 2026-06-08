"""Create final model comparison and error analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.features.mfcc import mfcc_mean_std
from src.training.train_baseline import LABELS, SPLITS, split_rows
from src.utils.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES, load_audio


FINAL_COMPARISON_FIELDS = [
    "model",
    "phase",
    "valid_accuracy",
    "valid_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "model_size_mb",
    "latency_seconds_per_sample",
    "metrics_path",
]
ERROR_FIELDS = [
    "sample_id",
    "filepath",
    "true_label",
    "predicted_label",
    "confidence",
    "duration",
    "notes",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def comparison_rows(
    baseline_path: Path,
    cnn_path: Path,
    phowhisper_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = read_json(baseline_path)
    if baseline:
        for model_name, model_results in baseline.get("models", {}).items():
            rows.append(
                {
                    "model": model_name,
                    "phase": baseline.get("phase", "phase4_mfcc_baseline"),
                    "valid_accuracy": model_results["valid"]["accuracy"],
                    "valid_macro_f1": model_results["valid"]["macro_f1"],
                    "test_accuracy": model_results["test"]["accuracy"],
                    "test_macro_f1": model_results["test"]["macro_f1"],
                    "model_size_mb": model_file_size_mb(
                        Path(model_results.get("model_path", ""))
                    ),
                    "latency_seconds_per_sample": "",
                    "metrics_path": baseline_path.as_posix(),
                }
            )

    cnn = read_json(cnn_path)
    if cnn:
        rows.append(
            {
                "model": "lightweight_cnn",
                "phase": cnn.get("phase", "phase5_lightweight_cnn"),
                "valid_accuracy": cnn["metrics"]["valid"]["accuracy"],
                "valid_macro_f1": cnn["metrics"]["valid"]["macro_f1"],
                "test_accuracy": cnn["metrics"]["test"]["accuracy"],
                "test_macro_f1": cnn["metrics"]["test"]["macro_f1"],
                "model_size_mb": model_file_size_mb(
                    Path(cnn.get("checkpoint_path", ""))
                ),
                "latency_seconds_per_sample": "",
                "metrics_path": cnn_path.as_posix(),
            }
        )

    phowhisper = read_json(phowhisper_path)
    if phowhisper:
        latency = phowhisper.get("latency_estimate", {})
        rows.append(
            {
                "model": "phowhisper_base",
                "phase": phowhisper.get("phase", "phase6_phowhisper_base"),
                "valid_accuracy": phowhisper["metrics"]["valid"]["accuracy"],
                "valid_macro_f1": phowhisper["metrics"]["valid"]["macro_f1"],
                "test_accuracy": phowhisper["metrics"]["test"]["accuracy"],
                "test_macro_f1": phowhisper["metrics"]["test"]["macro_f1"],
                "model_size_mb": phowhisper.get("model_size_mb", ""),
                "latency_seconds_per_sample": latency.get(
                    "mean_seconds_per_sample", ""
                ),
                "metrics_path": phowhisper_path.as_posix(),
            }
        )
    if not rows:
        raise ValueError("No metric JSON files found for final comparison.")
    return rows


def model_file_size_mb(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return f"{path.stat().st_size / (1024 * 1024):.4f}"


def best_model_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row["valid_macro_f1"]))


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FINAL_COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("valid_accuracy", "valid_macro_f1", "test_accuracy", "test_macro_f1"):
                formatted[key] = f"{float(row[key]):.6f}"
            if isinstance(formatted.get("model_size_mb"), float):
                formatted["model_size_mb"] = f"{formatted['model_size_mb']:.4f}"
            if isinstance(formatted.get("latency_seconds_per_sample"), float):
                formatted["latency_seconds_per_sample"] = (
                    f"{formatted['latency_seconds_per_sample']:.6f}"
                )
            writer.writerow({field: formatted.get(field, "") for field in FINAL_COMPARISON_FIELDS})


def read_preprocessed_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "sample_id",
            "source_split",
            "label",
            "preprocessed_audio_path",
            "preprocessed_duration_seconds",
            "preprocessing_status",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Preprocessed metadata missing fields: {missing}")
        return [
            row for row in reader if row["preprocessing_status"] == "preprocessed"
        ]


def load_baseline_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Baseline model not found: {path}")
    with path.open("rb") as input_file:
        payload = pickle.load(input_file)
    return payload["model"]


def baseline_test_predictions(
    metadata_path: Path,
    model_path: Path,
) -> list[dict[str, str]]:
    rows = read_preprocessed_metadata(metadata_path)
    test_rows = split_rows(rows)["test"]
    model = load_baseline_model(model_path)
    output_rows: list[dict[str, str]] = []
    for row in test_rows:
        audio_path = Path(row["preprocessed_audio_path"])
        waveform, sample_rate = load_audio(audio_path)
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"Wrong sample rate for {audio_path}: {sample_rate}")
        if waveform.shape != (TARGET_SAMPLES,):
            raise ValueError(f"Wrong waveform shape for {audio_path}: {waveform.shape}")
        feature = mfcc_mean_std(waveform, sample_rate=sample_rate).reshape(1, -1)
        predicted = str(model.predict(feature)[0])
        confidence = svm_margin_confidence(model, feature)
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "filepath": row["preprocessed_audio_path"],
                "true_label": row["label"],
                "predicted_label": predicted,
                "confidence": f"{confidence:.6f}",
                "duration": row.get("preprocessed_duration_seconds", ""),
                "notes": "svm_margin_not_probability",
            }
        )
    return output_rows


def svm_margin_confidence(model: Any, feature: np.ndarray) -> float:
    if not hasattr(model, "decision_function"):
        return 0.0
    scores = np.asarray(model.decision_function(feature), dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return 0.0
    return float(np.max(scores))


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = sorted(set(ERROR_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Prediction file missing fields: {missing}")
        return list(reader)


def prediction_rows_for_best_model(
    best_row: dict[str, Any],
    metadata_path: Path,
    phowhisper_predictions_path: Path,
) -> list[dict[str, str]]:
    model_name = best_row["model"]
    if model_name == "svm":
        return baseline_test_predictions(metadata_path, Path("outputs/models/svm_mfcc.pkl"))
    if model_name == "phowhisper_base":
        return read_prediction_rows(phowhisper_predictions_path)
    if model_name in {"logistic_regression", "lightweight_cnn"}:
        raise ValueError(
            f"Sample error generation for {model_name} is not implemented because "
            "Phase 7 expects SVM margins or PhoWhisper softmax predictions."
        )
    raise ValueError(f"Unsupported best model for error analysis: {model_name}")


def write_sample_errors(path: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    errors = [row for row in rows if row["true_label"] != row["predicted_label"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        writer.writerows(errors)
    return errors


def confusion_summary(rows: list[dict[str, str]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        if row["true_label"] == row["predicted_label"]:
            continue
        key = f"{row['true_label']}->{row['predicted_label']}"
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items(), key=lambda item: (-item[1], item[0])))


def write_error_report(
    path: Path,
    comparison_rows_: list[dict[str, Any]],
    best: dict[str, Any],
    errors: list[dict[str, str]],
    all_predictions: list[dict[str, str]],
) -> None:
    total = len(all_predictions)
    correct = total - len(errors)
    lines = [
        "# Final Error Analysis",
        "",
        f"Best model by validation macro F1: `{best['model']}`.",
        "",
        "| Model | Valid Macro F1 | Test Macro F1 |",
        "| --- | ---: | ---: |",
    ]
    for row in comparison_rows_:
        lines.append(
            f"| {row['model']} | {float(row['valid_macro_f1']):.4f} | "
            f"{float(row['test_macro_f1']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Test Error Summary",
            "",
            f"- Test predictions analyzed: {total}.",
            f"- Correct predictions: {correct}.",
            f"- Errors: {len(errors)}.",
            "",
        ]
    )
    summary = confusion_summary(all_predictions)
    if summary:
        lines.extend(["## Confusion Patterns", ""])
        for pattern, count in summary.items():
            lines.append(f"- `{pattern}`: {count}")
        lines.append("")
    lines.extend(
        [
            "Sample-level errors are saved to `outputs/metrics/final_sample_errors.csv`.",
            "",
            "Notes: SVM confidence is a decision margin, not a calibrated probability. "
            "PhoWhisper confidence is softmax probability.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing Phase 7 outputs: {formatted}. "
            "Pass --overwrite to regenerate them."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final comparison and error analysis reports."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/processed/preprocessed_metadata.csv"),
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=Path("outputs/metrics/baseline_results.json"),
    )
    parser.add_argument(
        "--cnn-path",
        type=Path,
        default=Path("outputs/metrics/cnn_results.json"),
    )
    parser.add_argument(
        "--phowhisper-path",
        type=Path,
        default=Path("outputs/metrics/phowhisper_results.json"),
    )
    parser.add_argument(
        "--phowhisper-predictions-path",
        type=Path,
        default=Path("outputs/metrics/phowhisper_test_predictions.csv"),
    )
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=Path("outputs/metrics/final_comparison.csv"),
    )
    parser.add_argument(
        "--sample-errors-path",
        type=Path,
        default=Path("outputs/metrics/final_sample_errors.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/error_analysis.md"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = [
        args.comparison_path,
        args.sample_errors_path,
        args.report_path,
    ]
    if not args.overwrite:
        ensure_outputs_absent(output_paths)

    rows = comparison_rows(args.baseline_path, args.cnn_path, args.phowhisper_path)
    best = best_model_row(rows)
    write_comparison(args.comparison_path, rows)
    predictions = prediction_rows_for_best_model(
        best,
        args.metadata_path,
        args.phowhisper_predictions_path,
    )
    errors = write_sample_errors(args.sample_errors_path, predictions)
    write_error_report(args.report_path, rows, best, errors, predictions)
    print(
        f"Phase 7 complete: best_model={best['model']}, "
        f"valid_macro_f1={float(best['valid_macro_f1']):.4f}, "
        f"errors={len(errors)}"
    )


if __name__ == "__main__":
    main()
