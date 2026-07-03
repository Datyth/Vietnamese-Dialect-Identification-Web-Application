"""Phase 9 E6 original Whisper-base experiment entrypoint."""

from __future__ import annotations

import sys

from src.training import train_phowhisper
from src.training.train_extended_deep_learning import (
    EXPERIMENTS,
    read_confusion_matrix_csv_from_metrics,
    read_json_or_none,
    write_confusion_matrix_figure,
    write_method_comparison_from_available,
    write_phase9_summary_from_available,
)


E6_DEFAULT_ARGS = (
    "--model-id",
    "openai/whisper-base",
    "--training-mode",
    "frozen_encoder",
    "--checkpoint-path",
    "outputs/models/e6_whisper_base_frozen_encoder.pt",
    "--metrics-path",
    "outputs/metrics/e6_whisper_base_results.json",
    "--training-log-path",
    "outputs/metrics/e6_whisper_base_training_log.csv",
    "--predictions-path",
    "outputs/metrics/e6_whisper_base_test_predictions.csv",
    "--report-path",
    "outputs/reports/e6_whisper_base_report.md",
    "--valid-confusion-path",
    "outputs/metrics/e6_whisper_base_valid_confusion_matrix.csv",
    "--test-confusion-path",
    "outputs/metrics/e6_whisper_base_test_confusion_matrix.csv",
    "--experiment-id",
    "e6_whisper_base",
    "--experiment-model-name",
    "Whisper-base original encoder + classifier",
    "--experiment-notes",
    "Original OpenAI Whisper-base checkpoint with the encoder frozen; same base-size family as PhoWhisper-base.",
)


def main() -> None:
    sys.argv = [sys.argv[0], *E6_DEFAULT_ARGS, *sys.argv[1:]]
    train_phowhisper.main()
    write_e6_confusion_figure()
    write_phase9_summary_from_available()
    write_method_comparison_from_available()


def write_e6_confusion_figure() -> None:
    spec = EXPERIMENTS["e6_whisper_base"]
    result = read_json_or_none(spec.metric_path)
    if result is None:
        return
    matrix = read_confusion_matrix_csv_from_metrics(
        result.get("metrics", {}).get("test", {})
    )
    if matrix is None:
        return
    write_confusion_matrix_figure(
        spec.confusion_figure_path,
        matrix,
        "e6_whisper_base test confusion matrix",
    )


if __name__ == "__main__":
    main()
