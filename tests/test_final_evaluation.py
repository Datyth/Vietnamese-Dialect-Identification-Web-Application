import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.final_evaluation import (
    ERROR_FIELDS,
    best_model_row,
    comparison_rows,
    read_prediction_rows,
    write_error_report,
    write_neural_model_report,
    write_sample_errors,
)


class FinalEvaluationTests(unittest.TestCase):
    def test_comparison_rows_aggregates_available_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.json"
            cnn = root / "cnn.json"
            phowhisper = root / "phowhisper.json"
            phowhisper_pretrained = root / "phowhisper_pretrained.json"
            baseline.write_text(
                json.dumps(
                    {
                        "phase": "phase4_mfcc_baseline",
                        "models": {
                            "svm": {
                                "model_path": str(root / "missing.pkl"),
                                "valid": {"accuracy": 0.5, "macro_f1": 0.4},
                                "test": {"accuracy": 0.6, "macro_f1": 0.5},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            cnn.write_text(
                json.dumps(
                    {
                        "phase": "phase5_lightweight_cnn",
                        "checkpoint_path": str(root / "missing.pt"),
                        "metrics": {
                            "valid": {"accuracy": 0.6, "macro_f1": 0.55},
                            "test": {"accuracy": 0.7, "macro_f1": 0.65},
                        },
                    }
                ),
                encoding="utf-8",
            )
            phowhisper.write_text(
                json.dumps(
                    {
                        "phase": "phase6_phowhisper_base",
                        "model_size_mb": 290.0,
                        "latency_estimate": {"mean_seconds_per_sample": 0.12},
                        "metrics": {
                            "valid": {"accuracy": 0.7, "macro_f1": 0.75},
                            "test": {"accuracy": 0.8, "macro_f1": 0.78},
                        },
                    }
                ),
                encoding="utf-8",
            )
            phowhisper_pretrained.write_text(
                json.dumps(
                    {
                        "phase": "phase6_phowhisper_pretrained_frozen_encoder",
                        "model_size_mb": 79.0,
                        "latency_estimate": {"mean_seconds_per_sample": 0.1},
                        "metrics": {
                            "valid": {"accuracy": 0.65, "macro_f1": 0.6},
                            "test": {"accuracy": 0.7, "macro_f1": 0.68},
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = comparison_rows(
                baseline, cnn, phowhisper, phowhisper_pretrained
            )
            best = best_model_row(rows)

            self.assertEqual(
                [row["model"] for row in rows],
                [
                    "svm",
                    "lightweight_cnn",
                    "phowhisper_pretrained_frozen_encoder",
                    "phowhisper_fine_tuned",
                ],
            )
            self.assertEqual(best["model"], "phowhisper_fine_tuned")

            report = root / "neural_comparison.md"
            write_neural_model_report(report, rows)
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Custom CNN", report_text)
            self.assertIn("pretrained, frozen encoder", report_text)
            self.assertIn("fine-tuned", report_text)

    def test_prediction_and_error_outputs_have_required_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.csv"
            sample_errors = root / "errors.csv"
            report = root / "error_analysis.md"
            rows = [
                {
                    "sample_id": "test:a.wav",
                    "filepath": "a.wav",
                    "true_label": "Northern",
                    "predicted_label": "Central",
                    "confidence": "0.700000",
                    "duration": "16.000000",
                    "notes": "softmax_probability",
                },
                {
                    "sample_id": "test:b.wav",
                    "filepath": "b.wav",
                    "true_label": "Central",
                    "predicted_label": "Central",
                    "confidence": "0.900000",
                    "duration": "16.000000",
                    "notes": "softmax_probability",
                },
            ]
            with predictions.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=ERROR_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            loaded = read_prediction_rows(predictions)
            errors = write_sample_errors(sample_errors, loaded)
            write_error_report(
                report,
                [
                    {
                        "model": "phowhisper_fine_tuned",
                        "valid_macro_f1": 0.75,
                        "test_macro_f1": 0.78,
                    }
                ],
                {
                    "model": "phowhisper_fine_tuned",
                    "valid_macro_f1": 0.75,
                    "test_macro_f1": 0.78,
                },
                errors,
                loaded,
            )

            self.assertEqual(len(errors), 1)
            self.assertIn("sample_id,filepath,true_label,predicted_label,confidence,duration,notes", sample_errors.read_text(encoding="utf-8"))
            self.assertIn("Best model", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
