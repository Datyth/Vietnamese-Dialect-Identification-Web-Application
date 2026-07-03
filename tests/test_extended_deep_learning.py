import csv
import tempfile
import unittest
from pathlib import Path

import torch

from src.models.efficientnet_classifier import EfficientNetB0Classifier
from src.models.mobilenetv3_classifier import MobileNetV3SmallClassifier
from src.models.vipvl_chunkformer_classifier import VipvlChunkFormerTinyClassifier
from src.models.wav2vec2_classifier import Wav2Vec2EmbeddingClassifier
from src.training.train_extended_deep_learning import (
    LABELS,
    METHOD_COMPARISON_FIELDS,
    canonical_experiment_id,
    read_rows_by_split,
    write_comparison_csv,
    write_method_comparison_csv,
)


class Phase9ModelTests(unittest.TestCase):
    def test_e1_mobilenetv3_forward_returns_three_class_logits(self):
        model = MobileNetV3SmallClassifier(num_classes=3)
        inputs = torch.randn(2, 1, 64, 99)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_e2_efficientnet_forward_returns_three_class_logits(self):
        model = EfficientNetB0Classifier(num_classes=3)
        inputs = torch.randn(2, 1, 64, 99)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_e3_wav2vec2_head_returns_three_class_logits(self):
        model = Wav2Vec2EmbeddingClassifier(embedding_dim=768, num_classes=3)
        inputs = torch.randn(2, 768)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_e5_chunkformer_forward_returns_three_class_logits(self):
        model = VipvlChunkFormerTinyClassifier(num_classes=3)
        inputs = torch.randn(2, 16000)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (2, 3))


class Phase9MetadataTests(unittest.TestCase):
    def test_current_preprocessed_metadata_can_be_limited_by_split(self):
        metadata_path = Path("data/processed/preprocessed_metadata.csv")

        rows_by_split, full_counts = read_rows_by_split(
            metadata_path,
            limit_per_split=6,
        )

        for split in ("train", "valid", "test"):
            self.assertLessEqual(len(rows_by_split[split]), 6)
            self.assertTrue(set(row["label"] for row in rows_by_split[split]) <= set(LABELS))
            self.assertGreater(sum(full_counts[split].values()), 0)


class Phase9ComparisonTests(unittest.TestCase):
    def test_e6_alias_resolves_to_whisper_base_experiment(self):
        self.assertEqual(canonical_experiment_id("e6"), "e6_whisper_base")
        self.assertEqual(
            canonical_experiment_id("e6_whisper"),
            "e6_whisper_base",
        )

    def test_comparison_csv_handles_trained_reused_and_skipped_rows(self):
        rows = [
            {
                "experiment_id": "e1_mobilenetv3",
                "model_name": "mobile",
                "status": "trained",
                "macro_f1": 0.2,
            },
            {
                "experiment_id": "e4_phowhisper",
                "model_name": "phowhisper",
                "status": "reused",
                "macro_f1": 0.7,
            },
            {
                "experiment_id": "e5_vipvl_chunkformer",
                "model_name": "chunkformer",
                "status": "skipped",
                "macro_f1": None,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "comparison.csv"

            write_comparison_csv(path, rows)

            with path.open(encoding="utf-8", newline="") as input_file:
                loaded = list(csv.DictReader(input_file))
        self.assertEqual([row["status"] for row in loaded], ["trained", "reused", "skipped"])
        self.assertEqual(loaded[2]["macro_f1"], "")

    def test_method_comparison_csv_writes_expected_fields(self):
        row = {
            "method_id": "e3_wav2vec2",
            "group": "phase9",
            "input_type": "waveform_16khz",
            "status": "trained",
            "valid_accuracy": 0.5,
            "valid_macro_f1": 0.4,
            "test_accuracy": 0.6,
            "test_macro_f1": 0.55,
            "model_size_mb": 378.0,
            "latency_ms_per_sample": 120.0,
            "device": "cpu",
            "metrics_path": "outputs/metrics/e3_wav2vec2_results.json",
            "notes": "ok",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "method_comparison.csv"

            write_method_comparison_csv(path, [row])

            with path.open(encoding="utf-8", newline="") as input_file:
                loaded = list(csv.DictReader(input_file))
        self.assertEqual(list(loaded[0]), METHOD_COMPARISON_FIELDS)
        self.assertEqual(loaded[0]["method_id"], "e3_wav2vec2")


if __name__ == "__main__":
    unittest.main()
