import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.models.whisper_cnn_fusion import (
    WhisperCnnFusionClassifier,
    count_parameters,
    infer_whisper_hidden_size,
)
from src.training.train_e7_whisper_cnn_fusion import (
    DEFAULT_FUSION_TYPE,
    DEFAULT_MODEL_ID,
    central_error_analysis,
    checkpoint_state,
)
from src.training.train_extended_deep_learning import (
    append_phase10_comparison_row,
)


class FakeWhisperEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=hidden_size)
        self.projection = torch.nn.Linear(80, hidden_size)

    def forward(self, input_features: torch.Tensor) -> SimpleNamespace:
        features = input_features.transpose(1, 2)
        return SimpleNamespace(last_hidden_state=self.projection(features))


class WhisperCnnFusionModelTests(unittest.TestCase):
    def test_concat_forward_returns_three_class_logits_and_freezes_encoder(self):
        encoder = FakeWhisperEncoder(hidden_size=32)
        model = WhisperCnnFusionClassifier(
            whisper_encoder=encoder,
            whisper_hidden_size=infer_whisper_hidden_size(encoder),
            num_classes=3,
            local_embedding_dim=16,
            fusion_dim=32,
            fusion_type="concat",
        )

        logits = model(
            torch.randn(2, 80, 100),
            torch.randn(2, 1, 64, 99),
        )
        counts = count_parameters(model)

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(counts["whisper_encoder_trainable"], 0)
        self.assertEqual(counts["local_encoder_trainable"], 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.local_encoder.parameters())
        )

    def test_default_fusion_model_has_no_dropout_modules(self):
        encoder = FakeWhisperEncoder(hidden_size=32)
        model = WhisperCnnFusionClassifier(
            whisper_encoder=encoder,
            whisper_hidden_size=32,
            num_classes=3,
            fusion_dim=32,
        )

        dropout_modules = [
            module for module in model.modules() if isinstance(module, torch.nn.Dropout)
        ]

        self.assertEqual(dropout_modules, [])

    def test_gated_forward_returns_three_class_logits(self):
        encoder = FakeWhisperEncoder(hidden_size=24)
        model = WhisperCnnFusionClassifier(
            whisper_encoder=encoder,
            whisper_hidden_size=24,
            num_classes=3,
            local_embedding_dim=12,
            fusion_dim=24,
            classifier_hidden_dim=10,
            fusion_type="gated",
        )

        logits = model(
            torch.randn(2, 80, 100),
            torch.randn(2, 1, 64, 99),
        )

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(model.classifier[1].weight.shape), (10, 24))
        self.assertEqual(tuple(model.classifier[4].weight.shape), (3, 10))

    def test_fusion_dim_must_match_global_embedding_dim(self):
        encoder = FakeWhisperEncoder(hidden_size=24)

        with self.assertRaisesRegex(ValueError, "fusion_dim must match"):
            WhisperCnnFusionClassifier(
                whisper_encoder=encoder,
                whisper_hidden_size=24,
                num_classes=3,
                local_embedding_dim=12,
                fusion_dim=32,
                fusion_type="gated",
            )


class WhisperCnnFusionTrainingTests(unittest.TestCase):
    def test_phase10_default_encoder_is_phowhisper(self):
        self.assertEqual(DEFAULT_MODEL_ID, "vinai/PhoWhisper-base")

    def test_phase10_default_fusion_is_gated(self):
        self.assertEqual(DEFAULT_FUSION_TYPE, "gated")

    def test_central_error_analysis_counts_targeted_confusions(self):
        matrix = np.asarray(
            [
                [5, 1, 0],
                [2, 6, 3],
                [0, 1, 7],
            ]
        )
        metrics = {
            "per_class": {
                "Central": {
                    "recall": 6 / 11,
                    "f1": 0.6,
                }
            }
        }

        analysis = central_error_analysis(matrix, metrics)

        self.assertEqual(analysis["central_to_northern_errors"], 2)
        self.assertEqual(analysis["central_to_southern_errors"], 3)
        self.assertAlmostEqual(analysis["test_central_recall"], 6 / 11)

    def test_checkpoint_state_excludes_frozen_encoders(self):
        encoder = FakeWhisperEncoder(hidden_size=16)
        local_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        model = WhisperCnnFusionClassifier(
            whisper_encoder=encoder,
            whisper_hidden_size=16,
            num_classes=3,
            local_encoder=local_encoder,
            local_embedding_dim=4,
            fusion_dim=16,
        )
        args = SimpleNamespace(
            model_id=DEFAULT_MODEL_ID,
            cnn_checkpoint_path=Path("outputs/models/e2_efficientnetb0_logmel.pt"),
            fusion_type="concat",
            fusion_dim=16,
            classifier_hidden_dim=256,
            seed=42,
        )

        state = checkpoint_state(
            model,
            epoch=1,
            valid_metrics={"macro_f1": 0.5},
            args=args,
            device=torch.device("cpu"),
            parameter_counts=count_parameters(model),
        )

        saved_keys = state["model_state_dict"]
        self.assertTrue(saved_keys)
        self.assertFalse(any(key.startswith("whisper_encoder.") for key in saved_keys))
        self.assertFalse(any(key.startswith("local_encoder.") for key in saved_keys))
        self.assertEqual(state["local_encoder"], "e2_efficientnetb0_features")
        self.assertTrue(state["local_encoder_frozen"])

    def test_phase10_comparison_row_is_collected_from_metrics_json(self):
        payload = {
            "phase": "phase10_whisper_cnn_fusion",
            "experiment_id": "e7_whisper_cnn_fusion",
            "input_type": "waveform_16khz_to_whisper_features_and_log_mel",
            "status": "trained",
            "model_size_mb": 100.0,
            "device": "cpu",
            "latency_estimate": {"mean_milliseconds_per_sample": 12.5},
            "metrics": {
                "valid": {"accuracy": 0.5, "macro_f1": 0.4},
                "test": {"accuracy": 0.6, "macro_f1": 0.55},
            },
            "notes": "ok",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "e7.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            rows = []

            append_phase10_comparison_row(rows, path)

        self.assertEqual(rows[0]["method_id"], "e7_whisper_cnn_fusion")
        self.assertEqual(rows[0]["group"], "phase10_whisper_cnn_fusion")
        self.assertEqual(rows[0]["test_macro_f1"], 0.55)


if __name__ == "__main__":
    unittest.main()
