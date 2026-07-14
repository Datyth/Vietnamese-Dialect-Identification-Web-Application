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
    DEFAULT_CNN_LEARNING_RATE,
    DEFAULT_CNN_TRAINABLE_LAYERS,
    DEFAULT_FUSION_TYPE,
    DEFAULT_MODEL_ID,
    central_error_analysis,
    checkpoint_state,
    optimizer_parameter_groups,
)
from src.training.train_e8_whisper_cnn_residual_fusion import (
    DEFAULT_BATCH_SIZE as E8_DEFAULT_BATCH_SIZE,
    DEFAULT_BEST_SCORE_TYPE as E8_DEFAULT_BEST_SCORE_TYPE,
    DEFAULT_BETA_INIT as E8_DEFAULT_BETA_INIT,
    DEFAULT_FUSION_TYPE as E8_DEFAULT_FUSION_TYPE,
    checkpoint_state as e8_checkpoint_state,
    load_phowhisper_head_weights,
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

    def test_residual_gated_forward_returns_three_class_logits_and_gate(self):
        encoder = FakeWhisperEncoder(hidden_size=16)
        model = WhisperCnnFusionClassifier(
            whisper_encoder=encoder,
            whisper_hidden_size=16,
            num_classes=3,
            local_embedding_dim=8,
            fusion_dim=16,
            classifier_hidden_dim=6,
            fusion_type="residual_gated",
            classifier_head_type="phowhisper_linear",
            beta_init=0.1,
        )

        logits, diagnostics = model.forward_with_diagnostics(
            torch.randn(2, 80, 100),
            torch.randn(2, 1, 64, 99),
        )

        gate = diagnostics["residual_gate"]
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(gate.shape), (2, 16))
        self.assertTrue(bool(torch.all(gate >= 0.0)))
        self.assertTrue(bool(torch.all(gate <= 1.0)))
        self.assertEqual(tuple(model.projector.weight.shape), (6, 16))
        self.assertEqual(tuple(model.classifier.weight.shape), (3, 6))

    def test_residual_beta_zero_reduces_to_global_embedding(self):
        encoder = FakeWhisperEncoder(hidden_size=16)
        model = WhisperCnnFusionClassifier(
            whisper_encoder=encoder,
            whisper_hidden_size=16,
            num_classes=3,
            local_embedding_dim=8,
            fusion_dim=16,
            classifier_hidden_dim=6,
            fusion_type="residual_gated",
            classifier_head_type="phowhisper_linear",
            beta_init=0.1,
        )
        with torch.no_grad():
            model.beta.fill_(0.0)

        logits, diagnostics = model.forward_with_diagnostics(
            torch.randn(2, 80, 100),
            torch.randn(2, 1, 64, 99),
        )
        expected_logits = model.classifier(model.projector(diagnostics["global_embedding"]))

        self.assertTrue(torch.allclose(diagnostics["fused"], diagnostics["global_embedding"]))
        self.assertTrue(torch.allclose(logits, expected_logits))

    def test_residual_gated_gradients_reach_trainable_parts_only(self):
        torch.manual_seed(42)
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
            classifier_hidden_dim=6,
            fusion_type="residual_gated",
            classifier_head_type="phowhisper_linear",
            beta_init=0.1,
        )
        selected = model.enable_local_encoder_finetuning(trainable_layers=1)

        logits = model(
            torch.randn(2, 80, 100),
            torch.randn(2, 1, 64, 99),
        )
        loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 2]))
        loss.backward()

        self.assertEqual(selected, ["0"])
        self.assertIsNotNone(model.beta.grad)
        self.assertGreater(float(model.beta.grad.detach().abs().sum()), 0.0)
        self.assertIsNotNone(model.residual_gate.weight.grad)
        self.assertGreater(float(model.residual_gate.weight.grad.detach().abs().sum()), 0.0)
        self.assertIsNotNone(model.local_projection[1].weight.grad)
        self.assertGreater(float(model.local_projection[1].weight.grad.detach().abs().sum()), 0.0)
        self.assertIsNotNone(model.local_encoder[0].weight.grad)
        self.assertGreater(float(model.local_encoder[0].weight.grad.detach().abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))

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

    def test_local_encoder_can_finetune_last_parameterized_blocks(self):
        encoder = FakeWhisperEncoder(hidden_size=16)
        local_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, kernel_size=3, padding=1),
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

        selected = model.enable_local_encoder_finetuning(trainable_layers=1)
        counts = count_parameters(model)

        self.assertEqual(selected, ["2"])
        self.assertFalse(model.freeze_local)
        self.assertEqual(model.local_trainable_child_names, {"2"})
        self.assertTrue(all(not parameter.requires_grad for parameter in model.local_encoder[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.local_encoder[2].parameters()))
        self.assertGreater(counts["local_encoder_trainable"], 0)


class WhisperCnnFusionTrainingTests(unittest.TestCase):
    def test_phase10_default_encoder_is_phowhisper(self):
        self.assertEqual(DEFAULT_MODEL_ID, "vinai/PhoWhisper-base")

    def test_phase10_default_fusion_is_gated(self):
        self.assertEqual(DEFAULT_FUSION_TYPE, "gated")

    def test_phase10_defaults_lightly_finetune_cnn_tail(self):
        self.assertEqual(DEFAULT_CNN_TRAINABLE_LAYERS, 2)
        self.assertEqual(DEFAULT_CNN_LEARNING_RATE, 1e-5)

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
            cnn_trainable_layers=0,
            cnn_learning_rate=1e-5,
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
        self.assertEqual(state["local_encoder_trainable_layers"], 0)

    def test_checkpoint_state_includes_finetuned_local_encoder(self):
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
        selected = model.enable_local_encoder_finetuning(trainable_layers=1)
        args = SimpleNamespace(
            model_id=DEFAULT_MODEL_ID,
            cnn_checkpoint_path=Path("outputs/models/e2_efficientnetb0_logmel.pt"),
            fusion_type="gated",
            fusion_dim=16,
            classifier_hidden_dim=256,
            cnn_trainable_layers=1,
            cnn_learning_rate=1e-5,
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
        self.assertTrue(any(key.startswith("local_encoder.") for key in saved_keys))
        self.assertFalse(any(key.startswith("whisper_encoder.") for key in saved_keys))
        self.assertFalse(state["local_encoder_frozen"])
        self.assertEqual(state["local_encoder_trainable_layers"], 1)
        self.assertEqual(state["local_encoder_trainable_child_names"], selected)

    def test_optimizer_uses_smaller_lr_for_trainable_cnn_layers(self):
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
        model.enable_local_encoder_finetuning(trainable_layers=1)

        groups = optimizer_parameter_groups(
            model,
            learning_rate=1e-4,
            cnn_learning_rate=1e-5,
        )

        self.assertEqual([group["lr"] for group in groups], [1e-4, 1e-5])

    def test_optimizer_can_use_separate_residual_head_lr(self):
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
            classifier_hidden_dim=6,
            fusion_type="residual_gated",
            classifier_head_type="phowhisper_linear",
        )
        model.enable_local_encoder_finetuning(trainable_layers=1)

        groups = optimizer_parameter_groups(
            model,
            learning_rate=1e-4,
            head_learning_rate=3e-5,
            cnn_learning_rate=1e-5,
        )

        self.assertEqual([group["lr"] for group in groups], [1e-4, 3e-5, 1e-5])

    def test_phase11_e8_defaults(self):
        self.assertEqual(E8_DEFAULT_FUSION_TYPE, "residual_gated")
        self.assertEqual(E8_DEFAULT_BATCH_SIZE, 14)
        self.assertEqual(E8_DEFAULT_BETA_INIT, 0.1)
        self.assertEqual(E8_DEFAULT_BEST_SCORE_TYPE, "hybrid_macro_central")

    def test_e8_loads_phowhisper_linear_head_weights(self):
        encoder = FakeWhisperEncoder(hidden_size=16)
        model = WhisperCnnFusionClassifier(
            whisper_encoder=encoder,
            whisper_hidden_size=16,
            num_classes=3,
            local_embedding_dim=4,
            fusion_dim=16,
            classifier_hidden_dim=6,
            fusion_type="residual_gated",
            classifier_head_type="phowhisper_linear",
        )
        checkpoint = {
            "label_order": ("Northern", "Central", "Southern"),
            "model_state_dict": {
                "projector.weight": torch.ones_like(model.projector.weight),
                "projector.bias": torch.ones_like(model.projector.bias),
                "classifier.weight": torch.ones_like(model.classifier.weight) * 2.0,
                "classifier.bias": torch.ones_like(model.classifier.bias) * 2.0,
            },
            "model": "PhoWhisperClassifier",
            "training_mode": "frozen_encoder",
            "valid_metrics": {"macro_f1": 0.84},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "phowhisper.pt"
            torch.save(checkpoint, path)

            summary = load_phowhisper_head_weights(model, path, torch.device("cpu"))

        self.assertTrue(summary["loaded"])
        self.assertAlmostEqual(summary["source_valid_macro_f1"], 0.84)
        self.assertTrue(torch.equal(model.projector.weight, checkpoint["model_state_dict"]["projector.weight"]))
        self.assertTrue(torch.equal(model.classifier.bias, checkpoint["model_state_dict"]["classifier.bias"]))

    def test_e8_checkpoint_preserves_beta_and_residual_config(self):
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
            classifier_hidden_dim=6,
            fusion_type="residual_gated",
            classifier_head_type="phowhisper_linear",
            beta_init=0.1,
        )
        model.enable_local_encoder_finetuning(trainable_layers=1)
        with torch.no_grad():
            model.beta.fill_(0.37)
        args = SimpleNamespace(
            model_id=DEFAULT_MODEL_ID,
            cnn_checkpoint_path=Path("outputs/models/e2_efficientnetb0_logmel.pt"),
            fusion_type="residual_gated",
            fusion_dim=16,
            local_embedding_dim=4,
            classifier_hidden_dim=6,
            cnn_trainable_layers=1,
            beta_init=0.1,
            learning_rate=1e-4,
            head_learning_rate=3e-5,
            cnn_learning_rate=1e-5,
            seed=42,
        )

        state = e8_checkpoint_state(
            model,
            epoch=2,
            valid_metrics={"macro_f1": 0.5, "per_class": {"Central": {"f1": 0.4}}},
            valid_gate_diagnostics={"overall_mean": 0.51},
            args=args,
            device=torch.device("cpu"),
            parameter_counts=count_parameters(model),
            head_warm_start={"loaded": True},
            score=0.47,
        )
        reloaded = WhisperCnnFusionClassifier(
            whisper_encoder=FakeWhisperEncoder(hidden_size=16),
            whisper_hidden_size=16,
            num_classes=3,
            local_encoder=torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
            ),
            local_embedding_dim=4,
            fusion_dim=16,
            classifier_hidden_dim=6,
            fusion_type="residual_gated",
            classifier_head_type="phowhisper_linear",
        )
        reloaded.load_state_dict(state["model_state_dict"], strict=False)

        self.assertEqual(state["experiment_id"], "e8_whisper_cnn_residual_fusion")
        self.assertEqual(state["fusion_type"], "residual_gated")
        self.assertAlmostEqual(state["beta_init"], 0.1)
        self.assertAlmostEqual(state["beta_learned"], 0.37, places=6)
        self.assertEqual(state["valid_gate_diagnostics"]["overall_mean"], 0.51)
        self.assertEqual(state["head_warm_start"]["loaded"], True)
        self.assertAlmostEqual(float(reloaded.beta.detach()), 0.37, places=6)

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
