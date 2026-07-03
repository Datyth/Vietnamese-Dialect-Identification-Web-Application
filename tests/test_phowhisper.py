import unittest
from argparse import Namespace
from pathlib import Path

import torch

from src.training.train_phowhisper import (
    apply_mode_output_defaults,
    configure_trainable_parameters,
    is_mps_available,
    limit_rows_by_split,
    resolve_device,
    split_rows,
)
from src.training.train_e6_whisper import E6_DEFAULT_ARGS


class PhoWhisperDeviceTests(unittest.TestCase):
    def test_auto_returns_available_device(self):
        device = resolve_device("auto")

        self.assertIn(device.type, {"mps", "cuda", "cpu"})

    def test_unavailable_explicit_device_errors(self):
        checked_unavailable_device = False
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "cuda"):
                resolve_device("cuda")
            checked_unavailable_device = True
        if not is_mps_available(torch):
            with self.assertRaisesRegex(ValueError, "mps"):
                resolve_device("mps")
            checked_unavailable_device = True
        if not checked_unavailable_device:
            with self.assertRaisesRegex(ValueError, "device must be"):
                resolve_device("invalid")


class PhoWhisperMetadataTests(unittest.TestCase):
    def test_split_rows_groups_supported_labels(self):
        rows = [
            {"sample_id": "train:a.wav", "source_split": "train", "label": "Northern"},
            {"sample_id": "valid:b.wav", "source_split": "valid", "label": "Central"},
            {"sample_id": "test:c.wav", "source_split": "test", "label": "Southern"},
        ]

        by_split = split_rows(rows)

        self.assertEqual(len(by_split["train"]), 1)
        self.assertEqual(len(by_split["valid"]), 1)
        self.assertEqual(len(by_split["test"]), 1)

    def test_split_rows_rejects_unknown_label(self):
        rows = [
            {"sample_id": "train:a.wav", "source_split": "train", "label": "Northern"},
            {"sample_id": "valid:b.wav", "source_split": "valid", "label": "Central"},
            {"sample_id": "test:c.wav", "source_split": "test", "label": "Unknown"},
        ]

        with self.assertRaisesRegex(ValueError, "Unsupported label"):
            split_rows(rows)

    def test_limit_rows_by_split_keeps_each_split_bounded(self):
        rows = []
        for split in ("train", "valid", "test"):
            for index, label in enumerate(("Northern", "Central", "Southern")):
                for item in range(3):
                    rows.append(
                        {
                            "sample_id": f"{split}:{index}:{item}",
                            "source_split": split,
                            "label": label,
                        }
                    )
        limited = limit_rows_by_split(split_rows(rows), limit_per_split=6)

        for split in ("train", "valid", "test"):
            self.assertEqual(len(limited[split]), 6)
            self.assertEqual(
                {row["label"] for row in limited[split]},
                {"Northern", "Central", "Southern"},
            )


class PhoWhisperTrainingModeTests(unittest.TestCase):
    def test_frozen_encoder_leaves_only_classification_stack_trainable(self):
        class TinyAudioClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(4, 4)
                self.projector = torch.nn.Linear(4, 2)
                self.classifier = torch.nn.Linear(2, 3)

        model = TinyAudioClassifier()
        counts = configure_trainable_parameters(model, "frozen_encoder")

        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.projector.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        self.assertLess(counts["trainable"], counts["total"])

    def test_frozen_mode_uses_separate_output_paths(self):
        args = Namespace(
            training_mode="frozen_encoder",
            checkpoint_path=None,
            metrics_path=None,
            training_log_path=None,
            predictions_path=None,
            report_path=None,
            valid_confusion_path=None,
            test_confusion_path=None,
        )

        resolved = apply_mode_output_defaults(args)

        self.assertEqual(
            resolved.metrics_path,
            Path("outputs/metrics/phowhisper_pretrained_results.json"),
        )
        self.assertIn("frozen_encoder", resolved.checkpoint_path.name)

    def test_e6_defaults_use_original_whisper_base_outputs(self):
        defaults = list(E6_DEFAULT_ARGS)

        self.assertIn("openai/whisper-base", defaults)
        self.assertIn("e6_whisper_base", defaults)
        self.assertIn("outputs/metrics/e6_whisper_base_results.json", defaults)


if __name__ == "__main__":
    unittest.main()
