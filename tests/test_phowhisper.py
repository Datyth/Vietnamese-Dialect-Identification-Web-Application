import unittest

import torch

from src.training.train_phowhisper import (
    is_mps_available,
    resolve_device,
    split_rows,
)


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


if __name__ == "__main__":
    unittest.main()
