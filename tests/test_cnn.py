import unittest

import torch

from src.models.cnn import LightweightCNN
from src.training.train_cnn import is_mps_available, resolve_device


class LightweightCnnTests(unittest.TestCase):
    def test_forward_returns_three_class_logits(self):
        model = LightweightCNN(num_classes=3)
        inputs = torch.randn(2, 1, 64, 99)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (2, 3))


class DeviceResolverTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
