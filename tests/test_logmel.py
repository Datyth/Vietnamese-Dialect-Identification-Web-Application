import unittest

import numpy as np

from src.features.logmel import DEFAULT_N_MELS, log_mel_spectrogram


class LogMelTests(unittest.TestCase):
    def test_log_mel_shape_and_finite_values(self):
        sample_rate = 16_000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = 0.05 * np.sin(2 * np.pi * 440 * t)

        feature = log_mel_spectrogram(waveform, sample_rate=sample_rate)

        self.assertEqual(feature.shape, (DEFAULT_N_MELS, 99))
        self.assertTrue(np.all(np.isfinite(feature)))
        self.assertAlmostEqual(float(feature.mean()), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
