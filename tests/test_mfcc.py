import unittest

import numpy as np

from src.features.mfcc import DEFAULT_N_MFCC, mfcc_matrix, mfcc_mean_std


class MfccTests(unittest.TestCase):
    def test_mfcc_mean_std_shape_and_finite_values(self):
        sample_rate = 16_000
        seconds = 1
        t = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
        waveform = 0.05 * np.sin(2 * np.pi * 440 * t)

        feature = mfcc_mean_std(waveform, sample_rate=sample_rate)

        self.assertEqual(feature.shape, (DEFAULT_N_MFCC * 2,))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_mfcc_matrix_has_requested_coefficients(self):
        waveform = np.zeros(3200, dtype=np.float32)

        matrix = mfcc_matrix(waveform, sample_rate=16_000, n_mfcc=10, n_mels=24)

        self.assertEqual(matrix.shape[1], 10)
        self.assertTrue(np.all(np.isfinite(matrix)))


if __name__ == "__main__":
    unittest.main()

