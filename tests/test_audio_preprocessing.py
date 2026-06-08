import unittest

import numpy as np

from src.utils.audio import (
    FIXED_DURATION_SECONDS,
    TARGET_RMS,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
    fix_length,
    preprocess_waveform,
    trim_silence,
)


class AudioPreprocessingTests(unittest.TestCase):
    def test_preprocess_waveform_returns_fixed_shape(self):
        sample_rate = 8_000
        tone = np.sin(
            2 * np.pi * 220 * np.arange(sample_rate, dtype=np.float32) / sample_rate
        )
        waveform = np.concatenate(
            [
                np.zeros(sample_rate // 4, dtype=np.float32),
                tone.astype(np.float32) * 0.05,
                np.zeros(sample_rate // 4, dtype=np.float32),
            ]
        )

        processed, stats = preprocess_waveform(waveform, sample_rate)

        self.assertEqual(processed.shape, (TARGET_SAMPLES,))
        self.assertEqual(stats.output_sample_rate, TARGET_SAMPLE_RATE)
        self.assertEqual(stats.output_samples, TARGET_SAMPLES)
        self.assertAlmostEqual(stats.output_duration_seconds, FIXED_DURATION_SECONDS)
        self.assertLessEqual(float(np.max(np.abs(processed))), 0.99)
        self.assertGreater(stats.output_rms, 0.0)

    def test_trim_silence_removes_edges(self):
        waveform = np.array([0.0, 0.001, 0.03, 0.02, 0.001, 0.0], dtype=np.float32)

        trimmed = trim_silence(waveform, absolute_threshold=0.005)

        self.assertTrue(np.allclose(trimmed, np.array([0.03, 0.02], dtype=np.float32)))

    def test_fix_length_pads_and_crops(self):
        short = np.ones(4, dtype=np.float32)
        long = np.arange(10, dtype=np.float32)

        padded = fix_length(short, target_samples=8)
        cropped = fix_length(long, target_samples=4)

        self.assertEqual(padded.shape, (8,))
        self.assertTrue(np.allclose(padded[2:6], short))
        self.assertTrue(np.allclose(cropped, np.array([3, 4, 5, 6], dtype=np.float32)))

    def test_preprocess_waveform_accepts_stereo_input(self):
        waveform = np.ones((1000, 2), dtype=np.float32) * TARGET_RMS

        processed, _ = preprocess_waveform(waveform, TARGET_SAMPLE_RATE)

        self.assertEqual(processed.shape, (TARGET_SAMPLES,))


if __name__ == "__main__":
    unittest.main()

