import csv
import unittest
from pathlib import Path

from src.inference.predict import (
    DEFAULT_CHECKPOINT_PATH,
    load_model,
    predict,
)
from src.training.train_cnn import LABELS


class CnnInferenceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_model(DEFAULT_CHECKPOINT_PATH, device="cpu")

    def test_predict_existing_preprocessed_sample(self):
        metadata_path = Path("data/processed/preprocessed_metadata.csv")
        with metadata_path.open(encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        audio_path = next(
            Path(row["preprocessed_audio_path"])
            for row in rows
            if row["preprocessing_status"] == "preprocessed"
            and Path(row["preprocessed_audio_path"]).exists()
        )

        result = predict(audio_path)

        self.assertIn(result["predicted_label"], LABELS)
        self.assertEqual(tuple(result["probabilities"]), LABELS)
        self.assertEqual(len(result["probabilities"]), 3)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertAlmostEqual(
            result["confidence"],
            max(result["probabilities"].values()),
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
