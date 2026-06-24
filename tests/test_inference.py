import csv
import unittest
from pathlib import Path

from src.inference.predict import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SVM_MODEL_PATH,
    load_model,
    load_svm_model,
    normalize_model_name,
    predict,
)
from src.training.train_cnn import LABELS


class CnnInferenceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_model(DEFAULT_CHECKPOINT_PATH, device="cpu")
        load_svm_model(DEFAULT_SVM_MODEL_PATH)

    def test_predict_existing_preprocessed_sample(self):
        audio_path = self.existing_preprocessed_sample()

        result = predict(audio_path)

        self.assert_valid_prediction(result)
        self.assertEqual(result["model"], "cnn")
        self.assertEqual(result["score_type"], "softmax_probability")

    def test_svm_predict_existing_preprocessed_sample(self):
        audio_path = self.existing_preprocessed_sample()

        result = predict(audio_path, model_name="svm")

        self.assert_valid_prediction(result)
        self.assertEqual(result["model"], "svm")
        self.assertEqual(
            result["score_type"],
            "decision_function_softmax_uncalibrated",
        )

    def test_model_name_aliases(self):
        self.assertEqual(normalize_model_name("lightweight-cnn"), "cnn")
        self.assertEqual(normalize_model_name("support_vector_machine"), "svm")
        self.assertEqual(normalize_model_name("phoWIshper"), "phowhisper")

    def existing_preprocessed_sample(self):
        metadata_path = Path("data/processed/preprocessed_metadata.csv")
        with metadata_path.open(encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        return next(
            Path(row["preprocessed_audio_path"])
            for row in rows
            if row["preprocessing_status"] == "preprocessed"
            and Path(row["preprocessed_audio_path"]).exists()
        )

    def assert_valid_prediction(self, result):
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
