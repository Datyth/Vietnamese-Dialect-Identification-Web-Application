import tempfile
import unittest
from pathlib import Path

from src.data.prepare_metadata import (
    choose_source_shards,
    normalize_source_row,
    select_rows_for_label,
    tree_size_bytes,
)


class PrepareMetadataTests(unittest.TestCase):
    def test_normalize_source_row_maps_region_and_gender(self):
        row = normalize_source_row(
            {
                "region": "North",
                "province_code": 11,
                "province_name": "CaoBang",
                "filename": "11_0307.wav",
                "text": "Sample transcript",
                "speakerID": "spk_11_0142",
                "gender": 1,
                "source_audio_path": "11_0307.wav",
                "source_parquet_url": (
                    "https://example.invalid/data/test-00000.parquet"
                ),
                "source_split": "test",
            }
        )

        self.assertEqual(row["sample_id"], "test:11_0307.wav")
        self.assertEqual(row["label"], "Northern")
        self.assertEqual(row["gender"], "male")
        self.assertEqual(row["audio_status"], "not_selected_under_data_budget")

    def test_normalize_source_row_rejects_unknown_region(self):
        with self.assertRaisesRegex(ValueError, "Unsupported source region"):
            normalize_source_row(
                {
                    "region": "Unknown",
                    "province_code": 1,
                    "province_name": "Unknown",
                    "filename": "1_0001.wav",
                    "text": "Sample transcript",
                    "speakerID": "spk_1_0001",
                    "gender": 0,
                    "source_parquet_url": (
                        "https://example.invalid/data/train-00000.parquet"
                    ),
                    "source_split": "train",
                }
            )

    def test_select_rows_prioritizes_distinct_speakers(self):
        candidates = [
            {
                "filename": f"{index}.wav",
                "speaker_id": "speaker-a" if index < 2 else f"speaker-{index}",
                "source_audio_bytes": index + 1,
            }
            for index in range(4)
        ]

        selected = select_rows_for_label(
            candidates,
            count=3,
            excluded_speakers=set(),
            seed=42,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(len({row["speaker_id"] for row in selected}), 3)

    def test_choose_source_shards_prefers_compact_mixed_shard(self):
        metadata = []
        for split in ("train", "valid", "test"):
            for label, region in (
                ("Northern", "North"),
                ("Central", "Central"),
                ("Southern", "South"),
            ):
                for index in range(2):
                    metadata.append(
                        {
                            "source_split": split,
                            "label": label,
                            "source_parquet_file": f"{split}-mixed.parquet",
                            "source_region": region,
                            "filename": f"{label}-{index}.wav",
                        }
                    )
        source_files = {
            split: [
                {
                    "path": f"data/{split}-mixed.parquet",
                    "url": f"https://example.invalid/{split}-mixed.parquet",
                    "size": 100,
                },
                {
                    "path": f"data/{split}-large.parquet",
                    "url": f"https://example.invalid/{split}-large.parquet",
                    "size": 500,
                },
            ]
            for split in ("train", "valid", "test")
        }

        selected = choose_source_shards(
            metadata,
            source_files,
            {"train": 1, "valid": 1, "test": 1},
        )

        self.assertEqual(
            [item["filename"] for item in selected],
            [
                "train-mixed.parquet",
                "valid-mixed.parquet",
                "test-mixed.parquet",
            ],
        )

    def test_tree_size_bytes_counts_nested_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "one.bin").write_bytes(b"123")
            (root / "nested" / "two.bin").write_bytes(b"4567")

            self.assertEqual(tree_size_bytes(root), 7)


if __name__ == "__main__":
    unittest.main()
