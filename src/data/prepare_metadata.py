"""Prepare ViMD metadata and a size-bounded local audio subset."""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATASET_ID = "nguyendv02/ViMD_Dataset"
DATASET_REVISION = "main"
DEFAULT_MAX_DATA_BYTES = 1_000_000_000
METADATA_RESERVE_BYTES = 25_000_000
TARGET_SAMPLE_RATE = 16_000
DEFAULT_TARGETS = {
    "train": 100,
    "valid": 15,
    "test": 15,
}
LABEL_MAPPING = {
    "North": "Northern",
    "Central": "Central",
    "South": "Southern",
}
LABELS = ("Northern", "Central", "Southern")
SPLITS = ("train", "valid", "test")
REQUIRED_SOURCE_FIELDS = (
    "region",
    "province_code",
    "province_name",
    "filename",
    "text",
    "gender",
    "source_parquet_url",
)
USER_AGENT = "vietnamese-dialect-id-phase1/0.1"


def request_json(url: str, attempts: int = 5) -> dict[str, Any] | list[Any]:
    """Fetch JSON with a short retry for transient network errors."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
            time.sleep(min(2**attempt, 15))
    raise AssertionError("unreachable")


def source_parquet_files() -> dict[str, list[dict[str, Any]]]:
    """Return official Parquet shard metadata grouped by source split."""
    endpoint = (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/"
        f"{DATASET_REVISION}?recursive=true&expand=false"
    )
    tree = request_json(endpoint)
    if not isinstance(tree, list):
        raise RuntimeError("Unexpected Hugging Face repository tree response.")

    files: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    pattern = re.compile(r"^data/(train|valid|test)-\d+-of-\d+\.parquet$")
    base = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/"
    )
    for item in tree:
        path = str(item.get("path", ""))
        match = pattern.match(path)
        if match:
            files[match.group(1)].append(
                {
                    "path": path,
                    "url": base + path,
                    "size": int(item.get("size") or 0),
                }
            )

    missing = [split for split, split_files in files.items() if not split_files]
    if missing:
        raise RuntimeError(f"No official Parquet shards found for: {missing}")
    for split_files in files.values():
        split_files.sort(key=lambda item: item["path"])
    return files


def source_parquet_urls() -> dict[str, list[str]]:
    return {
        split: [item["url"] for item in files]
        for split, files in source_parquet_files().items()
    }


def load_source_metadata() -> list[dict[str, Any]]:
    """Read only metadata columns from remote ViMD Parquet shards."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "DuckDB is required for metadata-only Parquet reads. "
            "Install dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc

    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar = false")
    rows: list[dict[str, Any]] = []
    query = """
        SELECT
            region,
            province_code,
            province_name,
            filename,
            text,
            speakerID,
            gender,
            audio.path AS source_audio_path,
            source_parquet_url
        FROM read_parquet(?, filename = 'source_parquet_url')
        ORDER BY province_code, filename
    """
    try:
        for split, urls in source_parquet_urls().items():
            result = connection.execute(query, [urls])
            columns = [description[0] for description in result.description]
            for values in result.fetchall():
                source_row = dict(zip(columns, values))
                source_row["source_split"] = split
                rows.append(normalize_source_row(source_row))
    finally:
        connection.close()

    sample_ids = [row["sample_id"] for row in rows]
    duplicates = [
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(
            "Duplicate split/filename sample IDs found, for example: "
            f"{duplicates[:5]}"
        )
    return rows


def normalize_source_row(source_row: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one official ViMD metadata row."""
    missing = [
        field
        for field in REQUIRED_SOURCE_FIELDS
        if source_row.get(field) is None or str(source_row[field]).strip() == ""
    ]
    if missing:
        raise ValueError(
            f"Source row is missing required fields {missing}: {source_row!r}"
        )

    source_region = str(source_row["region"]).strip()
    if source_region not in LABEL_MAPPING:
        raise ValueError(
            f"Unsupported source region {source_region!r}; "
            f"expected one of {sorted(LABEL_MAPPING)}."
        )

    split = str(source_row.get("source_split", "")).strip()
    if split not in SPLITS:
        raise ValueError(f"Unsupported source split {split!r}.")

    filename = str(source_row["filename"]).strip()
    if Path(filename).name != filename or not filename.lower().endswith(".wav"):
        raise ValueError(f"Unsafe or unsupported audio filename: {filename!r}")

    gender_value = int(source_row["gender"])
    if gender_value not in (0, 1):
        raise ValueError(f"Unsupported gender value {gender_value!r} for {filename}.")

    speaker_id = str(source_row.get("speakerID") or "").strip()
    return {
        "sample_id": f"{split}:{filename}",
        "source_split": split,
        "source_region": source_region,
        "label": LABEL_MAPPING[source_region],
        "province_code": int(source_row["province_code"]),
        "province_name": str(source_row["province_name"]).strip(),
        "filename": filename,
        "text": str(source_row["text"]).strip(),
        "speaker_id": speaker_id,
        "gender": "female" if gender_value == 0 else "male",
        "source_audio_path": str(
            source_row.get("source_audio_path") or filename
        ).strip(),
        "source_parquet_url": str(source_row["source_parquet_url"]),
        "source_parquet_file": str(source_row["source_parquet_url"]).rsplit(
            "/", 1
        )[-1],
        "audio_path": "",
        "audio_status": "not_selected_under_data_budget",
        "audio_bytes": 0,
    }


def download_file(
    url: str,
    destination: Path,
    expected_bytes: int,
    data_root: Path,
    max_data_bytes: int,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > expected_bytes:
                        raise ValueError(
                            f"Download exceeded declared size for {destination.name}."
                        )
                    if tree_size_bytes(data_root) + len(chunk) > max_data_bytes:
                        raise ValueError(
                            f"Download would exceed {max_data_bytes:,} data bytes."
                        )
                    output.write(chunk)
        if written != expected_bytes:
            raise ValueError(
                f"Download size mismatch for {destination.name}: "
                f"expected {expected_bytes}, received {written}."
            )
        temporary.replace(destination)
    except (OSError, ValueError):
        temporary.unlink(missing_ok=True)
        raise
    return written


def tree_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def choose_source_shards(
    metadata: list[dict[str, Any]],
    source_files: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
) -> list[dict[str, Any]]:
    """Greedily choose small shards that cover every split/label target."""
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in metadata:
        counts[(row["source_split"], row["source_parquet_file"])][
            row["label"]
        ] += 1

    selected: list[dict[str, Any]] = []
    for split in SPLITS:
        candidates = []
        for source_file in source_files[split]:
            filename = Path(source_file["path"]).name
            candidates.append(
                {
                    **source_file,
                    "filename": filename,
                    "split": split,
                    "counts": counts[(split, filename)],
                }
            )

        best_combination: tuple[dict[str, Any], ...] | None = None
        best_size: int | None = None
        for shard_count in range(1, 4):
            for combination in itertools.combinations(candidates, shard_count):
                if not all(
                    sum(item["counts"][label] for item in combination)
                    >= targets[split]
                    for label in LABELS
                ):
                    continue
                total_size = sum(item["size"] for item in combination)
                if best_size is None or total_size < best_size:
                    best_combination = combination
                    best_size = total_size
            if best_combination is not None:
                break

        if best_combination is None:
            raise RuntimeError(
                f"Cannot cover requested {split} targets with three shards."
            )
        selected.extend(
            sorted(
                best_combination,
                key=lambda item: (-item["size"], item["filename"]),
            )
        )
    return selected


def require_audio_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy
        import soundfile
        import soxr
    except ImportError as exc:
        raise RuntimeError(
            "NumPy, SoundFile, and soxr are required for bounded 16 kHz WAV "
            "storage. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return numpy, soundfile, soxr


def normalize_wav(
    source: bytes | Path,
    destination: Path,
    max_output_bytes: int,
) -> int:
    numpy, soundfile, soxr = require_audio_dependencies()
    input_source: Any = io.BytesIO(source) if isinstance(source, bytes) else source
    waveform, sample_rate = soundfile.read(
        input_source,
        dtype="float32",
        always_2d=True,
    )
    mono = waveform.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, TARGET_SAMPLE_RATE, quality="HQ")
    mono = numpy.clip(mono, -1.0, 1.0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        soundfile.write(
            temporary,
            mono,
            TARGET_SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
        output_bytes = temporary.stat().st_size
        if output_bytes > max_output_bytes:
            raise ValueError(
                f"Normalized audio exceeds remaining byte budget: "
                f"{destination.name}"
            )
        with temporary.open("rb") as audio_file:
            header = audio_file.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Invalid normalized WAV: {destination.name}")
        temporary.replace(destination)
    except (OSError, RuntimeError, ValueError):
        temporary.unlink(missing_ok=True)
        raise
    return output_bytes


def select_rows_for_label(
    candidates: list[dict[str, Any]],
    count: int,
    excluded_speakers: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    candidates = sorted(
        candidates,
        key=lambda row: (row["source_audio_bytes"], row["filename"]),
    )
    selected: list[dict[str, Any]] = []
    selected_speakers: set[str] = set()
    for row in candidates:
        speaker_id = row["speaker_id"]
        if speaker_id in excluded_speakers or speaker_id in selected_speakers:
            continue
        selected.append(row)
        selected_speakers.add(speaker_id)
        if len(selected) == count:
            return selected

    same_split_repeats = [
        row
        for row in candidates
        if row not in selected and row["speaker_id"] not in excluded_speakers
    ]
    randomizer.shuffle(same_split_repeats)
    same_split_repeats.sort(key=lambda row: row["source_audio_bytes"])
    selected.extend(same_split_repeats[: count - len(selected)])
    if len(selected) == count:
        return selected

    cross_split_fallback = [row for row in candidates if row not in selected]
    cross_split_fallback.sort(
        key=lambda row: (row["source_audio_bytes"], row["filename"])
    )
    selected.extend(cross_split_fallback[: count - len(selected)])
    return selected


def acquire_audio_subset(
    metadata: list[dict[str, Any]],
    data_root: Path,
    max_data_bytes: int,
    targets: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required to extract ViMD audio.") from exc
    require_audio_dependencies()

    by_id = {row["sample_id"]: row for row in metadata}
    audio_root = data_root / "processed" / "audio_16k"
    legacy_root = data_root / "raw" / "vimd_audio"
    issues: list[dict[str, str]] = []

    selected_counts: Counter[tuple[str, str]] = Counter()
    globally_selected_speakers: set[str] = set()

    for row in metadata:
        destination = (
            audio_root / row["source_split"] / row["label"] / row["filename"]
        )
        if not destination.exists():
            continue
        size = destination.stat().st_size
        row.update(
            {
                "audio_path": destination.relative_to(
                    data_root.parent
                ).as_posix(),
                "audio_status": "downloaded",
                "audio_bytes": size,
            }
        )
        selected_counts[(row["source_split"], row["label"])] += 1
        globally_selected_speakers.add(row["speaker_id"])

    # Reuse the earlier sanity WAVs when they fit an unfilled split/label bucket.
    for row in metadata:
        bucket = (row["source_split"], row["label"])
        if selected_counts[bucket] >= targets[row["source_split"]]:
            continue
        legacy = legacy_root / row["source_split"] / row["filename"]
        if not legacy.exists() or row["speaker_id"] in globally_selected_speakers:
            continue
        destination = (
            audio_root / row["source_split"] / row["label"] / row["filename"]
        )
        remaining_bytes = (
            max_data_bytes - METADATA_RESERVE_BYTES - tree_size_bytes(data_root)
        )
        try:
            size = normalize_wav(legacy, destination, remaining_bytes)
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(
                {
                    "sample_id": row["sample_id"],
                    "issue": f"legacy_audio_normalization_failed: {exc}",
                }
            )
            continue
        row.update(
            {
                "audio_path": destination.relative_to(
                    data_root.parent
                ).as_posix(),
                "audio_status": "downloaded",
                "audio_bytes": size,
            }
        )
        selected_counts[bucket] += 1
        globally_selected_speakers.add(row["speaker_id"])

    source_files = source_parquet_files()
    selected_shards = choose_source_shards(metadata, source_files, targets)
    temporary_root = data_root / ".phase1_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar = false")
    try:
        for shard in selected_shards:
            split = shard["split"]
            needed = {
                label: max(0, targets[split] - selected_counts[(split, label)])
                for label in LABELS
            }
            if not any(needed.values()):
                continue

            if (
                tree_size_bytes(data_root)
                + shard["size"]
                + METADATA_RESERVE_BYTES
                > max_data_bytes
            ):
                raise RuntimeError(
                    f"Temporary shard {shard['filename']} would exceed the "
                    f"{max_data_bytes:,}-byte data cap."
                )

            local_shard = temporary_root / shard["filename"]
            if not local_shard.exists():
                print(
                    f"Downloading {shard['filename']} "
                    f"({shard['size']:,} bytes)..."
                )
                download_file(
                    shard["url"],
                    local_shard,
                    shard["size"],
                    data_root,
                    max_data_bytes,
                )

            result = connection.execute(
                """
                SELECT
                    region,
                    filename,
                    speakerID,
                    octet_length(audio.bytes) AS source_audio_bytes
                FROM read_parquet(?)
                ORDER BY filename
                """,
                [str(local_shard)],
            )
            available: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for source_region, filename, speaker_id, source_audio_bytes in (
                result.fetchall()
            ):
                label = LABEL_MAPPING[source_region]
                sample_id = f"{split}:{filename}"
                if sample_id not in by_id:
                    continue
                available[label].append(
                    {
                        "sample_id": sample_id,
                        "filename": filename,
                        "speaker_id": speaker_id,
                        "source_audio_bytes": int(source_audio_bytes),
                    }
                )

            selected_rows: list[dict[str, Any]] = []
            for label in LABELS:
                if not needed[label]:
                    continue
                rows = select_rows_for_label(
                    available[label],
                    needed[label],
                    globally_selected_speakers,
                    seed + SPLITS.index(split) * 10 + LABELS.index(label),
                )
                for row in rows:
                    row["label"] = label
                selected_rows.extend(rows)

            filenames = [row["filename"] for row in selected_rows]
            if filenames:
                placeholders = ", ".join("?" for _ in filenames)
                result = connection.execute(
                    f"""
                    SELECT filename, audio.bytes
                    FROM read_parquet(?)
                    WHERE filename IN ({placeholders})
                    """,
                    [str(local_shard), *filenames],
                )
                audio_by_filename = {
                    str(filename): bytes(audio_bytes)
                    for filename, audio_bytes in result.fetchall()
                }
            else:
                audio_by_filename = {}

            for selected in selected_rows:
                row = by_id[selected["sample_id"]]
                raw_audio = audio_by_filename.get(selected["filename"])
                if raw_audio is None:
                    issues.append(
                        {
                            "sample_id": row["sample_id"],
                            "issue": "audio_bytes_missing_from_source_shard",
                        }
                    )
                    continue
                destination = (
                    audio_root / split / row["label"] / row["filename"]
                )
                remaining_bytes = (
                    max_data_bytes
                    - METADATA_RESERVE_BYTES
                    - tree_size_bytes(data_root)
                )
                try:
                    size = normalize_wav(raw_audio, destination, remaining_bytes)
                except (OSError, RuntimeError, ValueError) as exc:
                    issues.append(
                        {
                            "sample_id": row["sample_id"],
                            "issue": f"audio_normalization_failed: {exc}",
                        }
                    )
                    continue
                row.update(
                    {
                        "audio_path": destination.relative_to(
                            data_root.parent
                        ).as_posix(),
                        "audio_status": "downloaded",
                        "audio_bytes": size,
                    }
                )
                selected_counts[(split, row["label"])] += 1
                globally_selected_speakers.add(row["speaker_id"])

            local_shard.unlink(missing_ok=True)
            print(
                f"Current counts after {shard['filename']}: "
                + ", ".join(
                    f"{label}={selected_counts[(split, label)]}"
                    for label in LABELS
                )
            )
    finally:
        connection.close()
        shutil.rmtree(temporary_root, ignore_errors=True)

    missing_targets = {
        f"{split}:{label}": targets[split] - selected_counts[(split, label)]
        for split in SPLITS
        for label in LABELS
        if selected_counts[(split, label)] < targets[split]
    }
    if missing_targets:
        raise RuntimeError(f"Audio targets were not met: {missing_targets}")
    return issues


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    metadata: list[dict[str, Any]],
    issues: list[dict[str, str]],
    data_root: Path,
    summary_path: Path,
    max_data_bytes: int,
    seed: int,
    targets: dict[str, int],
) -> dict[str, Any]:
    processed = data_root / "processed"
    metadata_fields = [
        "sample_id",
        "source_split",
        "source_region",
        "label",
        "province_code",
        "province_name",
        "filename",
        "text",
        "speaker_id",
        "gender",
        "source_audio_path",
        "source_parquet_file",
        "audio_path",
        "audio_status",
        "audio_bytes",
    ]
    write_csv(processed / "metadata_clean.csv", metadata, metadata_fields)

    class_rows = []
    speaker_rows = []
    for label in LABELS:
        label_rows = [row for row in metadata if row["label"] == label]
        downloaded = [
            row for row in label_rows if row["audio_status"] == "downloaded"
        ]
        class_rows.append(
            {
                "label": label,
                "total_samples": len(label_rows),
                "downloaded_samples": len(downloaded),
            }
        )
        speaker_rows.append(
            {
                "label": label,
                "total_speakers": len(
                    {row["speaker_id"] for row in label_rows if row["speaker_id"]}
                ),
                "downloaded_speakers": len(
                    {row["speaker_id"] for row in downloaded if row["speaker_id"]}
                ),
                "missing_speaker_ids": sum(
                    not row["speaker_id"] for row in label_rows
                ),
            }
        )
    write_csv(
        processed / "class_counts.csv",
        class_rows,
        ["label", "total_samples", "downloaded_samples"],
    )
    write_csv(
        processed / "speaker_counts.csv",
        speaker_rows,
        [
            "label",
            "total_speakers",
            "downloaded_speakers",
            "missing_speaker_ids",
        ],
    )

    split_class_rows = []
    for split in SPLITS:
        for label in LABELS:
            rows = [
                row
                for row in metadata
                if row["source_split"] == split and row["label"] == label
            ]
            downloaded = [
                row for row in rows if row["audio_status"] == "downloaded"
            ]
            split_class_rows.append(
                {
                    "source_split": split,
                    "label": label,
                    "total_samples": len(rows),
                    "downloaded_samples": len(downloaded),
                    "target_samples": targets[split],
                }
            )
    write_csv(
        processed / "split_class_counts.csv",
        split_class_rows,
        [
            "source_split",
            "label",
            "total_samples",
            "downloaded_samples",
            "target_samples",
        ],
    )

    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        if row["speaker_id"]:
            speaker_splits[row["speaker_id"]].add(row["source_split"])
    speaker_overlap_rows = [
        {
            "speaker_id": speaker_id,
            "source_splits": "|".join(sorted(splits)),
        }
        for speaker_id, splits in sorted(speaker_splits.items())
        if len(splits) > 1
    ]
    write_csv(
        processed / "speaker_split_overlap.csv",
        speaker_overlap_rows,
        ["speaker_id", "source_splits"],
    )

    selected_speaker_splits: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        if row["speaker_id"] and row["audio_status"] == "downloaded":
            selected_speaker_splits[row["speaker_id"]].add(row["source_split"])
    selected_speaker_overlap_rows = [
        {
            "speaker_id": speaker_id,
            "source_splits": "|".join(sorted(splits)),
        }
        for speaker_id, splits in sorted(selected_speaker_splits.items())
        if len(splits) > 1
    ]
    write_csv(
        processed / "selected_speaker_split_overlap.csv",
        selected_speaker_overlap_rows,
        ["speaker_id", "source_splits"],
    )

    unavailable = [
        {
            "sample_id": row["sample_id"],
            "source_split": row["source_split"],
            "label": row["label"],
            "filename": row["filename"],
            "audio_status": row["audio_status"],
        }
        for row in metadata
        if row["audio_status"] != "downloaded"
    ]
    write_csv(
        processed / "missing_audio.csv",
        unavailable,
        ["sample_id", "source_split", "label", "filename", "audio_status"],
    )

    missing_speaker_issues = [
        {"sample_id": row["sample_id"], "issue": "missing_speaker_id"}
        for row in metadata
        if not row["speaker_id"]
    ]
    all_issues = issues + missing_speaker_issues
    write_csv(
        processed / "metadata_issues.csv",
        all_issues,
        ["sample_id", "issue"],
    )

    data_bytes = tree_size_bytes(data_root)
    if data_bytes > max_data_bytes:
        raise RuntimeError(
            f"Generated data exceeds cap: {data_bytes:,} > {max_data_bytes:,} bytes."
        )

    status_counts = Counter(row["audio_status"] for row in metadata)
    split_counts = Counter(row["source_split"] for row in metadata)
    downloaded_rows = [
        row for row in metadata if row["audio_status"] == "downloaded"
    ]
    summary = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "license": "CC-BY-NC-ND-4.0",
        "metadata_samples": len(metadata),
        "metadata_provinces": len(
            {(row["province_code"], row["province_name"]) for row in metadata}
        ),
        "metadata_speakers": len(
            {row["speaker_id"] for row in metadata if row["speaker_id"]}
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": {
            row["label"]: row["total_samples"] for row in class_rows
        },
        "speaker_counts": {
            row["label"]: row["total_speakers"] for row in speaker_rows
        },
        "downloaded_class_counts": dict(
            sorted(Counter(row["label"] for row in downloaded_rows).items())
        ),
        "downloaded_split_counts": dict(
            sorted(
                Counter(row["source_split"] for row in downloaded_rows).items()
            )
        ),
        "downloaded_split_class_counts": {
            f"{row['source_split']}:{row['label']}": row["downloaded_samples"]
            for row in split_class_rows
        },
        "source_speaker_split_overlap_count": len(speaker_overlap_rows),
        "selected_speaker_split_overlap_count": len(
            selected_speaker_overlap_rows
        ),
        "label_mapping": LABEL_MAPPING,
        "audio_status_counts": dict(sorted(status_counts.items())),
        "downloaded_audio_bytes": sum(
            int(row["audio_bytes"]) for row in downloaded_rows
        ),
        "data_directory_bytes": data_bytes,
        "max_data_bytes": max_data_bytes,
        "data_cap_respected": data_bytes <= max_data_bytes,
        "selection_seed": seed,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "target_samples_per_label": targets,
        "source_note": (
            "Complete metadata is read from official original Parquet shards. "
            "The local subset is extracted shard-by-shard, prioritizes shorter "
            "files and speaker diversity, and stores mono 16 kHz PCM WAV files "
            "under the local byte budget. Source splits are preserved."
        ),
        "dataset_card_record_count": 18_949,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return summary


def ensure_outputs_absent(data_root: Path, summary_path: Path) -> None:
    paths = [
        data_root / "processed" / "metadata_clean.csv",
        data_root / "processed" / "class_counts.csv",
        data_root / "processed" / "speaker_counts.csv",
        data_root / "processed" / "split_class_counts.csv",
        data_root / "processed" / "speaker_split_overlap.csv",
        data_root / "processed" / "selected_speaker_split_overlap.csv",
        data_root / "processed" / "missing_audio.csv",
        data_root / "processed" / "metadata_issues.csv",
        summary_path,
    ]
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing Phase 1 outputs: {formatted}. "
            "Pass --overwrite to regenerate them."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare complete ViMD metadata and a balanced audio subset under "
            "a strict local data budget."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/reports/phase1_dataset_summary.json"),
    )
    parser.add_argument(
        "--max-data-bytes",
        type=int,
        default=DEFAULT_MAX_DATA_BYTES,
    )
    parser.add_argument(
        "--train-per-label",
        type=int,
        default=DEFAULT_TARGETS["train"],
    )
    parser.add_argument(
        "--valid-per-label",
        type=int,
        default=DEFAULT_TARGETS["valid"],
    )
    parser.add_argument(
        "--test-per-label",
        type=int,
        default=DEFAULT_TARGETS["test"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Prepare complete metadata without downloading audio.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate CSV and JSON outputs; existing audio is reused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_data_bytes <= METADATA_RESERVE_BYTES:
        raise ValueError(
            f"--max-data-bytes must exceed {METADATA_RESERVE_BYTES:,}."
        )
    targets = {
        "train": args.train_per_label,
        "valid": args.valid_per_label,
        "test": args.test_per_label,
    }
    if any(count < 0 for count in targets.values()):
        raise ValueError("Per-label sample targets cannot be negative.")
    if not args.overwrite:
        ensure_outputs_absent(args.data_root, args.summary_path)

    print("Reading metadata columns from official ViMD Parquet shards...")
    metadata = load_source_metadata()
    print(f"Validated {len(metadata):,} metadata rows.")

    issues: list[dict[str, str]] = []
    if not args.metadata_only and any(targets.values()):
        print(
            "Acquiring a balanced audio subset with per-label targets: "
            + ", ".join(f"{split}={count}" for split, count in targets.items())
        )
        issues = acquire_audio_subset(
            metadata=metadata,
            data_root=args.data_root,
            max_data_bytes=args.max_data_bytes,
            targets=targets,
            seed=args.seed,
        )

    summary = write_outputs(
        metadata=metadata,
        issues=issues,
        data_root=args.data_root,
        summary_path=args.summary_path,
        max_data_bytes=args.max_data_bytes,
        seed=args.seed,
        targets=targets,
    )
    print(
        f"Phase 1 complete: {summary['metadata_samples']:,} metadata rows, "
        f"{summary['audio_status_counts'].get('downloaded', 0):,} local audio "
        f"files, {summary['data_directory_bytes']:,} data bytes."
    )


if __name__ == "__main__":
    main()
