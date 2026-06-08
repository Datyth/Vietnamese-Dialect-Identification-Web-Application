"""Preprocess selected ViMD audio into fixed-length waveforms."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import soundfile as sf

from src.utils.audio import (
    FIXED_DURATION_SECONDS,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
    load_audio,
    peak,
    preprocess_file,
    rms,
)


LABELS = ("Northern", "Central", "Southern")
SPLITS = ("train", "valid", "test")
REQUIRED_METADATA_FIELDS = (
    "sample_id",
    "source_split",
    "label",
    "filename",
    "audio_path",
    "audio_status",
)


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = [
            field
            for field in REQUIRED_METADATA_FIELDS
            if field not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"Metadata is missing required fields: {missing}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_rows(metadata: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [row for row in metadata if row["audio_status"] == "downloaded"]
    if not rows:
        raise ValueError("No rows with audio_status=downloaded found.")
    return rows


def source_audio_path(row: dict[str, str]) -> Path:
    path = Path(row["audio_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Audio path for {row['sample_id']} does not exist: {path}"
        )
    return path


def destination_path(row: dict[str, str], output_root: Path) -> Path:
    split = row["source_split"]
    label = row["label"]
    filename = row["filename"]
    if split not in SPLITS:
        raise ValueError(f"Unsupported source split {split!r} for {row['sample_id']}.")
    if label not in LABELS:
        raise ValueError(f"Unsupported label {label!r} for {row['sample_id']}.")
    if Path(filename).name != filename:
        raise ValueError(f"Unsafe filename {filename!r} for {row['sample_id']}.")
    return output_root / split / label / filename


def file_duration_seconds(path: Path) -> float:
    info = sf.info(path)
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate in {path}: {info.samplerate!r}")
    return info.frames / float(info.samplerate)


def validate_preprocessed(path: Path) -> tuple[float, float]:
    waveform, sample_rate = load_audio(path)
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Wrong sample rate in {path}: {sample_rate}")
    if waveform.shape != (TARGET_SAMPLES,):
        raise ValueError(f"Wrong waveform shape in {path}: {waveform.shape}")
    return rms(waveform), peak(waveform)


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def preprocess_metadata(
    metadata_path: Path,
    output_root: Path,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    metadata = read_metadata(metadata_path)
    rows = selected_rows(metadata)
    processed_rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    for row in rows:
        try:
            source = source_audio_path(row)
            destination = destination_path(row, output_root)
            original_duration = file_duration_seconds(source)
            stats = preprocess_file(source, destination, overwrite=overwrite)
            output_rms, output_peak = validate_preprocessed(destination)
            processed_rows.append(
                {
                    **row,
                    "preprocessed_audio_path": relative_path(destination),
                    "preprocessing_status": "preprocessed",
                    "original_duration_seconds": f"{original_duration:.6f}",
                    "preprocessed_sample_rate": TARGET_SAMPLE_RATE,
                    "preprocessed_samples": TARGET_SAMPLES,
                    "preprocessed_duration_seconds": (
                        f"{stats.output_duration_seconds:.6f}"
                    ),
                    "preprocessed_rms": f"{output_rms:.6f}",
                    "preprocessed_peak": f"{output_peak:.6f}",
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append({"sample_id": row["sample_id"], "issue": str(exc)})

    return processed_rows, issues


def split_class_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["source_split"], row["label"]) for row in rows)
    return [
        {
            "source_split": split,
            "label": label,
            "preprocessed_samples": counts[(split, label)],
        }
        for split in SPLITS
        for label in LABELS
    ]


def summarize_numbers(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def write_summary(
    path: Path,
    processed_rows: list[dict[str, Any]],
    issues: list[dict[str, str]],
    output_root: Path,
) -> dict[str, Any]:
    original_durations = [
        float(row["original_duration_seconds"]) for row in processed_rows
    ]
    processed_durations = [
        float(row["preprocessed_duration_seconds"]) for row in processed_rows
    ]
    exact_shape_count = sum(
        int(row["preprocessed_sample_rate"]) == TARGET_SAMPLE_RATE
        and int(row["preprocessed_samples"]) == TARGET_SAMPLES
        for row in processed_rows
    )
    summary = {
        "phase": "phase2_audio_preprocessing",
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "fixed_duration_seconds": FIXED_DURATION_SECONDS,
        "target_samples": TARGET_SAMPLES,
        "output_root": relative_path(output_root),
        "preprocessed_count": len(processed_rows),
        "issue_count": len(issues),
        "exact_shape_count": exact_shape_count,
        "split_class_counts": split_class_counts(processed_rows),
        "original_duration_seconds": summarize_numbers(original_durations),
        "preprocessed_duration_seconds": summarize_numbers(processed_durations),
        "preprocessed_rms": summarize_numbers(
            [float(row["preprocessed_rms"]) for row in processed_rows]
        ),
        "preprocessed_peak": summarize_numbers(
            [float(row["preprocessed_peak"]) for row in processed_rows]
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return summary


def write_minimal_eda(path: Path, summary: dict[str, Any]) -> None:
    duration = summary["original_duration_seconds"]
    processed = summary["preprocessed_duration_seconds"]
    lines = [
        "# Minimal Data EDA",
        "",
        "Phase 3 is intentionally limited to validation needed before the "
        "traditional MFCC baseline.",
        "",
        "## Split And Class Counts",
        "",
        "| Split | Northern | Central | Southern |",
        "| --- | ---: | ---: | ---: |",
    ]
    count_lookup = {
        (row["source_split"], row["label"]): row["preprocessed_samples"]
        for row in summary["split_class_counts"]
    }
    for split in SPLITS:
        lines.append(
            "| "
            + split
            + " | "
            + " | ".join(str(count_lookup[(split, label)]) for label in LABELS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Duration Summary",
            "",
            "| Source | Min | Median | Mean | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| Original selected audio | {duration['min']:.2f}s | "
                f"{duration['median']:.2f}s | {duration['mean']:.2f}s | "
                f"{duration['max']:.2f}s |"
            ),
            (
                f"| Preprocessed audio | {processed['min']:.2f}s | "
                f"{processed['median']:.2f}s | {processed['mean']:.2f}s | "
                f"{processed['max']:.2f}s |"
            ),
            "",
            "## Validation Summary",
            "",
            f"- Preprocessed files: {summary['preprocessed_count']}.",
            f"- Files with exact 16 kHz / 16 s shape: {summary['exact_shape_count']}.",
            f"- Logged preprocessing issues: {summary['issue_count']}.",
            "- Speaker split validation remains inherited from Phase 1.",
            "- Full figures are deferred; this project is moving to the Phase 4 "
            "MFCC baseline after this minimal check.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing generated outputs: {formatted}. "
            "Pass --overwrite to regenerate them."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess downloaded ViMD audio to fixed 16-second WAV files."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/processed/metadata_clean.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/audio_preprocessed_16s"),
    )
    parser.add_argument(
        "--processed-metadata-path",
        type=Path,
        default=Path("data/processed/preprocessed_metadata.csv"),
    )
    parser.add_argument(
        "--issues-path",
        type=Path,
        default=Path("data/processed/preprocess_audio_issues.csv"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/reports/phase2_preprocessing_summary.json"),
    )
    parser.add_argument(
        "--eda-path",
        type=Path,
        default=Path("outputs/reports/data_eda.md"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate generated CSV/report outputs and rewrite audio files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_paths = [
        args.processed_metadata_path,
        args.issues_path,
        args.summary_path,
        args.eda_path,
    ]
    if not args.overwrite:
        ensure_outputs_absent(generated_paths)

    processed_rows, issues = preprocess_metadata(
        metadata_path=args.metadata_path,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    metadata_fields = list(read_metadata(args.metadata_path)[0].keys()) + [
        "preprocessed_audio_path",
        "preprocessing_status",
        "original_duration_seconds",
        "preprocessed_sample_rate",
        "preprocessed_samples",
        "preprocessed_duration_seconds",
        "preprocessed_rms",
        "preprocessed_peak",
    ]
    write_csv(args.processed_metadata_path, processed_rows, metadata_fields)
    write_csv(args.issues_path, issues, ["sample_id", "issue"])
    summary = write_summary(
        args.summary_path,
        processed_rows,
        issues,
        args.output_root,
    )
    write_minimal_eda(args.eda_path, summary)
    print(
        f"Phase 2 complete: {summary['preprocessed_count']} files, "
        f"{summary['exact_shape_count']} exact-shape files, "
        f"{summary['issue_count']} issues."
    )


if __name__ == "__main__":
    main()

