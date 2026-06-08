"""Shared audio loading and fixed-length preprocessing utilities."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr


TARGET_SAMPLE_RATE = 16_000
FIXED_DURATION_SECONDS = 16.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * FIXED_DURATION_SECONDS)
TARGET_RMS = 0.08
SILENCE_ABSOLUTE_THRESHOLD = 0.005
SILENCE_RELATIVE_THRESHOLD = 0.02
MAX_GAIN = 10.0
PEAK_LIMIT = 0.99


@dataclass(frozen=True)
class AudioPreprocessStats:
    """Small summary of one preprocessing operation."""

    original_sample_rate: int
    original_samples: int
    original_duration_seconds: float
    trimmed_samples: int
    output_sample_rate: int
    output_samples: int
    output_duration_seconds: float
    output_rms: float
    output_peak: float


def load_audio(source: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Load an audio file or byte string as mono float32 waveform."""
    input_source: Any = io.BytesIO(source) if isinstance(source, bytes) else source
    waveform, sample_rate = sf.read(input_source, dtype="float32", always_2d=True)
    if waveform.size == 0:
        raise ValueError("Audio file is empty.")

    mono = waveform.mean(axis=1).astype(np.float32, copy=False)
    if not np.all(np.isfinite(mono)):
        raise ValueError("Audio contains non-finite samples.")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate!r}.")
    return mono, int(sample_rate)


def resample_audio(
    waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample waveform when needed."""
    waveform = ensure_1d_float32(waveform)
    if source_sample_rate == target_sample_rate:
        return waveform
    if source_sample_rate <= 0:
        raise ValueError(f"Invalid source sample rate: {source_sample_rate!r}.")
    return soxr.resample(
        waveform,
        source_sample_rate,
        target_sample_rate,
        quality="HQ",
    ).astype(np.float32)


def trim_silence(
    waveform: np.ndarray,
    absolute_threshold: float = SILENCE_ABSOLUTE_THRESHOLD,
    relative_threshold: float = SILENCE_RELATIVE_THRESHOLD,
) -> np.ndarray:
    """Remove leading and trailing low-amplitude samples."""
    waveform = ensure_1d_float32(waveform)
    if waveform.size == 0:
        return waveform

    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-8:
        return np.zeros(0, dtype=np.float32)

    threshold = max(float(absolute_threshold), peak * float(relative_threshold))
    voiced = np.flatnonzero(np.abs(waveform) >= threshold)
    if voiced.size == 0:
        return np.zeros(0, dtype=np.float32)
    return waveform[int(voiced[0]) : int(voiced[-1]) + 1]


def normalize_volume(
    waveform: np.ndarray,
    target_rms: float = TARGET_RMS,
    max_gain: float = MAX_GAIN,
    peak_limit: float = PEAK_LIMIT,
) -> np.ndarray:
    """Normalize RMS while keeping samples below the configured peak limit."""
    waveform = ensure_1d_float32(waveform)
    if waveform.size == 0:
        return waveform

    rms = float(np.sqrt(np.mean(np.square(waveform.astype(np.float64)))))
    if rms <= 1e-8:
        return waveform

    gain = min(float(target_rms) / rms, float(max_gain))
    normalized = waveform * gain
    peak = float(np.max(np.abs(normalized)))
    if peak > peak_limit:
        normalized = normalized * (float(peak_limit) / peak)
    return normalized.astype(np.float32)


def fix_length(
    waveform: np.ndarray,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """Center-crop or zero-pad a waveform to an exact number of samples."""
    waveform = ensure_1d_float32(waveform)
    if target_samples <= 0:
        raise ValueError("target_samples must be positive.")

    if waveform.size > target_samples:
        start = (waveform.size - target_samples) // 2
        return waveform[start : start + target_samples].astype(np.float32)

    if waveform.size < target_samples:
        missing = target_samples - waveform.size
        left = missing // 2
        right = missing - left
        return np.pad(waveform, (left, right), mode="constant").astype(np.float32)

    return waveform.astype(np.float32)


def preprocess_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> tuple[np.ndarray, AudioPreprocessStats]:
    """Apply the complete Phase 2 deterministic preprocessing pipeline."""
    waveform = ensure_1d_float32(waveform)
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate!r}.")
    original_samples = int(waveform.size)
    resampled = resample_audio(waveform, sample_rate, target_sample_rate)
    trimmed = trim_silence(resampled)
    normalized = normalize_volume(trimmed)
    fixed = fix_length(normalized, target_samples)
    fixed = np.clip(fixed, -PEAK_LIMIT, PEAK_LIMIT).astype(np.float32)

    stats = AudioPreprocessStats(
        original_sample_rate=int(sample_rate),
        original_samples=original_samples,
        original_duration_seconds=original_samples / float(sample_rate),
        trimmed_samples=int(trimmed.size),
        output_sample_rate=int(target_sample_rate),
        output_samples=int(fixed.size),
        output_duration_seconds=fixed.size / float(target_sample_rate),
        output_rms=rms(fixed),
        output_peak=peak(fixed),
    )
    return fixed, stats


def preprocess_file(
    source: str | Path,
    destination: str | Path,
    overwrite: bool = False,
) -> AudioPreprocessStats:
    """Preprocess one audio file and write a PCM 16-bit WAV."""
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        waveform, sample_rate = load_audio(destination_path)
        if sample_rate != TARGET_SAMPLE_RATE or waveform.size != TARGET_SAMPLES:
            raise ValueError(
                f"Existing preprocessed audio has wrong shape: {destination_path}"
            )
        return AudioPreprocessStats(
            original_sample_rate=sample_rate,
            original_samples=int(waveform.size),
            original_duration_seconds=waveform.size / float(sample_rate),
            trimmed_samples=int(waveform.size),
            output_sample_rate=sample_rate,
            output_samples=int(waveform.size),
            output_duration_seconds=waveform.size / float(sample_rate),
            output_rms=rms(waveform),
            output_peak=peak(waveform),
        )

    waveform, sample_rate = load_audio(source)
    processed, stats = preprocess_waveform(waveform, sample_rate)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        destination_path,
        processed,
        TARGET_SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    return stats


def ensure_1d_float32(waveform: np.ndarray) -> np.ndarray:
    """Validate a waveform and return a contiguous 1D float32 array."""
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError(f"Expected 1D waveform, received shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Audio contains non-finite samples.")
    return np.ascontiguousarray(array)


def rms(waveform: np.ndarray) -> float:
    waveform = ensure_1d_float32(waveform)
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(waveform.astype(np.float64)))))


def peak(waveform: np.ndarray) -> float:
    waveform = ensure_1d_float32(waveform)
    if waveform.size == 0:
        return 0.0
    return float(np.max(np.abs(waveform)))
