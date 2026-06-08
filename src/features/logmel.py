"""Log-Mel spectrogram extraction for the lightweight CNN."""

from __future__ import annotations

import numpy as np

from src.features.mfcc import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_SAMPLE_RATE,
    mel_filterbank,
    power_spectrogram,
)


DEFAULT_N_MELS = 64
LOG_FLOOR = 1e-10
STANDARDIZE_EPSILON = 1e-6


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    standardize: bool = True,
) -> np.ndarray:
    """Return a log-Mel spectrogram shaped as mel bins by time frames."""
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate!r}.")
    if n_mels <= 0:
        raise ValueError("n_mels must be positive.")

    power = power_spectrogram(
        waveform=waveform,
        n_fft=n_fft,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    filters = mel_filterbank(sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_power = np.maximum(
        np.einsum(
            "tf,mf->tm",
            power.astype(np.float64),
            filters.astype(np.float64),
        ),
        LOG_FLOOR,
    )
    feature = np.log(mel_power).T.astype(np.float32)
    if standardize:
        feature = standardize_feature(feature)
    return np.nan_to_num(feature, copy=False).astype(np.float32)


def standardize_feature(feature: np.ndarray) -> np.ndarray:
    """Standardize one spectrogram to zero mean and unit-ish variance."""
    array = np.asarray(feature, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D feature, received shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Feature contains non-finite values.")

    mean = float(array.mean())
    std = float(array.std())
    if std < STANDARDIZE_EPSILON:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - mean) / std).astype(np.float32)
