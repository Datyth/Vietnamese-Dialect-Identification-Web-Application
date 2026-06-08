"""MFCC feature extraction with NumPy-only signal processing."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_N_MFCC = 13
DEFAULT_N_MELS = 40
DEFAULT_N_FFT = 512
DEFAULT_FRAME_LENGTH = 400
DEFAULT_HOP_LENGTH = 160


def mfcc_mean_std(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Return MFCC mean/std aggregation for one waveform."""
    mfcc = mfcc_matrix(
        waveform=waveform,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    feature = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])
    return np.nan_to_num(feature, copy=False).astype(np.float32)


def mfcc_matrix(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Compute frame-level MFCC coefficients."""
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate!r}.")
    if n_mfcc <= 0 or n_mels <= 0:
        raise ValueError("n_mfcc and n_mels must be positive.")
    if n_mfcc > n_mels:
        raise ValueError("n_mfcc cannot exceed n_mels.")

    power = power_spectrogram(
        waveform=waveform,
        n_fft=n_fft,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    filters = mel_filterbank(sample_rate, n_fft, n_mels)
    mel_power = np.maximum(
        np.einsum(
            "tf,mf->tm",
            power.astype(np.float64),
            filters.astype(np.float64),
        ),
        1e-10,
    )
    log_mel = np.log(mel_power)
    basis = dct_basis(n_mfcc, n_mels)
    return np.einsum(
        "tm,cm->tc",
        log_mel,
        basis.astype(np.float64),
    ).astype(np.float32)


def power_spectrogram(
    waveform: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Compute a simple Hann-windowed power spectrogram."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, received shape {waveform.shape}.")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("Waveform contains non-finite samples.")
    if n_fft < frame_length:
        raise ValueError("n_fft must be at least frame_length.")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive.")

    if waveform.size < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.size))

    frame_count = 1 + int(np.ceil((waveform.size - frame_length) / hop_length))
    padded_length = (frame_count - 1) * hop_length + frame_length
    if padded_length > waveform.size:
        waveform = np.pad(waveform, (0, padded_length - waveform.size))

    frame_offsets = hop_length * np.arange(frame_count)[:, None]
    sample_offsets = np.arange(frame_length)[None, :]
    frames = waveform[frame_offsets + sample_offsets]
    frames = frames * np.hanning(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    return (np.abs(spectrum) ** 2 / float(n_fft)).astype(np.float32)


@lru_cache(maxsize=16)
def mel_filterbank(
    sample_rate: int,
    n_fft: int = DEFAULT_N_FFT,
    n_mels: int = DEFAULT_N_MELS,
    min_hz: float = 20.0,
    max_hz: float | None = None,
) -> np.ndarray:
    """Create triangular mel filters."""
    if max_hz is None:
        max_hz = sample_rate / 2
    if min_hz < 0 or max_hz <= min_hz:
        raise ValueError("Invalid mel frequency range.")

    mel_points = np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for bin_index in range(left, center):
            if 0 <= bin_index < filters.shape[1]:
                filters[index, bin_index] = (
                    bin_index - left
                ) / float(center - left)
        for bin_index in range(center, right):
            if 0 <= bin_index < filters.shape[1]:
                filters[index, bin_index] = (
                    right - bin_index
                ) / float(right - center)

        total = filters[index].sum()
        if total > 0:
            filters[index] /= total
    return filters


@lru_cache(maxsize=16)
def dct_basis(n_mfcc: int = DEFAULT_N_MFCC, n_mels: int = DEFAULT_N_MELS) -> np.ndarray:
    """Return an orthonormal DCT-II basis."""
    basis = np.empty((n_mfcc, n_mels), dtype=np.float32)
    samples = np.arange(n_mels, dtype=np.float32) + 0.5
    for coefficient in range(n_mfcc):
        basis[coefficient] = np.cos(np.pi * coefficient * samples / n_mels)
    basis[0] *= np.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        basis[1:] *= np.sqrt(2.0 / n_mels)
    return basis


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(value_mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(value_mel) / 2595.0) - 1.0)
