import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Import your project's custom feature extraction tools
from src.features.logmel import log_mel_spectrogram
from src.features.mfcc import mfcc_matrix

# 1. Load your preprocessed metadata to find a sample from each dialect
metadata_path = "data/processed/preprocessed_metadata.csv"
if not os.path.exists(metadata_path):
    raise FileNotFoundError("Please run your preprocessing pipeline first!")

df = pd.read_csv(metadata_path)

# 2. Grab the first available PREPROCESSED file for each dialect label
dialects = ['Northern', 'Central', 'Southern']
sample_files = {}

for d in dialects:
    # Filter for the specific dialect
    sample_rows = df[df['label'].str.lower() == d.lower()]
    if sample_rows.empty:
        raise ValueError(f"No samples found for dialect label: {d}")
    
    # We must explicitly use 'preprocessed_audio_path', not the raw 'audio_path'
    sample_files[d] = sample_rows.iloc[0]['preprocessed_audio_path']

# 3. Load the 16-second arrays
waveforms = {}
sample_rates = {}
for d, path in sample_files.items():
    sr, y = wav.read(path)
    # Convert to float32 if saved as int16
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    waveforms[d] = y
    sample_rates[d] = sr

# ─────────────────────────────────────────────────────────
# FIGURE 2.1: Waveform Plots
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
for i, d in enumerate(dialects):
    y = waveforms[d]
    sr = sample_rates[d]
    
    # Dynamically generate time axis for this specific array length (should be exactly 16s)
    time_axis = np.linspace(0, len(y) / sr, len(y))
    
    axes[i].plot(time_axis, y, color='#1f77b4', alpha=0.7)
    axes[i].set_title(f"{d} Dialect Preprocessed Waveform")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.savefig("waveform_comparison.png", dpi=300)
plt.close()
print("✓ Generated waveform_comparison.png")

# ─────────────────────────────────────────────────────────
# FIGURE 2.2: Log-Mel Spectrogram Comparison
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, d in enumerate(dialects):
    # Using your project's log_mel_spectrogram implementation
    mel_spec = log_mel_spectrogram(waveforms[d])
    
    img = axes[i].imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[i].set_title(f"{d} Log-Mel Spectrogram")
    axes[i].set_xlabel("Time Frames")

axes[0].set_ylabel("Mel Bins (64)")
plt.colorbar(img, ax=axes.tolist(), label="Log Energy")
plt.savefig("spectrogram_comparison.png", dpi=300)
plt.close()
print("✓ Generated spectrogram_comparison.png")

# ─────────────────────────────────────────────────────────
# FIGURE 2.3: MFCC Feature Vector (26-D)
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x_indices = np.arange(26)

for d in dialects:
    # Replicate your mfcc_mean_std logic explicitly
    mel_frames = mfcc_matrix(waveforms[d]) 
    
    # Calculate means and stds across frames
    means = np.mean(mel_frames, axis=1)[:13]
    stds = np.std(mel_frames, axis=1)[:13]
    feature_vector = np.concatenate([means, stds])
    
    ax.plot(x_indices, feature_vector, marker='o', label=f"{d} Dialect")

ax.set_title("26-D MFCC Mean & Standard Deviation Profile")
ax.set_xticks(x_indices)
ax.set_xticklabels([f"Mean_{i}" for i in range(13)] + [f"Std_{i}" for i in range(13)], rotation=90)
ax.set_xlabel("Feature Index")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("mfcc_comparison.png", dpi=300)
plt.close()
print("✓ Generated mfcc_comparison.png")