# E7 Central Error Analysis Summary

## E7 Test Confusion

- Accuracy: `0.8025`.
- Macro F1: `0.7947`.
- Central recall: `0.5914`.
- Central F1: `0.6849`.
- Central -> Northern errors: `85`.
- Central -> Southern errors: `114`.

## Dataset Signals To Inspect

- Samples: Northern `4680`, Central `4390`, Southern `4824`.
- Unique speakers: Northern `3419`, Central `3022`, Southern `3101`.
- Unique provinces: Northern `24`, Central `19`, Southern `19`.
- Central province entropy is not the highest; Central difficulty may come more from boundary overlap/model representation than province spread alone.

## Acoustic Nearest-Centroid Signal

- Central nearest Northern: `10`.
- Central nearest Central: `38`.
- Central nearest Southern: `32`.
- More sampled Central audio is closer to non-Central acoustic centroids than to Central, supporting the overlap hypothesis.

## Suggested Next Checks

1. Run the optional E7 forward section to export prediction confidence and fused embeddings.
2. Compare Central->Northern and Central->Southern samples by province and speaker.
3. If Central errors are low-confidence, try central logit bias or class weighting.
4. If Central errors are high-confidence, inspect embedding overlap and consider targeted augmentation/fine-tuning.

## Generated Figures

- `outputs/figures/e7_central_error_analysis/audio_feature_pca_by_label.png`
- `outputs/figures/e7_central_error_analysis/audio_logmel_mean_boxplot.png`
- `outputs/figures/e7_central_error_analysis/audio_logmel_std_boxplot.png`
- `outputs/figures/e7_central_error_analysis/audio_nearest_centroid_matrix.png`
- `outputs/figures/e7_central_error_analysis/audio_peak_boxplot.png`
- `outputs/figures/e7_central_error_analysis/audio_rms_boxplot.png`
- `outputs/figures/e7_central_error_analysis/audio_spectral_bandwidth_mean_boxplot.png`
- `outputs/figures/e7_central_error_analysis/audio_spectral_centroid_mean_boxplot.png`
- `outputs/figures/e7_central_error_analysis/audio_spectral_flatness_mean_boxplot.png`
- `outputs/figures/e7_central_error_analysis/audio_zcr_boxplot.png`
- `outputs/figures/e7_central_error_analysis/central_top_provinces.png`
- `outputs/figures/e7_central_error_analysis/e7_test_confusion_matrix.png`
- `outputs/figures/e7_central_error_analysis/metadata_audio_bytes_boxplot.png`
- `outputs/figures/e7_central_error_analysis/metadata_original_duration_boxplot.png`
- `outputs/figures/e7_central_error_analysis/metadata_preprocessed_peak_boxplot.png`
- `outputs/figures/e7_central_error_analysis/metadata_preprocessed_rms_boxplot.png`
- `outputs/figures/e7_central_error_analysis/metadata_samples_by_split_label.png`
- `outputs/figures/e7_central_error_analysis/metadata_speaker_province_diversity.png`
- `outputs/figures/e7_central_error_analysis/speaker_samples_by_label_boxplot.png`
