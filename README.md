# Vietnamese-Dialect-Identification-Web-Application
This project aims to build a lightweight system that classifies short Vietnamese speech recordings into regional dialect groups. The initial scope is three classes: Northern, Central, and Southern.

## Phase 1: Dataset Metadata

Phase 1 prepares the official
[ViMD dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)
for the three project labels:

| ViMD region | Project label |
| --- | --- |
| `North` | `Northern` |
| `Central` | `Central` |
| `South` | `Southern` |

The preparation command reads metadata columns from the remote Parquet shards,
then processes one selected shard at a time. It prioritizes short files and
speaker diversity, stores mono 16 kHz PCM WAV files, and removes each temporary
shard before continuing. The default balanced subset is:

| Split | Northern | Central | Southern |
| --- | ---: | ---: | ---: |
| Train | 100 | 100 | 100 |
| Validation | 15 | 15 | 15 |
| Test | 15 | 15 | 15 |

The complete `data/` directory remains below the strict 1,000,000,000-byte
limit:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.data.prepare_metadata
```

Custom targets:

```bash
python3 -m src.data.prepare_metadata --overwrite \
  --train-per-label 100 --valid-per-label 15 --test-per-label 15
```

Use metadata only, without downloading audio:

```bash
python3 -m src.data.prepare_metadata --metadata-only
```

Generated Phase 1 outputs:

- `data/processed/metadata_clean.csv`
- `data/processed/class_counts.csv`
- `data/processed/split_class_counts.csv`
- `data/processed/speaker_counts.csv`
- `data/processed/speaker_split_overlap.csv`
- `data/processed/selected_speaker_split_overlap.csv`
- `data/processed/missing_audio.csv`
- `data/processed/metadata_issues.csv`
- `outputs/reports/phase1_dataset_summary.json`

Phase 1 dependencies are intentionally small:

- `duckdb` reads metadata and embedded audio from one Parquet shard at a time.
- `soundfile` decodes both PCM and IEEE-float source WAV files.
- `soxr` performs reliable 16 kHz resampling.
- `numpy` handles mono conversion and sample clipping.

Using the Python standard library alone is insufficient because it cannot read
Parquet and rejects IEEE-float WAV input.

## Local Environment

Use a local uv virtual environment in the repository:

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

## Phase 2: Audio Preprocessing

Phase 2 turns the selected Phase 1 audio into fixed-length waveforms shared by
training and future inference:

- load audio with SoundFile;
- convert to mono;
- resample to 16 kHz when needed;
- trim leading/trailing silence;
- normalize RMS volume;
- center-crop or zero-pad to exactly 16 seconds.

Run:

```bash
.venv/bin/python -m src.data.preprocess_audio --overwrite
```

Generated outputs:

- `data/processed/audio_preprocessed_16s/`
- `data/processed/preprocessed_metadata.csv`
- `data/processed/preprocess_audio_issues.csv`
- `outputs/reports/phase2_preprocessing_summary.json`

The current run preprocessed 390/390 selected files with 0 issues. Every output
file is mono, 16 kHz, and 256,000 samples.

## Phase 3: Minimal Data EDA

The current Phase 3 report is intentionally minimal and validates the data
needed before the traditional baseline:

- `outputs/reports/data_eda.md`

It confirms the balanced 100/15/15 per-class split and fixed 16-second output
duration. Full plots are deferred.

## Phase 4: MFCC Baselines

Phase 4 trains traditional models from MFCC mean/std features:

- Logistic Regression;
- SVM.

Run:

```bash
.venv/bin/python -m src.training.train_baseline --overwrite
```

Generated outputs:

- `outputs/metrics/baseline_results.json`
- `outputs/metrics/*_confusion_matrix.csv`
- `outputs/models/logistic_regression_mfcc.pkl`
- `outputs/models/svm_mfcc.pkl`
- `outputs/reports/phase4_baseline_report.md`

Current validation results:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.6000 | 0.5981 |
| SVM | 0.6889 | 0.6918 |

`scikit-learn` is used only for the Phase 4 model and metric implementations.
MFCC extraction is implemented locally with NumPy.
