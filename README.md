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
shard before continuing. Large targets may span as many source shards as needed;
interrupted shard downloads retry with HTTP Range and preserve `.part` files for
the next run. The configured byte budget remains the limiting constraint. The
default balanced subset is:

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

## Automated Pipelines

Run metadata acquisition and audio preprocessing in order:

```bash
scripts/data.sh
```

Customize the balanced download subset and data budget:

```bash
scripts/data.sh \
  --max-data-bytes 1000000000 \
  --train-per-label 100 \
  --valid-per-label 15 \
  --test-per-label 15 \
  --seed 42 \
  --overwrite
```

Use `scripts/data.sh --metadata-only` to prepare metadata without downloading or
preprocessing audio.

Run all training experiments and final evaluation in order:

```bash
scripts/train.sh --device auto
```

Both scripts refuse to overwrite existing outputs by default. Pass
`--overwrite` only when the artifacts should be regenerated:

```bash
scripts/data.sh --overwrite
scripts/train.sh --device mps --overwrite
```

Set `PYTHON_BIN` to use a different Python executable. When PhoWhisper is
already cached, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` keeps training
offline.

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

## Phase 5: Lightweight CNN

Phase 5 trains a small CNN from standardized log-Mel spectrograms. The feature
extractor is implemented with NumPy and reuses the Phase 4 spectrogram/Mel
helpers, while model training uses PyTorch.

Install/update dependencies:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Run with automatic device selection. On Apple Silicon, `auto` prioritizes
PyTorch MPS, then CUDA, then CPU:

```bash
.venv/bin/python -m src.training.train_cnn --overwrite
```

Device can also be selected explicitly:

```bash
.venv/bin/python -m src.training.train_cnn --overwrite --device mps
.venv/bin/python -m src.training.train_cnn --overwrite --device cuda
.venv/bin/python -m src.training.train_cnn --overwrite --device cpu
```

For CUDA, install the CUDA-enabled PyTorch wheel that matches the target machine
from the official PyTorch selector before using `--device cuda`.

Generated outputs:

- `outputs/metrics/cnn_results.json`
- `outputs/metrics/cnn_training_log.csv`
- `outputs/metrics/cnn_valid_confusion_matrix.csv`
- `outputs/metrics/cnn_test_confusion_matrix.csv`
- `outputs/models/lightweight_cnn_logmel.pt`
- `outputs/reports/phase5_cnn_report.md`

The checkpoint under `outputs/models/` is intentionally ignored by Git. Metrics
and reports are small text artifacts and can be versioned.

Current CNN results from the local run:

| Split | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Train | 0.7900 | 0.7846 |
| Validation | 0.4222 | 0.4339 |
| Test | 0.6667 | 0.6668 |

The current environment selected `cpu` because PyTorch reported both MPS and
CUDA as unavailable. The script will still use MPS or CUDA automatically when
the installed PyTorch build exposes those devices.

## Phase 6: PhoWhisper-base Experiments

Phase 6 compares two uses of the same `vinai/PhoWhisper-base` checkpoint. The
pretrained baseline freezes the encoder and trains only the projector/classifier
head. The fine-tuned experiment updates the complete encoder/classification
stack. Both use `WhisperForAudioClassification` and do not run ASR generation.

The frozen baseline is not zero-shot: the original ASR checkpoint has no
Northern/Central/Southern output head, so a small supervised head is required.

Run the pretrained frozen-encoder baseline:

```bash
.venv/bin/python -m src.training.train_phowhisper \
  --training-mode frozen_encoder \
  --learning-rate 1e-3 \
  --max-epochs 20 \
  --patience 5 \
  --overwrite \
  --device auto
```

Run full fine-tuning:

```bash
.venv/bin/python -m src.training.train_phowhisper \
  --training-mode full_fine_tune \
  --overwrite \
  --device auto
```

The frozen run writes `phowhisper_pretrained_*` artifacts, including:

- `outputs/metrics/phowhisper_pretrained_results.json`
- `outputs/metrics/phowhisper_pretrained_training_log.csv`
- `outputs/metrics/phowhisper_pretrained_test_predictions.csv`
- `outputs/models/phowhisper_pretrained_frozen_encoder.pt`
- `outputs/reports/phase6_phowhisper_pretrained_report.md`

The fine-tuned run keeps the existing artifact names:

- `outputs/metrics/phowhisper_results.json`
- `outputs/metrics/phowhisper_training_log.csv`
- `outputs/metrics/phowhisper_valid_confusion_matrix.csv`
- `outputs/metrics/phowhisper_test_confusion_matrix.csv`
- `outputs/metrics/phowhisper_test_predictions.csv`
- `outputs/models/phowhisper_dialect.pt`
- `outputs/reports/phase6_phowhisper_report.md`

Current local comparison:

| Model | Validation Macro F1 | Test Macro F1 | Device |
| --- | ---: | ---: | --- |
| PhoWhisper pretrained, frozen encoder | 0.6720 | 0.7972 | MPS |
| PhoWhisper fine-tuned | 0.6623 | 0.7113 | MPS |

The frozen encoder was verified unchanged against the downloaded checkpoint;
only 132,099 of 20,722,691 classifier-model parameters were trainable. Local
checkpoints are ignored by Git.

## Phase 7: Final Evaluation And Error Analysis

Phase 7 compares all available models and analyzes errors for the best model by
validation macro F1.

Run:

```bash
.venv/bin/python -m src.evaluation.final_evaluation --overwrite
```

Generated outputs:

- `outputs/metrics/final_comparison.csv`
- `outputs/metrics/final_sample_errors.csv`
- `outputs/reports/error_analysis.md`
- `outputs/reports/neural_model_comparison.md`

Current best model by validation macro F1 is still the Phase 4 SVM baseline:

| Model | Validation Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.5981 | 0.6292 |
| SVM | 0.6918 | 0.6264 |
| Lightweight CNN | 0.4339 | 0.6668 |
| PhoWhisper pretrained, frozen encoder | 0.6720 | 0.7972 |
| PhoWhisper fine-tuned | 0.6623 | 0.7113 |

## Phase 8: Selectable Inference And FastAPI App

Phase 8 serves trained dialect classifiers through a Python inference module,
JSON API, and minimal browser interface. The app defaults to the lightweight
CNN, and the browser also lets the user choose the SVM MFCC baseline or
PhoWhisper when the local artifacts are available. Inference imports the same
Phase 2 `preprocess_file()`, Phase 4 `mfcc_mean_std()`, Phase 5
`log_mel_spectrogram()`, and Phase 6 PhoWhisper feature extraction conventions
used for training.

Install dependencies:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Run the app with the default model selection:

```bash
.venv/bin/python -m uvicorn src.app.main:app --reload
```

Override model artifacts, default model, or device when needed:

```bash
DEFAULT_MODEL=cnn \
CNN_CHECKPOINT_PATH=outputs/models/lightweight_cnn_logmel.pt \
SVM_MODEL_PATH=outputs/models/svm_mfcc.pkl \
PHOWHISPER_CHECKPOINT_PATH=outputs/models/phowhisper_pretrained_frozen_encoder.pt \
PHOWHISPER_CACHE_DIR=outputs/models/hf_cache \
CNN_DEVICE=auto \
PHOWHISPER_DEVICE=auto \
.venv/bin/python -m uvicorn src.app.main:app --reload
```

`CNN_DEVICE=auto` and `PHOWHISPER_DEVICE=auto` prefer CUDA, then Apple MPS, then
CPU. The PhoWhisper option defaults to the frozen-encoder checkpoint
`outputs/models/phowhisper_pretrained_frozen_encoder.pt`; set
`PHOWHISPER_CHECKPOINT_PATH=outputs/models/phowhisper_dialect.pt` only when you
want the fine-tuned checkpoint. PhoWhisper uses local Hugging Face cache files
by default; set `PHOWHISPER_LOCAL_FILES_ONLY=0` only in an environment where
network access is intended. Open `http://127.0.0.1:8000/`; model metadata is available at
`/models`, health information is available at `/health`, and multipart audio
prediction is available at `POST /predict` with form fields `file` and `model`
where `model` is one of `cnn`, `svm`, or `phowhisper`.
The browser page can also play or pause the selected local audio before sending
it for prediction; playback stays in the browser.

Run the read-only CPU inference smoke test:

```bash
.venv/bin/python -m unittest tests.test_inference -v
```

Model artifacts under `outputs/models/` must exist locally. CNN and PhoWhisper
return softmax confidence; SVM returns a softmax-normalized decision-margin score
for display only. All confidence values are uncalibrated, and the app predicts
only the three regional labels—it does not infer identity, hometown, ethnicity,
or background.
