# Vietnamese Dialect Identification Web Application

Lightweight Vietnamese speech dialect classification for three regional labels:
`Northern`, `Central`, and `Southern`.

The current project state includes metadata preparation, audio preprocessing,
MFCC baselines, a lightweight CNN, PhoWhisper-base, extended E1-E6 experiments,
a Phase 10 hybrid PhoWhisper + CNN fusion experiment scaffold, final comparison
artifacts, and a FastAPI demo app.

## Dataset

The project uses the official
[ViMD dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) and maps
its source regions into three project labels:

| ViMD region | Project label |
| --- | --- |
| `North` | `Northern` |
| `Central` | `Central` |
| `South` | `Southern` |

Current prepared metadata:

| Item | Count |
| --- | ---: |
| Official metadata rows | 18,949 |
| Provinces | 63 |
| Speakers in metadata | 12,953 |
| Downloaded local audio rows | 13,894 |
| Not selected under local data budget | 5,055 |
| Preprocessed rows | 13,894 |
| Preprocessing issues | 0 |

Current downloaded/preprocessed split:

| Split | Northern | Central | Southern | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 3,708 | 3,416 | 3,854 | 10,978 |
| Validation | 486 | 487 | 485 | 1,458 |
| Test | 486 | 487 | 485 | 1,458 |
| Total | 4,680 | 4,390 | 4,824 | 13,894 |

Speaker IDs in the prepared subset:

| Label | Speaker IDs |
| --- | ---: |
| Northern | 3,419 |
| Central | 3,022 |
| Southern | 3,101 |
| Total unique speaker IDs | 9,542 |

Preprocessing converts audio to mono 16 kHz waveform, trims silence, normalizes
volume, and center-crops or pads every sample to exactly 16 seconds
(`256,000` samples). Original duration statistics in the current subset are:
min `1.054s`, median `19.117s`, mean `19.157s`, max `32.240s`.

Key dataset artifacts:

- `data/processed/metadata_clean.csv`
- `data/processed/preprocessed_metadata.csv`
- `data/processed/audio_preprocessed_16s/`
- `data/processed/class_counts.csv`
- `data/processed/split_class_counts.csv`
- `data/processed/speaker_counts.csv`
- `outputs/reports/phase1_dataset_summary.json`
- `outputs/reports/phase2_preprocessing_summary.json`

## Environment

Create and install the local environment:

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

Apple Silicon training uses PyTorch MPS when the script is run from a normal
Terminal window. Codex sandboxed commands may not expose the Metal device even
when the machine supports MPS.

## Data Pipeline

Run metadata acquisition and preprocessing:

```bash
scripts/data.sh --overwrite
```

Useful options:

```bash
scripts/data.sh \
  --max-data-bytes 10000000000 \
  --train-per-label 4000 \
  --valid-per-label 500 \
  --test-per-label 500 \
  --seed 42 \
  --overwrite
```

Metadata-only mode:

```bash
scripts/data.sh --metadata-only
```

Run preprocessing directly:

```bash
.venv/bin/python -m src.data.preprocess_audio --overwrite
```

## Training Commands

Run the core MVP training pipeline:

```bash
scripts/train.sh --device mps --overwrite
```

Run E3 and E5 on Apple MPS:

```bash
scripts/train_e3_e5_mps.sh --overwrite
```

Run E6 original Whisper-base on Apple MPS:

```bash
scripts/train_e6_whisper_mps.sh --overwrite
```

Run Phase 10 E7 hybrid PhoWhisper + CNN fusion on Apple MPS:

```bash
scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite
```

Add `--allow-download` only if `vinai/PhoWhisper-base` is not already cached
under `outputs/models/hf_cache/`.

Smoke-test the heavier scripts on a tiny subset:

```bash
scripts/train_e3_e5_mps.sh --overwrite --smoke
scripts/train_e6_whisper_mps.sh --overwrite --smoke
scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite --smoke
```

## Current Model Results

The trained rows below are generated from the current local artifacts under
`outputs/metrics/`. E7 is listed as the next Phase 10 run until its artifacts
are generated. Model selection should use validation macro F1; test metrics are
reported for final comparison only.

| Model | Input | Status | Device | Valid Acc | Valid Macro F1 | Test Acc | Test Macro F1 | Size MB | Latency ms/sample |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | MFCC mean/std | trained | CPU | 0.5617 | 0.5612 | 0.5466 | 0.5463 | 0.002 | N/A |
| SVM | MFCC mean/std | trained | CPU | 0.5686 | 0.5688 | 0.5590 | 0.5592 | 1.783 | N/A |
| Lightweight CNN | Log-Mel | trained | MPS | 0.6091 | 0.6052 | 0.6187 | 0.6115 | 0.099 | N/A |
| E1 MobileNetV3-style | Log-Mel | trained | MPS | 0.6783 | 0.6719 | 0.6996 | 0.6941 | 0.202 | 152.69 |
| E2 EfficientNet-B0-style | Log-Mel | trained | MPS | 0.7188 | 0.7127 | 0.7003 | 0.6932 | 0.515 | 85.93 |
| E3 wav2vec2 Vietnamese | Waveform | trained | MPS | 0.6337 | 0.6063 | 0.6180 | 0.5809 | 1441.515 | 216.61 |
| E4 PhoWhisper-base frozen encoder | Whisper features | reused | MPS | 0.8477 | 0.8474 | 0.8368 | 0.8348 | 79.085 | 79.08 |
| E5 ChunkFormer-style local model | Waveform | trained | MPS | 0.6372 | 0.6286 | 0.6427 | 0.6330 | 1.634 | 27.55 |
| E6 Whisper-base original frozen encoder | Whisper features | trained | MPS | 0.8244 | 0.8229 | 0.8189 | 0.8163 | 79.084 | 86.29 |
| E7 PhoWhisper + CNN fusion | PhoWhisper features + Log-Mel | pending full run | MPS/CUDA/CPU | N/A | N/A | N/A | N/A | N/A | N/A |

Current best model by validation macro F1 is E4 PhoWhisper-base frozen encoder
with validation macro F1 `0.8474` and test macro F1 `0.8348`.

Important notes:

- E1 and E2 are PyTorch-only MobileNet/EfficientNet-inspired classifiers; they
  do not use `torchvision` pretrained ImageNet weights.
- E3 uses frozen Vietnamese wav2vec2 embeddings plus a classifier head.
- E4 reuses the Phase 6 PhoWhisper-base frozen-encoder run.
- E5 is a local ChunkFormer-style waveform model, not an official ViP-VL or
  ChunkFormer pretrained checkpoint.
- E6 uses original `openai/whisper-base` with the same frozen-encoder
  comparison setup as PhoWhisper-base.
- E7 freezes the PhoWhisper encoder and reuses the trained E2
  EfficientNetB0-style log-Mel features branch. It trains only the
  projection/fusion layers and 3-class classifier head. It does not use the
  decoder or ASR transcripts.
- PhoWhisper full fine-tuning exists as a smaller older run
  (`outputs/metrics/phowhisper_results.json`): validation macro F1 `0.6623`,
  test macro F1 `0.7113`, using 300/45/45 train/valid/test rows.

## Phase 10 Hybrid PhoWhisper + CNN Fusion

Phase 10 adds `e7_whisper_cnn_fusion`, a local-global research experiment for
Vietnamese dialect classification. The global branch uses a frozen
`vinai/PhoWhisper-base` encoder, currently the strongest baseline in this repo,
and mean-pools encoder hidden states into an utterance-level embedding. The
local branch converts the same 16 kHz / 16 s waveform into a standardized
log-Mel spectrogram and passes it through the trained E2 EfficientNetB0-style
features branch to reuse local time-frequency dialect cues. By default, E7
lightly fine-tunes the last two parameterized EfficientNetB0 feature blocks with
a smaller CNN learning rate while keeping PhoWhisper frozen. PhoWhisper global
embeddings stay at 512 dimensions. EfficientNetB0 features produce a
128-dimensional local vector, which is projected to 512 dimensions before gated
fusion. The classification head is `512 -> 256 -> 3`.

The default fusion is gated. A concat fusion ablation can be run with:

```bash
scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite --concat
```

The original frozen-CNN ablation can still be run with:

```bash
CNN_TRAINABLE_LAYERS=0 scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite
```

The E7 report focuses on validation/test macro F1 plus Central recall, Central
F1, and Central-to-Northern / Central-to-Southern confusion. Interpret E7
against E1/MobileNetV3-style, the Phase 5 CNN, E3 wav2vec2, E4 PhoWhisper, and
E6 Whisper-base. If Central recall/F1 improves over frozen PhoWhisper-base, the
local EfficientNetB0-style branch likely contributes complementary dialect cues.
If it does not, the extra fusion path may add complexity and latency without
useful gain.

E7 predicts only the dataset-defined `Northern`, `Central`, and `Southern`
labels. It does not infer hometown, identity, ethnicity, or personal background.

Comparison artifacts:

- `outputs/metrics/model_method_comparison.csv`
- `outputs/figures/model_method_comparison_metrics.png`
- `outputs/figures/model_method_comparison_tradeoffs.png`
- `outputs/metrics/deep_learning_comparison.csv`
- `outputs/figures/deep_learning_comparison.png`
- `outputs/reports/extended_deep_learning_experiments.md`

## Model Artifacts

Important checkpoints are written under `outputs/models/`:

- `logistic_regression_mfcc.pkl`
- `svm_mfcc.pkl`
- `lightweight_cnn_logmel.pt`
- `e1_mobilenetv3_logmel.pt`
- `e2_efficientnetb0_logmel.pt`
- `e3_wav2vec2_classifier.pt`
- `phowhisper_pretrained_frozen_encoder.pt`
- `e5_vipvl_chunkformer_classifier.pt`
- `e6_whisper_base_frozen_encoder.pt`
- `e7_whisper_cnn_fusion.pt`

The Hugging Face cache is under `outputs/models/hf_cache/`.

## App And Inference

Run the FastAPI app:

```bash
.venv/bin/python -m uvicorn src.app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Available API endpoints:

- `GET /models`
- `GET /health`
- `POST /predict` with multipart form fields `file` and `model`

Supported app model names are:

- `cnn`
- `svm`
- `phowhisper`

Override app artifacts and devices:

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

The app predicts only the three regional labels. It does not infer identity,
hometown, ethnicity, or personal background. Confidence values are uncalibrated:
CNN/PhoWhisper use softmax scores, and SVM uses a decision-margin display score.

## Verification

Focused tests:

```bash
.venv/bin/python -m unittest tests.test_whisper_cnn_fusion -v
.venv/bin/python -m unittest tests.test_phowhisper tests.test_extended_deep_learning -v
.venv/bin/python -m unittest tests.test_inference tests.test_app -v
```

Full test suite:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Compile check:

```bash
.venv/bin/python -m compileall -q src tests
```

## Dependencies

Core dependencies are intentionally small:

- `numpy` for feature and waveform arrays.
- `soundfile` for audio loading/writing.
- `soxr` for 16 kHz resampling.
- `duckdb` for ViMD Parquet metadata/audio extraction.
- `scikit-learn` for MFCC baselines and metrics.
- `torch` for neural training.
- `transformers` for wav2vec2, PhoWhisper, and Whisper-base checkpoints.
- `fastapi` and `uvicorn` for the local web app.
