# Implementation Report

## Latest Update: Configurable Training Parameters

### Task Summary

Added explicit, centralized training hyperparameters to `scripts/train.sh` for
the MFCC baseline seed, lightweight CNN, frozen PhoWhisper classifier, and full
PhoWhisper fine-tuning. Every value can be overridden through an environment
variable, and each run prints the resolved values before training starts.

### Files Changed

| File | Purpose |
| --- | --- |
| `scripts/train.sh` | Centralizes and forwards seed, batch size, epochs, patience, learning rate, and weight decay. |
| `reports/implementation_report.md` | Records the implementation and verification. |

### Scope And Decisions

- Preserved all previous training defaults and the existing phase order.
- Used environment variables instead of adding a large set of shell CLI flags.
- Passed the shared seed explicitly to all three training modules.
- Kept separate PhoWhisper parameters for frozen-encoder and full-fine-tune
  experiments because their learning rates and epoch budgets differ.

### Commands Run

```bash
bash -n scripts/train.sh
scripts/train.sh --help
env PYTHON_BIN=/bin/echo scripts/train.sh --device mps --overwrite
env PYTHON_BIN=/bin/echo CNN_LEARNING_RATE=5e-4 CNN_PATIENCE=10 scripts/train.sh --device mps --overwrite
git diff --check -- scripts/train.sh reports/implementation_report.md
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Bash syntax | Passed. |
| Help output | Lists every training parameter and default. |
| Default dry run | All parameters were forwarded to the intended training commands. |
| Override dry run | CNN learning rate `5e-4` and patience `10` were resolved and forwarded. |

### Known Limitations

- Full model training was not repeated because this change only forwards
  existing CLI options and a complete run is compute-intensive.
- Invalid numeric values are rejected by the existing Python training CLIs when
  their phase starts rather than by the shell script.

### Reviewer Priorities

1. Confirm experiment-specific defaults before starting a long training run.
2. Keep the printed parameter block with saved logs for reproducibility.

---

## Latest Update: Browser Audio Playback

### Task Summary

Added Play/Pause controls for the locally selected upload file in the Phase 8
frontend. Playback uses a browser object URL and does not send an extra request.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/app/static/index.html` | Added audio player state, Play/Pause button, object URL creation, and cleanup. |
| `README.md` | Documented local pre-prediction playback. |
| `reports/implementation_report.md` | Recorded implementation and verification. |

### Commands Run

```bash
sed -n '1,240p' PLAN.md
python -c "... parse src/app/static/index.html with html.parser ..."
git diff --check
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| HTML parsing | Passed. |
| Playback controls | Play button, audio element, object URL, Play/Pause handlers, and URL cleanup are present. |
| Backend impact | None; `/predict` and model inference are unchanged. |

### Known Limitations

- Playback depends on the browser's native codec support for the selected file.
- Browser playback was inspected statically because the managed sandbox cannot
  open an interactive browser.

---

## Latest Update: Phase 8 CNN Inference And FastAPI App

### Task Summary

Implemented a PyTorch inference pipeline and FastAPI upload app for the existing
Phase 5 lightweight CNN without changing preprocessing, feature extraction,
training logic, or model architecture.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/inference/predict.py` | Strict checkpoint validation, device selection, shared preprocessing/log-Mel extraction, and softmax prediction. |
| `src/inference/__init__.py` | Inference package marker. |
| `src/app/main.py` | FastAPI lifespan model loading plus `/`, `/health`, and `/predict`. |
| `src/app/static/index.html` | Minimal upload UI and per-class probability bars. |
| `src/app/__init__.py` | App package marker. |
| `tests/test_inference.py` | Read-only CPU smoke test against one existing preprocessed sample. |
| `requirements.txt`, `pyproject.toml` | Added FastAPI runtime dependencies. |
| `README.md` | Added exact app and smoke-test commands. |

### Implementation Scope And Decisions

- `predict()` calls the existing `preprocess_file()`, reloads its PCM16 output,
  and calls the existing `log_mel_spectrogram()`; feature math is not duplicated.
- Checkpoint metadata must match the code constants for model name, feature name,
  16 kHz sample rate, 256,000 samples, 64 Mel bins, and label order.
- Label order is imported from CNN training as Northern, Central, Southern.
- The model is reconstructed with `LightweightCNN`, loaded from
  `model_state_dict`, moved to the selected device, and put in eval mode.
- FastAPI lifespan loads the checkpoint once. Automatic device priority is CUDA,
  then MPS, then CPU.
- Uploaded files and inference preprocessing outputs use OS temporary storage and
  are removed after each request.

### Dependencies

- `fastapi` provides the HTTP routes and JSON responses.
- `uvicorn[standard]` provides the local ASGI server.
- `python-multipart` is required by FastAPI for multipart audio uploads.
- Existing `soundfile`, `soxr`, NumPy, and PyTorch dependencies remain the audio
  and model runtime; no additional audio library was added.

### Commands Run

```bash
sed -n '1,260p' PLAN.md
nl -ba src/utils/audio.py
nl -ba src/features/logmel.py
nl -ba src/features/mfcc.py
nl -ba src/training/train_cnn.py
nl -ba src/models/cnn.py
.venv/bin/python -c "... inspect CNN checkpoint metadata ..."
.venv/bin/python -m unittest tests.test_inference -v
.venv/bin/python -m compileall -q src tests
env UV_CACHE_DIR=/tmp/vimd-uv-cache uv pip install --python .venv/bin/python -r requirements.txt
env CNN_DEVICE=cpu .venv/bin/python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8765
.venv/bin/python -c "... run FastAPI lifespan, health, index, and predict_upload ..."
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| CNN checkpoint | Found at `outputs/models/lightweight_cnn_logmel.pt`; state-dict checkpoint, epoch 13. |
| CPU inference smoke test | Passed: three ordered classes, probability sum approximately 1, valid predicted label. |
| Full unit tests before final documentation | Passed: 26 tests. |
| Python compilation | Passed. |
| FastAPI dependencies | Installed and importable in `.venv`. |
| App startup | Passed through lifespan/model loading. Localhost bind was denied by the managed sandbox. |
| Direct endpoint smoke check | Passed: health reported CPU, index resolved `index.html`, upload returned all three classes. |

### Known Limitations

- The managed sandbox does not permit binding a localhost port, so browser access
  was not exercised here; run uvicorn from a normal terminal.
- The local CNN checkpoint is ignored by Git and is required at app startup.
- Softmax confidence is uncalibrated.
- The CNN remains the existing Phase 5 model; no retraining or architecture
  change was performed.

### Reviewer Priorities

1. Start uvicorn locally and verify one browser upload.
2. Keep checkpoint metadata validation intact to prevent silent preprocessing or
   label-order mismatch.
3. Preserve the regional-classification disclaimer in future UI work.

---

## Latest Update: Resumable ViMD Shard Downloads

### Task Summary

Fixed truncated large-shard downloads such as `train-00101-of-00103.parquet`,
where the connection ended at 317,249,630 of 381,107,172 expected bytes.

### Root Cause

The downloader treated an early EOF as a final size mismatch, deleted the
partial file, and raised immediately. Large 200–400 MB Hugging Face shards are
more likely to encounter transient connection interruption.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/data/prepare_metadata.py` | Added five-attempt retry, HTTP Range resume, partial-file preservation, and byte-budget accounting for resumed files. |
| `tests/test_prepare_metadata.py` | Added a truncated-response test that resumes from byte 6 and verifies exact output content. |
| `README.md` | Documented retry/resume behavior. |
| `reports/implementation_report.md` | Recorded diagnosis and verification. |

### Commands Run

```bash
sed -n '1,220p' PLAN.md
.venv/bin/python -m unittest tests.test_prepare_metadata -v
.venv/bin/python -m compileall -q src/data/prepare_metadata.py tests/test_prepare_metadata.py
curl -L --range 317249630-317250653 --output /dev/null --dump-header - https://huggingface.co/datasets/nguyendv02/ViMD_Dataset/resolve/main/data/train-00101-of-00103.parquet
.venv/bin/python -c "... urllib Range request for bytes 317249630-317250653 ..."
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Focused Phase 1 tests | Passed: 7 tests. |
| Python compilation | Passed. |
| Simulated truncated response | Passed: second request sent `Range: bytes=6-` and reconstructed the exact 10-byte file. |
| Real Hugging Face Range request | Passed: HTTP 206, `Content-Range: bytes 317249630-317250653/381107172`, 1,024 bytes received. |
| Pipeline urllib Range request | Passed through Hugging Face redirect with HTTP 206 and correct content range. |

### Known Limitations

- The full 381 MB shard was not redownloaded during verification; only a 1 KB
  range was requested from the real endpoint.
- After five failed attempts the script exits but preserves `.part` under
  `data/.phase1_tmp/`, allowing the same command to resume later.

### Reviewer Priorities

1. Rerun the same data command. Previously extracted WAV files are reused.
2. Do not manually delete `data/.phase1_tmp/*.part` unless restarting a shard
   from byte zero is intentional.

---

## Latest Update: Large Dataset Shard Selection Fix

### Task Summary

Fixed Phase 1 failure for large balanced targets such as 4,000 train and 500
validation/test samples per label. Shard selection is no longer limited to
three Parquet files per split.

### Root Cause

`choose_source_shards()` exhaustively searched only combinations of one to
three shards. The 10 GB configuration requires many shards, so it failed before
downloading audio with `Cannot cover requested train targets with three shards`.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/data/prepare_metadata.py` | Replaced the three-shard search with deterministic greedy coverage that scales to all available shards. |
| `tests/test_prepare_metadata.py` | Added coverage requiring four shards per split. |
| `README.md` | Documented scalable shard selection and byte-budget enforcement. |
| `reports/implementation_report.md` | Recorded diagnosis, fix, and verification. |

### Commands Run

```bash
sed -n '1,240p' PLAN.md
sed -n '220,560p' src/data/prepare_metadata.py
.venv/bin/python -m unittest tests.test_prepare_metadata -v
.venv/bin/python -m compileall -q src/data/prepare_metadata.py tests/test_prepare_metadata.py
.venv/bin/python -c "... simulate 4000/500/500 targets from metadata_clean.csv ..."
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Phase 1 focused tests | Passed: 6 tests. |
| Python compilation | Passed. |
| More-than-three-shard regression | Passed: selected four shards for every split. |
| Real metadata simulation | Passed: 107 shards selected for 4,000/500/500 targets—84 train, 11 valid, and 12 test. |

### Known Limitations

- The full 10 GB download was not repeated during verification because it is a
  long external transfer. The selection stage that raised the reported error
  was reproduced and now completes locally.
- Final downloaded counts still depend on the 10 GB cap, remote shard sizes,
  available audio, and normalization output sizes.

### Reviewer Priorities

1. Rerun the same `scripts/data.sh` command; existing downloaded audio is reused
   when `--overwrite` is supplied.
2. Keep at least 18–20 GB free for source and preprocessed copies together.

---

## Latest Update: Automated Data And Training Pipelines

### Task Summary

Implemented executable shell scripts that run the existing data phases and
training/evaluation phases in their required order.

### Files Changed

| File | Purpose |
| --- | --- |
| `scripts/data.sh` | Runs metadata/audio acquisition, preprocessing, and minimal EDA with configurable data budget, split sizes, and seed. |
| `scripts/train.sh` | Runs MFCC baselines, CNN, both PhoWhisper modes, and final evaluation. |
| `README.md` | Documents pipeline commands, overwrite behavior, device selection, and environment overrides. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Implementation Scope And Decisions

- Scripts resolve the repository root, so they can be called from any working
  directory.
- `.venv/bin/python` is the default; `PYTHON_BIN` can override it.
- Existing outputs are protected unless the caller explicitly passes
  `--overwrite`.
- `scripts/data.sh` exposes `--max-data-bytes`, per-label train/validation/test
  targets, `--seed`, and `--metadata-only` from the underlying metadata CLI.
- `scripts/train.sh` accepts `auto`, `mps`, `cuda`, or `cpu` and forwards the
  selected device to CNN and both PhoWhisper experiments.
- Frozen PhoWhisper uses the verified experiment settings: learning rate
  `1e-3`, 20 maximum epochs, and patience 5.
- No dependency or additional orchestration framework was added.

### Commands Run

```bash
sed -n '1,240p' PLAN.md
sed -n '1,260p' scripts/data.sh
sed -n '1,320p' scripts/train.sh
chmod +x scripts/data.sh scripts/train.sh
bash -n scripts/data.sh scripts/train.sh
scripts/data.sh --help
scripts/train.sh --help
env PYTHON_BIN=/bin/echo scripts/data.sh --max-data-bytes 1000000000 --train-per-label 120 --valid-per-label 20 --test-per-label 20 --seed 7 --overwrite
env PYTHON_BIN=/bin/echo scripts/data.sh --metadata-only --seed 9 --overwrite
env PYTHON_BIN=/usr/bin/true scripts/train.sh --device mps --overwrite
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Bash syntax | Passed for both scripts. |
| Help commands | Passed with exit code 0. |
| Data pipeline dry run | Passed in Phase 1 → Phase 2/3 order with all download arguments forwarded correctly. |
| Metadata-only dry run | Passed and skipped audio preprocessing. |
| Training pipeline dry run | Passed in Phase 4 → Phase 5 → Phase 6a → Phase 6b → Phase 7 order. |
| Executable permissions | Set on both scripts. |

### Known Limitations

- A full data run was not repeated because it requires remote ViMD access and
  would regenerate the existing dataset.
- A full training run was not repeated because CNN and both PhoWhisper runs are
  compute-intensive; their individual pipelines were already verified earlier.
- Explicit `--device mps` may require running outside a managed sandbox even
  when MPS works in the user's normal terminal.

### Reviewer Priorities

1. Run `scripts/data.sh --overwrite` only when dataset artifacts should be
   regenerated.
2. On Apple Silicon, use `scripts/train.sh --device mps --overwrite` from a
   normal terminal to force Apple GPU execution.

---

## Latest Update: Frozen PhoWhisper Baseline And Neural Comparison

### Task Summary

Added a pretrained PhoWhisper comparison run that keeps the encoder frozen,
trains only the dialect classification stack, and reports it separately from
the existing full fine-tuning result and custom CNN.

### Files Changed

| File or output | Purpose |
| --- | --- |
| `src/training/train_phowhisper.py` | Added `frozen_encoder` and `full_fine_tune` modes, mode-specific output paths, trainable-parameter reporting, and frozen-encoder handling. |
| `src/evaluation/final_evaluation.py` | Added both PhoWhisper variants and a focused three-model neural comparison report. |
| `tests/test_phowhisper.py`, `tests/test_final_evaluation.py` | Added frozen-parameter, output-path, aggregation, and neural-report checks. |
| `README.md`, `READING_GUIDE.md` | Documented the two PhoWhisper experiments, commands, semantics, and results. |
| `outputs/metrics/phowhisper_pretrained_*` | Frozen-encoder metrics, training log, confusion matrices, and predictions. |
| `outputs/reports/phase6_phowhisper_pretrained_report.md` | Frozen-encoder experiment report. |
| `outputs/metrics/final_comparison.csv`, `outputs/reports/neural_model_comparison.md`, `outputs/reports/error_analysis.md` | Updated final and focused comparisons. |

### Implementation Scope And Decisions

- The pretrained baseline freezes every PhoWhisper encoder parameter.
- Only the newly initialized projector and three-class classifier are trained:
  132,099 of 20,722,691 parameters in `WhisperForAudioClassification`.
- A completely untouched ASR model cannot produce dialect classes because it
  has no Northern/Central/Southern output head. The report therefore calls this
  a frozen-encoder baseline, not zero-shot inference.
- Full fine-tuning keeps its existing artifacts; frozen mode writes separate
  `phowhisper_pretrained_*` artifacts.
- Both runs use the same 300/45/45 metadata splits and seed 42.
- No dependency was added.

### Commands Run

```bash
sed -n '1,240p' PLAN.md
git status --short
head -n 8 data/processed/preprocessed_metadata.csv
.venv/bin/python -m unittest tests.test_phowhisper tests.test_final_evaluation -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.training.train_phowhisper --training-mode frozen_encoder --learning-rate 1e-3 --max-epochs 20 --patience 5 --overwrite --device auto
.venv/bin/python -c "import platform, torch; ..."
system_profiler SPDisplaysDataType
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.training.train_phowhisper --training-mode frozen_encoder --learning-rate 1e-3 --max-epochs 20 --patience 5 --overwrite --device mps
.venv/bin/python -m src.evaluation.final_evaluation --overwrite
.venv/bin/python -c "import torch; ... compare frozen checkpoint with pretrained encoder ..."
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Full unit tests | Passed: 23 tests. |
| Python compilation | Passed for `src` and `tests`. |
| Apple GPU probe outside sandbox | Passed: Apple M5, MPS built and available, tensor allocated on `mps:0`. |
| Frozen PhoWhisper training | Passed on MPS: early stopped at epoch 13, best epoch 8. |
| Encoder immutability | Passed: three sampled encoder tensors were bit-identical to pretrained weights (`changed_elements=0`). |
| Final evaluation | Passed: SVM remains best by validation macro F1. |

Neural model comparison:

| Model | Valid Macro F1 | Test Macro F1 | Device |
| --- | ---: | ---: | --- |
| Custom CNN | 0.4339 | 0.6668 | CPU |
| PhoWhisper pretrained, frozen encoder | 0.6720 | 0.7972 | MPS |
| PhoWhisper fine-tuned | 0.6623 | 0.7113 | MPS |

### Known Limitations

- Validation and test contain only 45 files each, so the metric differences are
  noisy; the frozen variant's higher test score is not enough to claim it is
  generally better.
- The frozen baseline still trains a small supervised classification stack; raw
  PhoWhisper ASR is not a zero-shot dialect classifier.
- Managed sandbox processes did not expose MPS and initially ran on CPU. The
  final saved frozen artifacts were regenerated outside the sandbox with
  explicit `--device mps` after an MPS tensor probe passed.
- Checkpoints and Hugging Face cache remain ignored under `outputs/models/`.

### Reviewer Priorities

1. Use `outputs/reports/neural_model_comparison.md` for the requested three-row
   comparison.
2. Keep model selection based on validation macro F1, not the highest test score.
3. Preserve the distinction between frozen pretrained encoder and full
   fine-tuning in future inference/app work.

---

## Latest Update: Phase 6 PhoWhisper And Phase 7 Final Evaluation

### Task Summary

Implemented full PhoWhisper-base fine-tuning for three-region dialect
classification and generated final model comparison plus error analysis.

### Files Changed

| File or output | Purpose |
| --- | --- |
| `src/training/train_phowhisper.py` | Full PhoWhisper-base fine-tuning script with device selection, early stopping, metrics, predictions, latency, and report outputs. |
| `src/evaluation/final_evaluation.py` | Final comparison and sample-level error analysis script. |
| `tests/test_phowhisper.py`, `tests/test_final_evaluation.py` | Unit tests for device/split behavior and final evaluation output contracts. |
| `requirements.txt`, `pyproject.toml` | Added `transformers>=4.41,<5`. |
| `README.md`, `READING_GUIDE.md` | Added Phase 6/7 commands, results, and reading path. |
| `outputs/metrics/phowhisper_*`, `outputs/reports/phase6_phowhisper_report.md` | Phase 6 metrics, training log, confusion matrices, predictions, and report. |
| `outputs/metrics/final_comparison.csv`, `outputs/metrics/final_sample_errors.csv`, `outputs/reports/error_analysis.md` | Phase 7 final comparison and error analysis artifacts. |

### Implementation Scope

Included:

- Full fine-tuning using `WhisperForAudioClassification` from
  `vinai/PhoWhisper-base`.
- All classification model parameters trainable; no frozen encoder path.
- `auto -> mps -> cuda -> cpu` device resolution with clear explicit-device
  errors.
- Best checkpoint by validation macro F1.
- Published model size estimates and local checkpoint size.
- MPS/CUDA synchronized latency estimate.
- Final comparison across Logistic Regression, SVM, CNN, and PhoWhisper.
- Sample-level error analysis for the best validation model.

Not included:

- No province-level classification, speaker/hometown prediction, ASR generation,
  PhoWhisper alternatives, ONNX, inference API, or web app.
- Model checkpoint and Hugging Face cache remain under ignored `outputs/models/`.

### Research Notes

- `vinai/PhoWhisper-base` is the PhoWhisper base model with an estimated 74M
  parameters.
- Hugging Face lists the model repository around 294 MB and PyTorch weights
  around 290 MB.
- The config is Whisper-base-like: 80 Mel bins, `d_model=512`, 6 encoder layers,
  and 6 decoder layers.

Sources checked:

- https://github.com/VinAIResearch/PhoWhisper
- https://huggingface.co/vinai/PhoWhisper-base/tree/main
- https://huggingface.co/vinai/PhoWhisper-base/blob/main/config.json
- https://arxiv.org/abs/2406.02555

### Commands Run

```bash
sed -n '1,260p' PLAN.md
git status --short
rg --files src tests outputs/metrics outputs/reports data/processed | sort
cat requirements.txt
cat pyproject.toml
sed -n '1,760p' src/training/train_cnn.py
sed -n '1,260p' src/training/train_baseline.py
head -n 5 data/processed/preprocessed_metadata.csv
cat outputs/metrics/baseline_results.json
cat outputs/metrics/cnn_results.json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
env UV_CACHE_DIR=/tmp/vimd-uv-cache uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import transformers; print(transformers.__version__); from transformers import AutoFeatureExtractor, WhisperForAudioClassification; print('ok')"
.venv/bin/python -c "import torch; print(torch.__version__); print('mps', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); print('cuda', torch.cuda.is_available())"
.venv/bin/python -m src.training.train_phowhisper --overwrite --device auto
.venv/bin/python -m src.evaluation.final_evaluation --overwrite
cat outputs/metrics/phowhisper_results.json
cat outputs/metrics/final_comparison.csv
head -n 8 outputs/metrics/phowhisper_training_log.csv
sed -n '1,220p' outputs/reports/phase6_phowhisper_report.md
sed -n '1,220p' outputs/reports/error_analysis.md
head -n 8 outputs/metrics/final_sample_errors.csv
ls -lh outputs/models/phowhisper_dialect.pt outputs/models/hf_cache
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Dependency install | Passed after network approval for `transformers`. |
| Unit tests | Passed: 21 tests. |
| Python compilation | Passed for `src` and `tests`. |
| PhoWhisper training | Passed: used `mps`, early stopped at epoch 6, best epoch 3. |
| Phase 7 final evaluation | Passed: selected SVM by validation macro F1. |

PhoWhisper metrics:

| Split | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Train | 0.9933 | 0.9933 |
| Validation | 0.6667 | 0.6623 |
| Test | 0.7111 | 0.7113 |

Final comparison:

| Model | Valid Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.5981 | 0.6292 |
| SVM | 0.6918 | 0.6264 |
| Lightweight CNN | 0.4339 | 0.6668 |
| PhoWhisper-base | 0.6623 | 0.7113 |

### Known Limitations

- PhoWhisper-base test macro F1 is highest, but final best model is selected by
  validation macro F1, so SVM remains best.
- Validation/test sets have only 45 files each; metrics are noisy.
- PhoWhisper overfits the 300-file train split quickly.
- Local checkpoint is `WhisperForAudioClassification` state for classification,
  not the full original ASR checkpoint with decoder generation usage.
- SVM confidence in final errors is a decision margin, not calibrated
  probability.

### Reviewer Priorities

1. Decide whether model selection should stay validation macro F1 or prefer
   PhoWhisper due to higher test macro F1.
2. Inspect `outputs/metrics/final_sample_errors.csv` before Phase 8 inference.
3. Keep Phase 8 scoped to regional dialect prediction only.

---

## Latest Update: Phase 5 Lightweight CNN

### Task Summary

Implemented Phase 5 with a PyTorch lightweight CNN trained from standardized
log-Mel spectrograms extracted from the Phase 2 fixed-length audio.

### Files Changed

| File or output | Purpose |
| --- | --- |
| `src/features/logmel.py` | NumPy log-Mel spectrogram extraction with per-sample standardization. |
| `src/models/cnn.py` | Lightweight 3-block CNN for 3-class dialect classification. |
| `src/training/train_cnn.py` | Phase 5 training script with auto/mps/cuda/cpu device selection, early stopping, checkpointing, metrics, and reports. |
| `tests/test_logmel.py`, `tests/test_cnn.py` | Unit tests for log-Mel shape, CNN forward pass, and device resolver behavior. |
| `requirements.txt`, `pyproject.toml` | Added `torch>=2.7,<3` for CNN training. |
| `README.md`, `READING_GUIDE.md` | Added Phase 5 commands, outputs, device notes, and reading path. |
| `outputs/metrics/cnn_results.json` | Phase 5 metrics JSON. |
| `outputs/metrics/cnn_training_log.csv` | Per-epoch train/validation log. |
| `outputs/metrics/cnn_valid_confusion_matrix.csv`, `outputs/metrics/cnn_test_confusion_matrix.csv` | CNN confusion matrices. |
| `outputs/reports/phase5_cnn_report.md` | Short Phase 5 report. |

### Implementation Scope

Included:

- Log-Mel input shaped `[batch, 1, 64, 1599]` from 16 kHz / 16 s audio.
- Lightweight CNN trained from scratch using AdamW and cross entropy.
- Device resolver with exact `auto -> mps -> cuda -> cpu` priority.
- Explicit `--device mps` or `--device cuda` raises a clear error when unavailable.
- Best checkpoint saved by validation macro F1.
- Validation/test confusion matrices and training log.

Not included:

- No pretrained model, PhoWhisper, inference API, web app, ONNX export, or audio
  augmentation.
- The checkpoint under `outputs/models/` remains intentionally ignored by Git.

### Design Decisions

- Added PyTorch because Phase 5 requires a real CNN; NumPy/scikit-learn is not a
  practical fit for training a CNN.
- Kept log-Mel extraction in NumPy and reused existing spectrogram/Mel helpers
  to avoid adding `librosa` or `torchaudio`.
- Used a small 3-block CNN to keep the model appropriate for the lightweight
  course project scope.
- Cloned the best model state before continuing training so the stored best
  checkpoint cannot be mutated by later epochs.

### Commands Run

```bash
sed -n '1,220p' PLAN.md
git status --short
sed -n '1,220p' pyproject.toml
sed -n '1,260p' src/features/mfcc.py
sed -n '1,420p' src/training/train_baseline.py
uv pip install --python .venv/bin/python -r requirements.txt
env UV_CACHE_DIR=/tmp/vimd-uv-cache uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
.venv/bin/python -c "import torch; print(torch.__version__); print('mps', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); print('cuda', torch.cuda.is_available())"
.venv/bin/python -m src.training.train_cnn --overwrite --device auto
cat outputs/metrics/cnn_results.json
sed -n '1,220p' outputs/reports/phase5_cnn_report.md
head -n 8 outputs/metrics/cnn_training_log.csv
cat outputs/metrics/cnn_valid_confusion_matrix.csv
cat outputs/metrics/cnn_test_confusion_matrix.csv
ls -lh outputs/models/lightweight_cnn_logmel.pt
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Dependency install | Passed after setting `UV_CACHE_DIR=/tmp/vimd-uv-cache` and allowing network for PyTorch. |
| Unit tests | Passed: 15 tests. |
| Python compilation | Passed for `src` and `tests`. |
| Phase 5 training | Passed: early stopped at epoch 21; best epoch 13. |
| Checkpoint | Created locally: `outputs/models/lightweight_cnn_logmel.pt` (~102 KB). |
| Metrics/report artifacts | Created under `outputs/metrics/` and `outputs/reports/`. |

PyTorch device probe:

```text
torch 2.12.0
mps False
cuda False
```

CNN metrics:

| Split | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Train | 0.7900 | 0.7846 |
| Validation | 0.4222 | 0.4339 |
| Test | 0.6667 | 0.6668 |

Current best Phase 4 validation macro F1 remains higher at 0.6918.

### Known Limitations

- The local PyTorch build/environment reported both MPS and CUDA unavailable, so
  the verified full run used CPU despite the requested Apple Silicon support.
- Validation/test sets contain only 45 samples each, so metrics are noisy.
- CNN validation macro F1 did not beat the Phase 4 SVM baseline in this run.
- Feature extraction is computed inside the training run and not cached.

### Reviewer Priorities

1. Confirm whether CPU-verified Phase 5 is acceptable or rerun on a PyTorch build
   where `torch.backends.mps.is_available()` returns true.
2. Review whether the CNN should become the deployment candidate despite lower
   validation macro F1 than SVM.
3. Keep future inference code routed through the same Phase 2 audio preprocessing
   and Phase 5 log-Mel feature extraction.

---

## Latest Update: Functional Commit Split

### Task Summary

Split the current repository work into small Git commits grouped by implemented
functionality.

### Files Changed

| File | Purpose |
| --- | --- |
| `reports/implementation_report.md` | Recorded the commit split and verification details. |

### Implementation Scope

Included:

- Reviewed the current phase plan and Git worktree before committing.
- Fixed `.gitignore` so local data/audio artifacts remain ignored without
  accidentally ignoring `src/data/`.
- Created focused commits for setup/scope, Phase 1 metadata, Phase 2 audio
  preprocessing, Phase 4 MFCC baseline, and project documentation.

Not included:

- No code behavior changes beyond the `.gitignore` correction needed to commit
  `src/data/`.
- No pushed remote commits.
- No WAV audio files, local virtual environment, Python cache files, or saved
  model pickles were committed.

### Commit Split

| Commit | Purpose |
| --- | --- |
| `6b4f2a9 chore: set up project scope and dependencies` | Scope docs, project dependencies, ignore rules, and minimal entrypoint. |
| `e9a4c6a feat(data): prepare ViMD metadata subset` | Phase 1 metadata script, label mapping, metadata tests, CSV metadata summaries, and Phase 1 report. |
| `b180213 feat(audio): add fixed-length preprocessing pipeline` | Shared audio utilities, preprocessing script, preprocessing tests, preprocessed metadata, and minimal EDA outputs. |
| `38f21ac feat(baseline): train MFCC classical models` | MFCC feature extraction, baseline training, MFCC tests, metrics, confusion matrices, and Phase 4 report. |
| `b1d076a docs: document workflow and reading guide` | README updates, beginner reading guide, and implementation report context. |

### Commands Run

```bash
sed -n '1,240p' PLAN.md
git status --short
git log --oneline --decorate -5
git diff --stat
git status --untracked-files=all --short
git ls-files
sed -n '1,220p' .gitignore
git diff -- .gitignore README.md
git status --ignored --short
rg --files data
rg --files outputs
find src -maxdepth 3 -type f -print
du -sh data/processed outputs configs src tests reports README.md READING_GUIDE.md OVERVIEW.md PLAN.md AGENTS.md pyproject.toml requirements.txt main.py
du -sh data/processed/*.csv outputs/metrics/* outputs/reports/*
.venv/bin/python -m unittest discover -s tests -v
git add ...
git diff --cached --name-status
git diff --cached --stat
git commit -m "chore: set up project scope and dependencies"
git commit -m "feat(data): prepare ViMD metadata subset"
git commit -m "feat(audio): add fixed-length preprocessing pipeline"
git commit -m "feat(baseline): train MFCC classical models"
git commit -m "docs: document workflow and reading guide"
git status --short
git status --ignored --short
git log --oneline --decorate -8
git diff --stat HEAD
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Unit tests before commit split | Passed: 11 tests. |
| Tracked worktree after split | Clean. |
| Ignored local artifacts | `.venv/`, WAV audio folders, `outputs/models/`, and `__pycache__/` remain ignored. |

Unit test output summary:

```text
Ran 11 tests in 0.008s
OK
```

### Known Limitations

- The commits are local only; nothing was pushed to `origin/main`.
- Large audio WAV files and saved model pickles remain intentionally untracked.
- Dataset metadata CSVs are committed because they are small enough compared with
  audio artifacts and are useful for reproducing the current phase outputs.

### Reviewer Priorities

1. Review the commit boundaries before pushing.
2. Confirm whether generated CSV/metric artifacts should stay versioned for the
   course submission.
3. Push the branch only after accepting the local commit history.

---

## Latest Update: Beginner Reading Guide

### Task Summary

Created a beginner-friendly `READING_GUIDE.md` in Vietnamese to explain the
current project scope, code layout, data workflow, key files, run commands, and
current limitations.

### Files Changed

| File | Purpose |
| --- | --- |
| `READING_GUIDE.md` | New project overview and reading path for newcomers. |
| `reports/implementation_report.md` | Added this latest implementation update. |

### Implementation Scope

Included:

- Project overview for the three supported classes: `Northern`, `Central`, and
  `Southern`.
- Current phase status based on `PLAN.md`.
- End-to-end workflow from ViMD metadata to Phase 4 MFCC baseline metrics.
- Beginner reading order for docs, data files, audio preprocessing, MFCC,
  baseline training, and tests.
- Current dataset subset counts and baseline results from generated outputs.
- Common commands and repo-specific cautions.

Not included:

- No code behavior changes.
- No new tests, dependencies, models, inference pipeline, or web app.

### Design Decisions

- Wrote the guide in Vietnamese because the request targets Vietnamese-speaking
  beginners.
- Kept the guide aligned with the implemented repo state, not the future target
  app name.
- Explicitly noted that CNN, PhoWhisper, inference, and web UI are not present
  yet to avoid confusing new readers.
- Used actual generated CSV/JSON outputs to describe schema, counts, and
  metrics instead of guessing.

### Commands Run

```bash
sed -n '1,240p' PLAN.md
rg --files
ls
git status --short
sed -n '1,260p' README.md
sed -n '1,260p' OVERVIEW.md
sed -n '1,240p' READING_GUIDE.md
sed -n '1,220p' pyproject.toml
sed -n '1,220p' requirements.txt
sed -n '1,260p' src/data/prepare_metadata.py
sed -n '260,620p' src/data/prepare_metadata.py
sed -n '1,280p' src/data/preprocess_audio.py
sed -n '280,620p' src/data/preprocess_audio.py
sed -n '1,280p' src/utils/audio.py
sed -n '1,280p' src/features/mfcc.py
sed -n '1,340p' src/training/train_baseline.py
sed -n '340,720p' src/training/train_baseline.py
sed -n '1,220p' main.py
sed -n '1,260p' tests/test_prepare_metadata.py
sed -n '1,260p' tests/test_audio_preprocessing.py
sed -n '1,260p' tests/test_mfcc.py
sed -n '1,220p' reports/implementation_report.md
head -n 6 data/processed/metadata_clean.csv
head -n 6 data/processed/preprocessed_metadata.csv
cat data/processed/class_counts.csv
cat data/processed/split_class_counts.csv
cat configs/label_mapping.csv
cat outputs/metrics/baseline_results.json
rg --files docs
sed -n '1,220p' outputs/reports/phase1_dataset_summary.json
sed -n '1,220p' outputs/reports/phase2_preprocessing_summary.json
sed -n '1,220p' outputs/reports/data_eda.md
sed -n '1,220p' outputs/reports/phase4_baseline_report.md
sed -n '1,320p' READING_GUIDE.md
.venv/bin/python -m unittest discover -s tests -v
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| `READING_GUIDE.md` readback | Passed; content is present and aligned with inspected files. |
| Unit tests | Passed: 11 tests. |

Unit test output summary:

```text
Ran 11 tests in 0.009s
OK
```

### Known Limitations

- `rg --files docs` returned no files; the `docs/` directory currently has no
  tracked guide files to reference.
- Metadata preparation commands were documented but not rerun, because this task
  only requested a project overview and the command may require network access.
- Guide reflects the current repo state: it does not document CNN or web app
  implementation details because those modules do not exist yet.

### Reviewer Priorities

1. Confirm the Vietnamese explanation is clear enough for a beginner.
2. Check whether the phase-status wording matches the intended class report
   narrative.
3. Keep future updates to the guide synchronized with Phase 5+ code when those
   modules are added.

---

## Previous Implementation Context

## Task Summary

Completed Phase 2 audio preprocessing, a minimal Phase 3 validation report, and
Phase 4 MFCC traditional baselines.

Phase 2 now preprocesses the 390 selected Phase 1 WAV files into shared
fixed-length audio for training and future inference:

| Split | Northern | Central | Southern | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 100 | 100 | 100 | 300 |
| Validation | 15 | 15 | 15 | 45 |
| Test | 15 | 15 | 15 | 45 |

## Files Changed

| File or output | Purpose |
| --- | --- |
| `src/utils/audio.py` | Shared load, mono/resample, silence trim, RMS normalize, and 16-second pad/crop pipeline. |
| `src/data/preprocess_audio.py` | Phase 2 script for selected metadata rows, fixed WAV outputs, issue log, JSON summary, and minimal EDA. |
| `src/features/mfcc.py` | NumPy MFCC extraction with mean/std aggregation. |
| `src/training/train_baseline.py` | Logistic Regression and SVM MFCC baseline training, metrics, models, and reports. |
| `tests/test_audio_preprocessing.py`, `tests/test_mfcc.py` | Unit tests for fixed shape preprocessing and MFCC feature stability. |
| `requirements.txt`, `pyproject.toml` | Added `scikit-learn` for Phase 4 models and metrics. |
| `README.md` | Added uv environment, Phase 2, minimal Phase 3, and Phase 4 commands/results. |
| `data/processed/*`, `outputs/*` | Generated Phase 2/3/4 artifacts. |

## Implementation Scope

### Included

- Deterministic shared audio preprocessing for training/inference.
- Fixed target: 16 kHz, mono, 16.00 seconds, 256,000 samples.
- Per-file preprocessing metadata and issue logging.
- Minimal EDA report validating split balance, duration, shape, and issue count.
- MFCC mean/std features with Logistic Regression and SVM baselines.
- Saved model pickles, metrics JSON, confusion matrix CSVs, and short baseline report.

### Not Included

- Full EDA figures, CNN training, PhoWhisper, inference API, or web app.
- Feature caching; MFCC features are computed inside the baseline training run.

## Design Decisions

- Used 16 seconds because it matches the selected audio median closely and avoids
  excessive truncation for the current MVP.
- Stored preprocessed audio under `data/processed/audio_preprocessed_16s/` instead
  of overwriting Phase 1 `audio_16k/`.
- Kept MFCC extraction in NumPy to avoid adding librosa for this phase.
- Added `scikit-learn` because Phase 4 requires standard Logistic Regression,
  SVM, classification metrics, and confusion matrices.
- Used Logistic Regression `newton-cg` solver to avoid runtime warnings seen with
  the default solver on the local NumPy/scikit-learn combination.

## How To Run

```bash
UV_CACHE_DIR=/tmp/vimd-uv-cache uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m src.data.preprocess_audio --overwrite
.venv/bin/python -m src.training.train_baseline --overwrite
```

## Outputs Produced

| Output | Result |
| --- | --- |
| `data/processed/audio_preprocessed_16s/` | 390 fixed-length WAV files. |
| `data/processed/preprocessed_metadata.csv` | 390 preprocessed rows. |
| `data/processed/preprocess_audio_issues.csv` | Header only; 0 issues. |
| `outputs/reports/phase2_preprocessing_summary.json` | Phase 2 counts, shape, duration, RMS, and peak summary. |
| `outputs/reports/data_eda.md` | Minimal Phase 3 validation report. |
| `outputs/metrics/baseline_results.json` | Phase 4 validation/test metrics. |
| `outputs/metrics/*_confusion_matrix.csv` | Confusion matrices for both models on validation and test. |
| `outputs/models/logistic_regression_mfcc.pkl` | Saved Logistic Regression baseline. |
| `outputs/models/svm_mfcc.pkl` | Saved SVM baseline. |
| `outputs/reports/phase4_baseline_report.md` | Short baseline report. |

## Verification

| Check | Result |
| --- | --- |
| Unit tests | Passed: 11 tests. |
| Python compilation | Passed for `src` and `tests`. |
| Phase 2 preprocessing | Passed: 390 files, 390 exact-shape files, 0 issues. |
| Preprocessed WAV shape check | Passed: 390/390 mono, 16 kHz, 256,000 frames. |
| Metadata rows | Passed: `preprocessed_metadata.csv` has 390 rows. |
| Phase 4 training | Passed: Logistic Regression and SVM trained and saved. |

Commands run:

```bash
env UV_CACHE_DIR=/tmp/vimd-uv-cache uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m src.data.preprocess_audio --overwrite
.venv/bin/python -m src.training.train_baseline --overwrite
```

## Outputs And Measurements

- Original selected audio duration: min 2.51s, median 15.94s, mean 16.01s,
  max 31.82s.
- Preprocessed duration: exactly 16.00s for every selected file.
- Validation metrics:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.6000 | 0.5981 |
| SVM | 0.6889 | 0.6918 |

- Test metrics:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.6222 | 0.6292 |
| SVM | 0.6222 | 0.6264 |

## Known Limitations

- Validation/test sets have only 45 files each, so metrics have high variance.
- Minimal Phase 3 does not include plots yet.
- MFCC features are a simple baseline, not expected to be the final best model.
- The uv install command required network access to PyPI in this environment.

## Reviewer Notes

1. Use `data/processed/preprocessed_metadata.csv` for Phase 4+ training inputs.
2. Keep future inference audio routed through `src/utils/audio.py`.
3. Prefer SVM as the current best baseline by validation macro F1.
4. Move to Phase 5 CNN only after accepting the Phase 4 baseline artifacts.
