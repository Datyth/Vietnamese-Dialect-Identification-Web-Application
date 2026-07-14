# Implementation Report

## Latest Update: E7 Light CNN Fine-Tuning

### Task Summary

Changed E7 so the PhoWhisper branch remains frozen, but the EfficientNetB0
local branch can be lightly fine-tuned instead of always being fully frozen.
The default runner now fine-tunes the last two parameterized CNN feature blocks
with a smaller CNN learning rate.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/models/whisper_cnn_fusion.py` | Adds controlled local-CNN fine-tuning for the last parameterized child modules while keeping frozen modules in eval mode. |
| `src/training/train_e7_whisper_cnn_fusion.py` | Adds `--cnn-trainable-layers`, `--cnn-learning-rate`, separate optimizer param groups, checkpoint metadata, and local encoder checkpoint saving when fine-tuned. |
| `scripts/train_e7_whisper_cnn_fusion_mps.sh` | Runs light CNN fine-tuning by default with `CNN_TRAINABLE_LAYERS=2`, `CNN_LR=1e-5`, and `DROPOUT=0.0`. |
| `configs/experiments/e7_whisper_cnn_fusion.yaml` | Records the light fine-tuning setup. |
| `PLAN.md`, `README.md` | Documents the updated E7 local-branch training mode and frozen-CNN ablation command. |
| `tests/test_whisper_cnn_fusion.py` | Covers local CNN layer selection, checkpoint state, optimizer learning rates, and defaults. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- PhoWhisper remains fully frozen.
- The default local branch setting is `cnn_trainable_layers=2`, meaning the last
  two parameterized top-level EfficientNetB0 feature modules are trainable.
- CNN trainable parameters use `cnn_learning_rate=1e-5`, while fusion/head
  parameters keep `learning_rate=1e-4`.
- `CNN_TRAINABLE_LAYERS=0` preserves the previous frozen-CNN ablation path.
- No full E7 retraining was run in this implementation step.

### Commands Run

```bash
bash -n scripts/train_e7_whisper_cnn_fusion_mps.sh
.venv/bin/python -m compileall -q src tests
scripts/train_e7_whisper_cnn_fusion_mps.sh --help
.venv/bin/python -m unittest tests.test_whisper_cnn_fusion -v
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python - <<'PY'
# Instantiate the repo EfficientNetB0 local branch and confirm the selected
# fine-tuned child modules plus trainable parameter counts.
PY
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax | Passed for `scripts/train_e7_whisper_cnn_fusion_mps.sh`. |
| Python compilation | Passed for `src` and `tests`. |
| Script help | Passed and documents `CNN_LR=1e-5`, `CNN_TRAINABLE_LAYERS=2`, `DROPOUT=0.0`, and `CLASSIFIER_HIDDEN_DIM=256`. |
| Focused E7 tests | Passed: 13 tests. |
| Full tests | Passed: 53 tests. |
| EfficientNet tail check | Passed: `CNN_TRAINABLE_LAYERS=2` selects child modules `5` and `6`, adding `69,400` local CNN trainable parameters. |

### Known Limitations

- Metrics and confusion matrices are not updated yet; run the E7 script to train
  the fine-tuned CNN variant and compare Central recall/F1.

### Reviewer Priorities

1. Run `scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite --smoke`.
2. Run full `scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite` and compare
   Central recall/F1 against the frozen-CNN E7 artifact.

---

## Latest Update: E7 Confusion Matrix Figure

### Task Summary

Generated a test confusion matrix figure for the current E7 Whisper-CNN fusion
metrics artifact.

### Files Changed

| File | Purpose |
| --- | --- |
| `outputs/figures/e7_whisper_cnn_fusion_confusion_matrix.png` | Adds the E7 test confusion matrix visualization with counts and row percentages. |
| `reports/implementation_report.md` | Records the figure generation and verification. |

### Scope And Decisions

- Used the existing `outputs/metrics/e7_whisper_cnn_fusion_test_confusion_matrix.csv`.
- Did not retrain or re-evaluate E7.
- Kept the figure naming aligned with existing E1-E6 confusion matrix artifacts.

### Commands Run

```bash
sed -n '1,12p' outputs/metrics/e7_whisper_cnn_fusion_test_confusion_matrix.csv
.venv/bin/python - <<'PY'
# Read the E7 test confusion matrix CSV and save
# outputs/figures/e7_whisper_cnn_fusion_confusion_matrix.png.
PY
ls -lh outputs/figures/e7_whisper_cnn_fusion_confusion_matrix.png
file outputs/figures/e7_whisper_cnn_fusion_confusion_matrix.png
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| CSV read | Passed: labels `Northern`, `Central`, `Southern`; matrix `[[451,16,19],[82,287,118],[13,53,419]]`. |
| Figure generation | Passed: PNG written to `outputs/figures/e7_whisper_cnn_fusion_confusion_matrix.png`. |
| File validation | Passed: PNG image data, `1350 x 1134`, 96 KB. |

### Known Limitations

- The figure reflects the existing saved E7 test CSV. If that CSV came from an
  older or interrupted run, regenerate E7 metrics before using the plot as final
  experiment evidence.

### Reviewer Priorities

1. Confirm the CSV corresponds to the intended final E7 checkpoint before
   including the figure in the final report.

---

## Latest Update: Reverted E7 Eval-Only Path

### Task Summary

Reverted the E7 `--eval-only` checkpoint-reporting path and restored the trainer
to its previous train-then-evaluate flow. The 512-dimensional gated-head
architecture remains unchanged.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/training/train_e7_whisper_cnn_fusion.py` | Removes eval-only checkpoint loading and restores inline post-training evaluation/report writing. |
| `scripts/train_e7_whisper_cnn_fusion_mps.sh` | Removes `--eval-only` CLI handling. |
| `README.md` | Removes the interrupted-training eval-only command. |
| `reports/implementation_report.md` | Records the revert and verification. |

### Scope And Decisions

- No architecture changes were reverted.
- E7 still saves the best checkpoint during training, but there is no resume or
  eval-only mode.

### Commands Run

```bash
bash -n scripts/train_e7_whisper_cnn_fusion_mps.sh
.venv/bin/python -m compileall -q src tests
rg -n -- "eval-only|eval_only|load_best_checkpoint|validate_checkpoint_contract" src/training/train_e7_whisper_cnn_fusion.py scripts/train_e7_whisper_cnn_fusion_mps.sh README.md reports/implementation_report.md
.venv/bin/python -m unittest tests.test_whisper_cnn_fusion -v
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax | Passed for `scripts/train_e7_whisper_cnn_fusion_mps.sh`. |
| Python compilation | Passed for `src` and `tests`. |
| Eval-only scan | Passed: no eval-only symbols remain. |
| Focused E7 tests | Passed: 9 tests. |
| Full tests | Passed: 49 tests. |

### Known Limitations

- Interrupted E7 training still cannot be resumed or evaluated via a dedicated
  eval-only command.

### Reviewer Priorities

1. Re-run E7 from the beginning when final metrics are needed.

---

## Latest Update: Phase 10 E7 512-Dim Gated Head

### Task Summary

Updated E7 so the PhoWhisper global branch keeps its native 512-dimensional
embedding, the frozen EfficientNetB0 local feature is projected from 128 to 512,
gated fusion operates at 512 dimensions, and the classifier head is
`512 -> 256 -> 3`.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/models/whisper_cnn_fusion.py` | Removes the global linear projection, adds a hidden classifier layer, and validates `fusion_dim` against the global embedding size. |
| `src/training/train_e7_whisper_cnn_fusion.py` | Sets `fusion_dim=512`, adds `classifier_hidden_dim=256`, records the new fusion/head metadata. |
| `scripts/train_e7_whisper_cnn_fusion_mps.sh` | Sets `FUSION_DIM=512` and adds `CLASSIFIER_HIDDEN_DIM`. |
| `configs/experiments/e7_whisper_cnn_fusion.yaml` | Records global/local/fusion/head dimensions. |
| `PLAN.md`, `README.md` | Documents the updated E7 data flow and dimensions. |
| `tests/test_whisper_cnn_fusion.py` | Covers the new dim contract and hidden classifier layer. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- PhoWhisper and EfficientNetB0 branches remain frozen.
- Global PhoWhisper embedding is normalized but not linearly projected.
- Local EfficientNet features are projected from 128 to 512 so both branches can
  be gated at the same dimensionality.
- No new dependencies were added.

### Commands Run

```bash
git status --short
sed -n '740,810p' PLAN.md
sed -n '1,190p' src/models/whisper_cnn_fusion.py
sed -n '40,90p' src/training/train_e7_whisper_cnn_fusion.py
git diff -- scripts/train_e7_whisper_cnn_fusion_mps.sh
sed -n '1,260p' scripts/train_e7_whisper_cnn_fusion_mps.sh
sed -n '1,220p' tests/test_whisper_cnn_fusion.py
bash -n scripts/train_e7_whisper_cnn_fusion_mps.sh
.venv/bin/python -m compileall -q src tests
scripts/train_e7_whisper_cnn_fusion_mps.sh --help
.venv/bin/python -m unittest tests.test_whisper_cnn_fusion -v
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m src.training.train_e7_whisper_cnn_fusion --device cpu --limit-per-split 3 --max-epochs 1 --batch-size 1 --patience 1 --latency-samples 1 --checkpoint-path /private/tmp/e7_512_head_smoke.pt --metrics-path /private/tmp/e7_512_head_smoke_results.json --training-log-path /private/tmp/e7_512_head_smoke_log.csv --report-path /private/tmp/e7_512_head_smoke_report.md --valid-confusion-path /private/tmp/e7_512_head_smoke_valid.csv --test-confusion-path /private/tmp/e7_512_head_smoke_test.csv --overwrite
.venv/bin/python -c "import json; data=json.load(open('/private/tmp/e7_512_head_smoke_results.json')); print(data['fusion']); print(data['parameter_counts']['trainable'])"
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax | Passed for `scripts/train_e7_whisper_cnn_fusion_mps.sh`. |
| Python compilation | Passed for `src` and `tests`. |
| Script help | Passed and documents `FUSION_DIM=512` plus `CLASSIFIER_HIDDEN_DIM=256`. |
| Focused E7 tests | Passed: 9 tests. |
| Full tests | Passed: 49 tests. |
| E7 CPU smoke | Passed with temp outputs under `/private/tmp`; trainable params `725,251`, fusion metadata records global `512`, local raw `128`, fusion `512`, classifier hidden `256`. |

### Known Limitations

- Full E7 retraining was not run during this change.
- Existing E7 metrics are stale until regenerated with the new 512-dimensional
  gated head.

### Reviewer Priorities

1. Run `scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite --smoke`.
2. Run full E7 and compare against E4 PhoWhisper and the prior E7 concat run.

---

## Latest Update: Phase 10 E7 Gated Mode And Bash Hyperparameters

### Task Summary

Changed E7 to use gated fusion as the default mode and updated the MPS runner so
training hyperparameters can be adjusted directly through environment variables.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/training/train_e7_whisper_cnn_fusion.py` | Sets default fusion to `gated` and records the correct default in metrics. |
| `scripts/train_e7_whisper_cnn_fusion_mps.sh` | Uses gated mode by default, adds `--concat`, validates `FUSION_TYPE`, and exposes more hyperparameter overrides. |
| `configs/experiments/e7_whisper_cnn_fusion.yaml` | Records gated as the default fusion mode. |
| `PLAN.md`, `README.md` | Updates Phase 10 docs and run command from gated ablation to concat ablation. |
| `tests/test_whisper_cnn_fusion.py` | Adds coverage for the default gated fusion mode. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- `gated` is now the default in Python, config, README, and bash.
- `concat` remains available as an explicit ablation via `--concat` or
  `FUSION_TYPE=concat`.
- The bash runner now exposes `SEED`, `MODEL_ID`, `CACHE_DIR`,
  `LOCAL_EMBED_DIM`, `FUSION_DIM`, `FUSION_TYPE`, `SMOKE_LIMIT_PER_SPLIT`, and
  `SMOKE_EPOCHS`, in addition to the existing training controls.
- No new dependencies were added.

### Commands Run

```bash
sed -n '730,820p' PLAN.md
sed -n '1,260p' scripts/train_e7_whisper_cnn_fusion_mps.sh
sed -n '502,595p' src/training/train_e7_whisper_cnn_fusion.py
sed -n '1,80p' configs/experiments/e7_whisper_cnn_fusion.yaml
rg -n "default concat|default fusion|concat|gated|--gated|--concat|Fusion type" README.md reports/implementation_report.md src/training/train_e7_whisper_cnn_fusion.py tests/test_whisper_cnn_fusion.py
bash -n scripts/train_e7_whisper_cnn_fusion_mps.sh
scripts/train_e7_whisper_cnn_fusion_mps.sh --help
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m unittest tests.test_whisper_cnn_fusion -v
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m src.training.train_e7_whisper_cnn_fusion --device cpu --limit-per-split 3 --max-epochs 1 --batch-size 1 --patience 1 --latency-samples 1 --checkpoint-path /private/tmp/e7_gated_smoke.pt --metrics-path /private/tmp/e7_gated_smoke_results.json --training-log-path /private/tmp/e7_gated_smoke_log.csv --report-path /private/tmp/e7_gated_smoke_report.md --valid-confusion-path /private/tmp/e7_gated_smoke_valid.csv --test-confusion-path /private/tmp/e7_gated_smoke_test.csv --overwrite
.venv/bin/python -c "import json; data=json.load(open('/private/tmp/e7_gated_smoke_results.json')); print(data['fusion']['type']); print(data['fusion']['default']); print(data['training']['learning_rate']); print(data['training']['weight_decay']); print(data['training']['batch_size'])"
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax | Passed for `scripts/train_e7_whisper_cnn_fusion_mps.sh`. |
| Script help | Passed and documents `--concat`, `FUSION_TYPE=gated`, and new hyperparameter overrides. |
| Python compilation | Passed for `src` and `tests`. |
| Focused E7 tests | Passed: 8 tests. |
| Full tests | Passed: 48 tests. |
| E7 CPU gated smoke | Passed with 3 rows per split, 1 epoch, temp outputs under `/private/tmp`; metrics JSON records `fusion.type=gated` and `fusion.default=gated`. |

### Known Limitations

- Full E7 retraining was not run during this change.
- Existing E7 metrics are stale until regenerated with the new gated default.

### Reviewer Priorities

1. Run `scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite --smoke`.
2. Tune hyperparameters through environment variables if the smoke run is stable.
3. Run full E7 and optionally run `--concat` as an ablation.

---

## Latest Update: Phase 10 E7 No-Dropout Head

### Task Summary

Changed E7 fusion training to use no dropout by default. When dropout is `0.0`,
the model now inserts `nn.Identity()` instead of `nn.Dropout`, so the default E7
head has no dropout modules.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/models/whisper_cnn_fusion.py` | Uses identity layers instead of dropout when dropout is zero. |
| `src/training/train_e7_whisper_cnn_fusion.py` | Sets E7 default dropout to `0.0`. |
| `scripts/train_e7_whisper_cnn_fusion_mps.sh` | Sets script default `DROPOUT=0.0` and keeps the override available. |
| `configs/experiments/e7_whisper_cnn_fusion.yaml` | Records `dropout: 0.0`. |
| `tests/test_whisper_cnn_fusion.py` | Adds coverage that the default E7 model has no `Dropout` modules. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- No new dependencies were added.
- Existing `--dropout` remains available only as an explicit ablation override.
- E7 metrics must still be regenerated after this architecture/training change.

### Commands Run

```bash
sed -n '730,820p' PLAN.md
rg -n "DROPOUT|dropout|Dropout" src/models/whisper_cnn_fusion.py src/training/train_e7_whisper_cnn_fusion.py scripts/train_e7_whisper_cnn_fusion_mps.sh configs/experiments/e7_whisper_cnn_fusion.yaml tests/test_whisper_cnn_fusion.py README.md reports/implementation_report.md
bash -n scripts/train_e7_whisper_cnn_fusion_mps.sh
.venv/bin/python -m compileall -q src tests
rg -n "dropout: 0\\.25|default: 0\\.25|DEFAULT_DROPOUT = 0\\.25|Dropout\\(p=dropout\\)|# DROPOUT" src/models/whisper_cnn_fusion.py src/training/train_e7_whisper_cnn_fusion.py scripts/train_e7_whisper_cnn_fusion_mps.sh configs/experiments/e7_whisper_cnn_fusion.yaml reports/implementation_report.md README.md PLAN.md
.venv/bin/python -m unittest tests.test_whisper_cnn_fusion -v
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax | Passed for `scripts/train_e7_whisper_cnn_fusion_mps.sh`. |
| Python compilation | Passed for `src` and `tests`. |
| Stale dropout scan | No E7 default `0.25` or direct `Dropout(p=dropout)` remains. |
| Focused E7 tests | Passed: 7 tests. |
| Full tests | Passed: 47 tests. |

### Known Limitations

- Full E7 retraining was not run during this change.

### Reviewer Priorities

1. Run `scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite --smoke`.
2. Run full E7 after the smoke run passes.

---

## Latest Update: Phase 10 E7 Frozen EfficientNetB0 Local Branch

### Task Summary

Changed E7 so the local log-Mel branch reuses the trained E2
EfficientNetB0-style checkpoint instead of training a new CNN from scratch.
E7 now freezes both PhoWhisper and EfficientNetB0 features, then trains only the
projection, fusion, and 3-class classification head.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/models/whisper_cnn_fusion.py` | Allows an injected frozen local encoder and keeps it in eval mode. |
| `src/training/train_e7_whisper_cnn_fusion.py` | Loads and validates the E2 checkpoint, freezes EfficientNet features, excludes frozen encoders from E7 checkpoints, and records local-branch metadata. |
| `scripts/train_e7_whisper_cnn_fusion_mps.sh` | Requires the E2 checkpoint before running E7 and documents the frozen EfficientNet branch. |
| `configs/experiments/e7_whisper_cnn_fusion.yaml` | Updates E7 config to use frozen `e2_efficientnetb0`. |
| `PLAN.md`, `README.md` | Clarify that E7 trains only the head/fusion layers. |
| `tests/test_whisper_cnn_fusion.py` | Covers frozen local-branch behavior and checkpoint filtering. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- No new dependencies were added.
- The default local checkpoint is `outputs/models/e2_efficientnetb0_logmel.pt`.
- The E2 checkpoint contract is validated before training: experiment ID,
  feature type, sample rate, target length, Mel bins, and label order.
- E7 checkpoints store trainable E7 layers only; they do not duplicate
  PhoWhisper or EfficientNetB0 weights.
- Existing E7 metrics are stale after this code change and should be regenerated.

### Commands Run

```bash
sed -n '1,260p' src/models/efficientnet_classifier.py
sed -n '1,260p' src/training/train_e2_efficientnet.py
sed -n '1,220p' tests/test_whisper_cnn_fusion.py
sed -n '1,120p' PLAN.md
rg -n "lightweight|checkpoint|load_state_dict|cnn|local_branch|freeze|trainable|state_dict|DEFAULT" src/models/whisper_cnn_fusion.py src/training/train_e7_whisper_cnn_fusion.py configs/experiments/e7_whisper_cnn_fusion.yaml
bash -n scripts/train_e7_whisper_cnn_fusion_mps.sh
.venv/bin/python -m compileall -q src tests
scripts/train_e7_whisper_cnn_fusion_mps.sh --help
.venv/bin/python -m unittest tests.test_whisper_cnn_fusion tests.test_extended_deep_learning -v
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -c "from types import SimpleNamespace; from pathlib import Path; import torch; from src.training.train_e7_whisper_cnn_fusion import load_frozen_efficientnet_encoder; args=SimpleNamespace(cnn_checkpoint_path=Path('outputs/models/e2_efficientnetb0_logmel.pt')); encoder, ckpt = load_frozen_efficientnet_encoder(args, torch.device('cpu')); print(ckpt.get('experiment_id')); print(sum(p.numel() for p in encoder.parameters() if p.requires_grad)); print(tuple(encoder(torch.randn(1,1,64,501)).shape))"
.venv/bin/python -m src.training.train_e7_whisper_cnn_fusion --device cpu --limit-per-split 3 --max-epochs 1 --batch-size 1 --patience 1 --latency-samples 1 --checkpoint-path /private/tmp/e7_smoke.pt --metrics-path /private/tmp/e7_smoke_results.json --training-log-path /private/tmp/e7_smoke_log.csv --report-path /private/tmp/e7_smoke_report.md --valid-confusion-path /private/tmp/e7_smoke_valid.csv --test-confusion-path /private/tmp/e7_smoke_test.csv --overwrite
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax | Passed for `scripts/train_e7_whisper_cnn_fusion_mps.sh`. |
| Python compilation | Passed for `src` and `tests`. |
| Script help | Passed and documents frozen E2 EfficientNetB0 branch plus `CNN_CHECKPOINT`. |
| Focused tests | Passed: 14 tests. |
| Full tests | Passed: 46 tests. |
| E2 checkpoint load | Passed: experiment `e2_efficientnetb0`, local encoder trainable params `0`, output shape `(1, 128, 1, 1)`. |
| E7 CPU smoke | Passed with 3 rows per split, 1 epoch, temp outputs under `/private/tmp`; trainable params `168,195/20,880,835`, PhoWhisper trainable `0`, EfficientNet trainable `0`. |

### Known Limitations

- Full E7 retraining was not run during this code change.
- The existing `outputs/metrics/e7_whisper_cnn_fusion_results.json` describes
  the previous trainable-CNN implementation until E7 is rerun.

### Reviewer Priorities

1. Run `scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite --smoke`.
2. Run full E7 after smoke passes.
3. Compare the rerun E7 against E2 and E4 before updating final conclusions.

---

## Latest Update: Phase 10 PhoWhisper + CNN Fusion Experiment

### Task Summary

Implemented the Phase 10 E7 hybrid research experiment: a frozen
`vinai/PhoWhisper-base` encoder branch fused with a trainable log-Mel CNN
branch for 3-class Vietnamese dialect classification.

### Files Changed

| File | Purpose |
| --- | --- |
| `PLAN.md` | Replaces the stale Phrase 10 placeholder with the hybrid PhoWhisper + CNN experiment scope and expected outputs. |
| `README.md` | Documents Phase 10 PhoWhisper motivation, commands, pending E7 result row, and interpretation guidance. |
| `configs/experiments/e7_whisper_cnn_fusion.yaml` | Records the E7 experiment setup and Central-focused analysis targets. |
| `src/models/whisper_cnn_fusion.py` | Adds the frozen PhoWhisper/Whisper-family encoder + local CNN fusion classifier with concat and gated fusion. |
| `src/training/train_e7_whisper_cnn_fusion.py` | Adds the Phase 10 training/evaluation runner, metrics JSON, confusion matrices, Central error analysis, and comparison refresh. |
| `src/training/train_extended_deep_learning.py` | Includes E7 in `model_method_comparison.csv` when E7 metrics exist. |
| `scripts/train_e7_whisper_cnn_fusion_mps.sh` | Adds an Apple MPS Terminal runner with batch size 4, smoke mode, gated mode, and optional download. |
| `tests/test_whisper_cnn_fusion.py` | Adds CPU-safe tests for fusion forward passes, encoder freezing, Central error analysis, and comparison row collection. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- PhoWhisper is used only as a frozen encoder; decoder and ASR transcripts are
  not used.
- The default fusion is concat; gated fusion is available via `--gated`.
- The runner trains only the CNN branch, projection/fusion layers, and
  classifier head.
- Feature extraction is on-demand from preprocessed waveforms to avoid storing
  all Whisper features in RAM on 16 GB machines.
- Checkpoints store trainable fusion/classifier weights plus metadata, not a
  duplicate copy of the frozen PhoWhisper encoder.
- No new dependencies were added; existing `torch`, `transformers`, `numpy`,
  and `scikit-learn` are sufficient.

### Commands Run

```bash
sed -n '1,260p' PLAN.md
rg --files src tests configs scripts reports outputs/reports | sort
git status --short
sed -n '1,260p' src/training/train_extended_deep_learning.py
sed -n '620,1240p' src/training/train_extended_deep_learning.py
sed -n '1240,2320p' src/training/train_extended_deep_learning.py
sed -n '1,620p' src/training/train_phowhisper.py
sed -n '1,520p' src/training/train_cnn.py
sed -n '1,240p' src/models/cnn.py
sed -n '1,220p' src/features/logmel.py
bash -n scripts/train_e7_whisper_cnn_fusion_mps.sh
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m unittest tests.test_whisper_cnn_fusion tests.test_extended_deep_learning -v
scripts/train_e7_whisper_cnn_fusion_mps.sh --help
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax | Passed for `scripts/train_e7_whisper_cnn_fusion_mps.sh`. |
| Python compilation | Passed for `src` and `tests`. |
| Focused tests | Passed: 13 tests. |
| Script help | Passed and documents `--overwrite`, `--smoke`, `--gated`, and `--allow-download`. |

### Known Limitations

- Full E7 training was not run in the sandbox. Run
  `scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite` from a normal
  Terminal to generate real metrics.
- If `vinai/PhoWhisper-base` is already cached from E4/Phase 6, `--allow-download`
  can be omitted.
- The current README lists E7 as pending until
  `outputs/metrics/e7_whisper_cnn_fusion_results.json` exists.

### Reviewer Priorities

1. Run the E7 smoke command from Terminal:
   `scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite --smoke`.
2. Run the full E7 concat experiment, then optionally the gated ablation.
3. Compare Central recall/F1 and Central confusion errors against E4
   PhoWhisper-base first, then E6 Whisper-base as the original-Whisper control.

---

## Latest Update: README Dataset And Training Results

### Task Summary

Updated `README.md` to reflect the current prepared ViMD subset, preprocessing
status, full model comparison results, E3/E5/E6 run scripts, and current best
model.

### Files Changed

| File | Purpose |
| --- | --- |
| `README.md` | Replaces stale 390-sample/subset documentation with current 13,894-row dataset and current model metrics. |
| `reports/implementation_report.md` | Records the documentation update and verification. |

### Scope And Decisions

- Used only local artifacts under `data/processed/`, `outputs/metrics/`, and
  `outputs/reports/`.
- Did not retrain models or regenerate metrics.
- Marked E5 as the local ChunkFormer-style model, not an official pretrained
  ViP-VL/ChunkFormer checkpoint.
- Kept README in English to match the existing file style.

### Commands Run

```bash
sed -n '1,260p' PLAN.md
sed -n '1,760p' README.md
find outputs/metrics -maxdepth 1 -type f | sort
find data/processed -maxdepth 1 -type f | sort
.venv/bin/python - <<'PY'  # summarize preprocessed metadata counts
...
PY
.venv/bin/python - <<'PY'  # summarize dataset reports and model metrics
...
PY
sed -n '1,520p' README.md
wc -l README.md
rg -n "390|0\\.6889|skipped|smoke metrics|E6|13,894|0\\.8474" README.md
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Dataset counts | README matches `13,894` preprocessed rows and the current train/valid/test class counts. |
| Model metrics | README table matches current local metric JSON/CSV artifacts. |
| Stale text scan | No stale `390` sample claim or old `0.6889` SVM-best result remains. |
| README readback | Passed manual readback with `sed`; table and command sections are present. |

### Known Limitations

- This was documentation-only; no training or full test suite was rerun.
- The README reports the current local artifact state, including the older
  smaller PhoWhisper full-fine-tune run as a note rather than a primary current
  comparison row.

### Reviewer Priorities

1. Review the README result table against `outputs/metrics/model_method_comparison.csv`.
2. Re-run comparison artifact generation only if model metrics are regenerated.

---

## Latest Update: E6 Original Whisper-Base Experiment

### Task Summary

Added Phase 9 E6 for original `openai/whisper-base`, matched to the
PhoWhisper-base size family, so it can be trained as a frozen-encoder baseline
and compared against PhoWhisper-base, E1-E5, MFCC baselines, and the main CNN.

### Files Changed

| File | Purpose |
| --- | --- |
| `PLAN.md` | Adds E6 scope, hyperparameters, and expected outputs. |
| `configs/experiments/e6_whisper_base.yaml` | Records the E6 experiment setup. |
| `src/training/train_e6_whisper.py` | Adds the E6 entrypoint with `openai/whisper-base` defaults and comparison refresh. |
| `src/training/train_phowhisper.py` | Adds reusable experiment metadata and `--limit-per-split` support for Whisper-family runs. |
| `src/training/train_extended_deep_learning.py` | Registers E6 in Phase 9 summaries and full method comparison output. |
| `scripts/train_e6_whisper_mps.sh` | Adds a Terminal-run MPS script for E6 only. |
| `tests/test_phowhisper.py`, `tests/test_extended_deep_learning.py` | Adds E6 default/alias and subset-limit coverage. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- E6 uses `openai/whisper-base`, the original Whisper base checkpoint.
- E6 uses `frozen_encoder` mode with batch size 4 by default, matching the
  lightweight comparison setup used for PhoWhisper-base.
- E6 writes separate artifacts under `e6_whisper_base_*` so it does not
  overwrite PhoWhisper outputs.
- No new dependencies were added; existing `torch` and `transformers` are enough.

### Commands Run

```bash
bash -n scripts/train_e6_whisper_mps.sh
scripts/train_e6_whisper_mps.sh --help
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m unittest tests.test_phowhisper tests.test_extended_deep_learning -v
scripts/train_e6_whisper_mps.sh --overwrite --smoke
.venv/bin/python - <<'PY'
from src.training.train_extended_deep_learning import write_method_comparison_from_available, write_phase9_summary_from_available
write_phase9_summary_from_available()
write_method_comparison_from_available()
PY
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax/help | Passed. |
| Python compilation | Passed for `src` and `tests`. |
| Focused tests | Passed: 16 tests. |
| E6 MPS smoke | Passed with `openai/whisper-base`, batch size 4, 6 rows per split, 1 epoch. |
| Summary refresh | Passed: E6 appears in `deep_learning_comparison.csv`, `model_method_comparison.csv`, and `extended_deep_learning_experiments.md`. |

### Known Limitations

- Current E6 metrics are smoke metrics from a tiny subset. Run
  `scripts/train_e6_whisper_mps.sh --overwrite` from Terminal to regenerate full
  E6 metrics and plots.
- The log about newly initialized `projector` and `classifier` weights is
  expected because Whisper ASR checkpoints do not contain a 3-class dialect head.

### Reviewer Priorities

1. Run full E6 after E3/E5 as a separate supplement:
   `scripts/train_e6_whisper_mps.sh --overwrite`.
2. Compare E6 against E4 PhoWhisper in
   `outputs/metrics/model_method_comparison.csv`.

---

## Latest Update: E3/E5 Script Bash 3 Fix

### Task Summary

Fixed `scripts/train_e3_e5_mps.sh` failing on macOS Bash 3.2 with
`LIMIT_ARGS[@]: unbound variable` when running the full command without
`--smoke`.

### Files Changed

| File | Purpose |
| --- | --- |
| `scripts/train_e3_e5_mps.sh` | Replaces empty optional argument arrays with conditionally built command arrays. |
| `reports/implementation_report.md` | Records the script compatibility fix and verification. |

### Scope And Decisions

- Kept the same user command: `scripts/train_e3_e5_mps.sh --overwrite`.
- Preserved batch size 4, MPS preflight, E3 checkpoint behavior, and E5
  training behavior.
- The fix targets macOS default Bash 3.2 with `set -u`; it avoids expanding
  empty arrays such as `LIMIT_ARGS[@]`.

### Commands Run

```bash
bash -n scripts/train_e3_e5_mps.sh
scripts/train_e3_e5_mps.sh --help
scripts/train_e3_e5_mps.sh --overwrite --smoke --skip-e5 --no-download
scripts/train_e3_e5_mps.sh --overwrite --smoke --skip-e3
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Shell syntax | Passed. |
| Help output | Passed. |
| E3 smoke on MPS | Passed with cached checkpoint, batch size 4. |
| E5 smoke on MPS | Passed with batch size 4. |

### Known Limitations

- Smoke verification overwrote the current E3/E5 metrics with tiny-subset
  metrics. Run the full command again to regenerate real metrics and plots.

### Reviewer Priorities

1. Re-run `scripts/train_e3_e5_mps.sh --overwrite` from a normal Terminal.
2. Confirm it proceeds past the run settings into E3 training without the
   `LIMIT_ARGS[@]` error.

---

## Latest Update: E3/E5 MPS Self-Run Script

### Task Summary

Added a Terminal-run script for training Phase 9 E3 and E5 on Apple MPS with
batch size 4, plus the lightweight code paths needed for E3 wav2vec2 frozen
embeddings, E5 ChunkFormer-style waveform training, and comparison plots.

### Files Changed

| File | Purpose |
| --- | --- |
| `scripts/train_e3_e5_mps.sh` | Runs E3/E5 outside the Codex sandbox, checks MPS, uses batch size 4, and updates comparison artifacts. |
| `src/training/train_extended_deep_learning.py` | Adds E3 wav2vec2 frozen-embedding training, E5 waveform training, and full method comparison CSV/figures. |
| `src/models/wav2vec2_classifier.py` | Adds the wav2vec2 embedding classifier head. |
| `src/models/vipvl_chunkformer_classifier.py` | Makes the local ChunkFormer-style classifier trainable on 16s waveforms without excessive sequence length. |
| `configs/experiments/e3_wav2vec2.yaml` | Records the Vietnamese wav2vec2 checkpoint and frozen-embedding training mode. |
| `configs/experiments/e5_vipvl_chunkformer.yaml` | Records the local ChunkFormer-style fallback training mode. |
| `tests/test_extended_deep_learning.py` | Adds forward/CSV coverage for E3/E5 and method comparison output. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- E3 uses `nguyenvulebinh/wav2vec2-base-vietnamese-250h`, freezes the encoder,
  extracts mean-pooled embeddings, and trains only a small classifier head.
- E5 trains the local plain-PyTorch ChunkFormer-style waveform classifier; the
  official ViP-VL/ChunkFormer checkpoint integration remains a limitation.
- The script requires `--overwrite` so existing metrics/checkpoints are not
  regenerated silently.
- Codex sandboxed commands reported `mps_is_available=False`, but the same venv
  outside the sandbox reports `mps_is_available=True` and can allocate `mps:0`.

### Commands Run

```bash
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m unittest tests.test_extended_deep_learning -v
.venv/bin/python - <<'PY'  # sandboxed MPS diagnostic
import torch
print(torch.backends.mps.is_built(), torch.backends.mps.is_available())
PY
# unsandboxed MPS diagnostic through Codex escalation
.venv/bin/python - <<'PY'
import torch
print(torch.backends.mps.is_built(), torch.backends.mps.is_available())
print(torch.ones(1, device="mps").cpu().item())
PY
scripts/train_e3_e5_mps.sh --help
scripts/train_e3_e5_mps.sh --overwrite --smoke --skip-e5
scripts/train_e3_e5_mps.sh --overwrite --smoke --skip-e3
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Python compilation | Passed for `src` and `tests`. |
| Focused Phase 9 tests | Passed: 7 tests. |
| MPS outside sandbox | Passed: `mps_is_available=True`, `device_count=1`, tensor allocation on `mps:0` succeeded. |
| E3 script smoke | Passed on MPS with batch size 4, 6 rows per split, 1 epoch. |
| E5 script smoke | Passed on MPS with batch size 4, 6 rows per split, 1 epoch. |

### Known Limitations

- Full E3/E5 training was not completed in Codex because sandboxed commands
  cannot access MPS and CPU wav2vec2 extraction is too slow.
- Current E3/E5 metric JSON files were overwritten by smoke verification; run
  `scripts/train_e3_e5_mps.sh --overwrite` from a normal Terminal to regenerate
  full metrics and plots.
- E3 first run may need network access unless the wav2vec2 checkpoint is already
  cached under `outputs/models/hf_cache`.
- E5 is a local ChunkFormer-style fallback, not an official ViP-VL/ChunkFormer
  pretrained checkpoint result.

### Reviewer Priorities

1. Run `scripts/train_e3_e5_mps.sh --overwrite` from a normal Terminal window.
2. Confirm the script preflight shows `mps_is_available=True`.
3. Review `outputs/metrics/model_method_comparison.csv` and the two comparison
   plots after the full run completes.

---

## Latest Update: PhoWhisper Frozen Default

### Task Summary

Changed the app's default PhoWhisper checkpoint from the fine-tuned checkpoint
to the pretrained frozen-encoder checkpoint requested for the demo.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/inference/predict.py` | Sets `DEFAULT_PHOWHISPER_CHECKPOINT_PATH` to `outputs/models/phowhisper_pretrained_frozen_encoder.pt`. |
| `tests/test_app.py` | Verifies `/models` exposes the frozen-encoder checkpoint path for the PhoWhisper option. |
| `README.md` | Documents the frozen-encoder default and how to override to the fine-tuned checkpoint. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- Kept the single UI option name `PhoWhisper`; only its default checkpoint path
  changed.
- Preserved `PHOWHISPER_CHECKPOINT_PATH` as the override for manually selecting
  `outputs/models/phowhisper_dialect.pt` or another compatible checkpoint.
- Did not retrain or modify model artifacts.

### Commands Run

```bash
sed -n '1,240p' PLAN.md
rg -n "PHOWHISPER|phowhisper_dialect|phowhisper_pretrained|DEFAULT_PHOWHISPER" src README.md tests reports/implementation_report.md
.venv/bin/python -m unittest tests.test_app tests.test_inference -v
.venv/bin/python -m compileall -q src tests
.venv/bin/python -c "... TestClient GET /models confirms PhoWhisper artifact path ..."
.venv/bin/python -c "... load_phowhisper_model() default path and predict one sample ..."
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| App shell/inference tests | Passed: `/models` exposes `outputs/models/phowhisper_pretrained_frozen_encoder.pt`; CNN/SVM inference smoke tests still pass. |
| Python compilation | Passed for `src` and `tests`. |
| PhoWhisper default smoke | Passed: default `load_phowhisper_model()` loaded and returned one CPU prediction. |

### Known Limitations

- The frozen-encoder checkpoint still requires the local PhoWhisper Hugging Face
  cache under `outputs/models/hf_cache`.

### Reviewer Priorities

1. Restart uvicorn so the new default path is loaded.
2. Confirm `/models` shows the PhoWhisper artifact path as
   `outputs/models/phowhisper_pretrained_frozen_encoder.pt`.

---

## Latest Update: App Model Selector Cache Fix

### Task Summary

Fixed the browser continuing to show the old single-model app shell after the
model selector was added.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/app/main.py` | Serves `/` with no-store/no-cache headers so `index.html` is not reused stale by the browser. |
| `tests/test_app.py` | Adds regression checks that `/` contains the model selector and no-cache header, and `/models` lists all selectable models. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- Kept the existing static HTML file and FastAPI route structure.
- Added only response headers for the app shell rather than changing the UI
  layout or prediction APIs.
- Added a narrow app-shell test because the reported issue was visible before
  any model inference happened.

### Commands Run

```bash
sed -n '1,240p' PLAN.md
sed -n '1,220p' src/app/static/index.html
sed -n '1,220p' src/app/main.py
.venv/bin/python -m unittest tests.test_app tests.test_inference -v
.venv/bin/python -m compileall -q src tests
.venv/bin/python -c "... TestClient GET / confirms cache-control and id=\"model\" ..."
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| App shell tests | Passed: `/` contains `id="model"` and `Cache-Control: no-store`; `/models` lists `cnn`, `svm`, `phowhisper`. |
| Inference focused tests | Passed: CNN/SVM smoke tests and model aliases. |
| Python compilation | Passed for `src` and `tests`. |
| Direct app-shell smoke | Passed: status 200, no-store header, model selector present. |

### Known Limitations

- A browser tab that already has the old HTML loaded still needs a reload after
  restarting uvicorn. The new response headers prevent future stale reuse.

### Reviewer Priorities

1. Stop the current uvicorn process, restart it, and reload
   `http://127.0.0.1:8000/`.
2. If an already-open tab still shows the old layout once, force reload with
   `Cmd+Shift+R` or open `http://127.0.0.1:8000/?v=2`.

---

## Latest Update: Selectable App Models

### Task Summary

Added user-selectable inference models to the FastAPI demo. The browser can now
choose between the existing lightweight CNN, the Phase 4 SVM MFCC baseline, and
the Phase 6 PhoWhisper classifier before submitting an audio file.

### Files Changed

| File | Purpose |
| --- | --- |
| `src/inference/predict.py` | Adds model-name normalization, SVM loading/prediction, PhoWhisper loading/prediction, and keeps CNN as the default path. |
| `src/app/main.py` | Adds `/models`, model-aware `/health`, lazy model loading, and `model` form handling for `/predict`. |
| `src/app/static/index.html` | Adds a model selector and sends the selected model with each prediction. |
| `tests/test_inference.py` | Adds SVM smoke coverage and model-name alias checks. |
| `README.md` | Documents selectable models, new environment variables, `/models`, and form field usage. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- Reused existing trained artifacts:
  `outputs/models/lightweight_cnn_logmel.pt`,
  `outputs/models/svm_mfcc.pkl`, and
  `outputs/models/phowhisper_dialect.pt`.
- Reused existing feature code instead of adding separate app preprocessing:
  CNN uses `log_mel_spectrogram()`, SVM uses `mfcc_mean_std()`, and PhoWhisper
  uses the training-time Hugging Face feature extractor convention.
- Kept `cnn` as the default model for backward compatibility.
- Loaded models lazily on first use so the app can start without immediately
  loading the larger PhoWhisper weights.
- Added tolerant aliases such as `lightweight-cnn`,
  `support_vector_machine`, and the misspelled `phoWIshper`.
- Kept SVM confidence clearly uncalibrated: its displayed class scores are
  softmax-normalized decision margins, not probabilities.

### Commands Run

```bash
sed -n '1,240p' PLAN.md
sed -n '1,280p' src/inference/predict.py
sed -n '1,280p' src/app/main.py
sed -n '1,320p' src/app/static/index.html
find outputs/models -maxdepth 3 -type f | sort
.venv/bin/python -c "... inspect SVM/CNN/PhoWhisper artifact metadata ..."
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m unittest tests.test_inference -v
.venv/bin/python -c "... parse src/app/static/index.html with html.parser ..."
.venv/bin/python -c "... FastAPI TestClient /models and SVM /predict smoke ..."
.venv/bin/python -c "... load PhoWhisper from local cache and run one CPU prediction ..."
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
env CNN_DEVICE=cpu PHOWHISPER_DEVICE=cpu .venv/bin/python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8765
.venv/bin/python -c "... GET http://127.0.0.1:8765/models ..."
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Python compilation | Passed for `src` and `tests`. |
| Focused inference tests | Passed: CNN default prediction, SVM prediction, and aliases. |
| HTML parsing | Passed. |
| FastAPI smoke | `/models` returned `cnn`, `svm`, `phowhisper`; `/predict` with `model=svm` returned 200. |
| PhoWhisper smoke | Loaded `outputs/models/phowhisper_dialect.pt` from local HF cache on CPU and returned a 3-class prediction. |
| Full test suite | Passed: 28 tests with explicit `unittest discover -s tests -p 'test_*.py' -v`. |
| Local app server | Started on `http://127.0.0.1:8765`; `/models` returned all three models as available. |

### Known Limitations

- PhoWhisper still needs the local Hugging Face cache under
  `outputs/models/hf_cache` unless `PHOWHISPER_LOCAL_FILES_ONLY=0` is used in
  an environment with network access.
- First PhoWhisper prediction is slower because the checkpoint is loaded lazily.
- CNN and PhoWhisper softmax confidence and SVM margin-derived scores are not
  calibrated probabilities.
- The app still predicts only Northern, Central, and Southern; it does not infer
  speaker identity, hometown, ethnicity, or personal background.

### Reviewer Priorities

1. Start the app and verify the dropdown behavior with one real audio upload for
   each available model.
2. Keep the uncalibrated-confidence disclaimer visible if the UI is restyled.
3. Prefer SVM for fastest CPU demo fallback when PhoWhisper startup latency is
   too high.

---

## Latest Update: Feature Visualization Notebook

### Task Summary

Added a presentation/report notebook for section 2.5 that selects real train
audio samples from all three dialect classes, exports original and preprocessed
clips, visualizes waveform/MFCC/log-Mel features, and writes a Vietnamese
summary report.

### Files Changed

| File | Purpose |
| --- | --- |
| `notebooks/feature_visualization.ipynb` | End-to-end feature visualization notebook with Vietnamese explanations and generated artifact validation. |
| `outputs/reports/feature_visualization_summary.md` | Generated presentation-ready Vietnamese summary from the executed notebook. |
| `requirements.txt`, `pyproject.toml` | Add direct `matplotlib` dependency for notebook plotting. |
| `reports/implementation_report.md` | Records implementation and verification. |

### Scope And Decisions

- Reused `src/utils/audio.py`, `src/features/mfcc.py`, and
  `src/features/logmel.py` instead of duplicating preprocessing or feature
  extraction logic.
- Preferred `data/processed/preprocessed_metadata.csv` and the train split when
  selecting samples, with deterministic `random_state = 42`.
- Selected 5 samples for each of Northern, Central, and Southern from real local
  dataset audio.
- Exported preprocessed and original WAV files under
  `outputs/audio/feature_visualization/`.
- Saved all figures at 300 DPI under `outputs/figures/feature_visualization/`.
- Kept generated visualizations exploratory and included limitations about
  speaker identity, sentence content, recording condition, and noise.

### Dependencies

- Added `matplotlib>=3.8,<4` because the notebook must generate waveform,
  heatmap, vector, and distribution figures. The standard library and existing
  NumPy/audio utilities cannot create the requested PNG charts by themselves.

### Commands Run

```bash
sed -n '1,240p' PLAN.md
rg --files
.venv/bin/python -c "... check train rows and existing audio by class ..."
.venv/bin/python -c "... check numpy, soundfile, matplotlib, IPython, nbformat, nbconvert imports ..."
mkdir -p notebooks outputs/figures/feature_visualization outputs/audio/feature_visualization outputs/reports
env JUPYTER_CONFIG_DIR=/private/tmp/vimd_jupyter_config JUPYTER_DATA_DIR=/private/tmp/vimd_jupyter_data JUPYTER_RUNTIME_DIR=/private/tmp/vimd_jupyter_runtime .venv/bin/jupyter nbconvert --to notebook --execute notebooks/feature_visualization.ipynb --output /private/tmp/feature_visualization_executed.ipynb
find outputs/audio/feature_visualization -type f | sort | wc -l
find outputs/figures/feature_visualization -type f | sort | wc -l
sed -n '1,220p' outputs/reports/feature_visualization_summary.md
```

### Outputs And Verification

| Check | Result |
| --- | --- |
| Dataset availability | Train split has existing audio for all classes: Northern 3,708; Central 3,416; Southern 3,854. |
| Notebook execution | Passed with `nbconvert`; generated artifacts from real local audio. |
| Audio exports | 30 WAV files: original and preprocessed clips for 15 selected samples. |
| Individual feature figures | 15 PNG files, one per selected sample. |
| Summary figures | 7 PNG files: waveform grid, MFCC grid, log-Mel grid, MFCC vectors, class-average MFCC, log-Mel energy, duration distribution. |
| Markdown summary | `outputs/reports/feature_visualization_summary.md` generated with sample, figure, audio, limitation, and presentation-script sections. |

### Known Limitations

- Jupyter execution in the managed sandbox required temp Jupyter directories and
  elevated execution because the local kernel binds loopback ports.
- The audio and figure artifacts are generated outputs from the current local
  dataset; rerunning after metadata or data changes can select different files.
- The figures are exploratory presentation aids, not final scientific proof of
  dialect differences.

### Reviewer Priorities

1. Open `notebooks/feature_visualization.ipynb` from the repository root and
   confirm the audio players load via the generated `/files/...` paths.
2. Review the generated figures for slide readability before placing them in
   the final report.
3. Keep the limitation language when reusing the visuals in presentation text.

---

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
