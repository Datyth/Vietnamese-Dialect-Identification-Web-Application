# Implementation Report

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
