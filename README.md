# Vietnamese Dialect Identification

A research pipeline and local web demo for classifying Vietnamese speech into three **dataset-defined regional labels**: `Northern`, `Central`, and `Southern`. The project covers ViMD metadata preparation, fixed-length audio preprocessing, MFCC baselines, neural experiments, evaluation, and a FastAPI upload interface.

![System architecture](outputs/figures/system_architecture.svg)

The diagram shows the offline research workflow and the models actually available in the web app. E7 and E8 are evaluated research models; the app currently offers SVM, lightweight CNN, and PhoWhisper.

## Results

The table below comes from [`outputs/metrics/model_method_comparison.csv`](outputs/metrics/model_method_comparison.csv). Model choice is based on **validation macro F1**; test results describe the saved evaluation runs. Values are rounded to four decimals.

| Model | Method | Validation macro F1 | Test macro F1 |
| --- | --- | ---: | ---: |
| Logistic Regression | MFCC mean/std | 0.5612 | 0.5463 |
| SVM | MFCC mean/std | 0.5688 | 0.5592 |
| Lightweight CNN | Log-Mel | 0.6052 | 0.6115 |
| E1 MobileNetV3-style | Log-Mel, local implementation | 0.6719 | 0.6941 |
| E2 EfficientNet-B0-style | Log-Mel, local implementation | 0.7127 | 0.6932 |
| E3 wav2vec2 Vietnamese | Frozen encoder | 0.6063 | 0.5809 |
| **E4 PhoWhisper-base** | **Frozen encoder** | **0.8474** | **0.8348** |
| E5 ChunkFormer-style | Waveform, local implementation | 0.6286 | 0.6330 |
| E6 Whisper-base | Frozen encoder | 0.8229 | 0.8163 |
| E7 PhoWhisper + CNN | Gated fusion | 0.8194 | 0.7947 |
| E8 PhoWhisper + CNN | Residual-gated fusion | 0.8395 | 0.8085 |

E4 has the highest saved validation macro F1. E8 improves on E7 but does not exceed E4 in this run. E1, E2, and E5 are locally implemented architectures; their names do not imply official pretrained MobileNetV3, EfficientNet-B0, or ViP-VL/ChunkFormer checkpoints. E4 reuses the Phase 6 frozen-encoder result. See the [method comparison](outputs/metrics/model_method_comparison.csv), [E7 report](outputs/reports/phase10_whisper_cnn_fusion_report.md), [E8 report](outputs/reports/phase11_whisper_cnn_residual_fusion_report.md), and [Central class analysis](docs/central_class_data_analysis.md) for details.

The older [`final_comparison.csv`](outputs/metrics/final_comparison.csv) was generated from a smaller Phase 7 run. Use `model_method_comparison.csv` for the full 13,894-sample research comparison; do not combine rows from the two tables.

## Data and preprocessing

The source is [ViMD](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset), recorded in the project metadata with license `CC-BY-NC-ND-4.0`. The source regions map as `North → Northern`, `Central → Central`, and `South → Southern`.

| Saved subset | Northern | Central | Southern | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 3,708 | 3,416 | 3,854 | 10,978 |
| Validation | 486 | 487 | 485 | 1,458 |
| Test | 486 | 487 | 485 | 1,458 |
| **Total** | **4,680** | **4,390** | **4,824** | **13,894** |

The saved summary records 18,949 source metadata rows and 13,894 selected audio rows under a 10 GB local data budget. The shared pipeline converts to mono, resamples to 16 kHz, trims silence, normalizes level, and center-crops or pads to 16 seconds (256,000 samples). Source splits are retained. One selected speaker ID appears in both validation and test; see [`selected_speaker_split_overlap.csv`](data/processed/selected_speaker_split_overlap.csv). Treat the reported test scores with that caveat.

The CSV metadata and metrics are committed, but raw/processed WAV files, model checkpoints, and Hugging Face cache are ignored. A fresh clone needs to acquire audio and train or supply compatible checkpoints before running predictions. Relevant paths:

- [`data/processed/preprocessed_metadata.csv`](data/processed/preprocessed_metadata.csv): audio manifest and preprocessing fields.
- [`outputs/reports/phase1_dataset_summary.json`](outputs/reports/phase1_dataset_summary.json): source and subset counts.
- [`outputs/reports/phase2_preprocessing_summary.json`](outputs/reports/phase2_preprocessing_summary.json): waveform shape and preprocessing summary.
- `data/processed/audio_preprocessed_16s/`: generated training audio, absent from Git.
- `outputs/models/`: generated checkpoints and Hugging Face cache, absent from Git.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The listed dependencies cover data extraction (`duckdb`), audio I/O and resampling (`soundfile`, `soxr`), features and baselines (`numpy`, `scikit-learn`), neural models (`torch`, `transformers`), plots (`matplotlib`), and the local app (`fastapi`, `python-multipart`, `uvicorn`). PyTorch device support depends on the platform and installation.

## Reproduce the pipeline

These commands download data and train models; allow for substantial disk space and compute. This checkout already contains generated metadata and metrics, so the commands use `--overwrite` to regenerate them. Copy those artifacts elsewhere first if you need to keep the saved results.

```bash
# Prepare the balanced local subset and fixed-length audio.
scripts/data.sh --overwrite

# Train MFCC baselines, lightweight CNN, and Phase 6 PhoWhisper models;
# then write the Phase 7 comparison/error analysis.
scripts/train.sh --device auto --overwrite

# Train E6 Whisper-base separately.
.venv/bin/python -m src.training.train_e6_whisper --device auto --overwrite

# Train E1/E2/E3/E5 and reuse existing E4/E6 results in the comparison.
.venv/bin/python -m src.training.train_extended_deep_learning --device auto --full --allow-download --overwrite
```

On Apple Silicon, the provided runners reproduce the later experiments on MPS:

```bash
scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite
scripts/train_e8_whisper_cnn_residual_fusion_mps.sh --overwrite
```

E7 requires the trained E2 checkpoint. E8 also uses the Phase 6 PhoWhisper head checkpoint by default. The E7/E8 runners use a local Hugging Face cache at `outputs/models/hf_cache/`; add `--allow-download` if PhoWhisper is not cached. E3 uses `--allow-download` in the extended runner above for its pretrained encoder. Experiment parameters are recorded in [`configs/experiments/`](configs/experiments/) and the runner scripts. For more options, run `scripts/data.sh --help` or `scripts/train.sh --help`.

## Run the web app

After the required checkpoints and cache exist:

```bash
.venv/bin/python -m uvicorn src.app.main:app --reload
```

Open <http://127.0.0.1:8000/> and upload an audio file. The interface offers `cnn`, `svm`, and `phowhisper` when their local artifacts are available. The default selection is CNN. Set `DEFAULT_MODEL=phowhisper` before starting the server to select the strongest saved research model by default; this still requires its checkpoint and cached encoder. Other overrides are `CNN_CHECKPOINT_PATH`, `SVM_MODEL_PATH`, `PHOWHISPER_CHECKPOINT_PATH`, `PHOWHISPER_CACHE_DIR`, `CNN_DEVICE`, and `PHOWHISPER_DEVICE`.

API endpoints: `GET /health`, `GET /models`, and `POST /predict` with multipart fields `file` and `model`. Example:

```bash
curl -F "file=@sample.wav" -F "model=phowhisper" http://127.0.0.1:8000/predict
```

The score shown by the app is uncalibrated. The system does not identify a speaker, hometown, ethnicity, or other personal background.

## Verify and explore

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m compileall -q src tests
```

Code lives in [`src/`](src/), shell runners in [`scripts/`](scripts/), tests in [`tests/`](tests/), saved figures and metrics in [`outputs/`](outputs/), and research notes in [`docs/`](docs/) and [`outputs/reports/`](outputs/reports/). The figure is an editable, standalone SVG styled with guidance from the locally cloned [figures4papers](https://github.com/ChenLiu-1996/figures4papers) reference; that clone is not required to view it.
