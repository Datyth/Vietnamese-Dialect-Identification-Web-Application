# Vietnamese Dialect Identification

A Python project for identifying three broad Vietnamese speech regions from an audio recording: **Northern**, **Central**, and **Southern**. It includes data preparation, model training, evaluation, and a local web app for trying predictions.

## Features

- A shared audio pipeline for training and inference
- Traditional and neural classification models
- A FastAPI web interface for uploading audio and viewing a prediction
- Scripts for preparing data, training models, and running checks

See the [system architecture diagram](outputs/figures/system_architecture.svg) for the main workflow.

## Requirements

- Python 3.10 or newer
- Enough local storage for the speech dataset and model files
- A compatible PyTorch installation for your CPU or accelerator

## Installation

From the repository root:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Usage

Prepare the dataset and train the main models:

```bash
scripts/data.sh --overwrite
scripts/train.sh --device auto --overwrite
```

These commands download data and create local audio and model files. They take time and can replace existing generated results. Run each script with `--help` for available options.

Start the web app after the required model files are available:

```bash
.venv/bin/python -m uvicorn src.app.main:app --reload
```

Open <http://127.0.0.1:8000/>. Upload an audio file, choose an available model, and select **Predict**. The app offers SVM, a lightweight CNN, and PhoWhisper when their local model files are present.

The API also provides `GET /health`, `GET /models`, and `POST /predict`. For example:

```bash
curl -F "file=@sample.wav" -F "model=cnn" http://127.0.0.1:8000/predict
```

## Project layout

| Path | Purpose |
| --- | --- |
| `src/` | Audio, features, models, training, inference, and web app code |
| `scripts/` | Data and training commands |
| `configs/` | Label mapping and experiment settings |
| `tests/` | Automated tests |
| `data/` | Metadata and locally generated audio |
| `outputs/` | Metrics, figures, and locally generated model files |

The dataset is [ViMD](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset). Check its license and terms before using or redistributing the data. Audio files, model checkpoints, and the local `figures4papers` reference clone are not included in Git.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Predictions are broad regional estimates, not a way to identify a speaker or their hometown. Displayed scores are not calibrated probabilities.
