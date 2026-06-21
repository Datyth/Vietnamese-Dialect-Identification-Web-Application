#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train.sh [--device auto|mps|cuda|cpu] [--overwrite]

Run the complete training and evaluation pipeline:
  1. MFCC Logistic Regression and SVM baselines.
  2. Lightweight custom CNN.
  3. PhoWhisper with a frozen pretrained encoder.
  4. PhoWhisper full fine-tuning.
  5. Final comparison and error analysis.

Options:
  --device     PyTorch device for CNN and PhoWhisper (default: auto).
  --overwrite  Regenerate existing checkpoints, metrics, and reports.
  -h, --help   Show this help message.

Environment:
  PYTHON_BIN   Python executable to use (default: .venv/bin/python).

Training parameters can be overridden with environment variables:
  TRAINING_SEED                         default: 42
  CNN_BATCH_SIZE                        default: 16
  CNN_MAX_EPOCHS                        default: 40
  CNN_PATIENCE                          default: 8
  CNN_LEARNING_RATE                     default: 1e-3
  CNN_WEIGHT_DECAY                      default: 1e-4
  PHOWHISPER_FROZEN_BATCH_SIZE          default: 2
  PHOWHISPER_FROZEN_MAX_EPOCHS          default: 20
  PHOWHISPER_FROZEN_PATIENCE            default: 5
  PHOWHISPER_FROZEN_LEARNING_RATE       default: 1e-3
  PHOWHISPER_FROZEN_WEIGHT_DECAY        default: 0.01
  PHOWHISPER_FINE_TUNE_BATCH_SIZE       default: 2
  PHOWHISPER_FINE_TUNE_MAX_EPOCHS       default: 8
  PHOWHISPER_FINE_TUNE_PATIENCE         default: 3
  PHOWHISPER_FINE_TUNE_LEARNING_RATE    default: 1e-5
  PHOWHISPER_FINE_TUNE_WEIGHT_DECAY     default: 0.01

For an already-downloaded PhoWhisper cache, offline execution can be requested:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/train.sh --overwrite
EOF
}

DEVICE="auto"
OVERWRITE_ARGS=()

# Keep training hyperparameters in one place. Environment overrides make
# experiments configurable without changing this script.
TRAINING_SEED="${TRAINING_SEED:-42}"

CNN_BATCH_SIZE="${CNN_BATCH_SIZE:-16}"
CNN_MAX_EPOCHS="${CNN_MAX_EPOCHS:-40}"
CNN_PATIENCE="${CNN_PATIENCE:-8}"
CNN_LEARNING_RATE="${CNN_LEARNING_RATE:-1e-3}"
CNN_WEIGHT_DECAY="${CNN_WEIGHT_DECAY:-1e-4}"

PHOWHISPER_FROZEN_BATCH_SIZE="${PHOWHISPER_FROZEN_BATCH_SIZE:-2}"
PHOWHISPER_FROZEN_MAX_EPOCHS="${PHOWHISPER_FROZEN_MAX_EPOCHS:-20}"
PHOWHISPER_FROZEN_PATIENCE="${PHOWHISPER_FROZEN_PATIENCE:-5}"
PHOWHISPER_FROZEN_LEARNING_RATE="${PHOWHISPER_FROZEN_LEARNING_RATE:-1e-3}"
PHOWHISPER_FROZEN_WEIGHT_DECAY="${PHOWHISPER_FROZEN_WEIGHT_DECAY:-0.01}"

PHOWHISPER_FINE_TUNE_BATCH_SIZE="${PHOWHISPER_FINE_TUNE_BATCH_SIZE:-2}"
PHOWHISPER_FINE_TUNE_MAX_EPOCHS="${PHOWHISPER_FINE_TUNE_MAX_EPOCHS:-8}"
PHOWHISPER_FINE_TUNE_PATIENCE="${PHOWHISPER_FINE_TUNE_PATIENCE:-3}"
PHOWHISPER_FINE_TUNE_LEARNING_RATE="${PHOWHISPER_FINE_TUNE_LEARNING_RATE:-1e-5}"
PHOWHISPER_FINE_TUNE_WEIGHT_DECAY="${PHOWHISPER_FINE_TUNE_WEIGHT_DECAY:-0.01}"

while (($# > 0)); do
  case "$1" in
    --device)
      if (($# < 2)); then
        echo "--device requires a value." >&2
        exit 2
      fi
      DEVICE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE_ARGS=(--overwrite)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${DEVICE}" in
  auto|mps|cuda|cpu) ;;
  *)
    echo "Unsupported device: ${DEVICE}. Use auto, mps, cuda, or cpu." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

cd "${ROOT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create .venv and install requirements, or set PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -f data/processed/preprocessed_metadata.csv ]]; then
  echo "Missing data/processed/preprocessed_metadata.csv." >&2
  echo "Run scripts/data.sh first." >&2
  exit 1
fi

run_step() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  "$@"
}

echo "Training parameters:"
echo "  seed=${TRAINING_SEED}"
echo "  cnn: batch_size=${CNN_BATCH_SIZE}, max_epochs=${CNN_MAX_EPOCHS}, patience=${CNN_PATIENCE}, learning_rate=${CNN_LEARNING_RATE}, weight_decay=${CNN_WEIGHT_DECAY}"
echo "  phowhisper_frozen: batch_size=${PHOWHISPER_FROZEN_BATCH_SIZE}, max_epochs=${PHOWHISPER_FROZEN_MAX_EPOCHS}, patience=${PHOWHISPER_FROZEN_PATIENCE}, learning_rate=${PHOWHISPER_FROZEN_LEARNING_RATE}, weight_decay=${PHOWHISPER_FROZEN_WEIGHT_DECAY}"
echo "  phowhisper_fine_tune: batch_size=${PHOWHISPER_FINE_TUNE_BATCH_SIZE}, max_epochs=${PHOWHISPER_FINE_TUNE_MAX_EPOCHS}, patience=${PHOWHISPER_FINE_TUNE_PATIENCE}, learning_rate=${PHOWHISPER_FINE_TUNE_LEARNING_RATE}, weight_decay=${PHOWHISPER_FINE_TUNE_WEIGHT_DECAY}"

run_step \
  "Phase 4: MFCC baselines" \
  "${PYTHON_BIN}" -m src.training.train_baseline \
  --seed "${TRAINING_SEED}" \
  "${OVERWRITE_ARGS[@]}"

run_step \
  "Phase 5: lightweight custom CNN" \
  "${PYTHON_BIN}" -m src.training.train_cnn \
  --device "${DEVICE}" \
  --seed "${TRAINING_SEED}" \
  --batch-size "${CNN_BATCH_SIZE}" \
  --max-epochs "${CNN_MAX_EPOCHS}" \
  --patience "${CNN_PATIENCE}" \
  --learning-rate "${CNN_LEARNING_RATE}" \
  --weight-decay "${CNN_WEIGHT_DECAY}" \
  "${OVERWRITE_ARGS[@]}"

run_step \
  "Phase 6a: PhoWhisper pretrained frozen encoder" \
  "${PYTHON_BIN}" -m src.training.train_phowhisper \
  --training-mode frozen_encoder \
  --device "${DEVICE}" \
  --seed "${TRAINING_SEED}" \
  --batch-size "${PHOWHISPER_FROZEN_BATCH_SIZE}" \
  --max-epochs "${PHOWHISPER_FROZEN_MAX_EPOCHS}" \
  --patience "${PHOWHISPER_FROZEN_PATIENCE}" \
  --learning-rate "${PHOWHISPER_FROZEN_LEARNING_RATE}" \
  --weight-decay "${PHOWHISPER_FROZEN_WEIGHT_DECAY}" \
  "${OVERWRITE_ARGS[@]}"

run_step \
  "Phase 6b: PhoWhisper full fine-tuning" \
  "${PYTHON_BIN}" -m src.training.train_phowhisper \
  --training-mode full_fine_tune \
  --device "${DEVICE}" \
  --seed "${TRAINING_SEED}" \
  --batch-size "${PHOWHISPER_FINE_TUNE_BATCH_SIZE}" \
  --max-epochs "${PHOWHISPER_FINE_TUNE_MAX_EPOCHS}" \
  --patience "${PHOWHISPER_FINE_TUNE_PATIENCE}" \
  --learning-rate "${PHOWHISPER_FINE_TUNE_LEARNING_RATE}" \
  --weight-decay "${PHOWHISPER_FINE_TUNE_WEIGHT_DECAY}" \
  "${OVERWRITE_ARGS[@]}"

run_step \
  "Phase 7: final comparison and error analysis" \
  "${PYTHON_BIN}" -m src.evaluation.final_evaluation "${OVERWRITE_ARGS[@]}"

echo
echo "Training pipeline completed successfully."
