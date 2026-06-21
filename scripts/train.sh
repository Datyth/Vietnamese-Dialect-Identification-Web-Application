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

For an already-downloaded PhoWhisper cache, offline execution can be requested:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/train.sh --overwrite
EOF
}

DEVICE="auto"
OVERWRITE_ARGS=()
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

run_step \
  "Phase 4: MFCC baselines" \
  "${PYTHON_BIN}" -m src.training.train_baseline "${OVERWRITE_ARGS[@]}"

run_step \
  "Phase 5: lightweight custom CNN" \
  "${PYTHON_BIN}" -m src.training.train_cnn \
  --device "${DEVICE}" "${OVERWRITE_ARGS[@]}"

run_step \
  "Phase 6a: PhoWhisper pretrained frozen encoder" \
  "${PYTHON_BIN}" -m src.training.train_phowhisper \
  --training-mode frozen_encoder \
  --learning-rate 1e-3 \
  --max-epochs 20 \
  --patience 5 \
  --device "${DEVICE}" \
  "${OVERWRITE_ARGS[@]}"

run_step \
  "Phase 6b: PhoWhisper full fine-tuning" \
  "${PYTHON_BIN}" -m src.training.train_phowhisper \
  --training-mode full_fine_tune \
  --device "${DEVICE}" \
  "${OVERWRITE_ARGS[@]}"

run_step \
  "Phase 7: final comparison and error analysis" \
  "${PYTHON_BIN}" -m src.evaluation.final_evaluation "${OVERWRITE_ARGS[@]}"

echo
echo "Training pipeline completed successfully."
