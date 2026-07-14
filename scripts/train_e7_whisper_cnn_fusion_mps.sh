#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite [--smoke] [--concat] [--allow-download]

Train Phase 10 E7 outside the Codex sandbox with Apple MPS:
  E7: frozen PhoWhisper-base encoder + lightly fine-tuned trained E2
      EfficientNetB0 branch + trainable projection/fusion/classification head.

Run from a normal Terminal window so PyTorch can see the Apple Metal device.

Options:
  --overwrite      Regenerate existing E7 checkpoint, metrics, report, and figures.
  --smoke          Use a tiny balanced subset and one epoch for a quick check.
  --concat         Use concat fusion instead of default gated fusion.
  --allow-download Allow Hugging Face downloads for vinai/PhoWhisper-base.
  -h, --help       Show this help message.

Environment overrides:
  PYTHON_BIN       default: .venv/bin/python
  DEVICE           default: mps
  SEED             default: 42
  BATCH_SIZE       default: 16
  E7_MAX_EPOCHS    default: 20
  PATIENCE         default: 10
  E7_LR            default: 1e-4
  CNN_LR           default: 1e-5
  CNN_TRAINABLE_LAYERS default: 2
  WEIGHT_DECAY     default: 1e-4
  DROPOUT          default: 0.0
  LOCAL_EMBED_DIM  default: 128
  FUSION_DIM       default: 512
  CLASSIFIER_HIDDEN_DIM default: 256
  FUSION_TYPE      default: gated
  LATENCY_SAMPLES  default: 5
  SMOKE_LIMIT_PER_SPLIT default: 6
  SMOKE_EPOCHS     default: 1
  MODEL_ID         default: vinai/PhoWhisper-base
  CACHE_DIR        default: outputs/models/hf_cache
  CNN_CHECKPOINT   default: outputs/models/e2_efficientnetb0_logmel.pt
EOF
}

OVERWRITE=0
SMOKE=0
CONCAT=0
ALLOW_DOWNLOAD=0

while (($# > 0)); do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --concat)
      CONCAT=1
      shift
      ;;
    --allow-download)
      ALLOW_DOWNLOAD=1
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

if [[ "${OVERWRITE}" != "1" ]]; then
  echo "Refusing to overwrite existing E7 artifacts without --overwrite." >&2
  echo "Run: scripts/train_e7_whisper_cnn_fusion_mps.sh --overwrite" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
E7_MAX_EPOCHS="${E7_MAX_EPOCHS:-20}"
PATIENCE="${PATIENCE:-10}"
E7_LR="${E7_LR:-1e-4}"
CNN_LR="${CNN_LR:-1e-5}"
CNN_TRAINABLE_LAYERS="${CNN_TRAINABLE_LAYERS:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DROPOUT="${DROPOUT:-0.0}"
LOCAL_EMBED_DIM="${LOCAL_EMBED_DIM:-128}"
FUSION_DIM="${FUSION_DIM:-512}"
CLASSIFIER_HIDDEN_DIM="${CLASSIFIER_HIDDEN_DIM:-256}"
LATENCY_SAMPLES="${LATENCY_SAMPLES:-5}"
MODEL_ID="${MODEL_ID:-vinai/PhoWhisper-base}"
CACHE_DIR="${CACHE_DIR:-outputs/models/hf_cache}"
CNN_CHECKPOINT="${CNN_CHECKPOINT:-outputs/models/e2_efficientnetb0_logmel.pt}"
FUSION_TYPE="${FUSION_TYPE:-gated}"

cd "${ROOT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f data/processed/preprocessed_metadata.csv ]]; then
  echo "Missing data/processed/preprocessed_metadata.csv." >&2
  echo "Run scripts/data.sh first." >&2
  exit 1
fi
if [[ ! -f "${CNN_CHECKPOINT}" ]]; then
  echo "Missing trained E2 EfficientNetB0 checkpoint: ${CNN_CHECKPOINT}" >&2
  echo "Run E2 first, then rerun E7." >&2
  exit 1
fi
if [[ "${SMOKE}" == "1" ]]; then
  E7_MAX_EPOCHS="${SMOKE_EPOCHS:-1}"
fi
if [[ "${CONCAT}" == "1" ]]; then
  FUSION_TYPE="concat"
fi
case "${FUSION_TYPE}" in
  concat|gated)
    ;;
  *)
    echo "Invalid FUSION_TYPE=${FUSION_TYPE}; expected concat or gated." >&2
    exit 2
    ;;
esac

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${DEVICE}" == "mps" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"mps_is_built={torch.backends.mps.is_built()}")
print(f"mps_is_available={torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit(
        "PyTorch cannot see MPS. Run this script from a normal Terminal, "
        "not inside the Codex sandbox."
    )
try:
    value = torch.ones(1, device="mps").cpu().item()
except Exception as exc:
    raise SystemExit(f"MPS allocation failed: {type(exc).__name__}: {exc}") from exc
print(f"mps_allocation_ok={value:.1f}")
PY
fi

echo
echo "Run settings:"
echo "  device=${DEVICE}"
echo "  seed=${SEED}"
echo "  batch_size=${BATCH_SIZE}"
echo "  e7_max_epochs=${E7_MAX_EPOCHS}"
echo "  patience=${PATIENCE}"
echo "  learning_rate=${E7_LR}"
echo "  cnn_learning_rate=${CNN_LR}"
echo "  cnn_trainable_layers=${CNN_TRAINABLE_LAYERS}"
echo "  weight_decay=${WEIGHT_DECAY}"
echo "  dropout=${DROPOUT}"
echo "  local_embed_dim=${LOCAL_EMBED_DIM}"
echo "  fusion_dim=${FUSION_DIM}"
echo "  classifier_hidden_dim=${CLASSIFIER_HIDDEN_DIM}"
echo "  fusion_type=${FUSION_TYPE}"
echo "  model_id=${MODEL_ID}"
echo "  cache_dir=${CACHE_DIR}"
echo "  latency_samples=${LATENCY_SAMPLES}"
echo "  cnn_checkpoint=${CNN_CHECKPOINT}"
echo "  pytorch_mps_fallback=${PYTORCH_ENABLE_MPS_FALLBACK}"
if [[ "${SMOKE}" == "1" ]]; then
  echo "  smoke_limit_per_split=${SMOKE_LIMIT_PER_SPLIT:-6}"
fi

run_python_module() {
  local module_name="$1"
  shift
  "${PYTHON_BIN}" - "${module_name}" "$@" <<'PY'
import os
import runpy
import sys
import traceback

module_name = sys.argv[1]
module_args = sys.argv[2:]
sys.argv = [module_name, *module_args]
exit_code = 0
try:
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
except SystemExit as exc:
    if exc.code is None:
        exit_code = 0
    elif isinstance(exc.code, int):
        exit_code = exc.code
    else:
        print(exc.code, file=sys.stderr)
        exit_code = 1
except BaseException:
    traceback.print_exc()
    exit_code = 1
sys.stdout.flush()
sys.stderr.flush()
os._exit(exit_code)
PY
}

E7_CMD=(
  run_python_module src.training.train_e7_whisper_cnn_fusion
  --batch-size "${BATCH_SIZE}"
  --max-epochs "${E7_MAX_EPOCHS}"
  --patience "${PATIENCE}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --model-id "${MODEL_ID}"
  --cache-dir "${CACHE_DIR}"
  --learning-rate "${E7_LR}"
  --cnn-learning-rate "${CNN_LR}"
  --cnn-trainable-layers "${CNN_TRAINABLE_LAYERS}"
  --weight-decay "${WEIGHT_DECAY}"
  --dropout "${DROPOUT}"
  --local-embedding-dim "${LOCAL_EMBED_DIM}"
  --fusion-dim "${FUSION_DIM}"
  --classifier-hidden-dim "${CLASSIFIER_HIDDEN_DIM}"
  --cnn-checkpoint-path "${CNN_CHECKPOINT}"
  --fusion-type "${FUSION_TYPE}"
  --latency-samples "${LATENCY_SAMPLES}"
  --overwrite
)
if [[ "${SMOKE}" == "1" ]]; then
  E7_CMD+=(--limit-per-split "${SMOKE_LIMIT_PER_SPLIT:-6}")
fi
if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
  E7_CMD+=(--allow-download)
fi
echo
echo "==> E7: frozen PhoWhisper-base + lightly fine-tuned E2 EfficientNetB0 fusion head"
"${E7_CMD[@]}"

echo
echo "Done. Key outputs:"
echo "  outputs/metrics/e7_whisper_cnn_fusion_results.json"
echo "  outputs/models/e7_whisper_cnn_fusion.pt"
echo "  outputs/reports/phase10_whisper_cnn_fusion_report.md"
echo "  outputs/metrics/model_method_comparison.csv"
