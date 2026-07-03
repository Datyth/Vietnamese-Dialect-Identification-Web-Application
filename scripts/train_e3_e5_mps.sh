#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_e3_e5_mps.sh --overwrite [--smoke] [--skip-e3] [--skip-e5] [--no-download]

Train Phase 9 E3 and E5 outside the Codex sandbox with Apple MPS:
  E3: frozen Vietnamese wav2vec2 encoder + trainable classifier head.
  E5: local ChunkFormer-style waveform classifier.

Run from a normal Terminal window so PyTorch can see the Apple Metal device.

Options:
  --overwrite    Regenerate existing E3/E5 checkpoints, metrics, and figures.
  --smoke        Use a tiny balanced subset and one epoch for a quick check.
  --skip-e3      Do not run E3 wav2vec2.
  --skip-e5      Do not run E5 ChunkFormer-style classifier.
  --no-download  Require the E3 wav2vec2 checkpoint to already be cached.
  -h, --help     Show this help message.

Environment overrides:
  PYTHON_BIN        default: .venv/bin/python
  DEVICE            default: mps
  BATCH_SIZE        default: 4
  E3_MAX_EPOCHS     default: 20
  E5_MAX_EPOCHS     default: 20
  PATIENCE          default: 5
  E3_LR_HEAD        default: 1e-3
  E5_LR             default: 1e-4
  WEIGHT_DECAY      default: 1e-4
  LATENCY_SAMPLES   default: 5
EOF
}

OVERWRITE=0
ALLOW_DOWNLOAD=1
RUN_E3=1
RUN_E5=1
SMOKE=0

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
    --skip-e3)
      RUN_E3=0
      shift
      ;;
    --skip-e5)
      RUN_E5=0
      shift
      ;;
    --no-download)
      ALLOW_DOWNLOAD=0
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
  echo "Refusing to overwrite existing E3/E5 artifacts without --overwrite." >&2
  echo "Run: scripts/train_e3_e5_mps.sh --overwrite" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-4}"
E3_MAX_EPOCHS="${E3_MAX_EPOCHS:-20}"
E5_MAX_EPOCHS="${E5_MAX_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
E3_LR_HEAD="${E3_LR_HEAD:-1e-3}"
E5_LR="${E5_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LATENCY_SAMPLES="${LATENCY_SAMPLES:-5}"

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

if [[ "${SMOKE}" == "1" ]]; then
  E3_MAX_EPOCHS="${SMOKE_EPOCHS:-1}"
  E5_MAX_EPOCHS="${SMOKE_EPOCHS:-1}"
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${DEVICE}" == "mps" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys
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
echo "  batch_size=${BATCH_SIZE}"
echo "  e3_max_epochs=${E3_MAX_EPOCHS}"
echo "  e5_max_epochs=${E5_MAX_EPOCHS}"
echo "  patience=${PATIENCE}"
echo "  pytorch_mps_fallback=${PYTORCH_ENABLE_MPS_FALLBACK}"
if [[ "${SMOKE}" == "1" ]]; then
  echo "  smoke_limit_per_split=${SMOKE_LIMIT_PER_SPLIT:-6}"
fi

run_step() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  "$@"
}

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

if [[ "${RUN_E3}" == "1" ]]; then
  E3_CMD=(
    run_python_module src.training.train_e3_wav2vec2
    --full
  )
  if [[ "${SMOKE}" == "1" ]]; then
    E3_CMD+=(--limit-per-split "${SMOKE_LIMIT_PER_SPLIT:-6}")
  fi
  E3_CMD+=(
    --batch-size "${BATCH_SIZE}"
    --max-epochs "${E3_MAX_EPOCHS}"
    --patience "${PATIENCE}"
    --device "${DEVICE}"
    --learning-rate-head "${E3_LR_HEAD}"
    --weight-decay "${WEIGHT_DECAY}"
    --latency-samples "${LATENCY_SAMPLES}"
  )
  if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
    E3_CMD+=(--allow-download)
  fi
  E3_CMD+=(--overwrite)
  run_step \
    "E3: wav2vec2 Vietnamese frozen encoder + classifier head" \
    "${E3_CMD[@]}"
fi

if [[ "${RUN_E5}" == "1" ]]; then
  E5_CMD=(
    run_python_module src.training.train_e5_vipvl_chunkformer
    --full
  )
  if [[ "${SMOKE}" == "1" ]]; then
    E5_CMD+=(--limit-per-split "${SMOKE_LIMIT_PER_SPLIT:-6}")
  fi
  E5_CMD+=(
    --batch-size "${BATCH_SIZE}"
    --max-epochs "${E5_MAX_EPOCHS}"
    --patience "${PATIENCE}"
    --device "${DEVICE}"
    --learning-rate "${E5_LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --latency-samples "${LATENCY_SAMPLES}"
    --overwrite
  )
  run_step \
    "E5: local ChunkFormer-style waveform classifier" \
    "${E5_CMD[@]}"
fi

echo
echo "Done. Key outputs:"
echo "  outputs/metrics/e3_wav2vec2_results.json"
echo "  outputs/metrics/e5_vipvl_chunkformer_results.json"
echo "  outputs/metrics/deep_learning_comparison.csv"
echo "  outputs/metrics/model_method_comparison.csv"
echo "  outputs/figures/model_method_comparison_metrics.png"
echo "  outputs/figures/model_method_comparison_tradeoffs.png"
echo "  outputs/reports/extended_deep_learning_experiments.md"
