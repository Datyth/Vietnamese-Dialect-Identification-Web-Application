#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_e8_whisper_cnn_residual_fusion_mps.sh --overwrite [--smoke] [--gated|--concat] [--allow-download]

Train Phase 11 E8 outside the Codex sandbox with Apple MPS:
  E8: frozen PhoWhisper-base encoder + residual-gated EfficientNetB0 local
      correction + PhoWhisper baseline projector/classifier warm-start.

Run from a normal Terminal window so PyTorch can see the Apple Metal device.

Options:
  --overwrite      Regenerate existing E8 checkpoint, metrics, report, and figures.
  --smoke          Use a tiny balanced subset and one epoch for a quick check.
  --gated          Use legacy gated fusion ablation instead of residual_gated.
  --concat         Use concat fusion ablation instead of residual_gated.
  --skip-head-warm-start
                   Do not load the PhoWhisper baseline projector/classifier.
  --allow-download Allow Hugging Face downloads for vinai/PhoWhisper-base.
  -h, --help       Show this help message.

Environment overrides:
  PYTHON_BIN       default: .venv/bin/python
  DEVICE           default: mps
  SEED             default: 42
  BATCH_SIZE       default: 14
  E8_MAX_EPOCHS    default: 20
  PATIENCE         default: 10
  FUSION_LR        default: 1e-4
  HEAD_LR          default: 3e-5
  CNN_LR           default: 1e-5
  CNN_TRAINABLE_LAYERS default: 2
  WEIGHT_DECAY     default: 1e-4
  DROPOUT          default: 0.0
  BETA_INIT        default: 0.1
  LOCAL_EMBED_DIM  default: 128
  FUSION_DIM       default: 512
  CLASSIFIER_HIDDEN_DIM default: 256
  FUSION_TYPE      default: residual_gated
  BEST_SCORE_TYPE  default: hybrid_macro_central
  LATENCY_SAMPLES  default: 5
  SMOKE_LIMIT_PER_SPLIT default: 6
  SMOKE_EPOCHS     default: 1
  MODEL_ID         default: vinai/PhoWhisper-base
  CACHE_DIR        default: outputs/models/hf_cache
  CNN_CHECKPOINT   default: outputs/models/e2_efficientnetb0_logmel.pt
  PHOWHISPER_HEAD_CHECKPOINT default: outputs/models/phowhisper_pretrained_frozen_encoder.pt
EOF
}

OVERWRITE=0
SMOKE=0
ALLOW_DOWNLOAD=0
SKIP_HEAD_WARM_START=0

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
    --gated)
      FUSION_TYPE="gated"
      shift
      ;;
    --concat)
      FUSION_TYPE="concat"
      shift
      ;;
    --skip-head-warm-start)
      SKIP_HEAD_WARM_START=1
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
  echo "Refusing to overwrite existing E8 artifacts without --overwrite." >&2
  echo "Run: scripts/train_e8_whisper_cnn_residual_fusion_mps.sh --overwrite" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-14}"
E8_MAX_EPOCHS="${E8_MAX_EPOCHS:-20}"
PATIENCE="${PATIENCE:-10}"
FUSION_LR="${FUSION_LR:-1e-4}"
HEAD_LR="${HEAD_LR:-3e-5}"
CNN_LR="${CNN_LR:-1e-5}"
CNN_TRAINABLE_LAYERS="${CNN_TRAINABLE_LAYERS:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DROPOUT="${DROPOUT:-0.0}"
BETA_INIT="${BETA_INIT:-0.1}"
LOCAL_EMBED_DIM="${LOCAL_EMBED_DIM:-128}"
FUSION_DIM="${FUSION_DIM:-512}"
CLASSIFIER_HIDDEN_DIM="${CLASSIFIER_HIDDEN_DIM:-256}"
LATENCY_SAMPLES="${LATENCY_SAMPLES:-5}"
MODEL_ID="${MODEL_ID:-vinai/PhoWhisper-base}"
CACHE_DIR="${CACHE_DIR:-outputs/models/hf_cache}"
CNN_CHECKPOINT="${CNN_CHECKPOINT:-outputs/models/e2_efficientnetb0_logmel.pt}"
PHOWHISPER_HEAD_CHECKPOINT="${PHOWHISPER_HEAD_CHECKPOINT:-outputs/models/phowhisper_pretrained_frozen_encoder.pt}"
FUSION_TYPE="${FUSION_TYPE:-residual_gated}"
BEST_SCORE_TYPE="${BEST_SCORE_TYPE:-hybrid_macro_central}"

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
  echo "Run E2 first, then rerun E8." >&2
  exit 1
fi
if [[ "${FUSION_TYPE}" == "residual_gated" && "${SKIP_HEAD_WARM_START}" != "1" && ! -f "${PHOWHISPER_HEAD_CHECKPOINT}" ]]; then
  echo "Missing PhoWhisper head checkpoint: ${PHOWHISPER_HEAD_CHECKPOINT}" >&2
  echo "Run E4/Phase 6 first, set PHOWHISPER_HEAD_CHECKPOINT, or pass --skip-head-warm-start." >&2
  exit 1
fi
if [[ "${SMOKE}" == "1" ]]; then
  E8_MAX_EPOCHS="${SMOKE_EPOCHS:-1}"
fi
case "${FUSION_TYPE}" in
  concat|gated|residual_gated)
    ;;
  *)
    echo "Invalid FUSION_TYPE=${FUSION_TYPE}; expected concat, gated, or residual_gated." >&2
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
echo "  e8_max_epochs=${E8_MAX_EPOCHS}"
echo "  patience=${PATIENCE}"
echo "  fusion_learning_rate=${FUSION_LR}"
echo "  head_learning_rate=${HEAD_LR}"
echo "  cnn_learning_rate=${CNN_LR}"
echo "  cnn_trainable_layers=${CNN_TRAINABLE_LAYERS}"
echo "  weight_decay=${WEIGHT_DECAY}"
echo "  dropout=${DROPOUT}"
echo "  beta_init=${BETA_INIT}"
echo "  local_embed_dim=${LOCAL_EMBED_DIM}"
echo "  fusion_dim=${FUSION_DIM}"
echo "  classifier_hidden_dim=${CLASSIFIER_HIDDEN_DIM}"
echo "  fusion_type=${FUSION_TYPE}"
echo "  best_score_type=${BEST_SCORE_TYPE}"
echo "  model_id=${MODEL_ID}"
echo "  cache_dir=${CACHE_DIR}"
echo "  latency_samples=${LATENCY_SAMPLES}"
echo "  cnn_checkpoint=${CNN_CHECKPOINT}"
echo "  phowhisper_head_checkpoint=${PHOWHISPER_HEAD_CHECKPOINT}"
echo "  skip_head_warm_start=${SKIP_HEAD_WARM_START}"
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

E8_CMD=(
  run_python_module src.training.train_e8_whisper_cnn_residual_fusion
  --batch-size "${BATCH_SIZE}"
  --max-epochs "${E8_MAX_EPOCHS}"
  --patience "${PATIENCE}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --model-id "${MODEL_ID}"
  --cache-dir "${CACHE_DIR}"
  --learning-rate "${FUSION_LR}"
  --head-learning-rate "${HEAD_LR}"
  --cnn-learning-rate "${CNN_LR}"
  --cnn-trainable-layers "${CNN_TRAINABLE_LAYERS}"
  --weight-decay "${WEIGHT_DECAY}"
  --dropout "${DROPOUT}"
  --beta-init "${BETA_INIT}"
  --local-embedding-dim "${LOCAL_EMBED_DIM}"
  --fusion-dim "${FUSION_DIM}"
  --classifier-hidden-dim "${CLASSIFIER_HIDDEN_DIM}"
  --cnn-checkpoint-path "${CNN_CHECKPOINT}"
  --phowhisper-head-checkpoint-path "${PHOWHISPER_HEAD_CHECKPOINT}"
  --fusion-type "${FUSION_TYPE}"
  --best-score-type "${BEST_SCORE_TYPE}"
  --latency-samples "${LATENCY_SAMPLES}"
  --overwrite
)
if [[ "${SMOKE}" == "1" ]]; then
  E8_CMD+=(--limit-per-split "${SMOKE_LIMIT_PER_SPLIT:-6}")
fi
if [[ "${ALLOW_DOWNLOAD}" == "1" ]]; then
  E8_CMD+=(--allow-download)
fi
if [[ "${SKIP_HEAD_WARM_START}" == "1" ]]; then
  E8_CMD+=(--skip-phowhisper-head-warm-start)
fi

echo
echo "==> E8: frozen PhoWhisper-base + residual-gated E2 EfficientNetB0 correction"
"${E8_CMD[@]}"

echo
echo "Done. Key outputs:"
echo "  outputs/metrics/e8_whisper_cnn_residual_fusion_results.json"
echo "  outputs/models/e8_whisper_cnn_residual_fusion.pt"
echo "  outputs/reports/phase11_whisper_cnn_residual_fusion_report.md"
echo "  outputs/metrics/model_method_comparison.csv"
