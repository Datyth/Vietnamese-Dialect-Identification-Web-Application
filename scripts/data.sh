#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/data.sh [options]

Run the complete data pipeline:
  1. Prepare ViMD metadata and acquire the balanced audio subset.
  2. Preprocess audio to mono 16 kHz, fixed 16-second WAV files.

Options:
  --max-data-bytes N   Maximum size of the data directory in bytes.
  --train-per-label N  Training samples to download per dialect label.
  --valid-per-label N  Validation samples to download per dialect label.
  --test-per-label N   Test samples to download per dialect label.
  --seed N             Dataset selection seed (default in Python: 42).
  --metadata-only      Prepare metadata without downloading or preprocessing audio.
  --overwrite          Regenerate existing metadata, reports, and preprocessed audio.
  -h, --help           Show this help message.

Environment:
  PYTHON_BIN   Python executable to use (default: .venv/bin/python).
EOF
}

OVERWRITE_ARGS=()
PREPARE_ARGS=()
METADATA_ONLY=0
while (($# > 0)); do
  case "$1" in
    --max-data-bytes|--train-per-label|--valid-per-label|--test-per-label|--seed)
      if (($# < 2)); then
        echo "$1 requires a value." >&2
        exit 2
      fi
      PREPARE_ARGS+=("$1" "$2")
      shift 2
      ;;
    --metadata-only)
      PREPARE_ARGS+=(--metadata-only)
      METADATA_ONLY=1
      shift
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

cd "${ROOT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create .venv and install requirements, or set PYTHON_BIN." >&2
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
  "Phase 1: metadata preparation and balanced audio acquisition" \
  "${PYTHON_BIN}" -m src.data.prepare_metadata \
  "${PREPARE_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

if ((METADATA_ONLY == 1)); then
  echo
  echo "Metadata-only pipeline completed successfully; audio preprocessing skipped."
  exit 0
fi

run_step \
  "Phase 2/3: audio preprocessing and minimal EDA" \
  "${PYTHON_BIN}" -m src.data.preprocess_audio "${OVERWRITE_ARGS[@]}"

echo
echo "Data pipeline completed successfully."
