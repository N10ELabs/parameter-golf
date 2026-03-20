#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_VARIANT="${DATA_VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/datasets/fineweb10B_${DATA_VARIANT}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

echo "Repo root: ${REPO_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not installed" >&2
  exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is required but not installed" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "Visible GPUs:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
  echo "nvidia-smi not found; skipping GPU visibility check"
fi

echo
python3 - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
PY

echo
if [ -d "${DATASET_DIR}" ] && compgen -G "${DATASET_DIR}/fineweb_train_*.bin" >/dev/null; then
  echo "Dataset already present at ${DATASET_DIR}"
else
  echo "Downloading cached FineWeb export for variant=${DATA_VARIANT}, train_shards=${TRAIN_SHARDS}"
  python3 data/cached_challenge_fineweb.py --variant "${DATA_VARIANT}" --train-shards "${TRAIN_SHARDS}"
fi

if [ ! -f "${TOKENIZER_PATH}" ]; then
  echo "Tokenizer not found at ${TOKENIZER_PATH}" >&2
  exit 1
fi

echo
echo "Setup complete. Training has not been started."
echo
cat <<EOF
When you are ready to launch the baseline 8-GPU run:

OMP_NUM_THREADS=1 \\
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \\
NCCL_IB_DISABLE=1 \\
RUN_ID=baseline_sp1024_8gpu \\
DATA_PATH=${DATASET_DIR} \\
TOKENIZER_PATH=${TOKENIZER_PATH} \\
VOCAB_SIZE=1024 \\
MAX_WALLCLOCK_SECONDS=600 \\
TRAIN_LOG_EVERY=50 \\
VAL_LOSS_EVERY=200 \\
torchrun --standalone --nproc_per_node=8 ${REPO_ROOT}/train_gpt.py
EOF
