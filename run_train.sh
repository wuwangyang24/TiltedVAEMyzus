#!/usr/bin/env bash
# Training launch script for the Convolutional VAE (bash / SageMaker / Linux).
#
# Usage:
#   chmod +x run_train.sh        # once, to make it executable
#   ./run_train.sh               # run with the defaults below
#
# Override any default via environment variables, e.g.:
#   DATA_DIR=../DATA BATCH_SIZE=512 EPOCHS=50 ./run_train.sh
#
# SECURITY: the W&B key is hard-coded below. Do NOT commit this file to git
# (add run_train.sh to .gitignore). Rotate the key if it leaks.

set -euo pipefail

# ---- Configuration (override via env vars) ----
DATA_DIR="${DATA_DIR:-/home/sagemaker-user/DATA/}"
IMG_SIZE="${IMG_SIZE:-96}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_SPLIT="${VAL_SPLIT:-0.05}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-500000}"
LATENT_DIM="${LATENT_DIM:-128}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-100}"
PRECISION="${PRECISION:-16-mixed}"
PROJECT="${PROJECT:-tilted-vae-myzus}"
ENTITY="${ENTITY:-wangyang-wu-bayer}"
RUN_NAME="${RUN_NAME:-vae-run}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# Hard-coded W&B API key (rotate if leaked; keep this file out of git).
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# Resolve paths relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_CACHE="${SCRIPT_DIR}/cache/image_index.npy"

python "${SCRIPT_DIR}/train.py" \
    --data_dir        "${DATA_DIR}" \
    --img_size        "${IMG_SIZE}" \
    --batch_size      "${BATCH_SIZE}" \
    --num_workers     "${NUM_WORKERS}" \
    --val_split       "${VAL_SPLIT}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" \
    --index_cache     "${INDEX_CACHE}" \
    --latent_dim      "${LATENT_DIM}" \
    --lr              "${LR}" \
    --epochs          "${EPOCHS}" \
    --precision       "${PRECISION}" \
    --anneal_kld \
    --anneal_end      1.0 \
    --anneal_k        3.5e-5 \
    --anneal_x0       200000 \
    --au_threshold    0.01 \
    --project         "${PROJECT}" \
    --entity          "${ENTITY}" \
    --run_name        "${RUN_NAME}" \
    --output_dir      "${OUTPUT_DIR}"
