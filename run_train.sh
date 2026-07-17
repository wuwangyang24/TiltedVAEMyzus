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
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_SPLIT="${VAL_SPLIT:-0.05}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-500000}"
MODEL="${MODEL:-tilted}"
LATENT_DIM="${LATENT_DIM:-128}"
TAU="${TAU:-}"
LR="${LR:-1e-4}"
KLD_WEIGHT="${KLD_WEIGHT:-0.0005}"
EPOCHS="${EPOCHS:-50}"
PRECISION="${PRECISION:-16-mixed}"
PROJECT="${PROJECT:-tilted-vae-myzus}"
ENTITY="${ENTITY:-wangyang-wu-bayer}"
RUN_NAME="${RUN_NAME:-${MODEL}-run}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# Hard-coded W&B API key (rotate if leaked; keep this file out of git).
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# W&B sync mode. Use "offline" to log locally without network (avoids the
# training hang from a blocked/slow connection); sync later with `wandb sync`.
# Set WANDB_MODE=online to stream live.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Resolve paths relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_CACHE="${SCRIPT_DIR}/cache/image_index.npy"

# ---- Chemical-class classifier callback ----
CLS_IMAGE_METADATA="${CLS_IMAGE_METADATA:-/home/sagemaker-user/METADATA/metadata_compound_all100ppm.json}"
CLS_LABEL_METADATA="${CLS_LABEL_METADATA:-/home/sagemaker-user/METADATA/synthesisprogram_compoundno.csv}"
CLS_ROOT_DIR="${CLS_ROOT_DIR:-/home/sagemaker-user/DATA_TEST/}"
CLS_EVERY_N_EPOCHS="${CLS_EVERY_N_EPOCHS:-1}"
CLS_LABEL_COL="${CLS_LABEL_COL:-synthesis_program}"
CLS_COMPOUND_COL="${CLS_COMPOUND_COL:-compound}"

# Pass --tau only when it is set (TiltedVAE defaults to sqrt(2 * latent_dim)).
TAU_ARG=()
if [[ -n "${TAU}" ]]; then
    TAU_ARG=(--tau "${TAU}")
fi

python "${SCRIPT_DIR}/train.py" \
    --data_dir        "${DATA_DIR}" \
    --img_size        "${IMG_SIZE}" \
    --batch_size      "${BATCH_SIZE}" \
    --num_workers     "${NUM_WORKERS}" \
    --val_split       "${VAL_SPLIT}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" \
    --index_cache     "${INDEX_CACHE}" \
    --model           "${MODEL}" \
    --latent_dim      "${LATENT_DIM}" \
    "${TAU_ARG[@]}" \
    --lr              "${LR}" \
    --kld_weight      "${KLD_WEIGHT}" \
    --epochs          "${EPOCHS}" \
    --precision       "${PRECISION}" \
    --anneal_kld \
    --anneal_k        6e-5 \
    --anneal_x0       310625 \
    --au_threshold    0.01 \
    --project         "${PROJECT}" \
    --entity          "${ENTITY}" \
    --run_name        "${RUN_NAME}" \
    --output_dir      "${OUTPUT_DIR}" \
    --cls_image_metadata  "${CLS_IMAGE_METADATA}" \
    --cls_label_metadata  "${CLS_LABEL_METADATA}" \
    --cls_root_dir        "${CLS_ROOT_DIR}" \
    --cls_every_n_epochs  "${CLS_EVERY_N_EPOCHS}" \
    --cls_label_col       "${CLS_LABEL_COL}" \
    --cls_compound_col    "${CLS_COMPOUND_COL}" \
    --cls_subtract_control \
    --cls_normalize_before_subtract
