#!/usr/bin/env bash
# Grafit (Touvron et al., 2020) training launcher.
#
#   L_tot(x) = L_knn(g(x), y) + GRAFIT_LAM * L_inst(x)
#
# L_knn is the coarse-label NCA loss at sigma = TEMPERATURE (0.05 in the paper),
# scored either in-batch or against a full-training-set memory bank (USE_BANK=1).
# L_inst is the BYOL-style term over GRAFIT_VIEWS augmented views of each image,
# using the model's predictor head and an EMA target network.
#
# Usage:
#   chmod +x run_grafit.sh
#   ./run_grafit.sh
#   DATASET=inat GRAFIT_VIEWS=4 USE_BANK=1 ./run_grafit.sh
#
# WARNING: USE_BANK=1 materializes a BATCH_SIZE x train_size logit matrix each
# step. Fine for FGVC-Aircraft (6.7k images); check memory before iNaturalist.

set -uo pipefail

# ─── EDIT THESE ────────────────────────────────────────────────────────────────
DATASET="${DATASET:-aircraft}"          # aircraft | inat | myzus
GPU="${GPU:-0}"
PROJECT="${PROJECT:-grafit}"
RUN_BASELINE="${RUN_BASELINE:-0}"       # 1 = also run a VanillaSupCon baseline

GRAFIT_LAM="${GRAFIT_LAM:-1.0}"         # paper: 1.0 (the two losses are summed)
GRAFIT_VIEWS="${GRAFIT_VIEWS:-4}"       # paper: T=4 on ImageNet, T=8 on CIFAR
USE_BANK="${USE_BANK:-1}"               # 1 = memory-bank L_knn (as in the paper)
TEMPERATURE="${TEMPERATURE:-0.05}"      # NCA sigma; the paper uses 0.05
EMA_MOMENTUM="${EMA_MOMENTUM:-0.996}"   # momentum of the BYOL target network

MODEL="${MODEL:-backbone}"              # backbone | dino_lora
BACKBONE="${BACKBONE:-vit_small_patch16_224}"
IMG_SIZE="${IMG_SIZE:-224}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-50}"
BS="${BS:-64}"                          # GRAFIT_VIEWS crops are encoded per item
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

TRAIN_CAT="${TRAIN_CAT:-variant}"       # coarse labels fed to L_knn
TEST_CATS=(${TEST_CATS:-family manufacturer})
AIRCRAFT_ROOT="${AIRCRAFT_ROOT:-data/fgvc_aircraft}"
INAT_ROOT="${INAT_ROOT:-data/inat2021}"
SUPERCLASS="${SUPERCLASS:-}"
# ───────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON=(
  --model "$MODEL"
  --project "$PROJECT"
  --train_cat "$TRAIN_CAT"
  --test_cat "${TEST_CATS[@]}"
  --img_size "$IMG_SIZE"
  --lr "$LR"
  --epochs "$EPOCHS"
  --batch_size "$BS"
  --num_workers "$NUM_WORKERS"
  --temperature "$TEMPERATURE"
  --EMA_momentum "$EMA_MOMENTUM"
  --scheduler cosine
  --warmup_epochs 3
  --precision bf16-mixed
  --val_every_n_epochs 1
  --log_every_n_steps 10
  --output_dir "$OUTPUT_DIR"
  --grad_checkpointing
)

if [[ "$MODEL" == "backbone" ]]; then
  COMMON+=(--backbone "$BACKBONE")
else
  COMMON+=(--dino_backbone "$BACKBONE")
fi

case "$DATASET" in
  aircraft)
    COMMON+=(
      --dataset aircraft
      --aircraft_root "$AIRCRAFT_ROOT"
      --aircraft_download
      --aircraft_train_split trainval
      --aircraft_val_split test
    )
    ;;
  inat)
    COMMON+=(
      --dataset inat
      --inat_train_metadata "$INAT_ROOT/train_mini.json"
      --inat_val_metadata "$INAT_ROOT/val.json"
      --inat_train_dir "$INAT_ROOT/train_mini"
      --inat_val_dir "$INAT_ROOT/val"
    )
    [[ -n "$SUPERCLASS" ]] && COMMON+=(--superclass "$SUPERCLASS")
    ;;
  myzus)
    COMMON+=(
      --dataset myzus
      --contrastive_metadata "${CONTRASTIVE_METADATA:?set CONTRASTIVE_METADATA}"
      --contrastive_labels "${CONTRASTIVE_LABELS:?set CONTRASTIVE_LABELS}"
      --contrastive_root_dir "${CONTRASTIVE_ROOT_DIR:?set CONTRASTIVE_ROOT_DIR}"
    )
    ;;
  *)
    echo "Unknown DATASET '$DATASET' (expected aircraft | inat | myzus)" >&2
    exit 1
    ;;
esac

GRAFIT=(
  --grafit
  --grafit_lam "$GRAFIT_LAM"
  --grafit_views "$GRAFIT_VIEWS"
)
[[ "$USE_BANK" == 1 ]] && GRAFIT+=(--grafit_bank)

run() {
  echo "=== [grafit:${DATASET}:${TRAIN_CAT}] $* ==="
  CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT_DIR/train.py" "${COMMON[@]}" "$@" \
    || echo "!!! FAILED: $*"
}

run "${GRAFIT[@]}"

if [[ "$RUN_BASELINE" == 1 ]]; then
  run --vanilla_supcon
fi
