#!/bin/bash
set -uo pipefail   # note: no -e, so one failed run doesn't kill the whole sweep

# ─── EDIT THESE ────────────────────────────────────────────────────────────────
TAUS=()                                 # supcon soft-positive taus to sweep
GPU=0
RUN_VANILLA=1                           # 1 = also run the VanillaSupCon baseline
RUN_CE=0
RUN_MS=0
PROJECT="aircraft-vits16-ema"
TRAIN_CAT=variant                       # manufacturer | family | variant
TEST_CATS=(family manufacturer)         # coarser levels for kNN / linear probe
AIRCRAFT_ROOT=data/fgvc_aircraft
LR=1e-4
EPOCHS=50
BS=256                                  # trainval is only 6,667 images
LOG_EVERY_N_STEPS=10
# ───────────────────────────────────────────────────────────────────────────────

COMMON=(
  --model backbone
  --backbone vit_small_patch16_224
  --dataset aircraft
  --train_cat "$TRAIN_CAT"
  --test_cat "${TEST_CATS[@]}"
  --project "$PROJECT"
  --aircraft_root "$AIRCRAFT_ROOT"
  --aircraft_download
  --aircraft_train_split trainval
  --aircraft_val_split test
  --img_size 224
  --lr "$LR"
  --epochs "$EPOCHS"
  --batch_size "$BS"
  --log_every_n_steps "$LOG_EVERY_N_STEPS"
  --scheduler cosine
  --warmup_epochs 3
  --precision bf16-mixed
  --num_workers 4
  --val_every_n_epochs 1
  --output_dir results
  --grad_checkpointing
)

run() {
  echo "=== [aircraft:${TRAIN_CAT}] $* ==="
  CUDA_VISIBLE_DEVICES="$GPU" python TiltedVAEMyzus/train.py "${COMMON[@]}" "$@" \
    || echo "!!! FAILED: aircraft $*"
}

if [[ "$RUN_MS" == 1 ]]; then
  run --ms_loss
fi

if [[ "$RUN_CE" == 1 ]]; then
  run --cross_entropy
fi

if [[ "$RUN_VANILLA" == 1 ]]; then
  run --vanilla_supcon
fi

for tau in "${TAUS[@]}"; do
  run --supcon_soft_pos_loss --supcon_soft_pos_tau "$tau"
done
