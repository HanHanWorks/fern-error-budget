#!/usr/bin/env bash
# Sequential fine-tuning comparison: BioCLIP 2 vs DINOv2, linear probe vs
# full fine-tune, under an identical protocol.
#
# Launch detached:
#   nohup setsid bash run_experiments.sh > ../logs/experiments.log 2>&1 &
set -u

PY=${FERN_PYTHON:-python3}
cd "$(dirname "$0")"

run () {
  local name="$1"; shift
  echo "=============================================================="
  echo "START $name  $(date '+%F %T')"
  echo "=============================================================="
  if "$PY" train_fern.py "$@" --out "runs/$name"; then
    echo "OK   $name  $(date '+%F %T')"
  else
    echo "FAIL $name  $(date '+%F %T')"
  fi
}

# 1-2) Linear probes: frozen trunk, head only. Fast, isolates feature quality.
run bioclip2_linear --backbone bioclip2 --mode linear \
    --epochs 15 --batch-size 64 --lr 1e-3 --workers 8
run dinov2_linear --backbone dinov2 --mode linear \
    --epochs 15 --batch-size 64 --lr 1e-3 --workers 8

# 3-4) Full fine-tune under the same recipe.
run bioclip2_ft --backbone bioclip2 --mode finetune \
    --epochs 15 --batch-size 32 --lr 2e-5 --workers 8
run dinov2_ft --backbone dinov2 --mode finetune \
    --epochs 15 --batch-size 32 --lr 2e-5 --workers 8

echo "ALL DONE $(date '+%F %T')"
