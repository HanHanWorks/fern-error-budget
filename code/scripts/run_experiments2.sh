#!/usr/bin/env bash
# Batch 2 — stress the winner (BioCLIP 2) and test the paper's architecture
# module (hierarchical head) plus long-tail handling and resolution.
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

# Zero-shot: specialist model with no training at all.
echo "=== zero-shot bioclip2 ==="
$PY zeroshot_bioclip.py --weights bioclip2 || echo "FAIL zeroshot"

# Fine-tune with a higher LR: batch 1 showed 2e-5 barely learned past the
# frozen features, so this tests whether the trunk can be improved at all.
run bioclip2_ft_lr5e5 --backbone bioclip2 --mode finetune \
    --epochs 15 --batch-size 32 --lr 5e-5 --workers 8

# Higher resolution (BioCLIP 2 is native 224; 336 may help small sori/venation).
run bioclip2_ft_336 --backbone bioclip2 --mode finetune --img-size 336 \
    --epochs 12 --batch-size 16 --lr 5e-5 --workers 8

# The paper's proposed module: hierarchical family->species head.
run bioclip2_ft_hier --backbone bioclip2 --mode finetune \
    --epochs 15 --batch-size 32 --lr 5e-5 --hier --hier-weight 0.3 --workers 8

# Long-tail handling.
run bioclip2_linear_bal --backbone bioclip2 --mode linear \
    --epochs 15 --batch-size 64 --lr 1e-3 --class-balance --workers 8

echo "ALL DONE $(date '+%F %T')"
