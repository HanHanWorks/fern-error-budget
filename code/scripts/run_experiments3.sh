#!/usr/bin/env bash
# Batch 3 — resolution study (224 -> 336) and the corrected zero-shot family
# evaluation (Latin family names; Chinese names embed to noise for CLIP).
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

echo "=== zero-shot bioclip2 (Latin family names) ==="
$PY zeroshot_bioclip.py --weights bioclip2 || echo "FAIL zeroshot"

run bioclip2_linear_336 --backbone bioclip2 --mode linear --img-size 336 \
    --epochs 15 --batch-size 32 --lr 1e-3 --workers 8
run bioclip2_ft_336 --backbone bioclip2 --mode finetune --img-size 336 \
    --epochs 12 --batch-size 16 --lr 2e-5 --workers 8
run dinov2_linear_336 --backbone dinov2 --mode linear --img-size 336 \
    --epochs 15 --batch-size 32 --lr 1e-3 --workers 8

echo "ALL DONE $(date '+%F %T')"
