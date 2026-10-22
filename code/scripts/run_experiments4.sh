#!/usr/bin/env bash
# Batch 4 — test-set stress tests with the winning configuration
# (BioCLIP 2, frozen trunk, linear head, 336 px).
#
# Measures the two shifts that matter for ecological deployment:
#   * source shift   GBIF  <-> iNaturalist
#   * domain shift   herbarium sheet <-> field photograph
set -u

PY=${FERN_PYTHON:-python3}
cd "$(dirname "$0")"

run () {
  local name="$1"; shift
  echo "=============================================================="
  echo "START $name  $(date '+%F %T')"
  echo "=============================================================="
  if "$PY" train_fern.py --backbone bioclip2 --mode linear --img-size 336 \
       --epochs 15 --batch-size 32 --lr 1e-3 --workers 8 "$@" --out "runs/$name"; then
    echo "OK   $name  $(date '+%F %T')"
  else
    echo "FAIL $name  $(date '+%F %T')"
  fi
}

run xfer_gbif2inat   --train-where source=GBIF       --eval-where source=iNaturalist
run xfer_inat2gbif   --train-where source=iNaturalist --eval-where source=GBIF
run xfer_spec2field  --train-where image_domain=specimen --eval-where image_domain=field
run xfer_field2spec  --train-where image_domain=field    --eval-where image_domain=specimen

echo "ALL DONE $(date '+%F %T')"
