#!/bin/bash
# Batch 5: geographic holdout experiments + Batch-4 protocol correction.
#
# Protocol correction (2026-10-13): the label space is now rebuilt from the
# FILTERED train split, so species with zero training rows no longer compete
# with random weights. The four Batch-4 transfer runs are re-run here under
# the corrected protocol (old values are preserved in EXPERIMENT_LOG.md).
#
# All runs use the best config from the main grid: BioCLIP 2, frozen trunk +
# linear head, 336 px, 15 epochs, bs 32, lr 1e-3, seed 20261010.
set -u
PY=${FERN_PYTHON:-python3}
cd "$(dirname "$0")"
COMMON="--backbone bioclip2 --mode linear --img-size 336 --epochs 15 --batch-size 32 --lr 1e-3"

run() {  # name  train-where  eval-where
    echo "===== $1 (train=$2 eval=$3) ====="
    $PY train_fern.py $COMMON --train-where "$2" --eval-where "$3" \
        --out "runs/$1" || echo "FAILED $1"
}

# --- Batch 4 re-run under the corrected protocol (same names as before) ---
run xfer_field2spec "image_domain=specimen" "image_domain=field"
run xfer_spec2field "image_domain=field" "image_domain=specimen"
run xfer_gbif2inat  "source=GBIF" "source=iNaturalist"
run xfer_inat2gbif  "source=iNaturalist" "source=GBIF"

# --- Batch 5: geographic holdout ---
run geo_ex-TW "country_clean!=TW" "country_clean=TW"
run geo_ex-CN "country_clean!=CN" "country_clean=CN"
run geo_ex-JP "country_clean!=JP" "country_clean=JP"
run geo_ex-RU "country_clean!=RU" "country_clean=RU"
run geo_ex-US "country_clean!=US" "country_clean=US"
run geo_ex-TH "country_clean!=TH" "country_clean=TH"
run geo_ex-VN "country_clean!=VN" "country_clean=VN"
# whole China region held out (mainland + HK + Macao + Taiwan)
run geo_ex-CN_region "country_clean!=CN,TW,HK,MO" "country_clean=CN,TW,HK,MO"

echo "ALL BATCH-5 + XFER-RERUN DONE"
