#!/bin/bash
# Seed-variance runs for the three key main-table rows (2026-10-13).
# Each config is re-run with three seeds; seed 20261010 reproduces the
# original single-seed run and doubles as a determinism check.
# Per-config hyper-parameters are copied verbatim from the original runs.
set -u
PY=${FERN_PYTHON:-python3}
cd "$(dirname "$0")"

run3() {  # name, then the train_fern.py args (without --seed/--out)
    name="$1"; shift
    for s in 20261010 20261011 20261012; do
        echo "===== ${name} seed=${s} ====="
        $PY train_fern.py "$@" --seed "$s" \
            --out "runs/${name}_s${s}" || echo "FAILED ${name}_s${s}"
    done
}

run3 bioclip2_linear_336 \
    --backbone bioclip2 --mode linear --img-size 336 \
    --epochs 15 --batch-size 32 --lr 1e-3

run3 bioclip2_ft_336 \
    --backbone bioclip2 --mode finetune --img-size 336 \
    --epochs 12 --batch-size 16 --lr 2e-5

run3 dinov2_linear_336 \
    --backbone dinov2 --mode linear --img-size 336 \
    --epochs 15 --batch-size 32 --lr 1e-3

echo "ALL SEED RUNS DONE"
