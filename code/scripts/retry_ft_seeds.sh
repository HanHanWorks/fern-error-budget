#!/bin/bash
# Retry the three bioclip2_ft_336 seed runs that OOM'd on 2026-10-13 because
# another job (ap_experiment_20261012) held ~12 GB of the 24 GB card.
# This wrapper waits until >= 16 GB of GPU memory is free before starting, so
# it completes on its own once the other job finishes; it does not touch that
# job in any way.
set -u
PY=${FERN_PYTHON:-python3}
cd "$(dirname "$0")"

while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
           | tr -d '[:space:]')
    echo "$(date +%H:%M:%S) GPU free = ${free} MiB (need >= 16000)"
    if [ "${free:-0}" -ge 16000 ]; then
        echo "enough memory, starting the ft_336 seeds"
        break
    fi
    sleep 300
done

for s in 20261010 20261011 20261012; do
    echo "===== bioclip2_ft_336 seed=$s (retry) ====="
    $PY train_fern.py --backbone bioclip2 --mode finetune --img-size 336 \
        --epochs 12 --batch-size 16 --lr 2e-5 --seed "$s" \
        --out "runs/bioclip2_ft_336_s$s" || echo "FAILED bioclip2_ft_336_s$s"
done
echo "FT SEED RETRY DONE"
