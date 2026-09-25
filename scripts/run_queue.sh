#!/usr/bin/env bash
# Run configs two at a time, one per GPU (Kaggle "GPU T4 x2").
#   bash scripts/run_queue.sh "ramp eat v3a_max" "seed=0"
set -uo pipefail
CONFIGS=($1); EXTRA=${2:-}
NGPU=$(python -c "import torch;print(max(1,torch.cuda.device_count()))")
i=0
for c in "${CONFIGS[@]}"; do
  gpu=$((i % NGPU))
  echo ">> $c on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -m aat.train --config configs/$c.yaml --set $EXTRA \
      > "logs_${c}_$(echo $EXTRA | tr ' =' '_-').txt" 2>&1 &
  i=$((i + 1))
  if (( i % NGPU == 0 )); then wait; fi
done
wait
