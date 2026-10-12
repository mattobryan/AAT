#!/usr/bin/env bash
# Run configs two at a time, one per GPU (Kaggle "GPU T4 x2").
#   bash scripts/run_queue.sh "ramp eat v3a_max" "seed=0"
# An entry may carry its own seed, which wins over EXTRA: "aat_full_nominal:1 aat_full_nominal:2" "seed=0"
set -uo pipefail
CONFIGS=($1); EXTRA=${2:-}
NGPU=$(python -c "import torch;print(max(1,torch.cuda.device_count()))")
i=0; FAIL=0; LOGS=()
for c in "${CONFIGS[@]}"; do
  gpu=$((i % NGPU))
  cfg=${c%%:*}; own=""; [[ $c == *:* ]] && own="seed=${c##*:}"
  echo ">> $c on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -m aat.train --config configs/$cfg.yaml --set $EXTRA $own \
      > "logs_${c/:/_s}_$(echo $EXTRA | tr ' =' '_-').txt" 2>&1 &
  LOGS+=("logs_${c/:/_s}_$(echo $EXTRA | tr ' =' '_-').txt")
  i=$((i + 1))
  if (( i % NGPU == 0 )); then wait; fi
done
# a failed job leaves a Python traceback in its log: surface it in this stage's output and exit non-zero
for l in "${LOGS[@]}"; do
  if grep -q "^Traceback" "$l"; then echo "=== FAILED: $l (tail) ==="; tail -n 25 "$l"; FAIL=1; fi
done
exit $FAIL
