#!/usr/bin/env bash
# Reproduce the RAMP-paper fine-tuning baselines (Table 24) with the OFFICIAL code, unmodified
# except for one missing import. Then evaluate every checkpoint with aat.evaluate (same AutoAttack
# protocol as the official eval.py, first 1000 test points).
#
#   bash scripts/ramp_official.sh setup
#   bash scripts/ramp_official.sh train ramp "0 1 2 3 4" 0      # method, seeds, gpu
#   bash scripts/ramp_official.sh eval  ramp "0 1 2 3 4" 0
#   bash scripts/ramp_official.sh pretr 0                       # sanity: evaluate pretr_Linf
# methods: ramp (lambda=1.5, repo default = Table 24 row), ramp05 (lambda=0.5), eat, max
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXT=$ROOT/external/ramp
COMMIT=be4971f04cf8e70bd8255874a1ed2ab489cae682
DATA=$ROOT/data
cmd=$1; shift

case $cmd in
setup)
  if [ ! -d "$EXT/.git" ]; then
    git clone -q https://github.com/uiuc-focal-lab/RAMP "$EXT"
  fi
  git -C "$EXT" checkout -q $COMMIT
  # utils.gp() calls copy.deepcopy without importing copy (only reached with --gp, i.e. from-scratch runs)
  grep -q "^import copy" "$EXT/utils.py" || sed -i '1i import copy' "$EXT/utils.py"
  pip install -q git+https://github.com/RobustBench/robustbench.git git+https://github.com/fra31/auto-attack
  python -c "import torchvision; torchvision.datasets.CIFAR10('$DATA', train=True, download=True); torchvision.datasets.CIFAR10('$DATA', train=False, download=True)"
  ls -la "$EXT/models"
  ;;
train)
  method=$1; seeds=$2; gpu=${3:-0}
  common="--finetune_model --lr-schedule=piecewise-ft --model_name pretr_Linf --at_iter 10 --epochs 3 --eval_freq 1 --lr-max 0.05 --data_dir $DATA"
  case $method in
    ramp)   script=RAMP.py;      extra="--kl --max" ;;
    ramp05) script=RAMP.py;      extra="--kl --max --lbd 0.5" ;;
    eat)    script=eat_train.py; extra="" ;;
    max)    script=MAX.py;       extra="" ;;
  esac
  cd "$EXT"
  for s in $seeds; do
    fname=${method}_ft_s$s
    [ -f "trained_models/$fname/ep_3_0.pth" ] && { echo "skip $fname"; continue; }
    fe=""; [ "$s" = "0" ] && fe="--final_eval --n_ex_final 1000"   # official eval on seed 0, to cross-check ours
    echo ">> $fname on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python $script $common $extra $fe --seed $s --fname $fname 2>&1 | grep -v "it/s\]" > "$ROOT/logs_official_$fname.txt"
  done
  ;;
eval)
  method=$1; seeds=$2; gpu=${3:-0}
  for s in $seeds; do
    fname=${method}_ft_s$s
    ck=$(ls "$EXT"/trained_models/$fname/ep_3*.pth | head -1)
    CUDA_VISIBLE_DEVICES=$gpu python -m aat.evaluate --run "$ROOT/runs_official/$fname" --ckpt "$ck" \
      --config "$ROOT/configs/official_eval.yaml" --name "${method}_official_ft"
  done
  ;;
pretr)
  gpu=${1:-0}
  CUDA_VISIBLE_DEVICES=$gpu python -m aat.evaluate --run "$ROOT/runs_official/pretr_linf" \
    --ckpt "$EXT/models/pretr_Linf.pth" --config "$ROOT/configs/official_eval.yaml" --name pretr_linf
  ;;
esac
