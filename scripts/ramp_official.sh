#!/usr/bin/env bash
# Reproduce the RAMP-paper fine-tuning baselines (Table 24) with the OFFICIAL code, unmodified
# except for one missing import. Then evaluate every checkpoint with aat.evaluate (same AutoAttack
# protocol as the official eval.py, first 1000 test points).
#
#   bash scripts/ramp_official.sh setup
#   bash scripts/ramp_official.sh train ramp "0 1 2 3 4" 0      # method, seeds, gpu
#   bash scripts/ramp_official.sh eval  ramp "0 1 2 3 4" 0
#   bash scripts/ramp_official.sh pretr 0                       # sanity: evaluate pretr_Linf
# methods:
#   ramp_scratch  thesis Table 7.1 / paper Table 3: 80 epochs from scratch, lambda=5, GP (multi-session)
#   ramp          fine-tuning, lambda=1.5 (repo default = paper Table 24 row); ramp05: lambda=0.5
#   eat, max      fine-tuning baselines from the same repo
# Checkpoints sync to the Hugging Face repo in $HF_REPO (token: $HF_TOKEN or the Kaggle secret HF_TOKEN).
# RAMP runs resume automatically; each session stops cleanly after $TIME_BUDGET_H hours (default 11).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXT=$ROOT/external/ramp
COMMIT=be4971f04cf8e70bd8255874a1ed2ab489cae682
DATA=$ROOT/data
cmd=$1; shift
run_name() { if [ "$1" = "ramp_scratch" ]; then echo "ramp_scratch_l5_s$2"; else echo "${1}_ft_s$2"; fi; }

case $cmd in
install)
  if [ ! -d "$EXT/.git" ]; then
    git clone -q https://github.com/uiuc-focal-lab/RAMP "$EXT"
  fi
  git -C "$EXT" checkout -q $COMMIT
  # utils.gp() calls copy.deepcopy without importing copy (only reached with --gp, i.e. from-scratch runs)
  grep -q "^import copy" "$EXT/utils.py" || sed -i '1i import copy' "$EXT/utils.py"
  # resume + HF Hub sync for RAMP.py (reviewable diff: baselines/ramp_resume_hub.patch)
  git -C "$EXT" apply --check "$ROOT/baselines/ramp_resume_hub.patch" 2>/dev/null && git -C "$EXT" apply "$ROOT/baselines/ramp_resume_hub.patch"
  cp "$ROOT/aat/hub.py" "$EXT/hub_sync.py"
  # dependencies first, then a full import check (also imports the official RAMP.py) before anything else
  pip install -q -r "$ROOT/requirements-kaggle.txt" || {
    echo "normal install failed; installing the pieces one by one"
    pip install -q huggingface_hub pyyaml geotorch timm gdown requests pandas
    pip install -q --no-deps git+https://github.com/RobustBench/robustbench.git
    python -c "import autoattack" 2>/dev/null || pip install -q git+https://github.com/fra31/auto-attack
  }
  python "$ROOT/scripts/check_env.py"
  ;;
data)
  # CIFAR-10: the Toronto mirror can be very slow, so keep a copy in the HF repo after the first download
  TAR="$DATA/cifar-10-python.tar.gz"; mkdir -p "$DATA"
  [ -f "$TAR" ] || python "$ROOT/aat/hub.py" pull "${HF_REPO:-none}" cache/cifar-10-python.tar.gz "$TAR" || true
  had_cache=0; [ -f "$TAR" ] && had_cache=1
  python -c "import torchvision; torchvision.datasets.CIFAR10('$DATA', train=True, download=True); torchvision.datasets.CIFAR10('$DATA', train=False, download=True)"
  [ $had_cache = 1 ] || python "$ROOT/aat/hub.py" push "${HF_REPO:-none}" "$TAR" cache/cifar-10-python.tar.gz || true
  echo "data OK"
  ;;
setup)
  bash "$0" install && bash "$0" data
  ;;
train)
  method=$1; seeds=$2; gpu=${3:-0}
  common="--finetune_model --lr-schedule=piecewise-ft --model_name pretr_Linf --at_iter 10 --epochs 3 --eval_freq 1 --lr-max 0.05 --data_dir $DATA"
  resume="--resume --time_budget_h ${TIME_BUDGET_H:-11}"
  case $method in
    ramp)         script=RAMP.py;      extra="--kl --max $resume" ;;
    ramp05)       script=RAMP.py;      extra="--kl --max --lbd 0.5 $resume" ;;
    ramp_scratch) script=RAMP.py;      extra="--kl --max --gp --lbd 5 $resume"
                  common="--lr-max 0.05 --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 10 --eval_freq 10 --data_dir $DATA" ;;
    eat)          script=eat_train.py; extra="" ;;
    max)          script=MAX.py;       extra="" ;;
  esac
  cd "$EXT"
  for s in $seeds; do
    fname=$(run_name $method $s)
    # finished = official final-eval log exists locally or on the Hub
    [ -f "trained_models/$fname/log_eval_final.txt" ] || python "$ROOT/aat/hub.py" pull "${HF_REPO:-none}" \
        "ramp_official/$fname/log_eval_final.txt" "trained_models/$fname/log_eval_final.txt" >/dev/null 2>&1 || true
    [ -f "trained_models/$fname/log_eval_final.txt" ] && { echo "done already: $fname"; continue; }
    fe="--final_eval --n_ex_final 1000"   # official eval on the same 1000 points, to cross-check ours
    echo ">> $fname on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $script $common $extra $fe --seed $s --fname $fname 2>&1 \
      | grep --line-buffered -v "it/s\]" | sed -u "s/^/[$fname] /" | tee "$ROOT/logs_official_$fname.txt"
  done
  ;;
eval)
  method=$1; seeds=$2; gpu=${3:-0}
  for s in $seeds; do
    fname=$(run_name $method $s)
    ep=3; [ "$method" = "ramp_scratch" ] && ep=80
    ck="$EXT/trained_models/$fname/ep_${ep}_0.pth"
    [ -f "$ck" ] || python "$ROOT/aat/hub.py" pull "${HF_REPO:-none}" "ramp_official/$fname/ep_${ep}_0.pth" "$ck"
    CUDA_VISIBLE_DEVICES=$gpu python -m aat.evaluate --run "$ROOT/runs_official/$fname" --ckpt "$ck" \
      --config "$ROOT/configs/official_eval.yaml" --name "${method}_official"
    python "$ROOT/aat/hub.py" push "${HF_REPO:-none}" "$ROOT/runs_official/$fname/eval_autoattack.json" \
      "ramp_official/$fname/eval_autoattack.json" || true
  done
  ;;
train2|eval2)
  # two seed lists in parallel, one per GPU; fails if either side fails
  method=$1; sub=${cmd%2}
  ea=0; eb=0
  bash "$0" $sub $method "$2" 0 & a=$!
  bash "$0" $sub $method "$3" 1 & b=$!
  wait $a || ea=$?
  wait $b || eb=$?
  exit $(( ea > eb ? ea : eb ))
  ;;
pretr)
  gpu=${1:-0}
  CUDA_VISIBLE_DEVICES=$gpu python -m aat.evaluate --run "$ROOT/runs_official/pretr_linf" \
    --ckpt "$EXT/models/pretr_Linf.pth" --config "$ROOT/configs/official_eval.yaml" --name pretr_linf
  ;;
esac
