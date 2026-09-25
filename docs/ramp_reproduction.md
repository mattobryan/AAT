# Step 1: reproduce the RAMP baseline

Before building AAT, the baseline has to be reproduced from the **official code**
([uiuc-focal-lab/RAMP](https://github.com/uiuc-focal-lab/RAMP), commit `be4971f`), with no
reimplementation. Only after that does our own RAMP config (`configs/ramp.yaml`) need to match it.

## Which RAMP numbers?

| Source | Setting | Clean | ℓ∞ | ℓ2 | ℓ1 | Union |
|---|---|---|---|---|---|---|
| RAMP paper, Table 24 (App. B.7) | RN-18 ℓ∞-AT **fine-tuned 3 epochs**, λ=1.5, 5 seeds | 81.1 | 45.4 | 66.1 | 47.2 | 43.1 |
| RAMP paper, Table 3 | RN-18 **from scratch, 80 epochs**, λ=5, GP, 5 seeds | 81.2 | 46.0 | 65.8 | 48.3 | 44.6 |
| Thesis, Table 7.1 | "baseline reproduction", 5 runs | 81.3 | 45.96 | 65.74 | 48.42 | 44.5 |
| Starting point `pretr_Linf` | before fine-tuning | 83.7 | 48.1 | 59.8 | 7.7 | 38.5 |

The thesis Table 7.1 averages match the paper's **from-scratch λ=5** row to within 0.1 pp in every
column. That setting is out of reach on Kaggle: the paper reports 157 s/epoch on an A100, which is
roughly 15–20 min/epoch on a T4 in fp32, or about 20–25 h per seed. That is longer than the 12 h
Kaggle session limit, and the official script cannot resume. On top of that, the public code's `--gp`
path crashes (`utils.gp` uses `copy` without importing it). We patch that one line.

**The target is therefore Table 24 (fine-tuning).** It uses the same method, loss and evaluation. It
starts from the official `pretr_Linf.pth` shipped in the repo and costs about 1 GPU-hour per seed.
The from-scratch row stays as a stretch goal and is reported as such.

## What has been verified (CPU, before this was paused)

* Our `preactresnet18_softplus` port reproduces the official model's logits to within 5e-7.
* `pretr_Linf` scores **83.7 %** clean on the first 1,000 test points (82.8 % on all 10k). The
  paper's 83.7 % means Table 24 uses the first-1,000-point protocol, so all evaluation here uses
  `eval.n = 1000`.
* `aat.evaluate` now uses the official evaluation settings: AutoAttack `version="standard"` restricted to
  APGD-CE + APGD-T. For ℓ1 that means 5 restarts, 5 target classes and large reps, as in RAMP's `eval.py`.

## Protocol (the official flags, from `scripts/cifar10/RAMP_finetune_cifar10.sh`)

`RAMP.py --finetune_model --model_name pretr_Linf --epochs 3 --lr-max 0.05 --lr-schedule piecewise-ft --at_iter 10 --kl --max`
with λ defaulting to 1.5, APGD-10 training attacks on ℓ∞ (source) and ℓ1 (target), CE on the
per-sample worst case, and KL(p_ℓ∞ ‖ p_ℓ1) on the points ℓ∞ gets right. There is no GP in fine-tuning.

## Kaggle run order (`notebooks/ramp_baseline_kaggle.ipynb`)

| Step | Command | Est. T4 time |
|---|---|---|
| Setup | `bash scripts/ramp_official.sh setup` | 3 min |
| Sanity | `bash scripts/ramp_official.sh pretr 0` → expect ≈ 83.7 / 48.1 / 59.8 / 7.7 / 38.5 | 10 min |
| RAMP ×5 seeds | `train ramp "0 2 4" 0` ‖ `train ramp "1 3" 1` | ~3 h wall |
| Evaluate | `eval ramp "0 1 2 3 4"` | ~45 min |
| Compare | `python scripts/compare_targets.py --map ramp_official_ft=ramp_l1.5 pretr_linf=pretr_linf` | – |
| Optional | E-AT and MAX, 3 seeds each, same commands | ~4 h |

**Acceptance:** each metric within 1 pp of the paper mean (OK), or within 2 pp (CLOSE, report it with
an explanation). Seed 0 also runs the official `--final_eval` on the same 1,000 points, so our
evaluator can be checked against theirs directly.

## After it passes

1. Point `configs/base.yaml` at the official setup: `model: preactresnet18_softplus`,
   `init_from: external/ramp/models/pretr_Linf.pth`, 3 epochs, `piecewise-ft` LR.
2. Switch our training attacks to APGD so that `configs/ramp.yaml` reproduces the official
   numbers. Only then do AAT differences mean something.
3. Build AAT on top of that pipeline.
