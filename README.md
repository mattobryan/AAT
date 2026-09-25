# AAT: Adaptive Adversarial Training

A clean rebuild of *Adaptive Adversarial Training: A Curriculum-Based Approach to Robustness*
(M. Brian, MSc thesis, University of Szeged, 2025). It is designed to run end to end on the free
Kaggle GPU tier (2×T4, ~30 GPU-h/week) in one week.

## Research question

> Does feedback-driven scheduling of the perturbation budget (per class) and of the attacked
> ℓp norm improve **union robustness over {ℓ1, ℓ2, ℓ∞}**, and robustness to **threats not seen in
> training**, compared with fixed-schedule multi-norm adversarial training (MAX, RAMP, E-AT)?
> Evaluation uses AutoAttack at fixed standard budgets, and training compute is reported next to
> every result.

| | Hypothesis | Test | Supported if |
|---|---|---|---|
| H1 | Full AAT beats fixed-schedule multi-norm AT on union robustness | `aat_full` vs `v3a_max`, `ramp`, `eat` | Union gain > 2× the seed std, clean drop ≤ 1 pp |
| H2 | A class-feedback ε narrows the class robustness gap | `v3b_classeps` vs `v3a_max` | Worst-class union ↑ and class std ↓ |
| H3 | Adaptive norm sampling beats cyclic sampling | `v2b_adaptive` vs `v2a_cyclic` | Union ↑ at equal compute |
| H4 | AAT generalizes better to unseen threats | ε-grid on all runs; `aat_full_no_l2` vs `ramp` / `eat` on ℓ2 (never trained on) | Higher accuracy off the training budgets |

The thesis numbers conflict with each other in several places, and it evaluates at adaptive ε.
See [`docs/thesis_audit.md`](docs/thesis_audit.md). This repo re-tests the claims under a clean
protocol. It does not reproduce the printed numbers.

## Protocol

* **Data / model:** CIFAR-10, PreActResNet-18, pixels in [0,1] with normalization inside the model.
  1,000 training images are held out as the *feedback set* that drives the schedulers. The test
  set is never used for adaptation or model selection; the last checkpoint is always used.
* **Training:** one shared ℓ∞-AT model (`pretrain_linf`, 30 epochs), then every method fine-tunes
  it for 10 epochs (RAMP / E-AT protocol), with SGD, cosine LR 0.05, AMP, and PGD steps 5/5/10 for ℓ∞/ℓ2/ℓ1.
* **Evaluation:** AutoAttack (APGD-CE + APGD-T) on the first 1,000 test points at ε = 8/255 (ℓ∞),
  0.5 (ℓ2), 12 (ℓ1). Union = per-sample AND. Also reported: per-class robustness, worst-class
  union, and an unseen-ε grid with APGD-CE (ℓ∞ 4/12/16 /255, ℓ2 0.25/1.0/1.5, ℓ1 6/18/24).

## Experiment matrix

| Config | Norms / batch | ε schedule | Norm schedule | Loss | Thesis name |
|---|---|---|---|---|---|
| `ramp` | ℓ∞+ℓ1 | static | both | max + KL(λ=2) + GP | RAMP baseline |
| `eat` | ℓ∞ or ℓ1 | static | ∝ robust error | CE | E-AT baseline |
| `v3a_max` | all 3 | static | all | per-sample max | 3A |
| `v1a_linear` | all 3 | global linear ramp | all | max | 1A |
| `v1b_sample` | all 3 | per-sample, grad-norm + class EMA (Eq. A.1) | all | max | 1B |
| `v2a_cyclic` | 1 | static | round-robin | CE | 2A |
| `v2b_adaptive` | 1 | static | Boltzmann on feedback (Eq. 4.20) | CE | 2B |
| `v3b_classeps` | all 3 | class-feedback controller (Eq. 4.19) | all | max | 3B |
| `v3b_thesis` | all 3 | label-index formula (Eq. A.5) | all | max | 3B as printed |
| `aat_full` | all 3 | class-feedback | adaptive loss weights | hybrid (4.8) + KL | full AAT / 3C |
| `aat_full_no_l2` | ℓ∞+ℓ1 | class-feedback | adaptive | hybrid + KL | unseen-norm test |

## One-week Kaggle plan

The costs are estimates for a T4 with AMP. **Replace them with the Day 1 timing numbers.**
Quota is counted per session hour, so running two jobs on the 2×T4 roughly halves wall-clock time.

| Day | Work | Est. GPU-h | Est. session-h |
|---|---|---|---|
| 1 | Smoke tests, timing, check RAMP details against the official code, `pretrain_linf` | 1.5 | 1.5 |
| 2 | `ramp eat v3a_max v1a_linear` | 4.5 | 2.3 |
| 3 | `v1b_sample v3b_classeps v2a_cyclic v2b_adaptive` | 4 | 2 |
| 4 | `aat_full aat_full_no_l2 v3b_thesis`; begin seeds 1–2 | 5 | 2.5 |
| 5 | Seeds 1–2 for `ramp eat v3a_max aat_full v3b_classeps v2b_adaptive` | 10 | 5 |
| 6 | AutoAttack plus the unseen grid on every final checkpoint (~25 min per model) | 10 | 5 |
| 7 | `scripts/aggregate.py`, figures, write-up. Buffer for re-runs | – | 2–4 |
| | **Total** | **~35** | **~20** of the 30 h quota |

If Day 1 timing shows this won't fit, cut in this order: `v3b_thesis`, then `v1a_linear`, then
seed 2, then drop unseen APGD to 500 points.

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q                                  # CPU unit + smoke tests
python -m aat.train --config configs/pretrain_linf.yaml
python -m aat.train --config configs/aat_full.yaml --set seed=1
bash scripts/run_queue.sh "ramp eat" "seed=0"        # one job per GPU
python -m aat.evaluate --run runs/aat_full_s0        # AutoAttack + unseen grid
python scripts/aggregate.py --runs runs              # results/summary_autoattack.md + trade-off plot
```

On Kaggle, use [`notebooks/aat_kaggle.ipynb`](notebooks/aat_kaggle.ipynb). Training resumes from
`last.pt`, so a session that times out can simply be re-run.

## Layout

```
aat/models.py      PreActResNet-18 with built-in normalization
aat/attacks.py     PGD ℓ∞ / ℓ2 / ℓ1 (SLIDE-style, exact ℓ1 projection), per-sample ε
aat/schedulers.py  EpsScheduler (static, linear, norm/class feedback, per-sample, class-index) and NormScheduler
aat/losses.py      max / mean / hybrid aggregation, RAMP KL pairing
aat/train.py       feedback -> adapt -> adversarial training loop; logs ε, norm entropy, zero-budget classes, KL
aat/evaluate.py    AutoAttack at fixed ε, union, per-class, unseen grid
configs/           one YAML per method (inherits base.yaml)
docs/              thesis audit
```

Per-epoch logs (`runs/<name>_s<seed>/log.jsonl`) record the thesis's auxiliary metrics (§7.7):
the ε per norm and class, zero-budget class count, norm-weight entropy, mean KL, the
secondary/primary loss ratio, feedback robustness, and test curves for monitoring only.
