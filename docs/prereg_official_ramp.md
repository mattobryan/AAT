# Pre-registration: budget allocation on the official RAMP fine-tune, and the held-out check

Written and committed on 23 October 2026, **before any of these runs**. Do not edit the predictions after results arrive; add a
dated "Results" section below instead. Background: `docs/report/roadmap.tex` (Programme A), `docs/thesis_audit.md` items 23–29.

## Question

In our own pipeline, moving training budget from ℓ2 to ℓ1 (ℓ∞ 1.0×, ℓ2 0.75×, ℓ1 1.5×) gave union robustness about 6 points above the
fixed-budget clean-versus-union frontier at matched clean accuracy (10-epoch fine-tune, 2 seeds, ratio chosen on test images 0–999).
Our ℓ1 training attack is probably weaker than RAMP's, and ℓ1 at the nominal budget was far weaker than ℓ∞ (about 30 against 45) in
our pipeline but not in RAMP's (47.2 against 45.4, paper Table 24). Is the effect a property of budget allocation, or of our pipeline?

## Part 1: held-out confirmation (our pipeline, test images 1000–1999)

Models: `v3a_max_realloc` and the fixed controls `v3a_max`, `_eps112`, `_eps125`, `_eps150`, seeds 0 and 1 (`HELD_RUNS` in the notebook).
Protocol: `scripts/eval_heldout.py` (APGD-CE + APGD-T seed 0, fixed ε 8/255, 0.5, 12, union = per-sample AND). Frontier = piecewise-linear
interpolation through the four fixed-budget runs **on the same held-out images**; gap = `realloc` union minus the frontier at its own clean accuracy.

| Held-out gap of `realloc` | Reading |
|---|---|
| ≥ +3 | the allocation gain is not a selection artifact (it was +6.2 on the images used to choose it) |
| between 0 and +3 | the gain shrank under held-out evaluation; report as partly selection |
| ≤ 0 | the +6 was selection; withdraw the allocation claim in our pipeline |

Binomial error on a 1000-image union is about 1.6 points per model; the comparison is paired (same images), so the gap is less noisy.

## Part 2: official RAMP fine-tune, six budget arms, seeds 0 and 1

Official code (commit `be4971f` plus our patches, `baselines/ramp_multi_eps.patch` for `--l_eps` with several norms), `--finetune_model`,
**3 epochs** (the default is 20), λ = 1.5, from `pretr_Linf`. Arms (`scripts/ramp_official.sh train rampeps_<arm>`):
`f100` default budgets; `f125`, `f150` both Linf and L1 ×1.25 / ×1.5 (the fixed frontier); `l1x150`, `l1x175` L1 only; `linfx125` Linf only.
RAMP trains no ℓ2, so these arms raise the total budget; they are **not** a reallocation, they test the direction of the effect.
Evaluation: our fixed-ε AutoAttack (`aat.evaluate`, `configs/official_eval.yaml`) only. The official script's own evaluation is at the training
ε (`RAMP.py` lines 193–195) and is never reported.

### Prediction (minimax theory)

In RAMP's fine-tune ℓ1 is the slack norm (published: ℓ∞ 45.4, ℓ1 47.2, union 43.1, clean 81.1). So:

1. `f100` should land near the published 43.1 union / 81.1 clean (within 2 points of each). If not, our evaluation or setup differs and Part 2 cannot be read.
2. The `l1x150` and `l1x175` arms land at about 0 or **below** the fixed frontier (`f100`, `f125`, `f150`).
3. The `linfx125` arm lands at most about +1 above the frontier (so it will usually count as "on" the frontier below).

### Definitions (fixed in advance)

* Frontier: piecewise-linear interpolation of union against clean through `f100`, `f125`, `f150` (seed means). The gap of an arm is its union minus the
  frontier at its own clean accuracy, **defined only if its clean accuracy lies between the smallest and largest clean of the frontier arms**
  (audit item 21); otherwise report "no gap" and the raw numbers.
* Threshold T = max(2.0, the larger of the arm's two-seed range and the nearest frontier arm's two-seed range). An arm is **above** if gap > T, **below** if gap < −T,
  **on** the frontier otherwise. Two seeds cannot establish significance; this is a decision rule, not a test.

### Outcomes (exhaustive; apply only if `f100` is within 2 of the published numbers)

Let U = the `l1x150` and `l1x175` arms, and L = the `linfx125` arm.

| Outcome | Condition | Reading |
|---|---|---|
| 1. Theory falsified | some arm in U is above, and its gap exceeds the gap of L | the weak-ℓ1-attack explanation of our +6 is weakened; ℓ1-heavy budgets help even where ℓ1 is not the weak norm |
| 2. Prediction 2 violated, theory not falsified | some arm in U is above, but its gap does not exceed the gap of L (or L has no gap) | raising budget in either direction helps; the direction is not resolved; report as inconclusive |
| 3. Supported | no arm in U is above, at least one is below, and L is above | raising ℓ1 hurts, raising ℓ∞ helps: the allocation story follows the weak norm |
| 4. Supported in ℓ1 only | no arm in U is above, at least one is below, and L is on or below | raising ℓ1 hurts as predicted; no ℓ∞ effect resolved |
| 5. No transfer | all arms in U are on the frontier, whatever L does | no allocation effect resolved on the official pipeline; consistent with, but not proof of, the weak-ℓ1-attack explanation; report the 10-epoch result as pipeline-specific |

Any arm without a defined gap is reported with raw numbers and excluded from the conditions above.

## Not claimed

Nothing here tests the controller, the 80-epoch schedule, or adaptive ε. If Part 2 supports transfer, controller v2 (evaluation-grade feedback)
is the next step; if not, the allocation result is reported as specific to our pipeline.

## Results

(to be added, dated, below this line)
