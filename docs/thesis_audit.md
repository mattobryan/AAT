# Thesis audit: what the Kaggle rebuild has to fix

These inconsistencies are in the May 2025 thesis. Any of them would be spotted by a PhD panel
or a reviewer, so the rebuild treats the thesis numbers as **hypotheses to re-test**, not results
to reproduce. Page numbers refer to the PDF.

## Results that contradict each other

| Claim | Where | Conflict |
|---|---|---|
| Variant 3B union robustness | p.41 Table 7.2: **33.1%** · p.49 §7.8: **72.5%** · p.73 Table A.6: **48.0%** | Three different numbers for one model. The 72.5 is 3B's ℓ2 accuracy from Table A.6. |
| Variant 3B clean accuracy | Table 7.2: 82.4 · §7.8 / A.6: 88.2 | Two numbers. |
| 3B is "best" vs "worst" | §7.2 says 3B "shows the lowest performance across all metrics". §7.8, §8.3 and §9.1 say it is Pareto-dominant | These conclusions contradict each other. |
| RAMP baseline | Table 7.1: (81.3 clean, 44.5 union) · §7.8: (85.0, 60.1) | Two baselines. |
| 1B: +9.7 pp union, −1.5 pp clean | Abstract, §7.8, §9.1 | Table 7.2 gives 1B 41.2 vs 1A 41.4 union, which is *lower*. Table A.3 gives +9.0 pp union and *+2.0* pp clean. No table gives +9.7 / −1.5. |
| Variant A / B labels | Table 6.1: A = custom, B = RAMP-like · §7.4–7.5: A = RAMP-buffered, B = custom | Swapped. |
| Variant B ℓ1 robust 73.8% > clean 63.4% | Table 7.5 | Robust accuracy cannot exceed clean accuracy on the same points under an untargeted attack. Either different subsets were used or the ℓ1 attack is broken. |
| Union = min over norms | §7.8 formula | Union is the per-sample AND (§4.4 defines it correctly), and it is ≤ the min. Table A.6 has union 48.0 < ℓ∞ 49.0, which is consistent with AND, not with min. |

## Method issues

1. **Evaluation at adaptive ε (the biggest issue).** Strategy A/B in §A.2 evaluate each sample
   at its *own* adaptive ε, with vulnerable samples getting smaller ε. That inflates robust
   accuracy and makes the numbers impossible to compare with RAMP's fixed-ε results.
   **Fix:** every evaluation uses fixed ε (8/255, 0.5, 12). Adaptive ε is for training only.
2. **Eq. A.5 (3B's ε) keys on the label index:** ε·(1 + 0.05·y). Class 9 gets 1.45× the
   budget of class 0 for no reason tied to vulnerability, and the ε exceeds ε_max. Kept as
   `v3b_thesis.yaml` for faithfulness only. The principled version is `v3b_classeps.yaml` (Eq. 4.19).
3. **Eq. A.1 sign.** As printed, harder samples get *larger* ε, but the text says they should
   progress more gradually. The code follows the text.
4. **Why the budget collapsed ("zero-budget classes", §7.4, §7.7).** Measuring robustness at the
   *nominal* ε and then lowering ε never raises that measurement, so ε ratchets down to the floor.
   **Fix:** the ε controller reads robustness at the *current* per-class ε (Eq. 4.2 as written).
   Norm weights read it at nominal ε, and the floor is 0.5×nominal instead of 0.
5. **1B and 3B are the same definition in §4.3** ("static multi-norm + adaptive ε"). They are
   separated here as 1B = per-sample (A.1) and 3B = class-feedback (4.19). Full AAT = 3B +
   adaptive norm weights + hybrid loss + KL (the thesis's "3C").
6. **E-AT (§3.4) uses the ℓ1/ℓ∞ extremes, not ℓ∞/ℓ2.** The convex hull of the ℓ1 and ℓ∞ balls
   is what covers the intermediate ℓp balls.
7. **RAMP reproduction (§6.2) is described as 50 ep ℓ∞ then 50 ep ℓ2.** RAMP pairs ℓ∞ and ℓ1 with
   logit pairing and GP. The rebuild implements the ℓ∞+ℓ1 form. As printed, Eq. 3.6 (GP) reduces
   to a per-layer rescaling of the adversarial gradient. **Check against the official code on day 1.**
8. **The unseen-threat evaluation promised in the task description is missing.** Added:
   an ε-grid beyond the training budgets, plus a held-out norm (ℓ2 for all ℓ∞+ℓ1 runs, including
   `aat_full_no_l2`).
9. **Robust numbers come from "full PGD", not AutoAttack.** The rebuild uses AutoAttack
   APGD-CE + APGD-T on the first 1000 test points, per norm, with the union as a per-sample AND.
10. **Compute is not matched.** 3-norm methods run ~3× more attack steps per batch than E-AT or
    2A/2B. The summary table reports train GPU-minutes for every run.

## Found while reproducing RAMP (step 1)

11. **Thesis Table 7.1 is the paper's from-scratch λ=5 result.** Its averages match RAMP Table 3
    (81.2 / 46.0 / 65.8 / 48.3 / 44.6) to within 0.1 pp. That setting takes about 3.5 h per seed on an
    A100, and the public code's `--gp` path raises a NameError (`utils.py` lacks `import copy`). How
    those five runs were produced needs to be documented. The rebuild reproduces the fine-tuning
    setting (Table 24) instead; see `docs/ramp_reproduction.md`.
12. **The evaluation protocol is the first 1,000 test points.** `pretr_Linf` gives 83.7 % there, which
    matches the paper, versus 82.8 % on the full 10k.

## Found while reviewing AAT-on-RAMP (commit 1806a84)

13. **Controller set-point vs. achievable training robustness.** The ε controller (τ = 0.5) reads
    10-step APGD robust accuracy on training batches. If a class's training robust accuracy stays
    under 0.5 (likely for cat/deer/bird, which sit at 13–37 % union on test for RAMP), its ε falls
    by up to 0.025×nominal per epoch and stays at the 0.5×nominal floor. AAT then mostly means
    "RAMP with smaller training ε for hard classes". A clean-up/robust-down shift against RAMP is
    then a budget effect, not an effect of adaptivity. **Fix:** check the `[aat-eps]` log lines
    early in the run. Report mean and per-class ε trajectories next to results. Attribute any
    difference to *adaptivity* only against a fixed-ε control with matched mean training ε.
14. **Run-vs-run comparisons must check that the protocol matches.** `scripts/compare_runs.py`
    must refuse to compare files whose `n`, `backend`, `attacks` or `unseen` grid differ. With two
    seeds per arm, differences are descriptive only.
    Addressed by `--aat_relative` (method `aat_rel`): ε moves with A_c − mean_c A_c and is projected
    so that mean_c ε_c = nominal (to fp32 precision, ~3e-7 relative) inside [0.5, 1.5]×nominal. The
    mean training budget then matches RAMP, but per-class ε still differs, so results remain descriptive.
15. **Per-class noise in figures (review of `scripts/plot_progress.py`).** Per-class cells have about
    100 images (binomial s.e. ≈ 5 pts). The seed-to-seed spread for the same method reaches 11.6 pts
    for union (RAMP frog) and 9.3 pts for clean (RAMP dog). A fixed "≥ 3 pts" threshold therefore
    marks noise as an effect. Annotate a per-class difference only when the two 2-seed ranges do not
    overlap, and call it descriptive. Also, the Hub copy of `ramp_scratch_l5_s0/eval_autoattack.json`
    differs from the locked `baselines/results` file by about 1 pt per class on plane/car/ship union.
    The locked file is authoritative. Do not re-pull the baseline from the Hub.
16. **Interpretation limits for the aat_rel / aat_flip results (results-deck review).** The ceiling
    union ≤ min single-norm applies only to a given model. RAMP's 1.3-pt gap (44.4 vs 45.7 ℓ∞) bounds
    better *alignment* for RAMP's weights, not what a different training schedule could reach. Two
    allocation rules at one step size do not show that reallocating ε "cannot" help. Statements that
    weak classes "lack robust features" or are "data-limited" are hypotheses (H1/H4), not findings.
17. **Hypothesis-label collision (results-note review, 5 Oct 2026).** Item 16, `scripts/train_vs_test.py` and
    the results note use "H1" for *data-limited* and "H4" for *optimisation/capacity-limited*, but README
    defines H1 as "full AAT beats fixed-schedule AT on union" and H4 as "AAT generalizes to unseen threats".
    Use distinct labels (e.g. D1 data-limited, D2 capacity-limited) for the weak-class diagnosis, and map the
    aat_rel/aat_flip results to README H2 (class-feedback ε narrows the class gap), which they do not support.
