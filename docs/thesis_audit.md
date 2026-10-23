# Thesis audit: what the Kaggle rebuild has to fix

These inconsistencies are in the May 2025 thesis. Any of them would be spotted by a PhD panel
or a reviewer, so the rebuild treats the thesis numbers as **hypotheses to re-test**, not results
to reproduce. Page numbers refer to the PDF (page index; the printed page number is one lower). Citations to Tables 7.2 and A.6,
§7.8 and the Content Summary were checked against the submitted 77-page version on 5 October 2026.

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
18. **Methods faithfulness of the AAT-on-RAMP arms (report review, 5 Oct 2026).** The ε controller changes only RAMP's
    ℓ∞ and ℓ1 training attacks (`baselines/aat_on_ramp.patch`), is a budget-neutral variant of Eq. 4.19 (τ replaced by the
    class-mean training-attack accuracy, α = 0.05, clipped to [0.5, 1.5]×nominal and projected to mean nominal), and is not
    the thesis controller. The class-sampling fine-tune runs at RAMP's final-phase lr 0.005 (`static` schedule is lr-max/10 after
    epoch 70, `external/ramp/utils.py`), not 0.05. Evaluation uses only APGD-CE and APGD-T of AutoAttack, and the unseen grid is
    APGD-CE only, so it is not comparable to standard-AA leaderboard numbers. The RobustBench Gowal2020 WRN checkpoint
    (`ramp_wrn`) was pre-trained with extra data, so a WRN fine-tune mixes capacity with data.
19. **CIFAR-100 and variance (check against the 77-page submitted thesis, 5 Oct 2026).** The Content Summary says evaluation is
    performed on CIFAR-10 and CIFAR-100, but no CIFAR-100 result appears in the text (the only other mentions are literature-review
    citations of other papers). The text contains no occurrence of "seed" or "standard deviation"; only the RAMP baseline (Table 7.1)
    is stated to be over five runs, so the ablation variants appear to be single runs without variance. Either run CIFAR-100 or drop
    the claim, and report seeds and spread for every variant.
20. **Evaluation of gated (squint) models (review, 12 Oct 2026).** A CE-trained gate saturates, so routing is almost
    piecewise constant and plain APGD can miss inputs pushed to a weaker expert. A robust number for a gated model counts only
    from `scripts/eval_gated_adaptive.py`: standard AutoAttack plus extra APGD-CE attackers (gate forced to each threat,
    gate gradient cut, straight-through temperature), per-sample AND, scored on the real model. It is an upper bound on true
    robustness. The comparison model (`v3a_max`) gets the same number of extra APGD-CE runs (seeds 1..N). AutoAttack in
    `aat.evaluate` is unseeded, so compare `robust` with `robust_standard` from the same `eval_adaptive.json`, never with the
    number in a different file. Squint has 2 seeds against 3 for the control and n = 1000, so a gap under about 2 pp is
    descriptive. `gate_acc` in `log.jsonl` is training-batch accuracy under 10-step PGD, not routing under AutoAttack.


## Found while building the per-norm reallocation study (15 Oct 2026, reviewer-drafted)

21. **Fixed-budget frontier gaps.** A frontier gap is defined only inside the clean-accuracy range of the fixed-budget runs (here
    74.95 to 85.87; `aat_full` at 86.9 lies outside). The frontier is also recipe-specific: the E-AT-style and RAMP-style fine-tunes land
    +3.3 and -1.0 off the `v3a_max` chord, so a gap of +1 to +2 for an adaptive variant is not attributable to adaptivity.
22. **Unseeded AutoAttack variance across files.** `aat.evaluate` is unseeded, so the same checkpoint scores differently in different
    files (`v3a_max_s0` union 30.0 in `eval_autoattack.json`, 29.1 in the in-file standard run of `eval_adaptive.json`). Compare
    `robust` with `robust_standard` from one file only. Lower-budget models lose more to seeded extra attacks (0.9 to 1.7) than
    higher-budget ones (0.2 to 0.3).
23. **`realloc` was selected on test results.** The ratio 1.0/0.75/1.5 is the better of two hand-picked tries (+6.2 against +3.1), chosen
    on the first 1000 test images. Confirm on held-out test images (indices 1000 to 9999) before calling it a finding.
24. **The "unseen" grid overlaps training budgets.** `realloc` trains l1 at 18, so l1@18 is a seen budget for it; the l-inf price of
    reallocation is 1.7 points against `v3a_max` (same l-inf budget), not the gap to the 1.5x run, which is a budget effect.
25. **`normctl_ft` starts at a test-selected point, and the controller's state must survive resume.** `v3a_max_normctl_ft` is initialised
    at the `realloc` ratio (item 23) and every normctl arm uses its mean fraction 1.0833, so comparisons with `realloc` and the fixed runs
    are conditional on that selection. The controller drives the norms toward equal accuracy at their nominal eps, but those accuracies are
    not on a common scale (l2 at 0.5 scores above l1 at 12), so it may saturate at the bounds (l2 0.5x, l1 2.0x); check the logged `frac`
    and say so if it did. The `frac` state was not saved in the resume checkpoint until 20 Oct 2026; runs resumed before that would have reset it.
26. **The RAMP-style `ramprecipe_st_e80` recipe is partial, and its "realloc" row is a budget increase.** Our gradient projection pairs the clean and adversarial gradients of the same model per batch (the update always points along the adversarial gradient); the official `gp()` adds a term from a separately trained clean copy once per epoch. The official attacks are 10-iteration APGD for both norms, BN and bias are excluded from weight decay, and the KL term is anchored on l-inf. `ramprecipe_st_realloc_e80` only raises l1 to 18 (no l2 is trained in that recipe), so it enlarges the total budget; do not call it a reallocation.
27. **The realloc monitor runs at the training eps.** For `realloc` runs `monitor` attacks at the run's own nominal eps (l2 0.375, l1 18), not the evaluation eps, so monitor numbers are not comparable across allocations. Reported results come from AutoAttack only.
28. **The official `RAMP.py` evaluates at the training eps.** Lines 193-195 set the internal evaluation eps to the training eps (`args.all_epss[idx] = args.l_eps[i]`), so with `--l_eps` changed its own evaluation output is at a different budget from the standard (8/255, 0.5, 12). Report only our fixed-eps AutoAttack numbers for any official run with a modified budget; `--finetune_model` takes `--epochs`, whose default is 20, so pass 3 explicitly.
29. **`union ~ min over norms` is weakest at equalisation.** Across the 45 three-norm runs the mean gap between min_p a_p and union is 0.5 points, but the largest gap (2.8) is the controller started at the static allocation, where l_inf and l_1 are nearly equal, and fixed 1.5x has 1.0 to 1.3. The approximation, and any allocation argument built on it, is weakest exactly where an allocation is good.
30. **Frontier gaps need a clean range, and the held-out slice is small.** `np.interp` clamps outside the frontier's clean range, which gives a gap against an endpoint; gaps are reported only inside the range (item 21). The held-out check uses 1000 images (1000 to 1999, about 1.5 points of binomial error per model), not the 1000 to 9999 proposed in item 23; a pass there is a first slice, not full confirmation.
