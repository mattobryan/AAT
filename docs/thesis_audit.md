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
