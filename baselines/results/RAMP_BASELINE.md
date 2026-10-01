# RAMP baseline (from scratch, λ=5, 80 epochs) — locked

Official RAMP code (commit be4971f) + resume/HF patch, PreActResNet-18, CIFAR-10, Kaggle 2×T4 across two sessions.
Evaluation: AutoAttack (APGD-CE + APGD-T, standard settings), first 1,000 test points, fixed ε (8/255, 0.5, 12), union = per-sample AND.

| | Clean | ℓ∞ | ℓ2 | ℓ1 | Union |
|---|---|---|---|---|---|
| Seed 0 | 80.7 | 47.3 | 66.2 | 49.0 | 45.7 |
| Seed 1 | 81.9 | 44.1 | 66.7 | 47.8 | 43.1 |
| **Mean ± sd (2 seeds)** | **81.3** ± 0.8 | **45.7** ± 2.3 | **66.4** ± 0.4 | **48.4** ± 0.8 | **44.4** ± 1.8 |
| Thesis Table 7.1 | 81.3 | 45.96 | 65.74 | 48.42 | 44.5 |
| RAMP paper Table 3 | 81.2 | 46.0 | 65.8 | 48.3 | 44.6 |
| Δ mean − thesis | -0.0 | -0.3 | +0.7 | -0.0 | -0.1 |

All five metrics are within 1 pp of the thesis and paper values (verdict: OK).
Cross-check: the official `eval.py` on the same 1,000 points gave seed 0: 80.7/47.3/66.2/48.6/45.5 and seed 1: 81.9/44.1/66.7/47.8/43.0 — within 0.4 pp of ours (AutoAttack randomness).

Checkpoints (Hugging Face, private): `matokebryan/aat-checkpoints` → `ramp_official/ramp_scratch_l5_s{0,1}/ep_80_0.pth`.
