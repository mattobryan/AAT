# Step 1: lock in the RAMP baseline

The method: reproduce the published numbers with the **official code**
([uiuc-focal-lab/RAMP](https://github.com/uiuc-focal-lab/RAMP), pinned commit `be4971f`). That gives
us RAMP's weights. Those weights and numbers become the fixed baseline that AAT is measured against.

## Targets (`baselines/targets.yaml`)

| Setting | Clean | ℓ∞ | ℓ2 | ℓ1 | Union | Role |
|---|---|---|---|---|---|---|
| **From scratch**, 80 ep, λ=5, GP (paper Table 3) | 81.2 | 46.0 | 65.8 | 48.3 | 44.6 | **Thesis baseline** (thesis Table 7.1: 81.3 / 45.96 / 65.74 / 48.42 / 44.5) |
| Fine-tune `pretr_Linf` 3 ep, λ=1.5 (paper Table 24) | 81.1 | 45.4 | 66.1 | 47.2 | 43.1 | Cheap cross-check of the pipeline |
| `pretr_Linf` (start point) | 83.7 | 48.1 | 59.8 | 7.7 | 38.5 | Sanity check of the evaluation |

## How the 12-hour Kaggle limit is handled

A from-scratch run takes longer than one Kaggle session, so every run can pause and resume:

* **`baselines/ramp_resume_hub.patch`** (≈90 lines) adds to the official `RAMP.py`:
  * `--resume`: restores both models, both optimizers, the stats, and all random-number-generator states.
  * `--time_budget_h`: exits cleanly after the last epoch that fits in the session.
  * `--hf_repo`: pushes the resume checkpoint, the logs, and `ep_*.pth` to Hugging Face after every epoch.

  The method itself is untouched; the diff is limited to the resume/sync hooks. One missing
  `import copy` in `utils.py`, which crashes the `--gp` path, is also patched.
* **`aat/hub.py`** does the syncing. It reads the token from the Kaggle secret `HF_TOKEN` and the
  repo from `HF_REPO`. The HF repo is created private. Resume checkpoints are overwritten in place
  and the history is squashed, so storage stays at roughly 200 MB per run rather than growing
  every epoch. Failed uploads only print a warning; they never stop training.
* **Our own trainer (`aat/train.py`)** does the same for AAT runs: it pulls `last.pt` at start, pushes
  it after every epoch, and stops cleanly after `train.time_budget_h`.

**Workflow:** in Kaggle, use *Save Version → Save & Run All*. The run continues in the background,
and at 11 h it pushes its checkpoint and exits. Run the notebook again and it continues from the
Hub. Finished runs are detected (the official final-eval log is on the Hub) and skipped.

## Verified so far (CPU)

* Our `preactresnet18_softplus` port gives the same logits as the official model class, to within 5e-7.
* `pretr_Linf` scores 83.7 % clean on the first 1,000 test points, matching the paper. So
  evaluation uses the first 1,000 points, with AutoAttack `standard` settings restricted to
  APGD-CE + APGD-T, the same as the official `eval.py`.
* The patch applies cleanly to the pinned commit, and the patched script compiles.

## Kaggle order (`notebooks/ramp_baseline_kaggle.ipynb`)

1. One-time: add an HF write token as the Kaggle secret `HF_TOKEN`, and set `HF_REPO`.
2. Sanity: evaluate `pretr_Linf`.
3. **A:** from-scratch RAMP, seeds 0 and 1 in parallel (one per GPU), resumed across sessions until
   both are done. Then seeds 2–4 if the quota allows. The first logged epoch time tells you how
   many sessions each seed needs.
4. **B:** fine-tuning cross-check on leftover GPU time (5 seeds, about 1 GPU-h each).
5. `compare_targets.py` marks each metric OK (≤1 pp), CLOSE (≤2 pp) or OFF.

Seed 0 also runs the official final evaluation on the same 1,000 points, so our evaluator
is checked against theirs.

## After the baseline is locked

The baseline checkpoints live on the Hub under `ramp_official/<run>/ep_*.pth`.

1. The main comparison: AAT trains from the same start as RAMP, under the same budget and the same
   evaluation, so any gain comes from the adaptive ε and norm scheduling.
2. Optional: AAT fine-tunes the RAMP weights. This needs a control where RAMP itself continues
   for the same number of epochs.
