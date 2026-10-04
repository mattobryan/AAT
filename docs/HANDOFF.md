# Handoff (4 October 2026)

Start a new session from this file. Read `CLAUDE.md` first (agent routing, the reviewer gate, never reintroduce
the thesis flaws), then this note. Branch: `matt/funny-planck-3j7md5`. Checkpoints and logs: the Hugging Face
repo in `$HF_REPO`, under `ramp_official/<run>/`. Never print tokens.

## What is established (numbers were recomputed by the reviewer from the eval JSONs)

All results: AutoAttack (APGD-CE + APGD-T), fixed ε (8/255, 0.5, 12), first 1000 CIFAR-10 test images, 2 seeds,
PreActResNet-18, 80 epochs from scratch, official RAMP code. Union is the per-sample AND over the three norms.

| Run (mean of 2 seeds) | Clean | ℓ∞ | ℓ2 | ℓ1 | Union | Worst class | Class std |
|---|---|---|---|---|---|---|---|
| RAMP (locked baseline) | 81.3 | 45.7 | 66.4 | 48.4 | 44.4 | 12.8 | 20.9 |
| aat_rel | 82.9 | 45.8 | 66.9 | 48.5 | 44.7 | 10.8 | 22.9 |
| aat_flip | 78.1 | 41.4 | 64.1 | 45.4 | 40.1 | 7.2 | 19.2 |

* RAMP reproduces the thesis Table 7.1 to within 1 point on every metric (`baselines/results/RAMP_BASELINE.md`).
* aat_rel / aat_flip = RAMP plus a per-class, per-norm training ε controller, budget-neutral (mean ε stays at the
  nominal value). aat_rel: classes with higher robust accuracy get more ε. aat_flip: the reverse.
  Result: no union gain for aat_rel (+1.6 clean), a loss for aat_flip (both seeds below both RAMP seeds).
* Diagnosis (RAMP): union is within 2.4 points of min(ℓ∞, ℓ1) in every class, overall within 1.3 points. The weak
  classes (cat, deer, dog, bird) are weak under every norm. Norm scheduling therefore has little headroom, and
  reallocating ε did not close the class gap.
* Per-class cells hold about 100 images (standard error about 5 points); per-class seed spread reaches 12
  points. Per-class differences are descriptive. n = 2 seeds.
* Files: `baselines/results/*_eval_autoattack.json`, `docs/figures/aat_three_way.png` and `panel_*.png`,
  `scripts/plot_progress.py` (needs `HF_REPO`, `HF_TOKEN`, matplotlib), `docs/thesis_audit.md` (16 items).
* Deck (private Slides artifact): https://claude.ai/artifact/UghsJMy4QWa6C6e2AQoQPq. Reviewed against the JSONs.

## Hypotheses (written down; see the deck)

H1 hard classes are data-limited; H2 ε reallocation cannot raise union (post hoc, observed); H3 norm scheduling has
little headroom; H4 weak classes are capacity-limited; H5 the worst-class number is largely noise.

## Open tasks, in order

1. **H5, checkpoint averaging.** Average RAMP's saved epoch 60/70/80 weights (HF `ramp_official/ramp_scratch_l5_s{0,1}/ep_*_0.pth`),
   evaluate with the same protocol. About 35 session-minutes. Shows how much of the worst-class number is noise.
2. **H1 proxy, class-sampling fine-tune.** From the locked epoch-80 weights, 10 more epochs, control arm (RAMP only) versus a
   budget-neutral feedback-driven class sampler. Cost about 2.5 session-hours per seed pair plus evaluation. Low expected
   power: a weak-class change under about 5 points cannot be resolved with ~100 images per class.
   Reviewer notes to carry into any write-up: the averaged model's BN running stats are an approximation (no BN recompute) and
   ep_60/70 come from the lr 0.05 phase, before the decay; the matched comparison for the sampler is `rampsamp` vs `rampcont` (both use the
   same with-replacement sampler, control with beta 0), not vs the 80-epoch RAMP. Both arms continue at lr 0.005 with momentum restarted.
   Code: `trainpair`/`evalpair` and `avg2` in `scripts/ramp_official.sh`; notebooks `ramp_avg_kaggle.ipynb`, `ramp_samp_kaggle.ipynb`.
3. **WRN-28-10 RAMP fine-tune (deferred).** Code is in (`ramp_wrn`, `notebooks/ramp_wrn_kaggle.ipynb`). The official script needs the
   500K aux pickle (no verified source) and doubles the images per epoch. Estimated 14+ GPU-hours per seed. A cheaper variant would
   fine-tune the same WRN on CIFAR-10 alone with plain `RAMP.py`. Decide only with a quota number in hand.
4. Short results note in the repo stating the two reallocation results and the class by norm diagnosis (reviewer-gated).
5. Optional: the full thesis method (`aat_full` in `aat/train.py`: adaptive norm weights, hybrid loss, KL pairing) has not been run.

## Constraints and how to work

* Kaggle: T4 x2, about 10 session-hours left this week (assumed; confirm). Save Version, then Save and Run All. Notebooks
  stop at the first failed stage, push checkpoints to HF, and send ntfy notifications. Re-import a notebook from GitHub before
  each run so it is the current version. `notebooks/aat_on_ramp_kaggle.ipynb` (METHOD = aat_rel / aat_flip / aat_ramp).
* Reviewer gate (CLAUDE.md): any change to `aat/attacks.py`, `aat/evaluate.py`, the `eval:` block of `configs/base.yaml`, or a
  sentence stating a quantitative result needs the Opus `reviewer` before commit. Evaluation always uses fixed ε.
* Kaggle's "Save to GitHub" pushes notebook commits to this branch: `git fetch` and merge before pushing, never force-push.
* Save tokens: start from this file, not from old history; read logs with `python -c` or `jq`, never whole files; one reviewer
  pass per experiment; use ntfy plus one check at the start and one at the end of a run instead of an hourly monitor
  (`trig_01WpyYX24gXUywpKi4NA1gqs` is paused).
