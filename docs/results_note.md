# Results note: does reallocating ε or samples help the weak classes? (4–5 October 2026)

Status: draft for reviewer sign-off. All numbers come from `eval_autoattack.json` files in `baselines/results/`
(AutoAttack APGD-CE + APGD-T at fixed ε = 8/255, 0.5, 12; first 1000 CIFAR-10 test images; union = per-sample
AND over the three norms; PreActResNet-18; 2 seeds per arm). Per-class cells hold about 100 images (binomial
standard error about 5 points), and 2 seeds cannot support significance claims, so every difference below is
descriptive.

## What was run

| Arm | Description |
|---|---|
| RAMP | Official RAMP, 80 epochs from scratch (locked baseline; reproduces thesis Table 7.1 within 1 point) |
| aat_rel / aat_flip | RAMP plus a per-class, per-norm training-ε controller with the mean ε held at nominal. aat_rel gives classes with higher training robust accuracy more ε; aat_flip does the reverse |
| cls_ctrl / cls_fb | Locked epoch-80 RAMP weights continued for 10 epochs (static lr 0.05) with a with-replacement sampler. cls_ctrl: uniform class weights. cls_fb: weights clip(1 + (Ā − A_c)/Ā, 0.5, 1.5), mean 1, from per-class training-attack robust accuracy |
| avg | Uniform average of the epoch 60/70/80 weights of RAMP (BN statistics averaged, not recalibrated) |

## Results

Overall (2-seed means, %):

| Run | Clean | ℓ∞ | ℓ2 | ℓ1 | Union | Worst-class union |
|---|---|---|---|---|---|---|
| RAMP | 81.30 | 45.70 | 66.45 | 48.40 | 44.40 | 12.78 |
| aat_rel | 82.90 | 45.80 | 66.85 | 48.55 | 44.65 | 10.83 |
| aat_flip | 78.15 | 41.40 | 64.15 | 45.40 | 40.15 | 7.22 |
| cls_ctrl | 82.50 | 45.00 | 65.40 | 48.50 | 43.45 | 12.78 |
| cls_fb | 82.65 | 45.70 | 65.80 | 48.60 | 44.55 | 13.33 |

1. **ε reallocation.** aat_rel did not raise union over RAMP (44.65 vs 44.40), gained 1.6 clean points, and
   lowered worst-class union in both seeds (10.0 and 11.7 vs RAMP 13.3 and 12.2). aat_flip lowered union
   (40.15); both of its seeds (42.0, 38.3) are below both RAMP seeds (45.7, 43.1). Neither supports README H2
   (class-feedback ε narrows the class gap): worst-class union fell in both arms.
2. **Sample reallocation.** Over 2 seeds, feedback class sampling was within noise of the uniform control
   (union 44.55 vs 43.45; worst-class 13.33 vs 12.78; mean union over bird, cat, deer, dog 21.1 vs 20.6), and
   neither arm exceeded locked epoch 80 by more than 0.2 points (cls_fb 44.55, cls_ctrl 43.45 vs 44.40). The unseen-ε grid differs by at most 1 point between the arms. This test
   is underpowered for effects under about 5 points per class, so a small benefit cannot be excluded. Oversampling
   repeats images, so it does not test whether more distinct data would help.
3. **Checkpoint averaging.** Uniform averaging of the epoch 60/70/80 weights lowered worst-class union in both
   seeds (13.3→10.7, 12.2→9.7; the worst class moved from deer to cat, about 3 images), changed union by −1.3/+2.8
   and clean by −0.8/−2.2. This does not support the claim that the worst-class number is mostly checkpoint noise,
   but given the size of the effect and the unrecalibrated BN statistics it does not refute it.
4. **Class by norm.** For RAMP, in the 2-seed mean, union is at most 2.5 points below min(ℓ∞, ℓ1) in every class
   (per seed up to 4.4, deer seed 1) and 1.3 points overall. The weak classes (cat, deer, dog, bird) are weak
   under every norm. For RAMP's weights, better alignment across norms could add at most about 1.3 points of
   union; this does not bound what a different schedule could reach.
5. **Train versus test (RAMP epoch 80, first 1000 train and test images; figures are unweighted class means
   over 2 seeds).** Union on the weak four classes is 32.8 on training images and 20.8 on test images (gap 12.0);
   on the other six classes it is 69.45 and 58.93 (gap 10.5). Clean accuracy of the weak four on training images
   is 79.1, against 95.3 for the others. The weak classes are therefore also poorly fit on images the model was
   trained on, and their train-test gap is not distinguishable from that of the other classes. Per class, cat
   (4 to 6) and dog (4 to 6) have small gaps; deer (20 to 28) and bird (11 to 17) have large ones, as do some
   strong classes (airplane 16 to 20, automobile 24), while frog is 0 and -7. Train images are unaugmented and
   each class cell has about 100 images. The test-pass union for seed 0 is 45.8 against 45.7 in the locked eval
   JSON (one image of attack variance); the locked eval JSON is authoritative.

## Interpretation (hypotheses, not findings)

None of the three reallocation experiments (ε by relative robustness, ε reversed, samples) raised union or the
weak classes; reversing the ε allocation (aat_flip) lowered both, and aat_rel lowered worst-class union. The weak
classes are also poorly fit on training images. That is consistent with an optimisation- or capacity-limited
explanation (D2), but equally with low power, narrow reallocation ranges, short fine-tuning and repeated images,
and nothing here changed model capacity or the number of distinct training images. D1 (the weak classes are
data-limited) and D2 remain open. (D1 and D2 are labels for this diagnosis only; they are not the README's H1 to H4.) A model-capacity test (WRN-28-10, or a wider PreActResNet-18) or longer training would be the discriminating experiment.
