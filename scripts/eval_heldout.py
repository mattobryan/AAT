"""Fixed-epsilon AutoAttack (APGD-CE + APGD-T, same settings as aat.evaluate) on a HELD-OUT slice of the CIFAR-10 test set.

    python scripts/eval_heldout.py --run runs/v3a_max_realloc_s0                     # test images 1000..1999
    python scripts/eval_heldout.py --run runs_official/rampeps_f100_ft_s0 --ckpt <ep_3_0.pth> --config configs/official_eval.yaml

The static reallocation ratio and the controller's start point were chosen from results on the first 1000 test images
(audit items 23, 25), so any claim about them needs images that took no part in that choice. This script scores the same
fixed budgets on images [start, start+n) (default 1000..1999) and writes eval_heldout.json: clean, per-norm robust,
union (per-sample AND), n and start. Seen budgets only, no unseen grid. Numbers are comparable across models within
this file's protocol, and are NOT directly comparable with eval_autoattack.json (different images, and the unseeded
AutoAttack there). aat/evaluate.py and aat/attacks.py are imported, not modified.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from aat.evaluate import per_class  # noqa: E402
from aat.models import build_model, load_weights  # noqa: E402
from aat.utils import device, load_config, set_seed  # noqa: E402
from eval_gated_adaptive import attack, predict  # noqa: E402

LABEL = "held-out test slice, AutoAttack APGD-CE + APGD-T (seed 0), per-sample AND; not comparable with eval_autoattack.json"


def heldout_eval(model, x, y, seen, bs=500, dev="cpu", attacks=("apgd-ce", "apgd-t"), C=10):
    """clean accuracy, per-norm robust accuracy and union on (x, y) at the fixed budgets in `seen`."""
    clean = predict(model, x, bs, dev) == y
    union, robust = clean.clone(), {}
    for p, eps in seen.items():
        x_adv = attack(model, x, y, p, eps, attacks, bs, dev, seed=0)
        m = (predict(model, x_adv, bs, dev) == y) & clean
        robust[p] = m.float().mean().item()
        union &= m
    uc = per_class(union, y, C)
    worst = [v for v in uc if v is not None]
    return {"clean": clean.float().mean().item(), "robust": robust, "union": union.float().mean().item(),
            "union_worst_class": min(worst) if worst else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None, help="default: <run>/final.pt")
    ap.add_argument("--config", default=None, help="default: <run>/config.yaml")
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=500)
    a = ap.parse_args()
    from aat.data import get_test_tensors

    cfg = load_config(a.config or os.path.join(a.run, "config.yaml"))
    dev = device()
    set_seed(0)
    model = build_model(cfg["model"], cfg["data"]["dataset"]).to(dev)
    load_weights(model, a.ckpt or os.path.join(a.run, "final.pt"), map_location=dev)
    model.eval()
    x, y = get_test_tensors(cfg, a.start + a.n)
    x, y = x[a.start:], y[a.start:]
    assert len(x) == a.n, (len(x), a.n)
    t0 = time.time()
    res = heldout_eval(model, x, y, cfg["eval"]["seen"], a.bs, dev, C=100 if cfg["data"]["dataset"] == "cifar100" else 10)
    res.update({"run": a.run, "name": cfg["name"], "seed": cfg.get("seed", 0), "backend": "heldout", "label": LABEL,
                "trained_norms": cfg["norms"], "start": a.start, "n": a.n, "eval_time_s": time.time() - t0})
    with open(os.path.join(a.run, "eval_heldout.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("name", "clean", "robust", "union", "start", "n")}))


if __name__ == "__main__":
    main()
