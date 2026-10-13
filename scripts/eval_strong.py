"""Stronger fixed-epsilon evaluation: the full AutoAttack `standard` suite (APGD-CE, APGD-T, FAB-T, Square) at the SAME
fixed seen budgets as aat.evaluate, with the contribution of each extra attack reported.

    python scripts/eval_strong.py --run runs/v3a_max_realloc_s0

aat.evaluate (and RAMP / E-AT's eval.py) use only APGD-CE + APGD-T. FAB-T is a minimum-norm attack and Square is
gradient-free, so a model whose robustness comes from masked or sparse-attack-specific gradients loses accuracy to them. For
each norm the script runs APGD-CE + APGD-T (seed 0, the reference), then each extra attack only on the points that survived
everything before it (the AutoAttack protocol), and scores every adversarial on the real model. Output: eval_strong.json,
with `robust` / `union` for the full suite and `robust_standard` / `union_standard` for APGD-CE + APGD-T in the same file,
`extra_drop` = points removed per extra attack, and the usual keys, so `scripts/aggregate.py --tag strong` works. Numbers here
are NOT comparable to the APGD-CE + APGD-T numbers in eval_autoattack.json or to RAMP's 44.4; compare only within the file.
aat/evaluate.py and aat/attacks.py are imported, not modified.
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
from aat.data import get_test_tensors  # noqa: E402
from aat.evaluate import per_class  # noqa: E402
from aat.models import build_model, load_weights  # noqa: E402
from aat.utils import device, load_config, set_seed  # noqa: E402
from eval_gated_adaptive import attack, predict  # noqa: E402

LABEL = "full AutoAttack standard (APGD-CE + APGD-T + FAB-T + Square), per-sample AND; not comparable to the 2-attack numbers"


def strong_norm(model, x, y, clean, p, eps, base=("apgd-ce", "apgd-t"), extras=("fab-t", "square"), bs=500, dev="cpu"):
    """Survival masks for one norm: (standard mask, full-suite mask, {extra attack: points it removed})."""
    x_adv = attack(model, x, y, p, eps, base, bs, dev, seed=0)
    m_std = (predict(model, x_adv, bs, dev) == y) & clean
    m, drops = m_std.clone(), {}
    for a in extras:
        idx = m.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            drops[a] = 0
            continue
        x_a = attack(model, x[idx], y[idx], p, eps, (a,), bs, dev, seed=0)
        still = predict(model, x_a, bs, dev) == y[idx]
        drops[a] = int((~still).sum())
        m[idx[~still]] = False
    return m_std, m, drops


def run(run_dir, n=None, bs=500, extras=("fab-t", "square")):
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    dev = device()
    set_seed(0)
    ecfg = cfg["eval"]
    n = n or ecfg["n"]
    C = 100 if cfg["data"]["dataset"] == "cifar100" else 10
    model = build_model(cfg["model"], cfg["data"]["dataset"]).to(dev)
    load_weights(model, os.path.join(run_dir, "final.pt"), map_location=dev)
    model.eval()
    x, y = get_test_tensors(cfg, n)
    t0 = time.time()
    clean = predict(model, x, bs, dev) == y
    res = {"run": run_dir, "name": cfg["name"], "seed": cfg.get("seed", 0), "backend": "strong", "label": LABEL,
           "attacks": ["apgd-ce", "apgd-t", *extras], "n": n, "trained_norms": cfg["norms"],
           "clean": clean.float().mean().item(), "clean_per_class": per_class(clean, y, C),
           "robust": {}, "robust_standard": {}, "robust_per_class": {}, "extra_drop": {}}
    union, union_std = clean.clone(), clean.clone()
    for p, eps in ecfg["seen"].items():
        m_std, m, drops = strong_norm(model, x, y, clean, p, eps, extras=extras, bs=bs, dev=dev)
        res["robust_standard"][p], res["robust"][p] = m_std.float().mean().item(), m.float().mean().item()
        res["robust_per_class"][p], res["extra_drop"][p] = per_class(m, y, C), drops
        union, union_std = union & m, union_std & m_std
        print(f"{cfg['name']}: {p} eps={eps:.4g} standard={res['robust_standard'][p]:.4f} "
              f"full={res['robust'][p]:.4f} extra_drop={drops}", flush=True)
    res["union"], res["union_standard"] = union.float().mean().item(), union_std.float().mean().item()
    uc = per_class(union, y, C)
    res["union_per_class"] = uc
    vals = [v for v in uc if v is not None]
    res["union_worst_class"] = min(vals)
    res["union_class_std"] = float(torch.tensor(vals).std()) if len(vals) > 1 else 0.0
    res["sanity"] = {"full_le_standard": all(res["robust"][p] <= res["robust_standard"][p] + 1e-9 for p in res["robust"]),
                     "robust_le_clean": all(res["robust"][p] <= res["clean"] + 1e-9 for p in res["robust"])}
    res["eval_time_s"] = time.time() - t0
    with open(os.path.join(run_dir, "eval_strong.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("name", "clean", "robust", "union", "union_standard", "sanity")}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--bs", type=int, default=500)
    ap.add_argument("--extras", nargs="*", default=["fab-t", "square"])
    a = ap.parse_args()
    run(a.run, n=a.n, bs=a.bs, extras=tuple(a.extras))


if __name__ == "__main__":
    main()
