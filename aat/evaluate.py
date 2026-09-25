"""Final evaluation at FIXED budgets (never the adaptive ones used in training).

Seen threats:   l_inf 8/255, l_2 0.5, l_1 12 (RAMP / E-AT / RobustBench convention)
Union:          per-sample AND over the three norms (a point counts only if it survives all)
Unseen threats: larger budgets per norm, and for runs trained without some norm, that norm.

    python -m aat.evaluate --run runs/aat_full_s0 --backend autoattack
"""
import argparse
import json
import os
import time

import torch

from .attacks import perturb
from .data import get_test_tensors
from .models import build_model
from .utils import device, load_config, set_seed

AA_NORM = {"linf": "Linf", "l2": "L2", "l1": "L1"}


def attack_batchwise(model, x, y, norm, eps, backend, attacks, bs, dev, pgd_steps):
    if backend == "autoattack":
        from autoattack import AutoAttack
        adv = AutoAttack(model, norm=AA_NORM[norm], eps=eps, version="custom",
                         attacks_to_run=list(attacks), verbose=False, device=dev)
        adv.apgd.n_restarts = 1
        adv.apgd_targeted.n_target_classes = 9
        if norm == "l1":  # as in AutoAttack's standard L1 setting
            adv.apgd.use_largereps = True
            adv.apgd_targeted.use_largereps = True
        x_adv = adv.run_standard_evaluation(x, y, bs=bs)
    else:
        x_adv = torch.cat([perturb(model, x[i:i + bs].to(dev), y[i:i + bs].to(dev), norm, eps,
                                   steps=pgd_steps, rand_init=True).cpu() for i in range(0, len(x), bs)])
    with torch.no_grad():
        pred = torch.cat([model(x_adv[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, len(x), bs)])
    return pred == y


def per_class(mask, y, C):
    return [float(mask[y == c].float().mean()) if (y == c).any() else None for c in range(C)]


def evaluate(run, backend="autoattack", attacks=("apgd-ce", "apgd-t"), n=None, bs=500, unseen=True,
             unseen_attacks=("apgd-ce",), pgd_steps=50, tag=None):
    cfg = load_config(os.path.join(run, "config.yaml"))
    dev = device()
    set_seed(0)
    ecfg = cfg["eval"]
    n = n or ecfg["n"]
    C = 100 if cfg["data"]["dataset"] == "cifar100" else 10
    model = build_model(cfg["model"], cfg["data"]["dataset"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(run, "final.pt"), map_location=dev)["model"])
    model.eval()
    x, y = get_test_tensors(cfg, n)

    t0 = time.time()
    with torch.no_grad():
        clean = torch.cat([model(x[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, n, bs)]) == y
    res = {"run": run, "name": cfg["name"], "seed": cfg.get("seed", 0), "backend": backend, "attacks": list(attacks),
           "n": n, "trained_norms": cfg["norms"], "clean": clean.float().mean().item(),
           "clean_per_class": per_class(clean, y, C), "robust": {}, "robust_per_class": {}}

    union = clean.clone()
    for p, eps in ecfg["seen"].items():
        m = attack_batchwise(model, x, y, p, eps, backend, attacks, bs, dev, pgd_steps) & clean
        res["robust"][p] = m.float().mean().item()
        res["robust_per_class"][p] = per_class(m, y, C)
        union &= m
        print(f"{cfg['name']}: {p} eps={eps:.4g} robust={res['robust'][p]:.4f}", flush=True)
    res["union"] = union.float().mean().item()
    uc = per_class(union, y, C)
    res["union_per_class"] = uc
    vals = [v for v in uc if v is not None]
    res["union_worst_class"] = min(vals)
    res["union_class_std"] = float(torch.tensor(vals).std())
    res["unseen_norms"] = [p for p in ecfg["seen"] if p not in cfg["norms"]]

    if unseen:
        res["unseen"] = {}
        for p, grid in ecfg["unseen"].items():
            for eps in grid:
                m = attack_batchwise(model, x, y, p, eps, backend, unseen_attacks, bs, dev, pgd_steps) & clean
                res["unseen"][f"{p}@{eps:.4g}"] = m.float().mean().item()
                print(f"{cfg['name']}: unseen {p} eps={eps:.4g} robust={m.float().mean():.4f}", flush=True)
    res["eval_time_s"] = time.time() - t0

    fn = f"eval_{tag or backend}.json"
    with open(os.path.join(run, fn), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("name", "clean", "robust", "union", "union_worst_class")}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--backend", default="autoattack", choices=["autoattack", "pgd"])
    ap.add_argument("--attacks", default="apgd-ce,apgd-t")
    ap.add_argument("--unseen-attacks", default="apgd-ce")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--bs", type=int, default=500)
    ap.add_argument("--no-unseen", action="store_true")
    ap.add_argument("--pgd-steps", type=int, default=50)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    for r in a.run:
        evaluate(r, a.backend, a.attacks.split(","), a.n, a.bs, not a.no_unseen,
                 a.unseen_attacks.split(","), a.pgd_steps, a.tag)


if __name__ == "__main__":
    main()
