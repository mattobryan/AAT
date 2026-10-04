"""Diagnostic for H1 vs H4: per-class robust accuracy on TRAINING images vs TEST images.

A large train-minus-test gap in a weak class = the model memorised it (data-limited, H1).
Low train accuracy as well = optimisation- or capacity-limited (H4). Same fixed-eps AutoAttack
(APGD-CE + APGD-T) as the main evaluation, only the images differ. Reuses aat.evaluate unchanged.

    python scripts/train_vs_test.py --ckpt ep_80_0.pth --out runs_official/ramp_scratch_l5_s0/train_vs_test.json
Train images: the first n of the CIFAR-10 training set, no augmentation (the official RAMP code trains on all 50k).
"""
import argparse
import json
import os
import sys

import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aat.evaluate import attack_batchwise, per_class  # noqa: E402
from aat.models import build_model, load_weights  # noqa: E402
from aat.utils import device, load_config, set_seed  # noqa: E402


def union_eval(model, x, y, ecfg, attacks, bs, dev):
    with torch.no_grad():
        clean = torch.cat([model(x[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, len(x), bs)]) == y
    union, out = clean.clone(), {"clean": per_class(clean, y, 10)}
    for p, eps in ecfg["seen"].items():
        m = attack_batchwise(model, x, y, p, eps, "autoattack", attacks, bs, dev, 50) & clean
        out[p] = per_class(m, y, 10)
        union &= m
    out["union"] = per_class(union, y, 10)
    out["union_overall"] = union.float().mean().item()
    out["clean_overall"] = clean.float().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/official_eval.yaml")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=500)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = device()
    set_seed(0)
    model = build_model(cfg["model"], cfg["data"]["dataset"]).to(dev)
    load_weights(model, a.ckpt, map_location=dev)
    model.eval()
    root = cfg["data"].get("root", "./data")
    res = {"ckpt": a.ckpt, "n": a.n, "attacks": ["apgd-ce", "apgd-t"]}
    for split, train in (("test", False), ("train", True)):
        ds = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=T.ToTensor())
        x, y = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(a.n)), a.n)))
        res[split] = union_eval(model, x, y, cfg["eval"], ["apgd-ce", "apgd-t"], a.bs, dev)
        print(split, "union", res[split]["union_overall"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
