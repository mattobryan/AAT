"""H5: average the saved epoch-60/70/80 weights of a RAMP run and evaluate with the unchanged protocol.

    python scripts/average_ckpts.py --seed 0 [--epochs 60 70 80] [--no-eval]

Pulls ramp_official/ramp_scratch_l5_s<seed>/ep_<e>_0.pth from the Hub (raw model state_dicts as saved by
RAMP.py), averages all float tensors (incl. BN running_mean/var), copies num_batches_tracked from the last
checkpoint, writes runs_official/rampavg_s<seed>/averaged.pth, then runs aat.evaluate (official_eval.yaml,
fixed eps, first 1000 test points) and pushes ramp_official/rampavg_s<seed>/eval_autoattack.json.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def average_state_dicts(sds):
    """Mean of float tensors; integer buffers (num_batches_tracked) come from the last state_dict."""
    out = {}
    for k, v in sds[-1].items():
        if torch.is_floating_point(v):
            out[k] = (sum(sd[k].double() for sd in sds) / len(sds)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    return sd.get("model", sd.get("state_dict", sd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, nargs="+", default=[60, 70, 80])
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    from aat import hub
    src = f"ramp_scratch_l5_s{a.seed}"
    run = os.path.join(ROOT, "runs_official", f"rampavg_s{a.seed}")
    os.makedirs(run, exist_ok=True)
    sds = []
    for e in a.epochs:
        p = os.path.join(ROOT, "external", "ramp", "trained_models", src, f"ep_{e}_0.pth")
        if not os.path.exists(p) and not hub.pull(f"ramp_official/{src}/ep_{e}_0.pth", p):
            raise SystemExit(f"missing {p}")
        sds.append(load_sd(p))
    avg = average_state_dicts(sds)
    out = os.path.join(run, "averaged.pth")
    torch.save(avg, out)
    print(f"averaged epochs {a.epochs} -> {out}")
    if a.no_eval:
        return
    from aat.evaluate import evaluate
    evaluate(run, "autoattack", ckpt=out, config=os.path.join(ROOT, "configs", "official_eval.yaml"), name=f"rampavg_s{a.seed}")
    hub.push(os.path.join(run, "eval_autoattack.json"), f"ramp_official/rampavg_s{a.seed}/eval_autoattack.json")


if __name__ == "__main__":
    main()
