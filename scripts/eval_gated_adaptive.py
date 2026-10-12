"""Gate-aware adaptive evaluation at the same FIXED budgets as aat.evaluate, for gated (squint) models and, as a
matched-compute baseline, for plain models.

    python scripts/eval_gated_adaptive.py --run runs/squint_v0_s0
    python scripts/eval_gated_adaptive.py --run runs/v3a_max_s0      # matched baseline: same number of extra attacks

Why: the gate softmax of a CE-trained gate saturates, so routing is almost piecewise constant and plain APGD can miss
inputs that are pushed to a weaker expert (Tramer et al. 2020). For every seen norm this script
  1. runs the standard AutoAttack (APGD-CE + APGD-T, seed 0) on the real model,
  2. runs extra APGD-CE attacks that try to defeat the routing, one candidate attacker each:
       force<k>   gate forced to one-hot(k) (constant) during the attack
       detach     forward pass identical to the real model, gate gradient cut (BPDA): the attack cannot be stopped
                  by the saturated softmax, and its success is judged on the real model
       sttemp<T>  straight-through temperature: forward value is the real gate, the backward pass uses
                  softmax(gate_logits / T), so the gradient is not saturated
     A plain (ungated) model gets the same NUMBER of extra APGD-CE runs with different seeds and no hook, so the
     comparison of attack budgets is matched.
  3. scores every adversarial on the REAL model and keeps a point only if it survives all of them (per-sample AND).
The union is the per-sample AND over the norms. Adaptive numbers are <= the standard numbers in the same file by
construction; always compare robust / union with robust_standard / union_standard from the same file, because
AutoAttack in aat.evaluate is not seeded. Label the result: "gate-aware adaptive (AutoAttack std + extra APGD-CE
attackers, per-sample AND), an upper bound on true robustness". The gate argmax and entropy on clean and on standard
adversarial inputs are logged. eval_adaptive.json has the same keys as eval_autoattack.json (no unseen grid), so
`scripts/aggregate.py --tag adaptive` works. aat/evaluate.py and aat/attacks.py are imported, not modified.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aat.data import get_test_tensors  # noqa: E402
from aat.evaluate import AA_NORM, per_class  # noqa: E402
from aat.models import GatedBN, build_model, load_weights  # noqa: E402
from aat.utils import device, load_config, set_seed  # noqa: E402

LABEL = ("gate-aware adaptive (AutoAttack std + extra APGD-CE attackers, per-sample AND); "
         "an upper bound on true robustness")


class GateAttack:
    """Context manager that changes how the gate is seen by the attacker; every hook is removed on exit, also when the
    attack raises. mode "force": constant one-hot(arg) gate. mode "detach": same forward as the real model, no gradient
    through the gate logits. mode "sttemp": forward uses the real gate, backward uses softmax(logits / arg)."""

    def __init__(self, net, mode, arg):
        self.net, self.mode, self.arg, self.handles = net, mode, arg, []

    def _gate_fc_hook(self, module, inputs, out):
        if self.mode == "force":
            o = torch.zeros_like(out)
            o[:, self.arg] = 1e4
            return o
        return out.detach()  # "detach": forward value unchanged, gradient cut

    def _gated_bn_pre_hook(self, module, inputs):
        gt = torch.softmax(self.net.last_gate_logits / self.arg, dim=1)
        module.g = module.g.detach() + gt - gt.detach()  # forward value stays the real gate
        return None

    def __enter__(self):
        if self.mode in ("force", "detach"):
            self.handles.append(self.net.gate_fc.register_forward_hook(self._gate_fc_hook))
        else:
            for m in self.net.modules():
                if isinstance(m, GatedBN):
                    self.handles.append(m.register_forward_pre_hook(self._gated_bn_pre_hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


def attack(model, x, y, norm, eps, attacks, bs, dev, seed=0):
    """AutoAttack with the given sub-attacks, same settings as aat.evaluate.attack_batchwise (but seeded); returns x_adv."""
    from autoattack import AutoAttack
    adv = AutoAttack(model, norm=AA_NORM[norm], eps=eps, seed=seed, version="standard", verbose=False, device=dev)
    adv.attacks_to_run = list(attacks)
    return adv.run_standard_evaluation(x.contiguous(), y, bs=bs)


@torch.no_grad()
def predict(model, x, bs, dev):
    return torch.cat([model(x[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, len(x), bs)])


@torch.no_grad()
def gate_stats(model, x, bs, dev):
    """Fraction of inputs routed (argmax) to each threat and the mean softmax entropy, from the real model."""
    net, logits = model[1], []
    for i in range(0, len(x), bs):
        model(x[i:i + bs].to(dev))
        logits.append(net.last_gate_logits.float().cpu())
    p = torch.cat(logits).softmax(1)
    return {"argmax_frac": [float((p.argmax(1) == k).float().mean()) for k in range(p.shape[1])],
            "entropy": float(-(p * p.clamp_min(1e-12).log()).sum(1).mean())}


def combine(*masks):
    """Per-sample AND of boolean survival masks."""
    out = masks[0].clone()
    for m in masks[1:]:
        out &= m
    return out


def variant_plan(K, temps):
    """Extra attackers: (label, mode, arg). Gated: force<k> for every threat, detach, sttemp<T>. Plain (K=0): the same
    number of unhooked APGD-CE runs with seeds 1..N, so the attack budget matches the gated model's."""
    gated = [(f"force{k}", "force", k) for k in range(K)] + [("detach", "detach", None)] \
        + [(f"sttemp{T:g}", "sttemp", T) for T in temps]
    n_extra = (K or 3) + 1 + len(temps)
    return gated if K else [(f"seed{s}", "plain", s) for s in range(1, n_extra + 1)]


def run(run_dir, n=None, bs=500, temps=(20.0,), std_attacks=("apgd-ce", "apgd-t"), var_attacks=("apgd-ce",)):
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    dev = device()
    set_seed(0)
    ecfg = cfg["eval"]
    n = n or ecfg["n"]
    C = 100 if cfg["data"]["dataset"] == "cifar100" else 10
    model = build_model(cfg["model"], cfg["data"]["dataset"]).to(dev)
    load_weights(model, os.path.join(run_dir, "final.pt"), map_location=dev)
    model.eval()
    net = model[1] if isinstance(model, torch.nn.Sequential) else model
    K = getattr(net, "gated_K", 0)
    x, y = get_test_tensors(cfg, n)
    plan = variant_plan(K, temps)
    t0 = time.time()

    clean = predict(model, x, bs, dev) == y
    res = {"run": run_dir, "name": cfg["name"], "seed": cfg.get("seed", 0), "backend": "adaptive-gated", "label": LABEL,
           "attacks": list(std_attacks), "variant_attacks": list(var_attacks), "variants_run": [v[0] for v in plan],
           "n": n, "trained_norms": cfg["norms"], "clean": clean.float().mean().item(),
           "clean_per_class": per_class(clean, y, C), "robust": {}, "robust_standard": {}, "robust_per_class": {},
           "variants": {}, "gate_clean": gate_stats(model, x, bs, dev) if K else None, "gate_adv": {}}

    union, union_std = clean.clone(), clean.clone()
    for p, eps in ecfg["seen"].items():
        x_std = attack(model, x, y, p, eps, std_attacks, bs, dev, seed=0)
        m_std = combine(predict(model, x_std, bs, dev) == y, clean)
        if K:
            res["gate_adv"][p] = gate_stats(model, x_std, bs, dev)
        m_ad, per = m_std.clone(), {}
        for label, mode, arg in plan:
            if mode == "plain":
                x_adv = attack(model, x, y, p, eps, var_attacks, bs, dev, seed=arg)
            else:
                with GateAttack(net, mode, arg):
                    x_adv = attack(model, x, y, p, eps, var_attacks, bs, dev, seed=0)
            m = combine(predict(model, x_adv, bs, dev) == y, clean)  # scored on the real model, no hook
            per[label] = m.float().mean().item()
            m_ad = combine(m_ad, m)
        res["variants"][p] = per
        res["robust_standard"][p] = m_std.float().mean().item()
        res["robust"][p] = m_ad.float().mean().item()
        res["robust_per_class"][p] = per_class(m_ad, y, C)
        union, union_std = combine(union, m_ad), combine(union_std, m_std)
        print(f"{cfg['name']}: {p} eps={eps:.4g} standard={res['robust_standard'][p]:.4f} "
              f"adaptive={res['robust'][p]:.4f} variants={per}", flush=True)
    res["union"], res["union_standard"] = union.float().mean().item(), union_std.float().mean().item()
    uc = per_class(union, y, C)
    res["union_per_class"] = uc
    vals = [v for v in uc if v is not None]
    res["union_worst_class"] = min(vals)
    res["union_class_std"] = float(torch.tensor(vals).std())
    res["sanity"] = {"adaptive_le_standard": all(res["robust"][p] <= res["robust_standard"][p] + 1e-9 for p in res["robust"]),
                     "robust_le_clean": all(res["robust"][p] <= res["clean"] + 1e-9 for p in res["robust"])}
    res["eval_time_s"] = time.time() - t0
    with open(os.path.join(run_dir, "eval_adaptive.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("name", "clean", "robust", "union", "union_standard", "sanity")}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--bs", type=int, default=500)
    ap.add_argument("--temps", type=float, nargs="*", default=[20.0])
    a = ap.parse_args()
    run(a.run, n=a.n, bs=a.bs, temps=tuple(a.temps))


if __name__ == "__main__":
    main()
