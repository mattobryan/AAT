"""AAT training loop: evaluation -> adaptation -> adversarial training (thesis Sec. 4.2).

    python -m aat.train --config configs/aat_full.yaml --set seed=1
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import yaml

from .attacks import perturb
from .data import get_loaders
from .losses import aggregate, pairing_loss
from .models import build_model
from .schedulers import EpsScheduler, NormScheduler
from .utils import JsonlLogger, device, load_config, run_dir, set_seed


# ----------------------------------------------------------------------------- helpers
def make_lr_lambda(cfg, steps_per_epoch):
    t = cfg["train"]
    total = t["epochs"] * steps_per_epoch
    warm = t.get("warmup_epochs", 0) * steps_per_epoch
    sched = t.get("lr_schedule", "cosine")

    def f(step):
        if sched == "onecycle":  # piecewise linear 0 -> 1 -> 0 (Wong et al., 2020)
            peak = total * t.get("onecycle_peak", 0.4)
            return step / peak if step < peak else max(0.0, (total - step) / (total - peak))
        if step < warm:
            return (step + 1) / warm
        if sched == "cosine":
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        if sched == "multistep":
            e = step / steps_per_epoch
            return 0.1 ** sum(e >= m for m in t.get("milestones", [0.5 * t["epochs"], 0.75 * t["epochs"]]))
        return 1.0

    return f


def input_grad_norm(model, x, y):
    was = model.training
    model.eval()
    x = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(F.cross_entropy(model(x), y, reduction="sum"), x)
    model.train(was)
    return g.flatten(1).norm(dim=1).detach()


@torch.no_grad()
def _correct(model, x, y):
    return model(x).argmax(1) == y


def robust_masks(model, loader, norm, eps_fn, steps, cfg, dev, limit=None):
    """Per-sample correctness under PGD for one norm; eps_fn(y) -> (B,) budget."""
    ys, masks, n = [], [], 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        xa = perturb(model, x, y, norm, eps_fn(y), steps=steps,
                     l1_sparsity=cfg["attack"].get("l1", {}).get("sparsity", 0.95), amp=cfg["train"]["amp"])
        masks.append(_correct(model.eval(), xa, y).cpu())
        ys.append(y.cpu())
        n += len(y)
        if limit and n >= limit:
            break
    return torch.cat(masks), torch.cat(ys)


def per_class(mask, ys, C):
    out = torch.zeros(C)
    for c in range(C):
        sel = ys == c
        out[c] = mask[sel].float().mean() if sel.any() else 0.0
    return out


def run_feedback(model, loader, eps_s, cfg, dev, C):
    """Robust accuracy on the held-out feedback slice, per norm and per class (Eq. 4.9)."""
    steps = cfg["feedback"].get("steps", 10)
    fb = {"nominal": {}, "current": {}, "class_nominal": {}, "class_current": {}}
    for p in eps_s.norms:
        m, ys = robust_masks(model, loader, p, lambda y: torch.full_like(y, eps_s.nominal[p], dtype=torch.float32),
                             steps, cfg, dev)
        fb["nominal"][p], fb["class_nominal"][p] = m.float().mean().item(), per_class(m, ys, C)
        if eps_s.mode(p) in ("norm_feedback", "class_feedback"):
            ce = eps_s.class_eps(p)
            m, ys = robust_masks(model, loader, p, lambda y: ce[y], steps, cfg, dev)
        fb["current"][p], fb["class_current"][p] = m.float().mean().item(), per_class(m, ys, C)
    return fb


def monitor(model, loader, cfg, dev, eps_s):
    """Test-set curves for plotting only; never used for model selection or adaptation."""
    mcfg = cfg["monitor"]
    x_all = []
    out = {}
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out.setdefault("clean", []).append(_correct(model.eval(), x, y).cpu())
        x_all.append(len(y))
        if sum(x_all) >= mcfg["n"]:
            break
    out = {"clean": torch.cat(out["clean"]).float().mean().item()}
    union = None
    for p in cfg.get("eval_norms", ["linf", "l2", "l1"]):
        nom = cfg["eps"][p]["nominal"]
        m, _ = robust_masks(model, loader, p, lambda y: torch.full_like(y, nom, dtype=torch.float32),
                            mcfg["steps"], cfg, dev, limit=mcfg["n"])
        out[p] = m.float().mean().item()
        union = m if union is None else union & m
    out["union"] = union.float().mean().item()
    return out


def combine_gp(adv_grads, nat_grads, beta, mode):
    """RAMP-style gradient projection, applied per parameter tensor.
    project: thesis Eq. 3.5/3.6, g = beta * GP(g_nat, g_adv) + (1 - beta) * g_adv
    pcgrad : keep g_nat minus its component conflicting with g_adv, then mix the same way."""
    out = []
    for ga, gn in zip(adv_grads, nat_grads):
        if ga is None:
            out.append(None)
            continue
        dot = (ga * gn).sum()
        nsq = (ga * ga).sum().clamp_min(1e-12)
        if mode == "project":
            gp = dot.clamp_min(0) / nsq * ga
        else:
            gp = gn - (dot.clamp_max(0) / nsq) * ga
        out.append(beta * gp + (1 - beta) * ga)
    return out


# ----------------------------------------------------------------------------- main
def train(cfg):
    dev = device()
    set_seed(cfg.get("seed", 0))
    out = run_dir(cfg)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    log = JsonlLogger(os.path.join(out, "log.jsonl"))

    train_loader, fb_loader, test_loader = get_loaders(cfg)
    C = 100 if cfg["data"]["dataset"] == "cifar100" else 10
    model = build_model(cfg["model"], cfg["data"]["dataset"]).to(dev).to(memory_format=torch.channels_last)
    if cfg.get("init_from"):
        sd = torch.load(cfg["init_from"], map_location=dev)
        model.load_state_dict(sd.get("model", sd))
        print(f"initialised from {cfg['init_from']}")

    t = cfg["train"]
    opt = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                          weight_decay=t["weight_decay"], nesterov=t.get("nesterov", False))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(cfg, len(train_loader)))
    amp = t["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    norms = cfg["norms"]
    eps_s = EpsScheduler(cfg["eps"], norms, C, t["epochs"], dev)
    norm_s = NormScheduler(cfg["norm_schedule"], norms, seed=cfg.get("seed", 0))
    lcfg, gcfg = cfg["loss"], cfg["gp"]
    needs_feedback = (any(eps_s.mode(p) in ("norm_feedback", "class_feedback", "sample") for p in norms)
                      or norm_s.mode in ("adaptive", "all_adaptive", "error"))

    start = 0
    ck_path = os.path.join(out, "last.pt")
    if os.path.exists(ck_path):  # resume across Kaggle sessions
        ck = torch.load(ck_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        lr_sched.load_state_dict(ck["lr_sched"]); scaler.load_state_dict(ck["scaler"])
        eps_s.state, eps_s.class_rob_ema = ck["eps_state"], ck["eps_ema"]
        norm_s.probs, norm_s.loss_gap = ck["norm_probs"], ck["norm_loss_gap"]
        start = ck["epoch"] + 1
        print(f"resumed from epoch {start}")

    params = [p for p in model.parameters() if p.requires_grad]
    ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else torch.enable_grad
    for epoch in range(start, t["epochs"]):
        t0 = time.time()
        eps_s.set_epoch(epoch); norm_s.set_epoch(epoch)
        fb = None
        fcfg = cfg["feedback"]
        if needs_feedback and (epoch % fcfg["every"] == 0) and (epoch > 0 or fcfg.get("at_start", True)):
            fb = run_feedback(model, fb_loader, eps_s, cfg, dev, C)
            eps_s.update(fb); norm_s.update(fb)
        t_fb = time.time() - t0

        model.train()
        agg = {"loss": 0.0, "n": 0, "kl": 0.0, "sec_ratio": 0.0, "norm_counts": {p: 0 for p in norms},
               "adv_correct": {p: 0 for p in norms}, "adv_seen": {p: 0 for p in norms}}
        for i, (x, y) in enumerate(train_loader):
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            active = norm_s.select(i)
            gn = input_grad_norm(model, x, y) if eps_s.needs_grad_signal() else None
            advs = {}
            for p in active:
                a = cfg["attack"][p]
                advs[p] = perturb(model, x, y, p, eps_s.sample_eps(p, y, gn), steps=a["steps"],
                                  step_size=a.get("step_size"), l1_sparsity=a.get("sparsity", 0.95), amp=amp)
                agg["norm_counts"][p] += 1

            need_clean = lcfg.get("lambda_clean", 0) > 0 or gcfg.get("enabled") or norm_s.mode == "loss"
            batch = torch.cat([advs[p] for p in active] + ([x] if need_clean else []))
            model.train()
            with ctx():
                logits_all = model(batch)
            chunks = logits_all.float().split(len(y))
            logits = dict(zip(active, chunks))
            ce = {p: F.cross_entropy(logits[p], y, reduction="none") for p in active}
            loss, st = aggregate(ce, norm_s.weights(), lcfg.get("aggregate", "max"), lcfg.get("lambda_sec", 0.5))
            if lcfg.get("lambda_kl", 0) > 0 and len(active) > 1:
                kl, kst = pairing_loss(logits, ce, y)
                loss = loss + lcfg["lambda_kl"] * kl
                agg["kl"] += kst["kl_mean"]
            clean_ce = F.cross_entropy(chunks[-1], y) if need_clean else None
            if lcfg.get("lambda_clean", 0) > 0:
                loss = loss + lcfg["lambda_clean"] * clean_ce
            if norm_s.mode == "loss":
                for p in active:
                    norm_s.observe_loss(p, float(ce[p].mean()), float(clean_ce))

            opt.zero_grad(set_to_none=True)
            if gcfg.get("enabled"):
                ga = torch.autograd.grad(scaler.scale(loss), params, retain_graph=True, allow_unused=True)
                gn_ = torch.autograd.grad(scaler.scale(clean_ce), params, allow_unused=True)
                for prm, g in zip(params, combine_gp(ga, gn_, gcfg.get("beta", 0.5), gcfg.get("mode", "project"))):
                    prm.grad = g
            else:
                scaler.scale(loss).backward()
            if t.get("clip_grad"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, t["clip_grad"])
            scaler.step(opt); scaler.update(); lr_sched.step()

            agg["loss"] += float(loss.detach()); agg["n"] += 1
            agg["sec_ratio"] += st.get("sec_primary_ratio", 0.0)
            for p in active:
                agg["adv_correct"][p] += int((logits[p].argmax(1) == y).sum())
                agg["adv_seen"][p] += len(y)
            if cfg.get("limit_batches") and i + 1 >= cfg["limit_batches"]:
                break

        n = max(1, agg["n"])
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": agg["loss"] / n,
               "kl_mean": agg["kl"] / n, "sec_primary_ratio": agg["sec_ratio"] / n,
               "train_adv_acc": {p: agg["adv_correct"][p] / max(1, agg["adv_seen"][p]) for p in norms},
               "norm_counts": agg["norm_counts"], "eps": eps_s.summary(), "norm_sched": norm_s.summary(),
               "epoch_time_s": time.time() - t0, "feedback_time_s": t_fb}
        if fb is not None:
            rec["feedback"] = {k: {p: (v.tolist() if torch.is_tensor(v) else v) for p, v in d.items()} for k, d in fb.items()}
        m = cfg["monitor"]
        if m.get("every") and ((epoch + 1) % m["every"] == 0 or epoch + 1 == t["epochs"]):
            rec["monitor_test"] = monitor(model, test_loader, cfg, dev, eps_s)
        log.log(rec)
        print(json.dumps({k: rec[k] for k in ("epoch", "train_loss", "train_adv_acc", "epoch_time_s")}
                         | ({"monitor_test": rec["monitor_test"]} if "monitor_test" in rec else {})))

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "lr_sched": lr_sched.state_dict(),
                    "scaler": scaler.state_dict(), "eps_state": eps_s.state, "eps_ema": eps_s.class_rob_ema,
                    "norm_probs": norm_s.probs, "norm_loss_gap": norm_s.loss_gap, "epoch": epoch}, ck_path)

    torch.save({"model": model.state_dict()}, os.path.join(out, "final.pt"))
    print(f"done -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=2 seed=1")
    a = ap.parse_args()
    train(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
