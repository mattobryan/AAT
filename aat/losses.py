"""Multi-norm loss aggregation (Eqs. 4.6-4.8, 4.16) and RAMP logit pairing (Eq. 3.1)."""
import torch
import torch.nn.functional as F


def kl_pairing(logits, y, weak, strong):
    """RAMP pairing: KL(p_weak || p_strong) averaged over the n_c samples that are correctly
    classified under the weaker attack. Returns 0 when n_c = 0."""
    lw, ls = logits[weak], logits[strong]
    correct = (lw.argmax(1) == y).float()
    kl = (F.softmax(lw, 1) * (F.log_softmax(lw, 1) - F.log_softmax(ls, 1))).sum(1)
    return (kl * correct).sum() / correct.sum().clamp_min(1.0)


def aggregate(per_norm_ce, weights, mode="max", lambda_sec=0.5):
    """per_norm_ce: {norm: (B,) CE}. Returns (loss, stats).

    max      per-sample max over norms (union / worst-case objective, = choosing the strongest
             adversarial example per sample as in Eq. A.4)
    mean     weighted average-case objective (Eq. 4.6, weights from the norm scheduler)
    hybrid   Eq. 4.8: per-sample max over the two currently weakest (highest-weight) norms
             + lambda_sec * weighted mean of the rest
    """
    norms = list(per_norm_ce)
    L = torch.stack([per_norm_ce[p] for p in norms], 1)  # (B, P)
    w = torch.tensor([weights.get(p, 1.0) for p in norms], device=L.device)
    stats = {}
    if len(norms) == 1 or mode == "mean":
        loss = (L * w).mean(1).mean() if len(norms) > 1 else L[:, 0].mean()
    elif mode == "max":
        loss = L.max(1).values.mean()
    elif mode == "hybrid":
        order = torch.argsort(w, descending=True)
        prim, sec = order[:2], order[2:]
        primary = L[:, prim].max(1).values.mean()
        secondary = (L[:, sec] * w[sec]).mean() if len(sec) else L.new_zeros(())
        loss = primary + lambda_sec * secondary
        stats["sec_primary_ratio"] = float(secondary.detach() / primary.detach().clamp_min(1e-8)) if len(sec) else 0.0
    else:
        raise ValueError(mode)
    return loss, stats


def pairing_loss(logits, per_norm_ce, y):
    """Pair every norm against the strongest attack of the batch (highest mean CE)."""
    norms = list(logits)
    if len(norms) < 2:
        return logits[norms[0]].new_zeros(()), {}
    strong = max(norms, key=lambda p: float(per_norm_ce[p].detach().mean()))
    kls = {q: kl_pairing(logits, y, q, strong) for q in norms if q != strong}
    loss = sum(kls.values()) / len(kls)
    return loss, {"kl_mean": float(loss.detach()), "strong_norm": strong}
