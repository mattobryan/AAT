"""Training-time PGD attacks under l_inf, l_2 and l_1 with per-sample budgets.

All attacks work in [0, 1] pixel space and accept `eps` as a float or a (B,) tensor, which
is what lets the class-/sample-aware schedulers hand every example its own budget.
Final evaluation does NOT use these attacks; it uses AutoAttack (see evaluate.py).
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F

NORMS = ("linf", "l2", "l1")


def _view(t, x):
    return t.view(-1, *([1] * (x.dim() - 1)))


def _as_tensor(eps, x):
    if not torch.is_tensor(eps):
        eps = torch.full((x.shape[0],), float(eps), device=x.device)
    return eps.to(x.device, torch.float32)


def project_l1(v: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each row of v onto the l1 ball of radius eps (Duchi et al., 2008)."""
    b = v.shape[0]
    flat = v.reshape(b, -1)
    abs_v = flat.abs()
    outside = abs_v.sum(1) > eps
    if not outside.any():
        return v
    mu, _ = abs_v.sort(1, descending=True)
    cs = mu.cumsum(1)
    ar = torch.arange(1, flat.shape[1] + 1, device=v.device, dtype=v.dtype)
    cond = (mu - (cs - eps[:, None]) / ar) > 0
    rho = (cond * ar).argmax(1, keepdim=True)  # last index where cond holds
    theta = ((cs.gather(1, rho) - eps[:, None]) / (rho + 1).to(v.dtype)).clamp_min(0)
    proj = flat.sign() * (abs_v - theta).clamp_min(0)
    return torch.where(outside[:, None], proj, flat).view_as(v)


def project(delta, x, norm, eps):
    if norm == "linf":
        e = _view(eps, x)
        delta = torch.max(torch.min(delta, e), -e)
    elif norm == "l2":
        n = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
        delta = delta * _view(torch.clamp(eps / n, max=1.0), x)
    elif norm == "l1":
        delta = project_l1(delta, eps)
    else:
        raise ValueError(norm)
    # box constraint; clipping only shrinks |delta_i| so the norm bound still holds
    return (x + delta).clamp(0, 1) - x


def _random_init(x, norm, eps):
    if norm == "linf":
        d = (torch.rand_like(x) * 2 - 1) * _view(eps, x)
    elif norm == "l2":
        d = torch.randn_like(x)
        d = d / _view(d.flatten(1).norm(dim=1), x) * _view(eps * torch.rand_like(eps), x)
    else:  # sparse-ish Laplace start, scaled to a fraction of the l1 budget
        d = torch.distributions.Laplace(0.0, 1.0).sample(x.shape).to(x.device)
        d = d / _view(d.flatten(1).abs().sum(1), x) * _view(eps * torch.rand_like(eps) * 0.5, x)
    return project(d, x, norm, eps)


def _l1_direction(g, x, delta, sparsity):
    """SLIDE-style steepest l1 ascent: move only the top-(1 - sparsity) coordinates by |g|,
    ignoring coordinates already saturated against the [0,1] box in the ascent direction."""
    b = g.shape[0]
    xa = x + delta
    blocked = ((xa <= 0) & (g < 0)) | ((xa >= 1) & (g > 0))
    g = g.masked_fill(blocked, 0)
    flat = g.reshape(b, -1)
    k = max(1, int(round((1 - sparsity) * flat.shape[1])))
    thr = flat.abs().topk(k, dim=1).values[:, -1:]
    d = flat.sign() * (flat.abs() >= thr)
    d = d / d.abs().sum(1, keepdim=True).clamp_min(1)
    return d.view_as(g)


def perturb(model, x, y, norm, eps, steps=10, step_size=None, rand_init=True,
            l1_sparsity=0.95, amp=False, return_loss=False):
    """Untargeted PGD maximising cross-entropy. `step_size` is relative: alpha = step_size * eps."""
    eps = _as_tensor(eps, x)
    rel = step_size if step_size is not None else (2.5 / steps if norm != "l1" else 5.0 / steps)
    alpha = eps * rel
    was_training = model.training
    model.eval()
    delta = _random_init(x, norm, eps) if rand_init else torch.zeros_like(x)
    ctx = torch.autocast("cuda", dtype=torch.float16) if (amp and x.is_cuda) else nullcontext()
    for _ in range(steps):
        delta.requires_grad_(True)
        with ctx:
            loss = F.cross_entropy(model(x + delta), y, reduction="sum")
        g, = torch.autograd.grad(loss, delta)
        g = g.float()
        with torch.no_grad():
            if norm == "linf":
                delta = delta + _view(alpha, x) * g.sign()
            elif norm == "l2":
                gn = g.flatten(1).norm(dim=1).clamp_min(1e-12)
                delta = delta + _view(alpha / gn, x) * g
            else:
                delta = delta + _view(alpha, x) * _l1_direction(g, x, delta, l1_sparsity)
            delta = project(delta, x, norm, eps)
    model.train(was_training)
    x_adv = (x + delta).detach()
    if return_loss:
        with torch.no_grad(), ctx:
            l = F.cross_entropy(model(x_adv), y, reduction="none").float()
        return x_adv, l
    return x_adv


def perturb_norm(delta, norm):
    f = delta.flatten(1)
    return {"linf": f.abs().amax(1), "l2": f.norm(dim=1), "l1": f.abs().sum(1)}[norm]
