"""Adaptive epsilon and norm schedulers (thesis Ch. 4, Eqs. 4.1-4.20, A.1-A.5).

Feedback dictionary produced by `train.run_feedback` and consumed by `update(...)`:
    fb["nominal"][norm]       robust acc on the feedback set at the nominal (target) eps
    fb["current"][norm]       robust acc at the eps currently used for training
    fb["class_current"][norm] (C,) per-class robust acc at the current per-class eps
    fb["class_nominal"][norm] (C,) per-class robust acc at the nominal eps
"""
import math

import torch


# --------------------------------------------------------------------------- epsilon
def project_mean(eps, target, lo, hi, iters=20):
    """Shift per-class budgets so that eps.mean() == target while staying inside [lo, hi].

    Only classes that can still move in the needed direction absorb the gap, so a class sitting at a bound
    does not block the projection. Used by the budget-neutral class-feedback controller (mean_preserve)."""
    eps = eps.clamp(lo, hi)
    for _ in range(iters):
        gap = float(target - eps.mean())
        if abs(gap) < 1e-9:
            break
        free = (eps < hi) if gap > 0 else (eps > lo)
        if not bool(free.any()):
            break
        eps = torch.where(free, eps + gap * eps.numel() / free.sum(), eps).clamp(lo, hi)
    return eps


class EpsScheduler:
    """Per-norm budget controller.

    modes (per norm, `cfg["eps"][norm]["mode"]`):
      static          eps = nominal                                                  (RAMP, 3A)
      linear          global ramp from start_frac*nominal to nominal over ramp_epochs (1A)
      norm_feedback   norm-level, budget-neutral across the norms in this mode: one fraction f_p per norm,
                      f_p <- Proj(f_p + alpha*(mean_q A_q - A_p)), eps_p = f_p*nominal_p, A_p = robust acc
                      at the NOMINAL eps (fb["nominal"]); weakest norm gains eps. alpha and mean_frac are
                      read from `cfg["norm_ctl"]`, bounds/init per norm (replaces the unused Eq. 4.18 form)
      class_feedback  per-class eps_c <- Proj(eps_c + alpha*(A_c - tau)*nominal)   (Eq. 4.19)
      sample          per-sample eps from gradient-norm + class-robustness difficulty (Eq. A.1)
      class_index     literal thesis Eq. A.5: nominal*(1+0.05*y)*min(1, t/T); kept only for
                      faithfulness, it keys on the label index and can exceed nominal by 45%
    """

    def __init__(self, cfg, norms, num_classes, total_epochs, device):
        self.cfg, self.norms, self.C, self.T, self.device = cfg, norms, num_classes, total_epochs, device
        self.nominal = {p: float(cfg[p]["nominal"]) for p in norms}
        self.state = {}
        for p in norms:
            c = cfg[p]
            init = c.get("init_frac", 1.0) * self.nominal[p]
            self.state[p] = torch.full((num_classes,), init, device=device)
        self.class_rob_ema = {p: torch.full((num_classes,), 0.5, device=device) for p in norms}
        self.epoch = 0
        self.ctl_norms = [p for p in norms if self.mode(p) == "norm_feedback"]
        self.frac = {p: float(cfg[p].get("init_frac", 1.0)) for p in self.ctl_norms}  # norm-level fractions
        self.ctl_acc = {}  # last nominal-eps robust acc seen by the norm-level controller

    def bounds(self, p):
        c = self.cfg[p]
        return c.get("min_frac", 0.25) * self.nominal[p], c.get("max_frac", 1.0) * self.nominal[p]

    def mode(self, p):
        return self.cfg[p].get("mode", "static")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _update_norm_feedback(self, fb):
        """f_p <- f_p + alpha*(mean_q A_q - A_p), clip to [min_frac, max_frac], project to mean(f) = mean_frac."""
        ps = self.ctl_norms
        ctl = self.cfg.get("norm_ctl", {})
        alpha, mean_frac = float(ctl.get("alpha", 0.5)), float(ctl.get("mean_frac", 1.0))
        A = torch.tensor([float(fb["nominal"][p]) for p in ps], dtype=torch.float64)
        f = torch.tensor([self.frac[p] for p in ps], dtype=torch.float64)
        lo = torch.tensor([self.cfg[p].get("min_frac", 0.25) for p in ps], dtype=torch.float64)
        hi = torch.tensor([self.cfg[p].get("max_frac", 2.0) for p in ps], dtype=torch.float64)
        f = project_mean((f + alpha * (A.mean() - A)).clamp(lo, hi), mean_frac, lo, hi)
        for p, fp, ap in zip(ps, f.tolist(), A.tolist()):
            self.frac[p], self.ctl_acc[p] = fp, ap
            self.state[p] = torch.full((self.C,), fp * self.nominal[p], device=self.device)

    def update(self, fb):
        if self.ctl_norms:
            self._update_norm_feedback(fb)
        for p in self.norms:
            c, m = self.cfg[p], self.mode(p)
            lo, hi = self.bounds(p)
            alpha, tau = c.get("alpha", 0.05), c.get("tau", 0.5)
            if m == "class_feedback":
                a = fb["class_current"][p].to(self.device)
                self.state[p] = (self.state[p] + alpha * (a - tau) * self.nominal[p]).clamp(lo, hi)
                if c.get("mean_preserve"):  # budget-neutral: reallocate eps across classes, keep the mean fixed
                    target = c.get("mean_frac", c.get("init_frac", 1.0)) * self.nominal[p]
                    self.state[p] = project_mean(self.state[p], target, lo, hi)
            if "class_nominal" in fb:
                mom = c.get("ema", 0.7)
                self.class_rob_ema[p] = mom * self.class_rob_ema[p] + (1 - mom) * fb["class_nominal"][p].to(self.device)

    def class_eps(self, p):
        """(C,) budget currently assigned to each class (used by feedback at 'current' eps)."""
        m, nom = self.mode(p), self.nominal[p]
        if m == "static":
            return torch.full((self.C,), nom, device=self.device)
        if m == "linear":
            c = self.cfg[p]
            ramp = max(1, c.get("ramp_epochs", self.T // 2))
            f = c.get("start_frac", 0.25) + (1 - c.get("start_frac", 0.25)) * min(1.0, self.epoch / ramp)
            return torch.full((self.C,), f * nom, device=self.device)
        if m == "class_index":
            f = min(1.0, (self.epoch + 1) / self.T)
            return nom * (1 + 0.05 * torch.arange(self.C, device=self.device)) * f
        if m == "sample":  # class-level fallback when no per-sample signal is available
            lo, hi = self.bounds(p)
            return lo + (hi - lo) * self.class_rob_ema[p]
        return self.state[p]

    def needs_grad_signal(self):
        return any(self.mode(p) == "sample" for p in self.norms)

    def sample_eps(self, p, y, grad_norm=None):
        """(B,) per-sample budget for norm p."""
        if self.mode(p) != "sample":
            return self.class_eps(p)[y]
        c = self.cfg[p]
        lo, hi = self.bounds(p)
        a, b, lam = c.get("a", 1.0), c.get("b", 1.0), c.get("lam", 2.0)
        z = torch.zeros_like(y, dtype=torch.float32)
        if grad_norm is not None:
            z = (grad_norm - grad_norm.mean()) / (grad_norm.std() + 1e-8)
        difficulty = a * z + b * (1 - self.class_rob_ema[p][y]) * 2  # class term roughly in [0, 2]
        difficulty = difficulty - difficulty.mean()
        # harder samples get *smaller* budgets (the text of A.1; the printed sign is reversed)
        return lo + (hi - lo) * torch.sigmoid(-lam * difficulty)

    def zero_budget_classes(self, p):
        lo, _ = self.bounds(p)
        return int((self.class_eps(p) <= lo * 1.0001).sum().item())

    def summary(self):
        out = {p: {"mean": float(self.class_eps(p).mean()), "min": float(self.class_eps(p).min()),
                   "max": float(self.class_eps(p).max()),
                   "zero_budget_classes": self.zero_budget_classes(p)} for p in self.norms}
        for p in self.ctl_norms:  # norm-level controller: fraction f_p and the nominal-eps accuracy A_p behind it
            out[p]["frac"] = self.frac[p]
            if p in self.ctl_acc:
                out[p]["acc_nominal"] = self.ctl_acc[p]
        return out


# --------------------------------------------------------------------------- norms
class NormScheduler:
    """Which norms each batch is attacked with, and their loss weights.

    modes:
      all              every norm every batch (multi-norm); weights uniform                 (3A/3B)
      all_adaptive     every norm every batch; loss weights w_p ~ exp(-gamma R_p)           (AAT full)
      cyclic           one norm per batch, cycling every `period` epochs or batches         (2A)
      random           one norm per batch, uniform                                          (SAT)
      adaptive         one norm per batch, pi_p ~ exp(-gamma R_p)                           (2B, Eq. 4.20)
      error            one norm per batch, pi_p ~ 1 - R_p                                   (E-AT, Eq. 4.4)
      loss             one norm per batch, pi_p ~ exp(gamma * EMA[L_adv - L_clean])         (Eq. A.2)
    `floor` mixes in a uniform distribution so no norm is ever starved.
    """

    def __init__(self, cfg, norms, seed=0):
        self.cfg, self.norms = cfg, list(norms)
        self.mode = cfg.get("mode", "all")
        self.gamma = cfg.get("gamma", 2.0)
        self.floor = cfg.get("floor", 0.1)
        self.probs = torch.full((len(norms),), 1.0 / len(norms))
        self.loss_gap = torch.zeros(len(norms))
        self.gen = torch.Generator().manual_seed(seed)
        self.epoch = 0

    @property
    def multi(self):
        return self.mode.startswith("all")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _mix(self, p):
        p = p / p.sum()
        return (1 - self.floor) * p + self.floor / len(p)

    def update(self, fb):
        R = torch.tensor([fb["nominal"][p] for p in self.norms])
        if self.mode in ("adaptive", "all_adaptive"):
            self.probs = self._mix(torch.softmax(-self.gamma * R, 0))
        elif self.mode == "error":
            self.probs = self._mix((1 - R).clamp_min(1e-3))
        elif self.mode == "loss":
            self.probs = self._mix(torch.softmax(self.gamma * self.loss_gap, 0))

    def observe_loss(self, norm, adv_loss, clean_loss, mom=0.9):
        i = self.norms.index(norm)
        self.loss_gap[i] = mom * self.loss_gap[i] + (1 - mom) * (adv_loss - clean_loss)

    def select(self, batch_idx):
        if self.multi:
            return list(self.norms)
        if self.mode == "cyclic":
            unit = self.cfg.get("cycle_unit", "epoch")
            period = self.cfg.get("period", 1)
            k = (self.epoch if unit == "epoch" else batch_idx) // period
            return [self.norms[k % len(self.norms)]]
        if self.mode == "random":
            return [self.norms[torch.randint(len(self.norms), (1,), generator=self.gen).item()]]
        i = torch.multinomial(self.probs, 1, generator=self.gen).item()
        return [self.norms[i]]

    def weights(self):
        """Loss weights for multi-norm aggregation, summing to len(norms)."""
        if self.mode == "all_adaptive":
            return {p: float(w) * len(self.norms) for p, w in zip(self.norms, self.probs)}
        return {p: 1.0 for p in self.norms}

    def entropy(self):
        p = self.probs.clamp_min(1e-12)
        return float(-(p * p.log()).sum())

    def summary(self):
        return {"probs": {p: float(v) for p, v in zip(self.norms, self.probs)}, "entropy": self.entropy(),
                "max_entropy": math.log(len(self.norms))}
