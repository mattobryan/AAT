import torch
import torch.nn as nn

from aat.attacks import perturb, perturb_norm, project_l1
from aat.losses import aggregate, pairing_loss
from aat.schedulers import EpsScheduler, NormScheduler


def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 10))


def test_l1_projection_is_exact():
    v = torch.randn(4, 50) * 3
    eps = torch.tensor([1.0, 5.0, 1000.0, 0.1])
    p = project_l1(v, eps)
    assert torch.all(p.abs().sum(1) <= eps + 1e-4)
    assert torch.allclose(p[2], v[2])  # already inside
    inside = project_l1(v, eps)
    assert torch.allclose(project_l1(inside, eps), inside, atol=1e-5)


def test_attacks_respect_budget_and_box():
    m = tiny_model()
    x, y = torch.rand(16, 3, 8, 8), torch.randint(0, 10, (16,))
    for norm, eps in [("linf", 8 / 255), ("l2", 0.5), ("l1", 3.0)]:
        per = torch.full((16,), eps)
        per[:8] *= 0.5  # per-sample budgets
        xa = perturb(m, x, y, norm, per, steps=5)
        assert xa.min() >= 0 and xa.max() <= 1
        assert torch.all(perturb_norm(xa - x, norm) <= per + 1e-4), norm


def test_attack_increases_loss():
    m = tiny_model()
    x, y = torch.rand(64, 3, 8, 8), torch.randint(0, 10, (64,))
    base = nn.functional.cross_entropy(m(x), y)
    for norm, eps in [("linf", 8 / 255), ("l2", 0.5), ("l1", 5.0)]:
        adv = nn.functional.cross_entropy(m(perturb(m, x, y, norm, eps, steps=10)), y)
        assert adv > base, norm


def _fb(norms, R, C=10):
    return {"nominal": dict(zip(norms, R)), "current": dict(zip(norms, R)),
            "class_nominal": {p: torch.full((C,), r) for p, r in zip(norms, R)},
            "class_current": {p: torch.linspace(0, 1, C) for p in norms}}


def test_class_feedback_eps_moves_toward_target():
    norms = ["linf"]
    cfg = {"linf": {"nominal": 1.0, "mode": "class_feedback", "init_frac": 0.75, "min_frac": 0.5,
                    "max_frac": 1.25, "alpha": 0.1, "tau": 0.5}}
    s = EpsScheduler(cfg, norms, 10, 10, "cpu")
    s.update(_fb(norms, [0.5]))
    e = s.class_eps("linf")
    assert e[0] < 0.75 < e[-1]  # weak class shrinks, strong class grows
    for _ in range(100):
        s.update(_fb(norms, [0.5]))
    assert s.class_eps("linf").min() >= 0.5 and s.class_eps("linf").max() <= 1.25


def test_sample_eps_gives_harder_samples_smaller_budget():
    cfg = {"l2": {"nominal": 1.0, "mode": "sample", "min_frac": 0.5, "max_frac": 1.5}}
    s = EpsScheduler(cfg, ["l2"], 10, 10, "cpu")
    y = torch.zeros(4, dtype=torch.long)
    e = s.sample_eps("l2", y, torch.tensor([0.1, 1.0, 2.0, 10.0]))
    assert torch.all(e[:-1] >= e[1:]) and e.min() >= 0.5 and e.max() <= 1.5


def test_adaptive_norm_scheduler_prefers_weak_norm():
    norms = ["linf", "l2", "l1"]
    s = NormScheduler({"mode": "adaptive", "gamma": 5.0, "floor": 0.0}, norms)
    s.update(_fb(norms, [0.4, 0.7, 0.2]))
    assert s.probs.argmax().item() == 2
    picks = [s.select(i)[0] for i in range(2000)]
    assert picks.count("l1") > picks.count("linf") > picks.count("l2")
    c = NormScheduler({"mode": "cyclic", "cycle_unit": "batch", "period": 1}, norms)
    assert [c.select(i)[0] for i in range(4)] == ["linf", "l2", "l1", "linf"]


def test_losses():
    ce = {"linf": torch.tensor([1.0, 3.0]), "l2": torch.tensor([2.0, 1.0]), "l1": torch.tensor([0.5, 0.5])}
    loss, _ = aggregate(ce, {p: 1.0 for p in ce}, "max")
    assert torch.isclose(loss, torch.tensor(2.5))
    loss, st = aggregate(ce, {"linf": 1.5, "l2": 1.0, "l1": 0.5}, "hybrid", lambda_sec=0.5)
    assert torch.isclose(loss, torch.tensor(2.5 + 0.5 * 0.25)) and "sec_primary_ratio" in st
    logits = {p: torch.randn(8, 10) for p in ce}
    y = torch.randint(0, 10, (8,))
    kl, _ = pairing_loss(logits, {p: torch.rand(8) for p in ce}, y)
    assert kl >= 0
