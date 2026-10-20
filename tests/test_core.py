import pytest
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


def test_mean_preserving_class_feedback_keeps_budget_and_bounds():
    norms = ["linf"]
    cfg = {"linf": {"nominal": 1.0, "mode": "class_feedback", "init_frac": 1.0, "min_frac": 0.5,
                    "max_frac": 1.5, "alpha": 0.2, "tau": 0.3, "mean_preserve": True}}
    s = EpsScheduler(cfg, norms, 10, 10, "cpu")
    for _ in range(30):
        s.update(_fb(norms, [0.8]))  # most classes sit above tau, so the raw controller would keep raising the budget
        e = s.class_eps("linf")
        assert abs(float(e.mean()) - 1.0) < 1e-5
        assert float(e.min()) >= 0.5 - 1e-6 and float(e.max()) <= 1.5 + 1e-6
    assert e[0] < e[-1]  # the ordering by robustness survives the projection
    # without mean_preserve the same feedback inflates the budget above nominal
    cfg["linf"]["mean_preserve"] = False
    s2 = EpsScheduler(cfg, norms, 10, 10, "cpu")
    for _ in range(30):
        s2.update(_fb(norms, [0.8]))
    assert float(s2.class_eps("linf").mean()) > 1.05


def test_gated_model_shapes_gate_and_gradients():
    from aat.models import GatedBN, build_model
    torch.manual_seed(0)
    m = build_model("preactresnet18_gated3")
    assert any(isinstance(mod, GatedBN) for mod in m.modules())
    # every block / final BN is a GatedBN: the only affine BatchNorm2d left is the gate head's own
    assert [n for n, mod in m.named_modules() if isinstance(mod, nn.BatchNorm2d) and mod.affine] == ["1.gate_bn"]
    m.eval()
    x = torch.rand(4, 3, 32, 32, requires_grad=True)
    out = m(x)
    gl = m[1].last_gate_logits
    assert out.shape == (4, 10) and gl.shape == (4, 3)
    g = next(mod.g for mod in m.modules() if isinstance(mod, GatedBN))
    assert torch.allclose(g.sum(1), torch.ones(4), atol=1e-5)
    gx, = torch.autograd.grad(out.sum(), x, retain_graph=True)
    assert gx.abs().sum() > 0
    gx_gate, = torch.autograd.grad(gl.sum(), x, retain_graph=True)
    assert gx_gate.abs().sum() > 0
    # gradients must flow through the gate into the per-threat affine and the gate head
    m.zero_grad()
    out.sum().backward()
    assert m[1].gate_fc.weight.grad.abs().sum() > 0 and m[1].bn.gamma.grad.abs().sum() > 0
    assert m[0].mean.shape == (1, 3, 1, 1) and set(k.split(".")[0] for k in m.state_dict()) == {"0", "1"}


def test_plain_model_path_unchanged():
    from aat.models import GatedBN, build_model
    m = build_model("preactresnet18")
    assert not any(isinstance(mod, GatedBN) for mod in m.modules())
    assert not hasattr(m[1], "gate_fc") and m[1].last_gate_logits is None
    assert "1.bn.weight" in m.state_dict() and "1.layer1.0.bn1.running_mean" in m.state_dict()
    m.eval()
    assert m(torch.rand(2, 3, 32, 32)).shape == (2, 10) and m[1].last_gate_logits is None


def test_gate_attack_modes_force_detach_sttemp_and_restore():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("eval_gated_adaptive", os.path.join(os.path.dirname(__file__), "..", "scripts", "eval_gated_adaptive.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    from aat.models import GatedBN, build_model
    torch.manual_seed(0)
    m = build_model("preactresnet18_gated3").eval()
    net = m[1]
    x = torch.rand(4, 3, 32, 32)
    gate = lambda: next(mm.g for mm in m.modules() if isinstance(mm, GatedBN))
    with torch.no_grad():
        ref = m(x).clone(); free = net.last_gate_logits.clone()
        with mod.GateAttack(net, "force", 2):
            m(x)
            assert torch.allclose(gate()[:, 2], torch.ones(4)) and torch.allclose(gate()[:, :2], torch.zeros(4, 2))
        with mod.GateAttack(net, "detach", None):  # forward identical to the real model
            assert torch.allclose(m(x), ref, atol=1e-6)
        with mod.GateAttack(net, "sttemp", 10.0):  # forward value is the real gate
            assert torch.allclose(m(x), ref, atol=1e-6)
        assert torch.allclose(m(x), ref, atol=1e-6) and torch.allclose(net.last_gate_logits, free, atol=1e-6)
    # gradients: detach cuts the gate path, the real model and the straight-through temperature keep it
    def gate_grad(mode, arg):
        xx = x.clone().requires_grad_(True)
        with mod.GateAttack(net, mode, arg):
            out = m(xx)
        return torch.autograd.grad(out.sum(), xx, allow_unused=True)[0]
    # make the per-threat affines distinct so the gate matters
    with torch.no_grad():
        for mm in m.modules():
            if isinstance(mm, GatedBN):
                mm.gamma.add_(0.3 * torch.randn_like(mm.gamma)); mm.beta.add_(0.3 * torch.randn_like(mm.beta))
    xx = x.clone().requires_grad_(True)
    g_real = torch.autograd.grad(m(xx).sum(), xx)[0]
    g_det, g_st = gate_grad("detach", None), gate_grad("sttemp", 10.0)
    assert not torch.allclose(g_det, g_real, atol=1e-7)  # the gate path contributed to the real gradient
    assert g_st is not None and g_st.abs().sum() > 0 and not torch.allclose(g_st, g_det, atol=1e-7)
    # hooks are removed even when the attack raises
    try:
        with mod.GateAttack(net, "force", 0):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    with torch.no_grad():
        assert torch.allclose(m(x), m(x)) and not (net.gate_fc._forward_hooks or any(mm._forward_pre_hooks for mm in m.modules()))
    a, b = torch.tensor([True, True, False, True]), torch.tensor([True, False, False, True])
    assert mod.combine(a, b).tolist() == [True, False, False, True]
    assert [v[0] for v in mod.variant_plan(3, (20.0,))] == ["force0", "force1", "force2", "detach", "sttemp20"]
    assert len(mod.variant_plan(0, (20.0,))) == 5  # matched number of extra attacks for a plain model


def test_gated_model_initialises_from_a_plain_checkpoint_and_matches_it():
    from aat.models import GatedBN, build_model, load_plain_into_gated
    torch.manual_seed(0)
    plain = build_model("preactresnet18")
    plain.train()
    for _ in range(3):  # give the BN running statistics non-trivial values
        plain(torch.rand(16, 3, 32, 32))
    with torch.no_grad():
        for mod in plain.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.weight.add_(0.2 * torch.randn_like(mod.weight)); mod.bias.add_(0.2 * torch.randn_like(mod.bias))
    sd = plain.state_dict()
    gated = build_model("preactresnet18_gated3")
    load_plain_into_gated(gated, sd)
    plain.eval(); gated.eval()
    x = torch.rand(6, 3, 32, 32)
    with torch.no_grad():  # identical affine rows: the gate cannot change the function at initialisation
        assert torch.allclose(plain(x), gated(x), atol=1e-4)
    # a plain checkpoint that does not fit is rejected, not silently half-loaded
    bad = dict(sd); bad.pop("1.linear.weight")
    try:
        load_plain_into_gated(build_model("preactresnet18_gated3"), bad)
        raise AssertionError("expected a RuntimeError")
    except RuntimeError:
        pass


def test_strong_eval_extras_run_on_survivors_only():
    """scripts/eval_strong.py: extra attacks run on the points that survived APGD-CE + APGD-T, the full-suite mask is a
    subset of the standard mask, and a tiny budget removes nothing."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    pytest.importorskip("autoattack")
    from eval_strong import strong_norm
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        y = model(x).argmax(1)
    clean = torch.ones(4, dtype=torch.bool)
    m_std, m, drops = strong_norm(model, x, y, clean, "linf", 1e-6, bs=4, dev="cpu")
    assert m_std.all() and m.all() and drops == {"fab-t": 0, "square": 0}


def _normctl_cfg(init=(1.0, 1.0, 1.0), alpha=0.5, mean_frac=1.0833):
    cfg = {"norm_ctl": {"alpha": alpha, "mean_frac": mean_frac}}
    for p, nom, f in zip(["linf", "l2", "l1"], [8 / 255, 0.5, 12.0], init):
        cfg[p] = {"nominal": nom, "mode": "norm_feedback", "init_frac": f, "min_frac": 0.25, "max_frac": 2.0}
    return cfg


def test_norm_feedback_weakest_norm_gains_strongest_loses():
    norms = ["linf", "l2", "l1"]
    s = EpsScheduler(_normctl_cfg(), norms, 10, 10, "cpu")
    s.update(_fb(norms, [0.45, 0.65, 0.30]))  # l1 weakest, l2 strongest
    assert s.frac["l1"] > s.frac["linf"] > s.frac["l2"]
    assert float(s.class_eps("l1").mean()) == pytest.approx(s.frac["l1"] * 12.0)
    summ = s.summary()
    assert summ["l1"]["frac"] == s.frac["l1"] and summ["l1"]["acc_nominal"] == pytest.approx(0.30)


def test_norm_feedback_keeps_mean_fraction_and_bounds():
    norms = ["linf", "l2", "l1"]
    s = EpsScheduler(_normctl_cfg(init=(1.0, 0.75, 1.5), alpha=0.9), norms, 10, 10, "cpu")
    for acc in ([0.9, 0.1, 0.5], [0.1, 0.9, 0.2], [0.5, 0.5, 0.0]) * 20:
        s.update(_fb(norms, list(acc)))
        f = torch.tensor([s.frac[p] for p in norms])
        assert abs(float(f.mean()) - 1.0833) < 1e-6
        assert float(f.min()) >= 0.25 - 1e-9 and float(f.max()) <= 2.0 + 1e-9


def test_norm_feedback_equal_accuracy_is_stationary():
    norms = ["linf", "l2", "l1"]
    s = EpsScheduler(_normctl_cfg(init=(1.0, 0.75, 1.5), mean_frac=(1.0 + 0.75 + 1.5) / 3), norms, 10, 10, "cpu")
    for _ in range(5):
        s.update(_fb(norms, [0.4, 0.4, 0.4]))
    assert [s.frac[p] for p in norms] == pytest.approx([1.0, 0.75, 1.5])


def test_static_scheduler_unchanged_by_norm_feedback_code():
    norms = ["linf", "l2", "l1"]
    cfg = {p: {"nominal": n, "mode": "static"} for p, n in zip(norms, [8 / 255, 0.5, 12.0])}
    cfg["norm_ctl"] = {"alpha": 0.5, "mean_frac": 2.0}  # ignored when no norm uses norm_feedback
    s = EpsScheduler(cfg, norms, 10, 10, "cpu")
    s.update(_fb(norms, [0.9, 0.1, 0.5]))
    assert s.ctl_norms == []
    for p, n in zip(norms, [8 / 255, 0.5, 12.0]):
        assert torch.allclose(s.class_eps(p), torch.full((10,), n))
        assert set(s.summary()[p]) == {"mean", "min", "max", "zero_budget_classes"}


def test_norm_feedback_resume_matches_uninterrupted_run():
    norms = ["linf", "l2", "l1"]
    fbs = [_fb(norms, a) for a in ([0.45, 0.65, 0.30], [0.44, 0.66, 0.34], [0.43, 0.66, 0.38], [0.43, 0.66, 0.40])]
    ref = EpsScheduler(_normctl_cfg(), norms, 10, 10, "cpu")
    for fb in fbs:
        ref.update(fb)
    a = EpsScheduler(_normctl_cfg(), norms, 10, 10, "cpu")
    for fb in fbs[:2]:
        a.update(fb)
    st, ctl = {p: v.clone() for p, v in a.state.items()}, a.ctl_state()
    b = EpsScheduler(_normctl_cfg(), norms, 10, 10, "cpu")  # a fresh process after a Kaggle restart
    b.state = st
    b.load_ctl_state(ctl)
    for fb in fbs[2:]:
        b.update(fb)
    assert all(b.frac[p] == pytest.approx(ref.frac[p]) for p in norms)
    c = EpsScheduler(_normctl_cfg(), norms, 10, 10, "cpu")  # checkpoint written before ctl state existed
    c.state = {p: v.clone() for p, v in a.state.items()}
    c.load_ctl_state(None)
    assert all(c.frac[p] == pytest.approx(a.frac[p]) for p in norms)
