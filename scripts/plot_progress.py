"""RAMP (fixed eps) vs two budget-neutral per-class eps controllers (aat_rel, aat_flip): progression and final results.

Needs HF_REPO and HF_TOKEN in the environment (reads logs and eval JSONs from the Hub) and matplotlib.
RAMP final results come from the locked files in baselines/results/.
"""
import json, os, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

repo, tok = os.environ["HF_REPO"], os.environ["HF_TOKEN"]
C = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated; aqua needs direct labels)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e9e8e4"
ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / "docs/figures/aat_three_way.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RUNS = [("RAMP", BLUE, "ramp_scratch_l5"), ("aat_rel", ORANGE, "aatrel_l5"), ("aat_flip", AQUA, "aatflip_l5")]


def log(run):
    return open(hf_hub_download(repo, f"ramp_official/{run}/log_train.txt", token=tok)).read().splitlines()


def checks(lines):  # {epoch: union%} from the every-10-epochs monitor (200 images)
    out = {}
    for l in lines:
        m = re.search(r"\[epoch\] (\d+) \[time\]", l)
        if m and "[eval test]" in l:
            u = re.search(r"union ([\d.]+)%", l.split("[eval test]")[1])
            if u:
                out[int(m.group(1))] = float(u.group(1))
    return out


def eps_traj(lines):  # per-class eps/nominal of the l_inf controller, per epoch
    rows = {}
    for l in lines:
        m = re.search(r"after epoch (\d+) \| Linf A_c [\d. ]+ eps_c/nominal ([\d. ]+?) mean", l)
        if m:
            rows[int(m.group(1))] = [float(x) for x in m.group(2).split()]
    ep = sorted(rows)
    return [0] + ep, [[1.0] * 10] + [rows[e] for e in ep]


def final(prefix, s):
    if prefix == "ramp_scratch_l5":  # locked baseline
        return json.load(open(ROOT / f"baselines/results/ramp_scratch_l5_s{s}_eval_autoattack.json"))
    return json.load(open(hf_hub_download(repo, f"ramp_official/{prefix}_s{s}/eval_autoattack.json", token=tok)))


def pc(r, key):
    v = r[key]
    return [v[i] if isinstance(v, list) else v[c] for i, c in enumerate(C)]


logs = {n: [log(f"{p}_s{s}") for s in (0, 1)] for n, _, p in RUNS}
fin = {n: [final(p, s) for s in (0, 1)] for n, _, p in RUNS}
for n in fin:
    assert all(r["n"] == 1000 for r in fin[n]), n

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig = plt.figure(figsize=(14, 14.4), facecolor=SURF)
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22, left=0.06, right=0.97, top=0.88, bottom=0.07)


def style(ax, title, sub):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def dodge(vals, gap, lo=-1e9, hi=1e9):
    """Spread label positions at least `gap` apart, keeping them inside [lo, hi]."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    p = [vals[i] for i in order]
    for k in range(1, len(p)):
        p[k] = max(p[k], p[k - 1] + gap)
    p[-1] = min(p[-1], hi)
    for k in range(len(p) - 2, -1, -1):
        p[k] = min(p[k], p[k + 1] - gap)
    p[0] = max(p[0], lo)
    for k in range(1, len(p)):
        p[k] = max(p[k], p[k - 1] + gap)
    return {i: p[k] for k, i in enumerate(order)}


panels = {}

# --- row 0: per-class eps trajectories of the two controllers (RAMP = fixed line at 1.0)
for col, (name, color, prefix) in enumerate(RUNS[1:]):
    ax = fig.add_subplot(gs[0, col])
    panels[f"eps_{name}"] = ax
    ep, traj = eps_traj(logs[name][0])
    ep1, traj1 = eps_traj(logs[name][1])
    gap01 = max(abs(a - b) for t0, t1 in zip(traj, traj1) for a, b in zip(t0, t1))  # same epochs from 0..80
    style(ax, f"How each class's training budget moved ({name})",
          f"ℓ∞ training ε as a multiple of 8/255, updated once per epoch · seed 0 (seed 1 within {gap01:.3f})")
    finals = traj[-1]
    weak = {C.index(c) for c in ("cat", "deer", "bird", "dog")}
    for i in range(10):
        ax.plot(ep, [t[i] for t in traj], color=color, lw=2.0 if i in weak else 1.2,
                alpha=1.0 if i in weak else 0.55, solid_capstyle="round")
    ax.axhline(1.0, color=BLUE, lw=2, ls=(0, (5, 3)))
    ax.text(28, 1.0, "RAMP: fixed at 1.0 for every class", color=BLUE, fontsize=9, va="center",
            bbox=dict(facecolor=SURF, edgecolor="none", pad=2))
    pos = dodge(finals, 0.062, lo=0.48, hi=1.50)
    for i in range(10):
        ax.text(ep[-1] + 1.2, pos[i], f"{C[i]} {finals[i]:.2f}", fontsize=8.5, va="center",
                color=INK if i in weak else INK2, fontweight="bold" if i in weak else "normal")
    ax.axhline(0.5, color=GRID, lw=1); ax.axhline(1.5, color=GRID, lw=1)
    ax.text(1, 0.505, "floor 0.5", fontsize=8, color=INK2, va="bottom"); ax.text(1, 1.495, "cap 1.5", fontsize=8, color=INK2, va="top")
    ax.set_xlim(0, ep[-1] + 14); ax.set_ylim(0.45, 1.55); ax.set_xlabel("epoch", fontsize=9)

# --- row 1 left: in-training union check, three runs
ax = fig.add_subplot(gs[1, 0])
panels["intrain"] = ax
style(ax, "In-training union check, every 10 epochs", "200 images, weaker attack, s.e. ≈ 3.5 pts · thin = single seeds, thick = mean of 2")
ends = {}
for name, color, _ in RUNS:
    cs = [checks(l) for l in logs[name]]
    eps_ = sorted(set(cs[0]) & set(cs[1]))
    for c_ in cs:
        ax.plot(eps_, [c_[e] for e in eps_], color=color, lw=1, alpha=0.35)
    mean = [sum(c_[e] for c_ in cs) / 2 for e in eps_]
    ax.plot(eps_, mean, color=color, lw=2.4, marker="o", ms=5.5, mfc=color, mec=SURF, mew=1.5, label=name)
    ends[name] = (eps_[-1], mean[-1])
pos = dodge([ends[n][1] for n, _, _ in RUNS], 1.1)
for k, (name, _, _) in enumerate(RUNS):
    ax.text(ends[name][0] + 1.2, pos[k], name, color=INK, fontsize=9.5, va="center", fontweight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_xlim(8, 92); ax.set_xlabel("epoch", fontsize=9); ax.set_ylabel("union robust accuracy (%)", fontsize=9)

# --- row 1 right: overall final metrics (mean of 2 seeds, dots = single seeds)
ax = fig.add_subplot(gs[1, 1])
panels["overall"] = ax
style(ax, "Final overall results", "AutoAttack, fixed ε, 1000 images · bars = mean of 2 seeds, dots = single seeds")
metrics = [("clean", lambda r: r["clean"]), ("union", lambda r: r["union"]), ("worst-class union", lambda r: r["union_worst_class"])]
w = 0.26
for j, (name, color, _) in enumerate(RUNS):
    for m, (_, f) in enumerate(metrics):
        vals = [f(r) * 100 for r in fin[name]]
        x = m + (j - 1) * w
        ax.bar(x, sum(vals) / 2, w, color=color, edgecolor=SURF, linewidth=2, label=name if m == 0 else None)
        ax.scatter([x] * 2, vals, s=14, color=INK, zorder=3)
        ax.text(x, max(vals) + 1.5, f"{sum(vals) / 2:.1f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([m for m, _ in metrics], fontsize=9.5)
ax.set_ylim(0, 95); ax.set_ylabel("accuracy (%)", fontsize=9)
ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=3)

# --- row 2: per-class final union and clean, three runs
ro = sorted(range(10), key=lambda i: sum(pc(r, "union_per_class")[i] for r in fin["RAMP"]))


def grouped(ax, key, title, sub, ylab):
    style(ax, title, sub)
    w = 0.27
    for j, (name, color, _) in enumerate(RUNS):
        m = [sum(pc(r, key)[i] for r in fin[name]) / 2 * 100 for i in ro]
        ax.bar([k + (j - 1) * w for k in range(10)], m, w, color=color, edgecolor=SURF, linewidth=1.5, label=name)
        if name == "RAMP":
            continue
        for k, i in enumerate(ro):
            a = [pc(r, key)[i] * 100 for r in fin[name]]; b = [pc(r, key)[i] * 100 for r in fin["RAMP"]]
            if min(a) > max(b) or max(a) < min(b):
                ax.text(k + (j - 1) * w, m[k] + 1.0, f"{m[k] - sum(b) / 2:+.0f}", ha="center", fontsize=7.5, color=INK, fontweight="bold")
    ax.set_xticks(range(10)); ax.set_xticklabels([C[i] for i in ro], fontsize=9)
    ax.set_ylabel(ylab, fontsize=9); ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=3)


axc = fig.add_subplot(gs[2, 0])
panels["perclass_union"] = axc
grouped(axc, "union_per_class", "Final union robustness per class", "mean of 2 seeds · labels = change vs RAMP where the 2-seed ranges don't overlap", "union robust accuracy (%)")
axc.set_ylim(0, 85)
axd = fig.add_subplot(gs[2, 1])
panels["perclass_clean"] = axd
grouped(axd, "clean_per_class", "Final clean accuracy per class", "same models · labels = change vs RAMP where the 2-seed ranges don't overlap", "clean accuracy (%)")
axd.set_ylim(0, 112)

fig.text(0.06, 0.975, "RAMP (fixed ε) vs two budget-neutral per-class ε controllers", fontsize=16, fontweight="bold", color=INK, va="top")
fig.text(0.06, 0.953, "aat_rel: classes with higher robust accuracy get more ε · aat_flip: the reverse · same mean ε as RAMP · CIFAR-10, PreActResNet-18, 80 epochs, 2 seeds each",
         fontsize=10, color=INK2, va="top")
fig.text(0.06, 0.02, "Evaluation always uses fixed ε (8/255, 0.5, 12); adaptive ε is training-only. Per-class cells hold ~100 images (s.e. ≈ 5 pts) and per-class seed spread reaches 12 pts, so per-class differences are descriptive (n=2 seeds).",
         fontsize=8.5, color=INK2)
fig.savefig(OUT, dpi=150, facecolor=SURF)
print("saved", OUT)

# one image per panel, for slides
renderer = fig.canvas.get_renderer()
for key, ax in panels.items():
    bb = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(0.12)
    fig.savefig(str(ROOT / f"docs/figures/panel_{key}.png"), dpi=150, facecolor=SURF, bbox_inches=bb)
print("panels:", ", ".join(panels))
