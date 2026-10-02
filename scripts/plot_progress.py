"""RAMP (fixed eps) vs AAT budget-neutral (aat_rel): progression and final per-class results.

Needs HF_REPO and HF_TOKEN in the environment (reads logs and eval JSONs from the Hub) and matplotlib.
"""
import json, os, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

repo, tok = os.environ["HF_REPO"], os.environ["HF_TOKEN"]
C = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e9e8e4"
ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / "docs/figures/aat_vs_ramp_progress.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def log(run):
    return open(hf_hub_download(repo, f"ramp_official/{run}/log_train.txt", token=tok)).read().splitlines()

def checks(lines):  # {epoch: union%} from the every-10-epochs monitor
    out = {}
    for l in lines:
        m = re.search(r"\[epoch\] (\d+) \[time\]", l)
        if m and "[eval test]" in l:
            u = re.search(r"union ([\d.]+)%", l.split("[eval test]")[1])
            if u:
                out[int(m.group(1))] = float(u.group(1))
    return out

def eps_traj(lines):  # {class: [eps/nominal per epoch]} for the l_inf controller
    rows = {}
    for l in lines:
        m = re.search(r"after epoch (\d+) \| Linf A_c [\d. ]+ eps_c/nominal ([\d. ]+?) mean", l)
        if m:
            rows[int(m.group(1))] = [float(x) for x in m.group(2).split()]
    ep = sorted(rows)
    return [0] + ep, [[1.0] * 10] + [rows[e] for e in ep]

def final(run_local=None, run_hf=None, s=0):
    if run_hf:
        return json.load(open(hf_hub_download(repo, f"ramp_official/{run_hf}_s{s}/eval_autoattack.json", token=tok)))
    return json.load(open(ROOT / f"baselines/results/ramp_scratch_l5_s{s}_eval_autoattack.json"))

def pc(r, key):
    v = r[key]
    return [v[i] if isinstance(v, list) else v[c] for i, c in enumerate(C)]

A_logs = [log(f"aatrel_l5_s{s}") for s in (0, 1)]
R_logs = [log(f"ramp_scratch_l5_s{s}") for s in (0, 1)]
A_fin = [final(run_hf="aatrel_l5", s=s) for s in (0, 1)]
R_fin = [final(s=s) for s in (0, 1)]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
fig = plt.figure(figsize=(14, 9.6), facecolor=SURF)
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22, left=0.06, right=0.97, top=0.80, bottom=0.10)

def style(ax, title, sub):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")

# --- A: per-class eps trajectories (AAT only; RAMP is the fixed line at 1.0)
ax = fig.add_subplot(gs[0, 0])
style(ax, "How each class's training budget moved (AAT)", "ℓ∞ training ε as a multiple of 8/255, updated once per epoch · seed 0 (seed 1 within 0.005)")
ep, traj = eps_traj(A_logs[0])
finals = traj[-1]
order = sorted(range(10), key=lambda i: finals[i])
weak = {C.index(c) for c in ("cat", "deer", "bird", "dog")}
for i in range(10):
    ys = [t[i] for t in traj]
    ax.plot(ep, ys, color=ORANGE, lw=2.0 if i in weak else 1.2, alpha=1.0 if i in weak else 0.55, solid_capstyle="round")
ax.axhline(1.0, color=BLUE, lw=2, ls=(0, (5, 3)))
ax.text(28, 1.0, "RAMP: fixed at 1.0 for every class", color=BLUE, fontsize=9, va="center", bbox=dict(facecolor=SURF, edgecolor="none", pad=2))
# direct labels at the right end, de-collided
ys = [finals[i] for i in order]; pos = ys[:]
for k in range(1, len(pos)):
    pos[k] = max(pos[k], pos[k - 1] + 0.055)
shift = (sum(ys) - sum(pos)) / len(pos)
pos = [p + shift for p in pos]
for k, i in enumerate(order):
    ax.text(ep[-1] + 1.2, pos[k], f"{C[i]} {finals[i]:.2f}", fontsize=8.5, va="center", color=INK if i in weak else INK2,
            fontweight="bold" if i in weak else "normal")
ax.axhline(0.5, color=GRID, lw=1); ax.axhline(1.5, color=GRID, lw=1)
ax.text(1, 0.505, "floor 0.5", fontsize=8, color=INK2, va="bottom"); ax.text(1, 1.495, "cap 1.5", fontsize=8, color=INK2, va="top")
ax.set_xlim(0, ep[-1] + 14); ax.set_ylim(0.45, 1.55)
ax.set_xlabel("epoch", fontsize=9)

# --- B: in-training union check
ax = fig.add_subplot(gs[0, 1])
style(ax, "In-training union check, every 10 epochs", "200 images, weaker attack, s.e. ≈ 3.5 pts · thin = single seeds, thick = mean")
for logs, col, name in ((R_logs, BLUE, "RAMP"), (A_logs, ORANGE, "AAT")):
    cs = [checks(l) for l in logs]
    eps_ = sorted(set(cs[0]) & set(cs[1]))
    for c_ in cs:
        ax.plot(eps_, [c_[e] for e in eps_], color=col, lw=1, alpha=0.4)
    mean = [sum(c_[e] for c_ in cs) / 2 for e in eps_]
    ax.plot(eps_, mean, color=col, lw=2.4, marker="o", ms=5.5, mfc=col, mec=SURF, mew=1.5, label=name)
    ax.text(eps_[-1] + 1.2, mean[-1], name, color=INK, fontsize=9.5, va="center", fontweight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_xlim(8, 88); ax.set_xlabel("epoch", fontsize=9); ax.set_ylabel("union robust accuracy (%)", fontsize=9)

# --- C / D: final per-class
ro = sorted(range(10), key=lambda i: sum(pc(r, "union_per_class")[i] for r in R_fin))
def grouped(ax, key, title, sub, note_thr, ylab):
    style(ax, title, sub)
    rm = [sum(pc(r, key)[i] for r in R_fin) / 2 * 100 for i in ro]
    am = [sum(pc(r, key)[i] for r in A_fin) / 2 * 100 for i in ro]
    x = list(range(10)); w = 0.38
    ax.bar([k - w / 2 for k in x], rm, w, color=BLUE, edgecolor=SURF, linewidth=2, label="RAMP")
    ax.bar([k + w / 2 for k in x], am, w, color=ORANGE, edgecolor=SURF, linewidth=2, label="AAT")
    rs = [[pc(r, key)[i] * 100 for r in R_fin] for i in ro]
    as_ = [[pc(r, key)[i] * 100 for r in A_fin] for i in ro]
    for k in x:
        d = am[k] - rm[k]
        if min(as_[k]) > max(rs[k]) or max(as_[k]) < min(rs[k]):
            ax.text(k, max(am[k], rm[k]) + 1.2, f"{d:+.1f}", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([C[i] for i in ro], fontsize=9)
    ax.set_ylabel(ylab, fontsize=9); ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
    return rm, am

axc = fig.add_subplot(gs[1, 0])
grouped(axc, "union_per_class", "Final union robustness per class", "AutoAttack, 1000 imgs, mean of 2 seeds · labels where the 2-seed ranges don't overlap", 3, "union robust accuracy (%)")
axd = fig.add_subplot(gs[1, 1])
grouped(axd, "clean_per_class", "Final clean accuracy per class", "same models · labels where the 2-seed ranges don't overlap", 3, "clean accuracy (%)")
axd.set_ylim(0, 105); axc.set_ylim(0, 80)

fig.text(0.06, 0.975, "RAMP (fixed ε) vs AAT with budget-neutral per-class ε", fontsize=16, fontweight="bold", color=INK, va="top")
fig.text(0.06, 0.940, "Same training budget on average, only the split across classes differs · CIFAR-10, PreActResNet-18, 80 epochs, 2 seeds each",
         fontsize=10, color=INK2, va="top")
fig.text(0.06, 0.03, "Evaluation always uses fixed ε (8/255, 0.5, 12); adaptive ε is training-only. Per-class cells hold ~100 images (s.e. ≈ 5 pts) and per-class seed spread reaches 12 pts, so per-class differences are descriptive (n=2 seeds).",
         fontsize=8.5, color=INK2)
fig.savefig(OUT, dpi=150, facecolor=SURF)
print("saved", OUT)
