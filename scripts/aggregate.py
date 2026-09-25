"""Collect eval_*.json across runs -> results/summary.{md,csv} (mean ± std over seeds) and a
clean-vs-union trade-off plot.

    python scripts/aggregate.py --runs runs --tag autoattack
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

COLS = ["clean", "linf", "l2", "l1", "union", "union_worst_class"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="autoattack")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    groups = defaultdict(list)
    train_min = {}
    for f in sorted(glob.glob(os.path.join(a.runs, "*", f"eval_{a.tag}.json"))):
        r = json.load(open(f))
        row = {"clean": r["clean"], **r["robust"], "union": r["union"], "union_worst_class": r["union_worst_class"]}
        for k, v in r.get("unseen", {}).items():
            row[f"unseen:{k}"] = v
        groups[r["name"]].append(row)
        log = os.path.join(os.path.dirname(f), "log.jsonl")
        if os.path.exists(log):
            recs = [json.loads(l) for l in open(log)]
            train_min.setdefault(r["name"], []).append(sum(x["epoch_time_s"] for x in recs) / 60)

    unseen_cols = sorted({k for rows in groups.values() for row in rows for k in row if k.startswith("unseen:")})
    cols = COLS + unseen_cols
    lines = ["| run | seeds | " + " | ".join(cols) + " | train GPU-min |", "|" + "---|" * (len(cols) + 3)]
    csv = ["run,seeds," + ",".join(f"{c}_mean,{c}_std" for c in cols) + ",train_min"]
    points = []
    for name, rows in sorted(groups.items()):
        cells, csvc = [], []
        for c in cols:
            v = np.array([row[c] for row in rows if c in row]) * 100
            if len(v) == 0:
                cells.append("–"); csvc += ["", ""]
                continue
            cells.append(f"{v.mean():.1f}" + (f" ± {v.std(ddof=1):.1f}" if len(v) > 1 else ""))
            csvc += [f"{v.mean():.3f}", f"{v.std(ddof=1) if len(v) > 1 else 0:.3f}"]
        tm = np.mean(train_min.get(name, [np.nan]))
        lines.append(f"| {name} | {len(rows)} | " + " | ".join(cells) + f" | {tm:.0f} |")
        csv.append(f"{name},{len(rows)}," + ",".join(csvc) + f",{tm:.1f}")
        points.append((name, np.mean([r["clean"] for r in rows]) * 100, np.mean([r["union"] for r in rows]) * 100))

    open(os.path.join(a.out, f"summary_{a.tag}.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(a.out, f"summary_{a.tag}.csv"), "w").write("\n".join(csv) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, c, u in points:
            ax.scatter(c, u, s=36)
            ax.annotate(name, (c, u), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("clean accuracy (%)")
        ax.set_ylabel("union robust accuracy (%)")
        ax.set_title(f"Clean vs union robustness ({a.tag})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"tradeoff_{a.tag}.png"), dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
