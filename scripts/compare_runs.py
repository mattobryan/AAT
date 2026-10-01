"""Run-vs-run comparison of eval_autoattack.json files (e.g. AAT vs the locked RAMP baseline).

    python scripts/compare_runs.py --a "runs_official/aat_ramp_l5_s*/eval_autoattack.json" \
        --b "baselines/results/ramp_scratch_l5_s*_eval_autoattack.json"

Prints per-seed values and the per-metric mean of A, mean of B and mean(A) - mean(B) in percentage
points. Descriptive only: no significance claim is made (n seeds is tiny).
"""
import argparse
import glob
import json
import re

import numpy as np

METRICS = ["clean", "linf", "l2", "l1", "union", "union_worst_class", "union_class_std"]
SETUP = ["n", "backend", "attacks", "trained_norms"]


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        vals = {"clean": r["clean"], **r["robust"], "union": r["union"],
                "union_worst_class": r["union_worst_class"], "union_class_std": r["union_class_std"]}
        setup = {k: r[k] for k in SETUP}
        setup["unseen_keys"] = sorted(r.get("unseen", {}))
        m = re.search(r"_s(\d+)", f.split("/")[-1] if "_s" in f.split("/")[-1] else f)
        seed = int(m.group(1)) if m else r.get("seed")  # file name wins: some eval JSONs carry the eval seed
        rows.append((seed, f, vals, setup))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="glob for runs A (e.g. AAT)")
    ap.add_argument("--b", required=True, help="glob for runs B (e.g. RAMP baseline)")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    a = ap.parse_args()
    ra, rb = load(a.a), load(a.b)
    if not ra or not rb:
        raise SystemExit(f"no files matched: A={len(ra)} B={len(rb)}")
    na, nb = a.names
    ref = ra[0][3]
    for r in ra + rb:
        if r[3] != ref:
            raise SystemExit(f"evaluation setups differ ({r[1]}): {r[3]} vs {ref}; not comparable")
    ra, rb = sorted(ra, key=lambda r: r[0]), sorted(rb, key=lambda r: r[0])
    if [r[0] for r in ra] != [r[0] for r in rb]:
        print(f"note: seed sets differ, A={[r[0] for r in ra]} B={[r[0] for r in rb]}; per-seed columns follow each side's seeds")
    print(f"| metric | {na} per seed | {nb} per seed | {na} mean | {nb} mean | mean delta (pp) |")
    print("|---|---|---|---|---|---|")
    for m in METRICS:
        va = [100 * r[2][m] for r in ra]
        vb = [100 * r[2][m] for r in rb]
        fmt = lambda v: " / ".join(f"{x:.1f}" for x in v)
        print(f"| {m} | {fmt(va)} | {fmt(vb)} | {np.mean(va):.2f} | {np.mean(vb):.2f} | {np.mean(va) - np.mean(vb):+.2f} |")
    print(f"seeds {na}: {[r[0] for r in ra]}  seeds {nb}: {[r[0] for r in rb]}  (n={len(ra)} vs {len(rb)}; descriptive only)")


if __name__ == "__main__":
    main()
