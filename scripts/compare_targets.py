"""Compare our evaluated runs with the published numbers in baselines/targets.yaml.

    python scripts/compare_targets.py --runs runs_official --table finetune_table24 \
        --map ramp_official=ramp_l1.5 eat_official=eat max_official=max pretr_linf=pretr_linf

Verdict per metric: OK |delta| <= 1.0 pp, CLOSE <= 2.0 pp, OFF otherwise.
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml

METRICS = ["clean", "linf", "l2", "l1", "union"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_official")
    ap.add_argument("--tag", default="autoattack")
    ap.add_argument("--targets", default=os.path.join(os.path.dirname(__file__), "..", "baselines", "targets.yaml"))
    ap.add_argument("--table", default="finetune_table24")
    ap.add_argument("--map", nargs="+", required=True, help="our_name=target_key")
    a = ap.parse_args()
    table = yaml.safe_load(open(a.targets))[a.table]

    rows = {}
    for f in glob.glob(os.path.join(a.runs, "*", f"eval_{a.tag}.json")):
        r = json.load(open(f))
        rows.setdefault(r["name"], []).append({"clean": r["clean"], **r["robust"], "union": r["union"]})

    out = ["| run | n | metric | ours | paper | Δ | verdict |", "|---|---|---|---|---|---|---|"]
    worst = "OK"
    for pair in a.map:
        ours, key = pair.split("=")
        if ours not in rows:
            out.append(f"| {ours} | 0 | – | not evaluated | | | |")
            worst = "NOT EVALUATED"
            continue
        tgt = table[key]
        for m in METRICS:
            if tgt.get(m) is None:
                continue
            v = np.array([row[m] for row in rows[ours]]) * 100
            d = v.mean() - tgt[m]
            verdict = "OK" if abs(d) <= 1.0 else "CLOSE" if abs(d) <= 2.0 else "OFF"
            if worst != "NOT EVALUATED" and (verdict == "OFF" or (verdict == "CLOSE" and worst == "OK")):
                worst = verdict
            sd = f" ± {v.std(ddof=1):.1f}" if len(v) > 1 else ""
            out.append(f"| {ours} | {len(v)} | {m} | {v.mean():.1f}{sd} | {tgt[m]} | {d:+.1f} | {verdict} |")
    print("\n".join(out))
    print(f"\noverall: {worst}")
    return 0 if worst in ("OK", "CLOSE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
