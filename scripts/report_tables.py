"""Emit the LaTeX tables of docs/report/report.tex from baselines/results/*.json (2-seed means).
    python scripts/report_tables.py > docs/report/tables.tex"""
import json
R = "baselines/results/"
runs = [("RAMP", "ramp_scratch_l5"), ("aat\\_rel", "aatrel_l5"), ("aat\\_flip", "aatflip_l5"),
        ("cls\\_ctrl", "clsctrl_l5"), ("cls\\_fb", "clsfb_l5")]
names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
D = {k: [json.load(open(f"{R}{v}_s{s}_eval_autoattack.json")) for s in (0, 1)] for k, v in runs}
m = lambda k, f: 100 * sum(f(d) for d in D[k]) / 2

print("% overall")
print("\\begin{tabular}{lrrrrrr}\\toprule\nRun & Clean & $\\ell_\\infty$ & $\\ell_2$ & $\\ell_1$ & Union & Worst class \\\\\\midrule")
for k, _ in runs:
    print(f"{k} & {m(k,lambda d:d['clean']):.2f} & {m(k,lambda d:d['robust']['linf']):.2f} & {m(k,lambda d:d['robust']['l2']):.2f} & "
          f"{m(k,lambda d:d['robust']['l1']):.2f} & {m(k,lambda d:d['union']):.2f} & {m(k,lambda d:d['union_worst_class']):.2f} \\\\")
print("\\bottomrule\\end{tabular}\n")
print("% per-class union")
print("\\begin{tabular}{l" + "r" * len(runs) + "}\\toprule\nClass & " + " & ".join(k for k, _ in runs) + " \\\\\\midrule")
for i, n in enumerate(names):
    print(n + " & " + " & ".join(f"{m(k,lambda d:d['union_per_class'][i]):.1f}" for k, _ in runs) + " \\\\")
print("\\bottomrule\\end{tabular}\n")
print("% unseen")
keys = ["linf@0.04706", "linf@0.06275", "l2@1", "l2@1.5", "l1@18", "l1@24"]
hdr = ["$\\ell_\\infty$ 12/255", "$\\ell_\\infty$ 16/255", "$\\ell_2$ 1.0", "$\\ell_2$ 1.5", "$\\ell_1$ 18", "$\\ell_1$ 24"]
print("\\begin{tabular}{l" + "r" * len(keys) + "}\\toprule\nRun & " + " & ".join(hdr) + " \\\\\\midrule")
for k, _ in runs:
    print(k + " & " + " & ".join(f"{m(k,lambda d:d['unseen'][u]):.1f}" for u in keys) + " \\\\")
print("\\bottomrule\\end{tabular}")
