"""Progress report for every run that syncs to the Hugging Face repo, readable from anywhere.

    HF_TOKEN=... HF_REPO=user/aat-checkpoints python scripts/status.py

Reads only what the training scripts push after each epoch (official RAMP runs:
ramp_official/<run>/log_train.txt; AAT runs: aat/<run>/log.jsonl; evaluations: eval_autoattack.json).
Flags a run as STALLED when nothing was pushed for 2x its mean epoch time + 20 min, which
usually means the Kaggle session ended and the notebook needs to be re-run.
"""
import datetime as dt
import json
import os
import re
import sys

EPOCHS = {"ramp_scratch": 80}  # everything else defaults to the 3-epoch fine-tuning protocol


PCT = re.compile(r"(Linf|L2|L1|clean|union) ([\d.]+)%")
KEYS = {"Linf": "linf", "L2": "l2", "L1": "l1", "clean": "clean", "union": "union"}


def _curve_ramp(text):
    """Per-epoch loss and (every eval_freq epochs) test accuracies from the official log_train.txt."""
    loss, acc = [], []
    for line in text.splitlines():
        m = re.search(r"\[epoch\] (\d+) \[time\] ([\d.]+) s \[train\] loss ([\d.]+)", line)
        if not m:
            continue
        ep = int(m.group(1))
        loss.append({"epoch": ep, "loss": float(m.group(3))})
        if "[eval test]" in line:
            test = line.split("[eval test]")[1]
            acc.append({"epoch": ep, **{KEYS[k]: float(v) for k, v in PCT.findall(test)}})
    return loss, acc


def _emit(report, as_json):
    if as_json:
        print(json.dumps(report))
    else:
        print(report.get("message") or f"AAT status {report['generated_at']} UTC ({report['repo']})")
        print("\n".join(r["line"] for r in report.get("runs", [])))


def main():
    as_json = "--json" in sys.argv
    repo, token = os.environ.get("HF_REPO"), os.environ.get("HF_TOKEN")
    now = dt.datetime.now(dt.timezone.utc)
    report = {"generated_at": now.strftime("%Y-%m-%d %H:%M"), "repo": repo, "runs": []}
    if not repo or not token:
        report["message"] = "monitor not configured: set HF_REPO and HF_TOKEN in the environment"
        _emit(report, as_json)
        return 0
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo)
    except Exception as e:
        report["message"] = f"cannot read {repo}: {type(e).__name__}"
        _emit(report, as_json)
        return 1
    logs = [f for f in files if f.endswith(("log_train.txt", "log.jsonl"))]
    if not logs:
        report["message"] = f"{repo}: no runs have pushed yet"
        _emit(report, as_json)
        return 0
    info = {p.path: p for p in api.get_paths_info(repo, logs, expand=True)}

    for f in sorted(logs):
        run = f.split("/")[-2]
        text = open(hf_hub_download(repo, f, token=token)).read()
        age_min = (now - info[f].last_commit.date).total_seconds() / 60 if info[f].last_commit else float("nan")
        if f.endswith("log_train.txt"):  # official RAMP log
            ep = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"\[epoch\] (\d+) \[time\] ([\d.]+) s", text)]
            done = ep[-1][0] if ep else 0
            times = [t for _, t in ep]
            total = next((v for k, v in EPOCHS.items() if run.startswith(k)), 3)
            last_eval = re.findall(r"\[eval test\]([^\n]*)", text)
            metric = last_eval[-1].strip() if last_eval else ""
            loss_curve, acc_curve = _curve_ramp(text)
        else:  # our trainer
            recs = [json.loads(l) for l in text.splitlines() if l.strip()]
            done = recs[-1]["epoch"] + 1 if recs else 0
            times = [r["epoch_time_s"] for r in recs]
            cfg_file = f.replace("log.jsonl", "config.yaml")
            total = None
            if cfg_file in files:
                import yaml
                total = yaml.safe_load(open(hf_hub_download(repo, cfg_file, token=token)))["train"]["epochs"]
            mon = [r["monitor_test"] for r in recs if "monitor_test" in r]
            metric = " ".join(f"{k} {v:.1%}" for k, v in mon[-1].items()) if mon else ""
            loss_curve = [{"epoch": r["epoch"] + 1, "loss": r["train_loss"]} for r in recs]
            acc_curve = [{"epoch": r["epoch"] + 1, **{k: v * 100 for k, v in r["monitor_test"].items()}}
                         for r in recs if "monitor_test" in r]
        mean_t = sum(times) / len(times) if times else 0
        finished = total is not None and done >= total
        if finished:
            state = "DONE"
        elif mean_t and age_min > 2 * mean_t / 60 + 20:
            state = "STALLED - re-run the notebook"
        else:
            state = "running"
        eta_h = (total - done) * mean_t / 3600 if (total and mean_t and not finished) else 0
        ev = f.rsplit("/", 1)[0] + "/eval_autoattack.json"
        final = None
        if ev in files:
            r = json.load(open(hf_hub_download(repo, ev, token=token)))
            final = {"clean": r["clean"] * 100, "union": r["union"] * 100, **{k: v * 100 for k, v in r["robust"].items()}}
            metric = f"AA: clean {r['clean']:.1%} union {r['union']:.1%} " + \
                     " ".join(f"{k} {v:.1%}" for k, v in r["robust"].items())
        line = (f"- {run}: {state} | epoch {done}/{total or '?'} | {mean_t / 60:.0f} min/epoch"
                f" | last push {age_min:.0f} min ago" + (f" | ETA {eta_h:.1f} GPU-h" if eta_h else "")
                + (f"\n    {metric}" if metric else ""))
        report["runs"].append({"run": run, "state": state.split(" ")[0], "done": done, "total": total,
                               "min_per_epoch": round(mean_t / 60, 1), "last_push_min": round(age_min),
                               "eta_h": round(eta_h, 1), "loss": loss_curve, "acc": acc_curve,
                               "final": final, "line": line})
    _emit(report, as_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
