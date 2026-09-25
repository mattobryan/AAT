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


def main():
    repo, token = os.environ.get("HF_REPO"), os.environ.get("HF_TOKEN")
    if not repo or not token:
        print("monitor not configured: set HF_REPO and HF_TOKEN in the environment")
        return 0
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo)
    except Exception as e:
        print(f"cannot read {repo}: {type(e).__name__}: {e}")
        return 1
    now = dt.datetime.now(dt.timezone.utc)
    logs = [f for f in files if f.endswith(("log_train.txt", "log.jsonl"))]
    if not logs:
        print(f"{repo}: no runs have pushed yet")
        return 0
    info = {p.path: p for p in api.get_paths_info(repo, logs, expand=True)}

    rows = []
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
        if ev in files:
            r = json.load(open(hf_hub_download(repo, ev, token=token)))
            metric = f"AA: clean {r['clean']:.1%} union {r['union']:.1%} " + \
                     " ".join(f"{k} {v:.1%}" for k, v in r["robust"].items())
        rows.append(f"- {run}: {state} | epoch {done}/{total or '?'} | {mean_t / 60:.0f} min/epoch"
                    f" | last push {age_min:.0f} min ago" + (f" | ETA {eta_h:.1f} GPU-h" if eta_h else "")
                    + (f"\n    {metric}" if metric else ""))

    print(f"AAT status {now:%Y-%m-%d %H:%M} UTC ({repo})")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
