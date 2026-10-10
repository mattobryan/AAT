"""Evaluate finished runs that have no eval_autoattack.json yet, one run per GPU, and persist each result.

    python scripts/eval_pending.py --runs v3a_max_s0 aat_full_s0 ...

Sits after training in the Kaggle notebook. For every run it restores final.pt and any earlier result from the
Hugging Face repo, skips runs that already have a result, runs `python -m aat.evaluate --run runs/<run>` (fixed-epsilon
AutoAttack plus the unseen grid, defaults unchanged) on a free GPU, and pushes eval_autoattack.json right after each
run finishes, so a session that dies half way loses at most the run in flight. aat/evaluate.py is not modified.
"""
import argparse
import concurrent.futures as cf
import os
import queue
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aat import hub  # noqa: E402  (stdlib-only module)


def restore(run, fname):
    """True if runs/<run>/<fname> exists locally or could be pulled from the Hub."""
    path = os.path.join(ROOT, "runs", run, fname)
    return os.path.exists(path) or hub.pull(f"aat/{run}/{fname}", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ngpu", type=int, default=None)
    ap.add_argument("evaluate_args", nargs=argparse.REMAINDER, help="extra args for aat.evaluate, after --")
    a = ap.parse_args()

    todo, missing = [], []
    for r in a.runs:
        if restore(r, "eval_autoattack.json"):
            print(f"[skip] {r}: result exists", flush=True)
        elif restore(r, "final.pt"):
            restore(r, "config.yaml")
            todo.append(r)
        else:
            missing.append(r)
    print(f"{len(todo)} to evaluate, {len(missing)} not trained yet: {missing}", flush=True)

    if a.ngpu is None:
        import torch
        a.ngpu = max(1, torch.cuda.device_count())
    gpus = queue.Queue()
    for g in range(a.ngpu):
        gpus.put(g)
    extra = [x for x in a.evaluate_args if x != "--"]
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)

    def one(run):
        g = gpus.get()
        try:
            log = os.path.join(ROOT, "logs", f"eval_{run}.txt")
            with open(log, "w") as f:
                rc = subprocess.call([sys.executable, "-m", "aat.evaluate", "--run", os.path.join("runs", run), *extra],
                                     cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)},
                                     stdout=f, stderr=subprocess.STDOUT)
            res = os.path.join(ROOT, "runs", run, "eval_autoattack.json")
            ok = rc == 0 and os.path.exists(res)
            if ok:
                hub.push(res, f"aat/{run}/eval_autoattack.json")
                hub.push(log, f"logs/eval_{run}.txt")
            print(f"[{'ok' if ok else 'FAILED'}] {run} on GPU {g} (exit {rc}), log {log}", flush=True)
            return ok
        finally:
            gpus.put(g)

    with cf.ThreadPoolExecutor(a.ngpu) as ex:
        results = list(ex.map(one, todo))
    if not all(results):
        sys.exit(1)
    if missing:
        print(f"note: {len(missing)} planned runs have no checkpoint yet", flush=True)


if __name__ == "__main__":
    main()
