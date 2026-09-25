"""Run one notebook stage and tell your phone how it went.

    python scripts/stage.py NAME "shell command"

* streams the command's output live (so it shows in the Kaggle log) and saves logs/NAME.txt
* push notification via ntfy (env NTFY_TOPIC) when the stage starts and when it ends:
  OK / FAILED (with the last lines of output) / PAUSED (time budget reached, re-run to resume)
* uploads the log to the Hugging Face repo (logs/NAME.txt) and records the outcome in
  logs/stages.json, which the monitor and dashboard read
* exits with the command's exit code, so the notebook can stop at the first failure
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aat import hub  # noqa: E402  (stdlib-only module)


def notify(title, body, tags, priority="default"):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode("utf-8")[:3800],
                                     headers={"Title": title, "Tags": tags, "Priority": priority})
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print(f"[notify] could not send: {e}", flush=True)


def record(entry):
    """Append this stage's outcome to logs/stages.json on the Hub (last 50 kept)."""
    path = os.path.join(ROOT, "logs", "stages.json")
    events = []
    if hub.pull("logs/stages.json", path):
        try:
            events = json.load(open(path))
        except Exception:
            events = []
    events = (events + [entry])[-50:]
    json.dump(events, open(path, "w"), indent=1)
    hub.push(path, "logs/stages.json")


def main():
    name, cmd = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log_path = os.path.join(ROOT, "logs", f"{name}.txt")
    start = time.time()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    notify(f"AAT: {name} started", f"{stamp}\n{cmd}", "arrow_forward", "low")
    print(f"=== stage {name} started {stamp} ===", flush=True)

    tail, paused = [], False
    with open(log_path, "w") as log:
        p = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            clean = line.rstrip()
            if clean and "it/s]" not in clean and "s/it]" not in clean:
                tail = (tail + [clean[-200:]])[-15:]
            if "time budget reached" in line:
                paused = True
        code = p.wait()

    minutes = (time.time() - start) / 60
    if code != 0:
        state, title, tags, prio = "failed", f"AAT: {name} FAILED", "x", "high"
    elif paused:
        state, title, tags, prio = "paused", f"AAT: {name} PAUSED - re-run the notebook", "pause_button", "default"
    else:
        state, title, tags, prio = "ok", f"AAT: {name} OK", "white_check_mark", "default"
    body = f"{minutes:.0f} min, exit code {code}\nlog: {hub.get_repo() or 'local'} logs/{name}.txt\n\n" + "\n".join(tail[-12:])
    print(f"=== stage {name} {state.upper()} after {minutes:.0f} min (exit {code}) ===", flush=True)
    hub.push(log_path, f"logs/{name}.txt")
    record({"stage": name, "state": state, "started": stamp, "minutes": round(minutes, 1), "exit": code,
            "tail": tail[-8:]})
    notify(title, body, tags, prio)
    sys.exit(code)


if __name__ == "__main__":
    main()
