"""Fail fast: verify every import and setting the pipeline needs before any download or training.

    python scripts/check_env.py            # exit 1 if anything is missing
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "external", "ramp")

checks = []


def check(name, fn):
    try:
        detail = fn() or ""
        checks.append((name, True, str(detail)))
    except Exception as e:
        checks.append((name, False, f"{type(e).__name__}: {e}"))


def version(mod):
    return lambda: getattr(importlib.import_module(mod), "__version__", "ok")


for mod in ["torch", "torchvision", "numpy", "yaml", "huggingface_hub", "autoattack", "robustbench", "geotorch", "timm"]:
    check(f"import {mod}", version(mod))


def official_ramp():
    if not os.path.isdir(EXT):
        raise FileNotFoundError(f"{EXT} missing (run: bash scripts/ramp_official.sh install)")
    sys.path.insert(0, EXT)
    cwd = os.getcwd()
    os.chdir(EXT)
    try:
        for mod in ["RAMP", "eat_train", "MAX", "eval", "autopgd_train", "hub_sync"]:
            importlib.import_module(mod)
    finally:
        os.chdir(cwd)
    if not os.path.exists(os.path.join(EXT, "models", "pretr_Linf.pth")):
        raise FileNotFoundError("models/pretr_Linf.pth missing")
    if "[aat patch]" not in open(os.path.join(EXT, "RAMP.py")).read():
        raise RuntimeError("resume/HF patch not applied to RAMP.py")
    return "RAMP.py, eat_train.py, MAX.py, eval.py import; patch applied; pretr_Linf.pth present"


check("official RAMP code", official_ramp)
check("our package", lambda: [importlib.import_module(m) for m in ["aat.models", "aat.evaluate", "aat.train"]] and "aat imports")


def gpus():
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no GPU visible: set Accelerator to GPU T4 x2")
    return f"{n} x {torch.cuda.get_device_name(0)}" + ("  (only 1 GPU: seeds will not run in parallel)" if n == 1 else "")


check("GPU", gpus)


def hf():
    if not os.environ.get("HF_REPO"):
        raise RuntimeError("HF_REPO not set")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not set (Kaggle secret not attached?)")
    from huggingface_hub import HfApi
    user = HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
    return f"token OK for {user}, repo {os.environ['HF_REPO']}"


check("Hugging Face", hf)
check("ntfy topic", lambda: os.environ.get("NTFY_TOPIC") or "(not set: no phone notifications)")

width = max(len(n) for n, _, _ in checks)
for name, ok, detail in checks:
    print(f"{'OK  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")
failed = [n for n, ok, _ in checks if not ok]
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
sys.exit(1 if failed else 0)
