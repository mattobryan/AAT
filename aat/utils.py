import copy
import json
import os
import random
import time
from fractions import Fraction

import numpy as np
import torch
import yaml


def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_fractions(obj):
    """Allow '8/255' style epsilons in YAML."""
    if isinstance(obj, dict):
        return {k: _parse_fractions(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_parse_fractions(v) for v in obj]
    if isinstance(obj, str) and "/" in obj:
        try:
            return float(Fraction(obj))
        except ValueError:
            return obj
    return obj


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("base"))
        cfg = _deep_merge(load_config(base_path), cfg)
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return _parse_fractions(cfg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dir(cfg):
    d = os.path.join(cfg.get("out_dir", "runs"), f"{cfg['name']}_s{cfg.get('seed', 0)}")
    os.makedirs(d, exist_ok=True)
    return d


class JsonlLogger:
    def __init__(self, path):
        self.path = path

    def log(self, record):
        record = {"time": time.time(), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=_to_py) + "\n")


def _to_py(o):
    if torch.is_tensor(o):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
