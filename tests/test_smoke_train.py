"""End-to-end CPU smoke test of train + evaluate on FakeData (no downloads)."""
import json
import os

import torch
import torchvision
import torchvision.transforms as T

import aat.data as data
from aat.evaluate import evaluate
from aat.train import train
from aat.utils import load_config

CFG = os.path.join(os.path.dirname(__file__), "..", "configs")


def _fake(root, train=True, download=False, transform=None):
    return torchvision.datasets.FakeData(size=1200 if train else 200, image_size=(3, 32, 32), num_classes=10,
                                         transform=transform or T.ToTensor(), random_offset=0 if train else 7)


def test_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_dataset_cls", lambda name: _fake)
    for name in ["aat_full", "ramp", "v1b_sample"]:
        cfg = load_config(os.path.join(CFG, f"{name}.yaml"), [
            f"out_dir={tmp_path}", "init_from=null", "train.epochs=2", "train.batch_size=32", "limit_batches=2",
            "data.feedback_size=100", "data.num_workers=0", "monitor.n=64", "monitor.every=1", "feedback.steps=2",
            "eval.n=64", "eval.unseen={linf: [0.0627]}"])
        out = train(cfg)
        res = evaluate(out, backend="pgd", n=64, pgd_steps=3)
        assert 0 <= res["union"] <= min(res["robust"].values()) <= res["clean"] <= 1
        recs = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
        assert len(recs) == 2 and "eps" in recs[-1]
