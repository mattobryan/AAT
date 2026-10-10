#!/usr/bin/env bash
# GPU smoke test and timing for the Kaggle notebook (about 15 min). HF_REPO is emptied so nothing reaches the Hub.
# Real epoch time is about 7.7 x the epoch_time_s printed here (limit_batches=50 of ~383 batches).
set -euo pipefail
export HF_REPO=
python -m pytest -q tests/test_core.py
python -m aat.train --config configs/v3a_max.yaml --set name=timing_max init_from=null train.epochs=1 limit_batches=50 monitor.every=0
python -m aat.train --config configs/aat_full.yaml --set name=timing_full init_from=null train.epochs=2 limit_batches=50 monitor.every=0
tail -qn1 runs/timing_max_s0/log.jsonl runs/timing_full_s0/log.jsonl | grep -o '"epoch_time_s": [0-9.]*\|"feedback_time_s": [0-9.]*'
python -m aat.evaluate --run runs/timing_full_s0 --n 100 --bs 100
python -c "import json;r=json.load(open('runs/timing_full_s0/eval_autoattack.json'));print({k:r[k] for k in ('clean','robust','union')}, list(r.get('unseen',{}))[:3])"
rm -rf runs/timing_max_s0 runs/timing_full_s0   # keep the throwaway n=100 result out of aggregate.py
echo SMOKE OK
