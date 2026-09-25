# CLAUDE.md: AAT repository

Context: this repo rebuilds an MSc thesis on Adaptive Adversarial Training (multi-norm adversarial
robustness, CIFAR-10, PreActResNet-18) to run on Kaggle within one week. Read `README.md` for the
research question and protocol, and `docs/thesis_audit.md` for known flaws in the thesis. Never
reintroduce those flaws. In particular, evaluation always uses fixed ε and never the adaptive ε.

## Token budget: route work to the cheapest model that can do it well

The main session is the **orchestrator**. It plans, makes decisions, and talks to the user. Delegate
bounded work to the subagents in `.claude/agents/`. Each one pins its model, so cost follows the task:

| Tier | Agent | Model | Use for | Never use for |
|---|---|---|---|---|
| 1 (cheap) | `scout` | haiku | Finding files and symbols, reading/grepping `log.jsonl` and `eval_*.json`, summarizing run status, checking configs, formatting tables | Judging whether a result is valid; editing code |
| 2 (default) | `builder` | sonnet | Scoped code edits with a clear spec, new configs, tests, Kaggle notebook cells, running pytest, fixing failing tests, plots/aggregation | Changing attack or evaluation semantics without a tier-3 sign-off |
| 3 (expensive) | `reviewer` | opus | Attack/eval correctness, threat-model or protocol changes, interpreting results against the hypotheses (H1–H4), statistical claims, thesis/paper text that makes claims, audit updates | Mechanical edits, log reading, anything tier 1–2 can do |

Routing rules:
1. **Default down.** Start at the lowest tier that can finish the task. Escalate only when that
   tier reports it is blocked, or when the task touches a tier-3 area listed above.
2. **Mandatory tier-3 gate** for any change to `aat/attacks.py`, `aat/evaluate.py`, the `eval:` block of
   `configs/base.yaml`, or any sentence stating a quantitative result. The `builder` may draft it,
   but the `reviewer` must check it before it is committed.
3. **Don't spawn for trivia.** A single grep, a one-line edit, or a question answered from context is
   done inline by the orchestrator. A spawn re-reads context and costs more than it saves.
4. **Give narrow briefs.** Every delegation states: goal, files, done-criteria, and the output format
   (e.g. "return a ≤15-line summary, no file dumps"). Ask subagents for conclusions, not transcripts.
5. **Read logs cheaply.** Use `tail -n1`, `jq`, or `python -c` to pull fields out of JSONL/JSON. Never cat whole
   logs or checkpoints into context. `scripts/aggregate.py` is the source of truth for results tables.
6. **Batch parallel work.** Independent tier-1/2 tasks, such as checking several run directories,
   go out as parallel subagents in one message.
7. **Reuse context.** Continue an existing subagent (SendMessage) instead of spawning a fresh one for a
   follow-up on the same task.

## Conventions

* Python ≥3.10, PyTorch ≥2.1. Pixels are in [0,1] and normalization lives inside the model.
* New methods go in as configs that inherit `base.yaml`. Add a code path only when a config can't express the method.
* `python -m pytest -q` must pass before any commit. `tests/test_core.py` runs in seconds; the smoke test takes a few minutes on CPU.
* Never commit `data/`, `runs/`, or `*.pt`.
* Results reported anywhere must come from `eval_autoattack.json`, never from `monitor_test` in the training logs.
