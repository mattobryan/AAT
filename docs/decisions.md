# Decision log

Each entry: what was decided, what was rejected, and what it costs.
Rewrite entries in your own words when you review them; this file is your defence of the design.

## D1. Reframe the question around deployment context
Chosen: test transfer across attack families and include a Swahili/Sheng translation family,
so the result speaks to whether developer evaluations hold in Kenya.
Rejected: a pure replication of the vision multi norm result on English attacks only.
Why: cross family failures are already partly shown (Wei et al. 2023; Andriushchenko and
Flammarion 2024; Scale AI multi turn study 2024). The language axis is the open gap.
Cost: Sheng has no reliable machine translation, so translations need manual checking.

## D2. Held out means held out twice
Chosen: test prompts differ from training prompts in both the harmful behaviour and the
attack template.
Rejected: a random split of one prompt pool.
Why: a random split lets the model pass by memorising templates, which inflates the
diagonal of the transfer matrix and confirms H1 for the wrong reason.
Cost: fewer usable prompts per split.

## D3. Over refusal is the main cost metric
Chosen: XSTest and OR-Bench for over refusal; MMLU only as a secondary check.
Rejected: MMLU as the main utility measure.
Why: safety training rarely damages knowledge; it makes models refuse harmless requests.
Cost: two extra benchmarks to run.

## D4. Decision threshold fixed before running
Chosen: transfer ratio tau(X to Y) = ASR drop on Y / ASR drop on X.
H1 if tau < 0.3 off the diagonal, H0 if tau > 0.7, bootstrap confidence intervals over 3 seeds.
Rejected: judging "little" versus "comparable" transfer after seeing results.
Why: a threshold set after the data can be bent to fit any outcome.
Cost: results between 0.3 and 0.7 are reported as inconclusive.
