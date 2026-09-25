# AAT: does LLM safety training transfer across attack families and languages?

## Purpose
Research code for one question: if a model is safety trained against one jailbreak family,
does that protection carry over to other families (paraphrase, Swahili/Sheng translation,
multi turn)? The result feeds a policy argument: developer safety evaluations may not hold
in countries like Kenya that deploy models they did not build.

The owner must be able to explain every line of this repo. Code exists to be understood first.

## Writing rules (prose, comments, commit messages, docs)
- Never use em dashes or en dashes. Use a comma, a colon, a full stop, or parentheses.
- No decorative hyphens in prose. Hyphens only where the term requires one or inside code.
- Plain sentences. No filler, no restating, no closing summaries.

## Code rules
- Minimal code. Fewest files, fewest abstractions, no framework unless it removes real work.
- Every module starts with a short docstring: what it does and where it sits in the pipeline.
- Every function has a docstring: inputs, output, and why it exists.
- Comments explain why, not what. Skip comments that repeat the code.
- Prefer plain functions over classes. Prefer the standard library over new dependencies.
- Every new dependency is justified in docs/decisions.md.

## Workflow
- Before writing a module, state in two or three sentences what it will do and why.
- After writing it, explain how it fits the pipeline so the owner can review it.
- Every design choice with a real alternative goes in docs/decisions.md with the option
  chosen, the options rejected, and the cost of the choice.
- This container has no GPU. Write and test code here on CPU with a tiny model;
  run training and full evaluation on a rented GPU.
