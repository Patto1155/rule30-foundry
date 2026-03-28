# Agent Notes

Use this workspace as a disciplined scratchpad, not as an ad hoc notebook dump.

## Before Proposing Experiments

**Read these first — in order:**

1. `docs/idea-bank/theoretical-reframe-2026-03-28.md` — explains why experiments A–L hit a ceiling and what the right question hierarchy is. Prevents proposing more "looks random" tests when the system has already been characterised at that level.
2. `docs/theory/README.md` — synthesis of academic literature on Rule 30 (computational irreducibility, P-completeness, topological entropy, open problems). Read this before suggesting any theoretically-motivated experiment. *(Populating in progress — check if it exists.)*
3. `docs/experiment-logs/README.md` — naming convention (letters A–P so far; next is Q)

**Key instinct:** "Does this experiment answer a question that theory says is answerable? Or is it another 'looks random' test?" If the latter, step back and consult the theory docs.

**Current experiment frontier:** M (causal sensitivity), N (column MI), O (2D fractal), P (invariant measure) — all planned, none run. See their log stubs.

## Defaults

- Put concise problem framing in `docs/problem-statements/`
- Put speculative approaches in `docs/idea-bank/`
- Put experiment results in `docs/experiment-logs/`
- Put citations or source links in `docs/references/`
- Put reusable markdown templates in `docs/templates/`

## Experiment Naming

Experiments are lettered sequentially: A, B, C... currently at **P**.

- Log files: `docs/experiment-logs/{LETTER}_{slug}.md`
- Scripts: `experiments/{slug}.py`
- Next experiment is **Q**

## Logging Standard

For each experiment log, capture:

- date
- goal
- setup
- result
- interpretation
- next step

## Good Behavior

- Prefer short, atomic notes over long essays.
- Keep each experiment reproducible.
- If an idea fails, write down why it failed.
- If a result is ambiguous, record what would disambiguate it.
