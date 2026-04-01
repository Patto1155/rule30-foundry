# Agent Notes

Use this workspace as a disciplined scratchpad, not as an ad hoc notebook dump.

## Before Proposing Experiments

Read these first, in order:

1. `docs/idea-bank/theoretical-reframe-2026-03-28.md`
   Why A-L hit a ceiling and why more "looks random" tests are low value.
2. `docs/theory/README.md`
   Read this before proposing theory-motivated work. Check whether it exists.
3. `docs/experiment-logs/README.md`
   Current log and naming conventions.

Key instinct: "Does this experiment answer a question that theory says is answerable, or is it just another random-looking test?"

If it is the latter, step back and consult the theory docs before proposing anything.

## Current Frontier

Canonical frontier:

- `M`: causal sensitivity / dynamical geometry
- `N`: column mutual information / transfer entropy
- `O`: 2D fractal or spacetime-complexity analysis
- `P`: invariant measure / entropy-rate work

Current state as of `2026-04-01`:

- `M` has been run with a corrected packed-bit implementation. See `docs/experiment-logs/M_causal_sensitivity.md`.
- Column MI / TE has also been run via `experiments/column_mi.py`.
- `O` (2D fractal) and `P` (invariant measure) are still open.
- Auxiliary cleanup experiments `fft_autocorr.py` and `compress_probe.py` have also been run.

Important naming caveat:

- A prior session introduced filename drift: `docs/experiment-logs/N_compress_probe.md` and `docs/experiment-logs/O_column_mi.md` do not match the original theory-driven `M/N/O/P` mapping.
- Do not assign new experiment letters casually.
- Do not assume "next experiment is Q" unless the canonical mapping has been reconciled first.
- If a session adds an auxiliary experiment, prefer a dated filename or an `aux_` slug instead of consuming a frontier letter.

## Defaults

- Put concise problem framing in `docs/problem-statements/`
- Put speculative approaches in `docs/idea-bank/`
- Put experiment results in `docs/experiment-logs/`
- Put citations or source links in `docs/references/`
- Put reusable markdown templates in `docs/templates/`

## Experiment Naming

Canonical frontier experiments should use:

- Log files: `docs/experiment-logs/{LETTER}_{slug}.md`
- Scripts: `experiments/{slug}.py`

Auxiliary runs, cleanup experiments, postmortems, and implementation notes may use dated filenames:

- `docs/experiment-logs/YYYY-MM-DD-short-slug.md`

Do not reuse a canonical frontier letter for an auxiliary experiment.

## Logging Standard

For each experiment log, capture:

- date
- goal
- setup
- result
- interpretation
- next step

If the run depended on GPU code or packed-bit kernels, also record:

- what was verified against a naive reference
- what sanity checks were used to reject impossible outputs
- whether the result is a direct consequence of the local rule or a higher-level empirical finding

## Implementation Guardrails

For bit-packed Rule 30 code, treat these as mandatory:

- Verify packed CPU/GPU kernels against a naive cell-by-cell implementation before trusting any result.
- Do not rely only on the first 20 center bits. Also test perturbation propagation across at least one 64-bit word boundary.
- Never infer geometric left/right from variable names alone. Confirm bit ordering and neighbor direction with a tiny naive test.
- In a radius-1 cellular automaton, `first_divergence < distance` is impossible. Treat that as a hard failure.
- "Never reached within N steps" is right-censored. Do not report it as "never" without qualification.
- If a metric is near zero, define a noise floor or baseline before calling it asymmetric or structured.
- One-step TE at distance 1 may just restate the update rule. Do not oversell it as a novel discovery without literature support.

## Good Behavior

- Prefer short, atomic notes over long essays.
- Keep each experiment reproducible.
- If an idea fails, write down why it failed.
- If a result is ambiguous, record what would disambiguate it.
- Clean up transient progress logs or ignore them in git.
