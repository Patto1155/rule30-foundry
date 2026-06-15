# Agent Quickstart

This repository is a disciplined research scratchpad for Rule 30 experiments.
Before proposing or running new experiments, read the root `AGENTS.md` and the
theory/logging notes it names.

## First Reads

Read these in order:

1. `AGENTS.md`
2. `docs/idea-bank/theoretical-reframe-2026-03-28.md`
3. `docs/experiment-logs/README.md`

If a task proposes theory-motivated work, also check whether `docs/theory/`
exists in the checkout before adding new theory notes.

## Current Tool Map

- `gpu/rule30_sim.py`: GPU center-column simulation entry point from the README.
- `experiments/rule30_open_utils.py`: shared open-boundary packed-bit helpers and
  naive-reference verification routines for follow-up experiments.
- `experiments/causal_sensitivity.py`: canonical M causal-sensitivity experiment.
- `experiments/column_mi.py`: column mutual information / transfer entropy run.
- `experiments/fractal_dimension.py`: O 2D spacetime complexity probe.
- `experiments/invariant_measure.py`: P finite-order invariant-measure probe.
- `run_all.py` and `run_after_sim.py`: orchestration scripts; inspect paths and
  target artifacts before running them.

Prefer these existing tools over ad hoc scripts. If a script has `--test`, use
that smoke path before any expensive GPU run.

## Triage Rules

- Prefer small reproducibility or correctness fixes over new experiments.
- Do not add another "looks random" test unless the theory notes identify the
  question as useful and answerable.
- Do not consume canonical frontier letters for auxiliary or cleanup work.
- Keep outputs reproducible: write logs under `docs/experiment-logs/`, data under
  `data/`, and plots under `docs/plots/`.

## Verification Checklist

Before trusting packed-bit or GPU results:

- Compare against a naive cell-by-cell implementation on a small case.
- Include a perturbation case crossing at least one 64-bit word boundary.
- Confirm bit ordering and geometric left/right with a tiny explicit case.
- Treat `first_divergence < distance` as an impossible result and fail hard.
- Report unreached distances as right-censored, not as permanent non-arrivals.

Before committing:

- Run `python -m py_compile` for changed Python modules.
- Run the relevant `--test` experiment or utility verification when cheap.
- Stage explicit files only; do not use `git add -A` in mixed worktrees.
