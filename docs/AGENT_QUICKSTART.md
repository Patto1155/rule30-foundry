# Agent Quickstart

This repo is a research foundry, not a notebook dump. Your job as an agent is to
preserve reproducibility, avoid rerunning expensive work blindly, and move claims
toward checkable artifacts.

## First 10 Minutes

1. Check state before touching files:

   ```bash
   git status --short --branch
   git branch -vv --all
   ```

2. Read these files in order:

   - **`docs/theory/README.md` — the theory gate. Read this first.** It lists what
     is already proved (do not re-measure it), which routes are closed, the
     single-seed rule for prize-facing work, and the counting bound that decides
     whether a planned search can produce information at all. A repo certificate
     was retracted in 2026-08 because this file did not exist yet.
   - `docs/WORKFLOW.md` for the operating loop and GPU/coarse-grain discipline.
   - `AGENTS.md` for naming, logging, and guardrails.
   - `docs/CLAIM_LEDGER.md` for what is observation vs certificate.
   - The latest relevant file in `docs/experiment-logs/`.

   Before proposing any "search class `M`, find no fit" experiment, run
   `python experiments/counting_bound.py --pretty` and check `log2|M| >= n`.
   If not, the negative is guaranteed and the run is worthless.

3. Prefer existing tools over one-off scripts. If you need coarse-grain or CA
   exploration, start with `ca_lab.py`.

4. Treat the worktree as mixed by default. Stage explicit files only. There are
   often untracked experiment outputs in `data/`, `docs/experiment-logs/`, and
   `experiments/`.

## Tool Map

| Need | Use | Notes |
|---|---|---|
| Rule 30 trusted simulation | `experiments/rule30_open_utils.py` | Reference path for packed Rule 30 work. |
| Fast Rule 30 GPU kernel | `gpu/rule30_fast.py` | Read `docs/GPU_KERNELS.md` before editing halo or `K`. |
| Any elementary CA rule | `experiments/eca_sim.py` | Verified arbitrary Wolfram-rule simulator. |
| Coarse-grain sweeps | `python ca_lab.py sweep ...` | JSON to stdout; use `--pretty` for stderr table. |
| Single b=2 closure | `python ca_lab.py closure ...` | Exhaustive over all 2^16 projections. |
| b>=3 projection search | `python ca_lab.py search ...` | Heuristic search; full re-score required. |
| b=3 verdict reproduction | `experiments/coarse_grain_bk_verdict.py` | `--test` writes test JSON by default; full run is expensive. |
| Batch old experiments | `run_all.py --test` | Smoke-test runner; now self-relative to this checkout. |
| Prize center-column artifacts | `prize_lab.py` | Exact prefix, GF(2) recurrences, finite kernel lower bounds, DFAO SAT encodings. |
| Counting bound for a search class | `experiments/counting_bound.py` | Run BEFORE any "no fit found" experiment. `--verdict S:N` evaluates a specific claim. |
| Settled-wedge decomposition | `experiments/wedge_profile.py` | Single-seed cone structure; settling law, entropy split, cap invariance. |
| Diagonal recursion + period-16 | `experiments/diagonal_recursion.py` | Bit-exact identity check, period histogram, O(1) pattern map. |

## Prize-Facing Filter

Before proposing a new experiment, answer these:

- Does it target the single-black-cell center column, one of the prize objects,
  or a named shortcut class?
- Does it emit a candidate object or a checkable obstruction, not just a plot?
- What is the null, control, or impossible-output invariant?
- What would promote the result from observation to certificate?

High-upside current direction: extend `prize_lab.py` exact center-column work and
shortcut searches (`n -> center_bit(n)`) toward finite automata, transducers,
recurrences, divide-and-conquer summaries, SAT encodings, and
machine-checkable counterexamples.

## Claim Levels

Use these labels in logs and PRs:

- `Observation`: one run, one scale, or exploratory score.
- `Robust observation`: controls, nulls, multiple scales/seeds, held-out checks.
- `Certificate`: finite artifact that another agent can verify mechanically.
- `Theorem`: proved outright; no experiment needed. Do not re-measure these.
- `Proof candidate`: theorem-shaped argument with checkable lemmas or code.
- `Retracted`: was recorded too high and has been withdrawn, with the reason.

Most empirical Rule 30 work starts at observation. Prize-relevant work should aim
for certificate or proof candidate.

## Common Traps

- Do not call another "looks random" experiment prize progress unless it narrows
  a named hypothesis or emits a reusable artifact.
- Do not record a "no fit in class `M`" negative without the counting check. If
  the matched random control returns the *same* negative, that is a red flag the
  experiment is vacuous — not a reassuring baseline. This is exactly how the
  1-5 state / 128-bit DFAO certificate got recorded and later retracted.
- Do not report an ensemble/random-IC quantity as prize progress. All three
  prizes concern the single deterministic seed; see `docs/theory/README.md` §0.
- Do not use linear rules 90/150 as b>=3 coarse-grain controls. Use shift rules
  170/240.
- Do not report searched closure without searching the null at equal budget and
  re-scoring the winning projection on the full transition set.
- Do not overwrite canonical JSON with smoke-test output. Use `--test` defaults
  or pass `--out`.
- Do not use `D:\APATPROJECTS\rule30-research` as the active clone. This checkout
  is the canonical repo.
