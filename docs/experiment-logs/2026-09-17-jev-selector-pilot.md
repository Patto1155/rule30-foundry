# Jev selector pilot — 2026-09-17

## Goal

Build the smallest adaptive search loop with a measurable outcome: choose a
finite DFAO question, solve it, check its evidence, and prune implied questions.
See `docs/JEV_SEARCH.md` for the reproducible commands and scope.

## Setup

- Base: `5b139f7` (main, PR #42).
- CaDiCaL 3.0.1 plus drat-trim, built with the existing script. The existing
  self-test passed, including rejection of truncated and impossible proofs.
- CPU only; no canonical 10M/46M packed streams required. Each center prefix
  checked against `gen_golden_reference.center_naive` before encoding.
- Fixed and seeded-random ordering on `queue/jev/smoke.json`.
- A fixed-order n=64 trial, 10 seconds per solve/check and 90 seconds overall.
- TypeSafe key absent: **zero live Jev calls**. HTTP adapter tested with supplied
  responses; these are software tests, not model capability evidence.

## Results

| Run | Direct checked answers | Implied cases | Interpretation |
|---|---:|---:|---|
| Small fixed replay | 7 | 3 | Controls and bounded pruning work |
| Small random replay, seed 30 | 7 | 3 | Same finite answers; different order |
| n=64 fixed trial | 3 (including two controls) | 0 | Center s=15 SAT; other five trial cases UNKNOWN |

Summaries and the original n=64 trial plan are in `runs/jev-selector-demo/`.
The n=64 witness can be checked without Jev or a SAT solver:

```bash
python prize_lab.py check-dfao --candidate runs/jev-selector-demo/center64-s15-witness.json --sequence center --steps 63
```

The harness also checked this witness with
`experiments.dfao_drat_proofs.eval_dfao`, independently of the search encoder's
evaluator. The retained witness establishes only a finite upper bound. The
saved August curve already has the same upper bound, so this run is a
reproduction, not a new frontier result. The final frontier plan treats it as
calibration and keeps the 13/14-state center cases unresolved. Trial wall times
were about 0.26 s, 0.21 s and 50.49 s respectively; tiny replay times are not
evidence that either selection policy is better.

## Defect found and fixed

Postflight's `seed-echo` ignored experiment purpose, rejecting valid random and
Thue-Morse controls that preflight admitted. It now delegates to the same seed
gate as preflight, including the replication seed constraint. Regression tests
retain rejection of random-seed prize claims and mismatched replications.

## Interpretation and next step

The execution and evidence loop works. Jev's usefulness remains unmeasured.
Connect a TypeSafe key, freeze a genuinely unresolved plan and run fixed,
random and Jev policies at equal budgets. Report all unknowns and total wall
time. If the small menu is exhausted, the lead needs a better hypothesis,
not more calls. No claim-ledger promotion is made by this pilot.
