# Periodic trace controls and finite-strip synchronization

- Date: 2026-09-08
- Goal: test whether periodic-column constraints can yield a seed-specific
  nonperiodicity argument, and identify which proposed implications fail.
- Setup: standard-library CPU code, open boundaries, no packed bitstreams.
- Derivations: `docs/theory/periodic-trace-constraints.md`.
- Output directory: `runs/periodic-trace-2026-09-08/`.

## Gate and scope

This is a feasibility/control and finite-state implication check, not a search
for a small model fitting a long random-looking prefix. Trace inversion has
exactly 2^N initial words and 2^N target traces at length N. Its surjectivity is
already implied by left-permutivity. The implementation checks do not constitute
new evidence for that elementary fact. No prize-facing claim is promoted.

Positive controls: arbitrary prescribed traces must be realizable after changing
initial conditions; the true seed trace must invert back to the seed; constant
center words must synchronize their left neighbor once the strip is wide enough.
Negative controls: an open width-3 zero-center strip admits neighbor changes;
candidate tails with no comparison beyond the first block remain unresolved.
No random baseline is needed for the exhaustive local implication check.

## Commands and results

```bash
python experiments/periodic_trace_constraints.py --self-test
python -m unittest discover -s tests -p 'test_periodic*.py' -v
python experiments/periodic_trace_constraints.py --tail-scan
python experiments/periodic_strip_graph.py --scan
python experiments/periodic_strip_graph.py --scan --words 01 --widths 11 13 15
```

- All 64 paired local neighborhoods satisfy the exact defect formula.
- All 510 binary words of lengths 1 through 8 are realized by inverse initial
  words supported on the left. Independent expanding-row, rule-number lookup
  tests agree with the shrinking-cone implementation.
- The actual seed trace through t=80 inverts to the seed; reference checks
  include initial cells beyond distance 64.
- All 1,040 tails with onset 0..64 and period 1..16 first mismatch within the
  horizon t=0..512. These are finite exclusions only and do not improve the
  existing large-prefix period search.
- A tail starting at t=8 with period 3 first mismatches at t=11. Its inverse
  requires a first extra initial one at x=-11. The modified initial row realizes
  the full prescribed trace through t=64. This is the ordinary mismatch in
  spatial coordinates, not a stronger certificate.

Finite-strip model checking uses every state and every allowed transition,
then recurrent strongly connected components (SCCs). A p-step neighbor change
inside a recurrent SCC is saved together with a path returning to its start.
Repeating this closed walk violates eventual period p in the strip.

| Center word | Widths | Eventually same-period left neighbor? |
|---|---|---|
| 1 | 3, 5, 7, 9 | Yes |
| 0 | 3 | No: boundary freedom supplies a counterexample |
| 0 | 5, 7, 9 | Yes |
| 01 | 3, 5, 7, 9, 11, 13, 15 | No: closed-walk witnesses at every width |

The alternating cases have 4, 8, 16, 28, 46, 84, and 150 recurrent states,
respectively, all within one recurrent SCC at each width. These counts describe
the finite model only. A neighbor with a larger period suffices for a witness;
the test does not establish an aperiodic neighbor or an infinite-plane extension.

## Interpretation

Unanchored periodic-strip contradictions are unavailable: every infinite trace
has a generally infinite left-half-line initial realization. Fixing a finite
patch of seed cells still leaves every continuation after its forced prefix
possible. The whole single-seed condition cannot be replaced by that patch.

Constant-center synchronization has a short direct derivation: center one
forces left zero; center zero makes left=right and right evolve by OR, hence
monotonically. The width-5 control recovers that implication. Alternating-center
synchronization to the same period does not follow from open strips through
width 15. Wider strips or a different eventual neighbor period remain possible.

## Next step

Seek a seed-derived boundary constraint that eliminates the saved alternating
closed walks, or a proof of eventual neighbor periodicity with a period allowed
to exceed two. Increasing the prefix-period scan adds no new mechanism.

## Verification environment

Eleven new focused tests pass on Windows. The full suite passes in an isolated
native Linux checkout: eight runnable `verify_all` stages pass; the two absent
canonical bitstreams and absent SAT toolchain are explicitly skipped. The suite
contains 457 tests, including seven existing optional skips. Stage output is
saved in `runs/periodic-trace-2026-09-08/verify-linux.txt`.

The pre-existing Windows worker tests fail before these changes. A Linux run
on the mounted Windows drive also failed its existing 4.5-second pool timing
threshold (4.57 seconds; isolated retry 4.95 seconds); the same checks pass on
the native Linux filesystem. No worker code or timing threshold was changed.
