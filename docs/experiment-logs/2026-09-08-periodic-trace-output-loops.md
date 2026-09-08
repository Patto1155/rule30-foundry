# Output-loop certificates and the exterior constraint

- Date: 2026-09-08
- Goal: distinguish nonperiodic neighbors from larger-period neighbors, then
  test whether a finite right-hand initial state supplies the missing constraint.
- Source base: main after PR #33, `1f71bd4`.
- Scope: exact open-strip implications and finite-horizon driven subsystems.
  No single-seed nonperiodicity claim is made.
- Derivations: `docs/theory/periodic-trace-constraints.md`, sections 6 and 7.
- Artifacts: `runs/periodic-boundary-2026-09-08/`.

## 1. Arbitrary-period synchronization fails in the tested strips

```bash
python experiments/periodic_strip_graph.py --output-loops --words 01 --widths 3 5 7 9 11 13 15
python experiments/periodic_strip_graph.py --verify-loops runs/periodic-boundary-2026-09-08/output-loops.json
```

At every width, the synchronized-product search finds two equal-length closed
walks at the same full strip state, with different left-neighbor output blocks.
Widths 3, 5, 7, 9, 11, 13, 15 give block lengths 4, 4, 8, 8, 8, 24, 28.
The direct verifier checks all transitions and output blocks without rerunning
or trusting the search. Tampered transition/output tests are rejected.

Concatenating the two loops according to a non-eventually-periodic selector
gives a non-eventually-periodic neighbor while the center alternates. The
fixed-length distinct output code makes this implication rigorous. This is a
stronger counterexample than merely violating period 2. It remains an open-strip
construction with freely supplied boundaries, not an infinite-plane seed orbit.

Constant-center width-5 and width-9 controls admit no such output-loop pairs.

## 2. A complete right initial state constrains the construction

```bash
python experiments/periodic_right_boundary.py --check-loops runs/periodic-boundary-2026-09-08/output-loops.json
```

For each common root, keep its initial positive strip cells and set the exterior
to the right to zero. The right half-line is then uniquely driven by the center
word, and the center rule specifies the required left output. Both saved loop
choices fail this specific completion at every width. The artifact records
the first right-edge, positive-strip, and required-left mismatches.

This establishes precisely where the free choices entered the strip argument.
It does not exclude other exterior completions. Uniqueness of a driven
continuation is not periodicity of that continuation.

## 3. Finite support does not produce rapid small-period settling

```bash
python experiments/periodic_right_boundary.py --scan --steps 512
python experiments/periodic_right_boundary.py --scan --steps 2048
python -m unittest discover -s tests -p 'test_periodic*.py' -v
```

All 256 eight-bit positive initial words are tested under each of `0`, `1`,
`01`, `10`. Another family uses the actual positive seed rows at onsets 0..64,
then imposes alternating center forcing beginning with the observed center bit.
The horizon includes t=0..H; the tested suffix is floor(H/2)..H inclusive.

| Family | H=512 | H=2048 |
|---|---|---|
| boundary 0, 256 inputs | 256 period 1 | 256 period 1 |
| boundary 1, 256 inputs | 256 period 1 | 256 period 1 |
| boundary 01, 256 inputs | 241 unresolved; 9 period 4, 5 period 14, 1 period 36 | 245 unresolved; 9 period 4, 2 period 14 |
| boundary 10, 256 inputs | 242 unresolved; 6 period 4, 5 period 14, 3 period 36 | 248 unresolved; 6 period 4, 2 period 14 |
| 65 seed snapshots, alternating forcing | 55 unresolved; 2 period 10, 2 period 14, 6 period 36 | 65 unresolved |

Unresolved means no exact period 1..64 within the tested suffix. It does not
mean nonperiodic forever. Apparent periodic windows can disappear at longer
horizons, as the seed-snapshot family demonstrates.

The packed half-line updates are tested against an independent cell-by-cell
truth-table implementation, including random inputs and propagation beyond bit
64. The tape holds initial support plus H+1 cells, so its right boundary lies
beyond the reachable support throughout the requested run. No canonical
bitstream, GPU, or statistical null is involved. Constants are settling
controls; this is not a learned-model fit or an aggregate randomness test.

## Interpretation and next step

The simple strip-synchronization route fails through width 15. A full exterior
state supplies a necessary uniqueness constraint, but no sufficient inductive
seed invariant was found. The remaining condition is compatibility of the
required left output with evolution from the actual left seed row at the same
onset. The finite experiments do not decide that for unbounded onsets.

A separate proposed direction is recorded in
`docs/idea-bank/period-word-sieve.md`: certify forbidden periodic words for
all onsets via strip implications, then seek a rule covering a family of words.
