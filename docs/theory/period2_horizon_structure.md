# Finite period-two horizon structure

**Rule 90 filter.** The horizon encoder uses the complete Rule 30 truth table,
including its nonlinear OR latch, rather than left permutivity as an exclusion
argument. Its Rule 90 countercontrol accepts the finite seed with centre
`1,0,0,...`. Sequence fitting below is descriptive and advances no universal
period-exclusion argument.

**Zero filter.** Every radius search explicitly requires a nonzero initial row.
The separate zero-row control allows zero and accepts its constant-zero trace.
Alternating targets are nonconstant. Changing the initial condition changes
both feasibility and the finite radius; no single-seed prize result is claimed.

## Instrument and scope

`experiments/period_horizon_sat.py` now accepts `--min-length`, `--max-length`
and `--targets alternating,alternating-one`. Defaults and the original retained
artifact remain compatible. Both target phases are explicit; `alternating-one`
means `1010...`. Every query runs the finite-configuration counting bound before
search. CNFs are regenerated during verification; SAT rows are replayed by an
independent sparse-cell simulator; every retained UNSAT proof is DRAT checked.
Metadata verification checks exactly the requested length/target Cartesian
product, not just the original fixed 171 records. Tamper tests now also corrupt
the design range and target set.

The radius counts support in `[-R,R]`; lengths count observed bits including
time zero. An exact minimum requires SAT at R and checked UNSAT at R-1.
A timeout supplies neither UNSAT nor an exact minimum. These are bounded
findings over arbitrary finite rows, not measurements of the prize seed.

## Runs and useful witnesses

The broad run uses both phases, L=8..96, radius at most 96, three-second solver
limits and a 480-second search budget. A second targeted L=96 run uses ten-second
solver limits and a 100-second budget. A focused L=96 run probes R=80 and
R=53 using twenty-second solver limits and a 200-second budget. The budget controls solver launching,
not subsequent independent proof verification or archive compression.

Files:

- `runs/period2-horizon-extended-2026-10-14/results.json`
- `runs/period2-horizon-L96-2026-10-14/results.json` (initial weak brackets)
- `runs/period2-horizon-L96-focused-2026-10-14/results.json`
- `runs/period2-horizon-L96-focused-2026-10-14/constructive-witnesses.json`
- `runs/period2-horizon-extended-2026-10-14/sequences.json`
- `runs/period2-horizon-extended-2026-10-14/edge-witnesses.json`

`sequences.json` exports exact minima, first differences, and observed plateau
start/end sequences. It combines the pre-existing certified phase-zero curve
with new exact rows, rejecting any conflicting duplicate. Equal certified endpoint minima can determine intervening lengths by prefix
monotonicity; those inferred points retain both certificate sources. Guessers
use only the initial contiguous exact or explicitly derived block; censored holes are never silently joined.
An observed plateau touching a boundary or censored length has no certified
terminal endpoint beyond the displayed block.

Edge witnesses carry their source/query identity, entire finite initial row and
positive-half integer (R1 is bit zero). Phase-one rows must be advanced one
Rule 30 step, including their negative and centre cells, before their positive
half is used with a phase-zero forced-completion instrument. Using the original
phase-one positive half with a phase-zero target measures a different fiber.

The most distinctive early contrast is:

| Phase | Radius | Observed exact plateau | Next exact jump |
|---|---:|---|---|
| `0101...` | 20 | L=27..35 | R=29 at L=36 |
| `1010...` | 19 | L=26..36 | R=30 at L=37 |

The broad run contains 1000 queries: 548 SAT, 381 checked UNSAT, and 71
UNKNOWN, taking 486.243 seconds including checking overhead. New phase-zero
exact minima cover L=8..61 and L=66. Combining the old certificates gives directly certified
phase-zero L=8..64 plus L=66. Since R_0(64)=R_0(66)=54, prefix monotonicity
also fixes R_0(65)=54. The raw L=65 query record stays censored; the sequence
export marks that point as derived and records both certificate sources.
Thus the combined phase-zero curve is exactly determined through L=66. The new L=66 minimum is
R=54. Opposite-phase exact minima cover every L=8..63. Later requested lengths
are censored; a total-budget UNKNOWN is not a solver attempt.

At L=96 the focused run retains checked UNSAT at R=53 for **both** phases.
Independent sparse-cell simulation also validates two constructive upper rows
supplied by the lead from triangular inversion:

| Phase | Positive half r | Negative half n | Certified interval for minimum |
|---|---:|---:|---|
| `0101...` | 287 | 2894922832184770530049302 | [54,82] |
| `1010...` | 929 | 9508728737446193460969909 | [54,83] |

Set centre zero/one respectively; integer bit j of r occupies coordinate j+1,
and bit j of n occupies coordinate -j-1. Replay 96 steps with the independent
sparse simulator; no solver is needed to certify SAT. The `--append-witness`
operation records these rows as explicit SAT queries, regenerated CNF hashes,
and tightened upper bounds while preserving the checked lower certificates.
It validates metadata and the full trace before writing. The intervals remain
censored: no adjacent UNSAT certificate establishes either upper as minimum.

The lead obtained these constructive witnesses by testing all positive halves
r<2^16 at depth 95, separately for both initial centre phases, minimizing the
maximum of left and right bit lengths. This is a restricted 65,536-row class,
with log2 class cardinality 16 versus 96 target bits; any exclusion or minimum
within it must stay restricted and supplies no complexity evidence. C replayed
the two final witnesses, not the lead's entire 16-bit enumeration. The result
files retain the construction integers and scope, so those witnesses do not
depend on the search algorithm for verification.

## Candidate kills and automatic guessers

Three simple structural conjectures fail exactly:

| Candidate statement (for every L >= 8) | Kill test | First counterexample |
|---|---|---|
| R_0(L)=R_1(L) | Compare exact phases at each L | L=8: 5 versus 6 |
| abs(R_0(L)-R_1(L)) <= 1 | Compare exact phases at each L | L=9: 5 versus 7 |
| R_1(L)=R_0(L+1) | Compare shifted exact phases | L=8: 6 versus 5 |

These would simplify the sequence classification, but none would itself exclude
period two. They are finite kills, not complexity evidence. The instrument
passes the stated nonlinear-rule and zero controls; these numeric guesses
were never proposed as seed-independent impossibility arguments.

`experiments/period2_sequence_probe.py` uses rational Gaussian elimination with
Python `Fraction`, requiring no optional mathematical package. It attempts
homogeneous recurrences

`sum_{j=0}^o sum_{d=0}^D c[j,d] * L^d * R_phi(L+j) = 0`,

for order o=1..5 and degree D=0..3. Training uses the first two-thirds of the
contiguous exact terms, only when at least as many equations as coefficients
are available. Relations must use the highest-order term; surviving training
nullspace basis vectors are checked on every remaining equation. Full-data
nullspaces are exported separately, since killing individual basis vectors
does not in general kill every linear combination of them. Order D=0 is the
constant-coefficient linear subfamily; D>0 is the tested holonomic subfamily.
No nonzero recurrence in these tested classes fits the phase-zero training
block from L=8..66 (including the explicitly derived L=65 point). This excludes these specific equations on this finite
interval; it does not exclude a later-onset recurrence, larger order/degree,
or a recurrence with auxiliary variables.

For bases k=2,3 it also attempts the scalar digit-affine subclass
`R_phi(kL+j)=c*R_phi(L)+d*L+e` for every digit j. Finite sampled k-kernel ranks
through depth two are exported. They are diagnostics, not a complete detector
or exclusion of k-regular sequences: finite data cannot establish stabilization
of an infinite kernel. In particular, rank saturation on a short sample is an
observation about the sample width, not the true kernel dimension.

A preflight `counting_bound.py --pretty` was rerun before the final probes.
Each tested coefficient box [-8,8] has cardinality at most 17^dimension,
recorded in the JSON with purpose `exact-exclusion` and no complexity claim.
Exact rational nullspace rejection excludes a larger coefficient class than
that finite box. The bitstring randomness counting heuristic does not license
any inference from these integer-valued sequence diagnostics.

Positive and negative controls: a Fibonacci recurrence survives training and
heldout checks; perturbing its final term is rejected on heldout data. Singular
and inconsistent exact systems and both target phases have unit tests.

OEIS was searched through public indexed results using the exact initial
phase-zero prefix `5,5,6,8,8,8,8,8,9,9,14` and `Rule 30 alternating radius`.
The results were unrelated [OEIS sequences](https://oeis.org/). No match was
identified; this is an incomplete indexed search, not evidence of absence
from OEIS. No conjectural closed form survived these tests. No recursive extremizer
family was established; recursive witness-bit families were not exhaustively searched. A recurrence claim should be accompanied by its coefficients,
first valid L, phase, and a heldout exact radius certificate that could kill it.

## Acceptance and limits

Targeted instrument verification, tamper rejection, and four exact-algebra/phase/interpolation
unit tests were run. The lead runs the full repository verifier before the commit. Proofs
must be retained via Git LFS before treating the new exact rows as portable
certificates. The lead owns artifact upload, all grades, and ledger updates.
No infinite support-growth result, period-two exclusion, single-seed theorem,
Lean proof, complete OEIS search, GPU run, full infinite k-regularity test, or
outside-lineage audit is supplied by this work.


```bash
python experiments/period_horizon_sat.py --self-test
python experiments/period_horizon_sat.py --verify runs/period2-horizon-extended-2026-10-14/results.json --tamper-test runs/period2-horizon-extended-2026-10-14/results.json
python experiments/period_horizon_sat.py --verify runs/period2-horizon-L96-focused-2026-10-14/results.json --tamper-test runs/period2-horizon-L96-focused-2026-10-14/results.json
python experiments/period2_sequence_probe.py --inputs runs/period-horizon-2026-10-14/results.json runs/period2-horizon-extended-2026-10-14/results.json --verify runs/period2-horizon-extended-2026-10-14/sequences.json
python -m unittest tests.test_period2_sequence_probe
```

The sequence verifier regenerates both the interpolation and diagnostics from
source query records, rather than trusting the exported sequence JSON. The
SAT verifier was run on all 1000 broad queries, all 63 focused queries, and all
59 weak-run queries. All passed, including tamper rejection. After archiving,
archive extraction/hash validation is checked again; the expensive DRATs need
not be redundantly rechecked merely because lossless packaging changed.

Publication limit: the lead reported no usable Git LFS upload route. These
proof objects are locally retained and packaged for durable preservation;
a portable certificate claim requires recipients to receive the actual
objects. Git LFS pointers alone are not usable proof evidence. No ordinary Git
blob fallback is used. The PR remains blocked on proof delivery and the
required cross-lineage audit.
