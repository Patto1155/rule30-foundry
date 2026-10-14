# 2026-10-14 — realignment and the period ladder

## Goal and setup

Repair documentation and local verification inputs, independently audit the
period-one mechanism, locate its period-two failure, build a bounded horizon
SAT instrument, and identify the exact nonlinear forcing. Base main was
`b450406deace9f8adabc7b5b95d45b85e2c32432`. Implementation/research used
isolated worktrees and available native Codex subagents; no outside provider
was configured here. Claude subagent turns were not used. Subagents did not edit the two state
files or experiment logs. The lead checked arguments, replayed acceptance,
and assigned grades. Agreement between agents was not used as corroboration.

Theory gate preceded Phase 2 dispatch: `queue/results/period-ladder-theory-gate.md`.
Counting ran before SAT search, using the actual nonzero finite-row class
`2^(2R+1)-1`, not the DFAO count. Negative proofs are exact finite exclusions,
not complexity evidence. All arguments state the Rule 90 and zero filters.

## Lead-owned grade decisions

| Claim | Before → after | Evidence |
|---|---|---|
| Eventual period one for nonzero finite rows | External proof candidate, unchecked → independently audited analytic Theorem; publication remains unrefereed | `docs/theory/period2_attempt.md` §§1–2; `experiments/period2_identity_check.py` |
| Nonlinear obstruction | Informal quadratic-only wording → exact forcing identity and superposition-defect Theorem; hardness remains open | `docs/theory/nonlinear_obstruction.md`; `experiments/nonlinear_obstruction_check.py` |
| Period-two horizon instrument | Unmeasured → Certificate of bounded minimum-radius findings, T=0 and L=8..64 | `runs/period-horizon-2026-10-14/results.json`; retained DRAT chunks |
| Eventual period two | Open → open; permanent first-one latch fails under alternating return | `docs/theory/period2_attempt.md` §§3–5 |
| Diagonal unboundedness / growth law | Robust observation → unchanged; the recovered abstract branch tree does not identify the seed's branch | Sept8 B1 recovery log; replay section below |

Condrey's main argument was reconstructed analytically: left-permutive
triangular uniqueness plus explicit OR-latched fibers force an infinite
left tail, except for the zero row; expanding support endpoints preserve
nonzeroness under time shifts. The all-one case likewise has an infinite
checkerboard, or reduces to an eventually-zero neighbor. This passes both
filters. The unsupported inference in Condrey §5 from a linear lower bound
to the absence of a fixed-radius bound is not part of the accepted proof.
No Lean or referee verification was performed.

For alternation the first right-neighbor return is
`s_next=(1-s)(1-u)(1-v)`, which resets a one instead of preserving the
constant-zero latch. Right seeds 1 and 7 have the same first-one position but
force distinct initial left bits at depth 9. Transfer and renewal identities
are credited to the published research branch. The proposed positive-density
support-growth lemma is entirely open; its weaker unbounded-weight version
is equivalent to period-two exclusion, not a reduction in difficulty.

Rule 30's additive part is Rule 150, not Rule 90. The lead checked the all-time
identity `F_t=L^tF_0+sum_{s<t}L^(t-1-s)Q_s` and its superposition defect.
The first quadratic centre correction on the prize seed is at t=2. The
identity depends on intermediate orbit forcing and is not a shortcut or a
computational lower bound. Novelty of these elementary identities was not
established. The checker exhausts 32 five-cell seeds at every t=0..64, not
all width129 light cones.

## SAT result and replay

All 171 minima are exact, with no UNKNOWNs. At L=64 the minimum radii are
54 (alternation), 63 (zero), 62 (one). All 24 fixed-radius Condrey control
queries pass, and independent enumeration through radius 6 reproduces
extremizer counts. Four random controls are instrument checks, not a
statistical separation. The run took 271.504s and made 1249 distinct queries:
710 SAT, 539 UNSAT. Every UNSAT CNF, DRAT proof and checker log is retained
in 28 authenticated pieces of a 135,976,180-byte xz-compressed tar stream.

The lead inspected cone boundaries, infinite zero exterior, support nesting,
and target/proof identity, then replayed self-test, all CNF hashes, all
independently simulated witnesses, all 539 DRAT proofs and seven tamper
rejections. Record-label verification was strengthened during review; an
altered target or length can no longer reuse unrelated radius evidence.
The plot omits empty censored-series legend entries.

```bash
python experiments/period_horizon_sat.py --self-test
python experiments/period_horizon_sat.py --verify runs/period-horizon-2026-10-14/results.json --drat-trim third_party/drat-trim
python experiments/period_horizon_sat.py --tamper-test runs/period-horizon-2026-10-14/results.json --drat-trim third_party/drat-trim
python experiments/period2_identity_check.py
python experiments/nonlinear_obstruction_check.py
```

Finite configurations are a lemma-discovery family, not the single-seed
prize measurement. This grid supplies no asymptotic growth or eventual-period
theorem. Different start times, the opposite alternating phase and radii
above64 were not searched.

## Infrastructure and drift

- Golden 10M is deliberately MSB-first. Read as little-endian it first differs
  from a fresh CPU simulation at bit 0, with 49,944 differing bits in 100k.
  MSB decoding agrees with the fresh 100k CPU prefix; naive 4096 also agrees.
  Reencoding to LSB-first yields exactly the canonical manifest SHA
  `6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e`.
  All 10M decoded bits verify. Golden and manifest were unchanged.
- CaDiCaL 3.0.1 and drat-trim built. The n=56 refutation was freshly solved
  (36.589s), independently checked (41.342s), and its 197,932,048-byte proof
  hash exactly matches the existing certificate. This is a replay, not a
  grade promotion.
- Verifier timing, fallback delegation, duplicated section numbering and
  references, GPU script paths, STATUS date/default frontier and historical
  Jev claims were corrected. No Jev live comparison was rerun.
- Counting documentation and gate instructions no longer claim every word
  must fail a sparse-class search. A true exhaustive exclusion is distinct
  from evidence of complexity. No executable admission policy was weakened.
- The obligation queue no longer calls diagonal unboundedness or the
  empirical settling coefficient a settled all-horizon theorem.

Initial full-suite races arose because worktrees were being created or
removed by overlapping suites. After serializing suite runs, the integrated
Phase 1 verifier gave10PASS,1SKIP,0FAIL. Final gate: `verify_all: 10 passed, 1 skipped (0 unexpected), 0 failed  OK`.
The sole stage SKIP is the absent 46M stream, which checked nothing.
`queue/results/period-ladder-final-verifier.txt` and the companion report
retain the raw verdict and exclusions. A failed pre-commit attempt caught
invalid prize/mechanism labels in the new obligation entries; these were
corrected without relaxing the schema or gate, all 29 focused workflow tests
passed, and the complete verifier was rerun. The failed attempt is retained.

## D1 trajectory replay

The historical walk already existed; the Sept8 recovery audit checked its
recorded events, not all intervening transitions. A fresh script-harness
replay to the exclusive boundary d=12,000,000,000 completed. It used
`workhorse.py --agent script --dry-run` to avoid an automatic extra commit or
PR. Dry-run disables recursive external gates; separate repository verifier
runs and the accepted theory gate provide those checks. Numba was initially
absent; the first attempt refused immediately, then numba 0.68.0 / llvmlite 0.50.0
were installed before the rerun. No failed attempt is reported as a result.
The fresh run completed 1797.25s after launch; the map walk itself took
1785.71s. It reproduced all 36,617,159,481 steps, nine branch points, two
doubling leaves and eight undoubled leaves. All events, continuations,
leaves and step counts match the recovered artifact exactly. The 50,000
map-vs-simulation comparisons have zero mismatches; the lemma-A gate has
zero violations. `runs/d1-pattern-map-replay-2026-10-14/comparison.json`
records the comparison, and `outputs/pattern_map_walk32.json` retains the
fresh artifact with harness provenance and postflight. The standard-library
consistency verifier and its four tamper checks were replayed on the fresh
artifact, including fresh seed gates.

The abstract doubling leaves occur at d=2,107,985,255 and d=9,958,700,505.
Eight other leaves reach the exclusive 12-billion boundary without an
encountered doubling. This neither identifies the actual seed branch nor
establishes unbounded diagonal periods; the growth-law grade is unchanged.
The queue manifest is retained as a reproducible recipe, not an unrun task.
A throughput check on 10M map steps cost 0.431s (projected 26.3min for the
historical step total); full-run contention made the replay somewhat slower.

## Not run / not verified and next step

46M bitstream absent; its verifier stage checks nothing. No GPU or CuPy is
available. Lean executable absent; Mathlib page was reachable (actual HTTP 200),
but no optional installation or formal build was attempted. Historical Jev,
LSD n=56 and n=64 campaigns, the full old 207-instance DRAT campaign, recovered
renewal-core reproducers and reported 740-state work were not rerun. No
outside-model review or publication novelty review was obtained. No period two
exclusion, forcing complexity bound, or actual D1 seed-branch identification
is claimed.

Next: derive a physical ordered-support lemma for one fixed right trajectory,
with an explicit falsifiable inequality. Do not extrapolate the SAT curve,
replace physical boundaries with free choices, or promote the elementary
forcing identity into hardness. B3 LSD n=56 remains optional bounded work.
