# Checked Rule 30 foundations

Pinned Lean 4.19.0 and Mathlib v4.19.0; the lockfile pins Mathlib and its
transitive dependencies by commit. These proofs import Lean core only;
Mathlib is available for subsequent extensions, so no Mathlib cache download
is required for this build.

```sh
cd formal
lake update
lake build
lake env lean Rule30/Basic.lean
# Or, from repository root:
python tools/verify_lean.py
```

The verifier builds, rechecks the source, requires an axiom printout for every
theorem declaration, rejects `sorryAx`, and permits only `propext`,
`Classical.choice`, and `Quot.sound`. All 19 declarations were checked locally; `checked-axioms.txt` records the
actual build/recheck printouts (not a substitute for rerunning the verifier).

| Lemmas | Exact scope |
|---|---|
| `zero_latch`, `alternating_return` | Rule 30 OR identities, checked by Boolean case analysis |
| `step_finite`, `evolve_finite` | Finite support stays finite |
| `evolve_add`, `trace_shift` | Semigroup/time-shift identities |
| `onset_reduction` | Either phase of an eventually alternating trace yields a finite, nonzero row with trace `n mod 2` from onset zero |
| `oneLeft_adjacent`, `oneLeft_invariant` | Constant-one checkerboard's local invariant |
| `leftOne_step`, `leftOne_evolve`, `leftOne_constant_trace`, `leftOne_not_finite` | Condrey Theorem 3's **construction half** and its infinite support; no fiber-completeness claim |
| `temporal_phase_slip` | Recovered depth-seven Boolean/integer coboundary under adjacent-11-free triple hypotheses; algebra only, no spatial transport |
| `ring7_step0`–`ring7_step3`, `ring7_boundary` | Four-step orbit on a seven-cell ring, plus boundary bits; no claim of a compatible finite physical right half |

`AlternatingTail` explicitly asserts the alternating formula with either
phase. A generic recurrence-only definition of eventual period two is not
formalized here. No theorem claims exclusion of eventual period one or two.
Condrey Theorem 2, Theorem 3's uniqueness/completeness half, and Corollaries
4–5 remain unformalized. The checkerboard results cannot exclude arbitrary
constant traces without triangular uniqueness.

The nonlinear identities and checkerboard evolution use OR explicitly
(Rule 90 filter). The onset hypothesis excludes the zero orbit; the
checkerboard has arbitrarily distant ones (zero filter). Finiteness/time-shift
identities are general infrastructure rather than period-exclusion arguments.

On this sandbox Lean asks for `/proc/<pid>/exe`, which is absent even though
`/proc/self/exe` exists. `verify_lean.py` detects that exact startup error,
compiles `tools/lean_proc_path.c`, and retries with a readlink compatibility
shim. This changes executable-path discovery only, not Lean's kernel or
proof evaluation. Normal machines use the unmodified path. Local acceptance
used this shim. CI names `lean` as a permitted skip because neither CI job
installs Lean; that skip checks nothing.

Not run: full Condrey formalization, global exclusion/support-growth proof,
Mathlib cache download (no imported Mathlib modules), external lineage audit,
GPU jobs. Full repository verification is owned by the lead after integration;
this agent runs only focused tests while repository suites are serialized.
