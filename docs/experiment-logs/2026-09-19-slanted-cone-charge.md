# 2026-09-19 — Slanted cone-charge attempt

## Goal and outcome

Attempt a uniform cone-charge proof from `d83f767`. **No period-two exclusion
or cone-charge proof was obtained.** The useful outcome is exact obstruction
certificates and a specific missing predecessor dependency. Prize-status grades
were not changed. All work stays local.

The proposed sharp prefix bound is equivalent to `8W_N-N>=-1`, not `>=-6`.
The latter would still suffice for exclusion if proved uniformly.

## Exact geometry and potential falsifiers

The growing characteristic cut consists of complete diagonals
`(D_(q-1),D_q)`, where `D_q(t)=a(t,-q+t)`. Its recurrence preserves the
alternating centre endpoint. The macrostep tiling and both telescoping
endpoints are derived in [the geometry note](../theory/slanted-cone-geometry.md).
No exact finite-state encoder is assumed.

Across 75 predeclared endpoint/population feature sets, genuine cone edges
through depth 24 give 27 verified negative quotient cycles. A cycle rules out
every potential depending only on its declared features. The shortest is a
one-edge cycle of charge -8. It comes from `n=129`, whose first 16 initial
cells are `10000001 00000000`: the second block consumes carried credit.
These quotient cycles merge different complete cuts, so they are **not**
infinite Rule 30 orbits or counterexamples to cone charge.

Targeted tests of frozen bottom-three-bit and bottom-four-bit sample potentials
then gave two more family refutations, each with a two-edge cycle of charge
-8. The bottom-four-bit cycle uses depth `24->32` edges from initial integers
`2126145` and `536874262`, with charges -8 and 0. Both seams first disagree at
temporal index 5 and also discard growing tails. These were tests of specific
potentials, not a new raw survival-horizon scan. The 48 baseline no-cycle
outcomes are only unresolved sampled constraints; two were later refuted.
Exact population counts are globally unbounded, not finite-state features.

Every cycle is independently checked by forward elementary-rule truth tables,
including latch validity, common rule, exact tile dimensions, feature equality,
and integer charge. Rule 15 retains its genuine zero-initial-left alternating
fibre; no truncated zero background is imposed on its odd rows.

## Local density synthesis

Fourteen sufficient local density/flux languages were attempted. Refinements
included both spatial/time parity, one/two/three simultaneous diagonal-update
equations, bottom-pair corrections, and signed densities with a second flux
bounding the total terminal potential. All fourteen are infeasible, with
exact nonnegative-integer combinations of constraints proving contradictions.
Floating-point solver statuses are not accepted as proof.

The smallest certificate sums six constraints to `0>=2`. The strongest signed
language has a 27-row certificate giving `0>=6`. None of these certificates
uses initialization rows, so increasing the starting allowance alone cannot
repair the language. A standalone checker reconstructs labeled inequalities
without importing the synthesizer or a numeric library, validates the local
transitions against Rule 30's truth table, and checks cancellation as integers.

The strongest witness exposes a precise missing dependency:

```
000 -> 100 -> 111
```

Both individual transitions are locally legal, but their concatenation is
impossible. The first forces older bits `z=0,w=1`; their next update forces
`z'=1`, making the final first bit zero. Independent enumeration of all 128
seven-cell inputs confirms this and finds four such spurious two-edge walks.
The failure comes from resampling predecessor bits between transitions.

## Other tracks and controls

- The independent sparse-prefix checker reproduces sharp bounds
  `B_r=0,3,8,11,16,19` for weight caps `r=0,...,5`. Locality makes these valid
  for arbitrary support width at each fixed weight; they do not establish a
  bound for every weight. These recheck previously reached horizons.
- Removing one initial one while losing at most four latches is impossible:
  weight two can survive eight, while weight at most one survives at most
  three. Plain paired deletion also fails on a saved exact witness. See
  [renewal note](../theory/alternating-renewal.md).
- Rule 90's unconstrained right boundary correctly defeats generic left-latch
  exclusion. An all-period extension requires phase-specific constraints and
  endpoint bounds; latch gaps alone lose information. Constant-zero traces
  have no phase-one latches and still need a separate argument. See
  [period bridge](../theory/period-charge-bridge.md).

## JEV and review

The deterministic baseline and seed-20260919 random order were frozen before
attaching exact graph results. A metadata field initially called an encoding
bound was corrected to a heuristic feature cost; candidate order was unchanged
and both hashes are recorded. Metadata is explicitly scoped to sampled graphs;
unmeasured quantities remain null. Baseline's first unresolved candidate was
bottom-three-bit/no-count, subsequently refuted in targeted holdout.

An authenticated JEV request was rejected by automatic approval review as
external credential disclosure to OpenRouter. There were **zero successful
calls**, no JEV decisions, and no matched JEV performance comparison. No
workaround or retry was attempted. Separate agents supplied a same-lineage
hostile audit, not independent external-model corroboration. The lead checked
the reductions and exact witnesses; no outside review of a successful theorem
is claimed. [Audit](../theory/slanted-cone-hostile-review.md).

## Reproduce

```bash
python experiments/slanted_cone_potential.py --verify runs/theory/slanted-cone-potential-2026-09-19.json
python experiments/slanted_cone_potential.py --verify runs/theory/slanted-cone-potential-holdout-2026-09-19.json
python tools/verify_slanted_density.py runs/theory/slanted-density-obstructions.json
python experiments/alternating_renewal.py --weight 5 --bound 19
python tools/jev_cone_triage.py --check queue/theory/cone-candidates.json
python -m unittest tests.test_slanted_cone_potential tests.test_slanted_density_certificate tests.test_alternating_renewal tests.test_jev_cone_triage
```

Regenerating density candidates requires SciPy; verifying the retained integer
certificates does not. Baseline and final `verify_all` each reported 8 passed,
3 expected skips, 0 failures; all 21 focused tests passed. The skipped inputs are the two canonical bitstreams and SAT
proof toolchain; their absence does not affect these direct local checks.

## Single next obligation

Synthesize a local density/flux potential on **two successive cut rows whose
predecessor bits are shared across updates**. In particular the state following
`000->100` must retain `z'=1`. Enumerate transitions from seven original bits,
not two independently chosen sets of older bits, and retain exact top/bottom
conditions. Seek `8e_q-1+Phi_q-Phi_(q-1)>=0` with a bounded terminal upper
value. The refined state is specified; existence of such a potential remains
unproved. This narrows a failed proof route, but does not materially establish
period-two exclusion.
