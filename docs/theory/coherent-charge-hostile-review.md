# Hostile review of the coherent-charge follow-up

## Result

No unsound all-size theorem was found in the reviewed charge derivations.  The
pair-density obstruction and the temporal-pair obstruction are correctly
scoped as refutations of sufficient relaxed languages.  In particular, the
new `0 >= 4` identity is not evidence that the charge needs a third temporal
observation.  Its terminal cancellation uses a projected four-cycle which is
not a Rule 30 strip, and two of its four spatial paths are bulk strips which
cannot share the alternating centre latch.

The periodic-boundary result is also stated at its actual finite scope:
periods at most six, 126 boundary words, no censored orbit, observed minimum
cycle density `1/7`.  The 7-by-4 ether is exact, but it is a bi-infinite
periodic spacetime witness.  Its `alpha >= 7` consequence applies to a charge
inequality required on that relaxed periodic family; it is not by itself a
necessary coefficient bound for finite-support single-seed cones.

Two evidence gaps found in the first pass were repaired during review.

1. `runs/theory/periodic-boundary-density.json` originally was not replayed by
   any verifier or test.  The final version adds `--verify`, recomputes the
   declared period-6, 100,000-state scan, compares the complete parsed report,
   and has mutation tests for counts, cycle words, densities, and scope.
2. The Rule 15/90 controls exercise `forced_left_prefix` at depth 12 and right
   support 2.  They do not exercise the periodic word-state transition, orbit
   detection, censoring, or cycle-density calculation.  The final prose now
   calls them finite-cone constructor baselines and explicitly says they do not
   test the periodic scanner.  That resolves the scope overstatement.

## Evidence checked

- The focused 36-test set passed:
  `test_coherent_density_certificate`, `test_temporal_pair_density`,
  `test_periodic_boundary_density`, and `test_verify_coherent_density`.
- The independent verifier reconstructs the temporal-pair rows from saved
  labels, independently enumerates the Rule 30 truth table, checks all four
  integer identities, and obtains `0 >= 4` in every declared scope.  It also
  checks the spatial telescope at `q=3,...,7`, including the empty `q=3`
  interior and the finite endpoint domains.
- Direct finite-cone enumeration gives the two `q=6` paths from initial words
  0 and 1 respectively, and the two `q=7` paths from initial words 0 and 1.
  Thus all four are genuine bulk paths, while the paths from word 0 violate the
  first alternating latch.  This confirms the distinction in
  `coherent-density-obstruction.md`: an exact top cap is not an exact
  centre-capped cone endpoint.
- Exhausting all 12-bit predecessor words under the shrinking Rule 30 word map
  confirms that every occurrence of terminal pairs
  `00,00,00,11,00` has next pair `11`, never `00`.  The projected terminal
  four-cycle is therefore spurious exactly where claimed.
- The older obstruction was rechecked from the genuine `n=129` cuts: its
  spatial combination is `Phi_5 + Phi_7 - 2 Phi_3 >= 6`, its bottom terms
  cancel, and the three genuine terminal cycles give the opposing upper bound
  zero.  This supports the all-spatial-`psi` obstruction only while the same
  pair density and terminal-flow requirements are retained.
- The periodic nonzero-density lemma is sound.  A zero-output spatial cycle
  forces every temporal phase of both `E` and `O` to zero; the all-zero
  four-word state has only itself as a predecessor and cannot be reached from
  the prescribed start with `O=allones`.
- The doc-only characteristic-front calculations were spot-checked.  The map
  sends `65` to `406`; the reported defect masses are 2 and 5; and the two
  infinite halves are verified directly by `T_0(B)=A` and
  `T_0(A)=T_1(A)=A`.

## Remaining gaps

The charge and periodic-boundary results do not establish a uniform
cone-charge bound, exclude arbitrary finite-support survivors, or prove
density at least `1/8` for arbitrary
periodic boundary period.  Positivity for each fixed period gives no uniform
lower density because the eventual cycle length is unbounded with the period.
Likewise, the exact ether does not transfer from a bi-infinite periodic row to
the zero-tail finite-integer problem.

The temporal-pair Farkas witness identifies common predecessor/centre lineage
as the present missing dependency.  It does not show that a larger density
table is needed.  Any next terminal certificate must exclude both the false
six-pair terminal word and the two word-0 bulk paths while retaining every
actual centre-capped cut.  Merely increasing the projected window length
would repeat the relaxation error.

`alternating-alternative-mechanism.md` is mathematically candid, but its exact
return-map, `n=65` mass counterexample, and infinite-branch falsifier currently
have no dedicated executable artifact or regression test.  The manual checks
above found no error.  A small truth-table test would make this evidence
durable and would be preferable before citing the note as a certificate.

## Next implication

Prioritize an endpoint-conditioned terminal/credit lemma carrying common
predecessor lineage from the bottom latch to the centre cap.  Do not infer a
third temporal density observation, a uniform period bound, or a finite-support
exclusion from the current exact obstructions.

## Addendum: period-7 ether interface theorem

The new infinite-family argument in `period7-ether-interface.md` is sound.  In
particular, its 18-bit calculation is not an extrapolation from the first 512
truncations.  Sampling the closed triangular interface system every 14
half-steps gives a deterministic bijection on `(delta, time mod 28)`.  The
saved initial state returns after 512 samples with no earlier repeat, so every
sample index `m`, and hence every even `r=2m`, is covered by that exact cycle.

The potentially dangerous boundary switch is also correct.  The interface is
followed for `7r=14m` half-steps, an even phase at which the prescribed next
boundaries are again `0,1`.  `arrival_half` reconstructs the low finite-half
prefix at that phase, and `failure_delay` applies `T_0` followed by `T_1`.
Testing through delay eight means at most 16 further half-steps; the low-pair
test then depends on arrival indices only through 17, exactly the retained
18-bit prefix.  No unmodelled high bit or ether boundary is read.

Independent checks made during review were:

- direct integer evolution agrees with the interface reconstruction at every
  arrival through `r=32`, as required by the test;
- selected large cases `r=1024` and `r=1026` fail at macrotimes 3584 and 3592,
  within the certified bounds, so the cycle reduction is being interpreted in
  the correct phase and direction;
- the odd case has no hidden early-failure gap: before half-time `7r-1`, its
  low pair equals the boundary-compatible ether at every latch time, and the
  nearest discrepancy at that even time changes exactly one member of the low
  pair.  This proves equality, not merely an upper bound.

The two implementation issues found in the first interface pass were repaired:

1. The documentation now distinguishes the input coordinate `-time+d` from
   the output coordinate `-time-1+d`, matching the implementation.
2. The corollary `q(n_r) <= 4r` now records its finite bridge.  The uniform even
   bound implies it for `r>=16`; odd `r` are automatic.  The remaining evens are
   `r=2,4,...,14`, with exact failures `8,16,21,28,39,43,49`, all at most
   `4r`; the artifact and unit test now include `r=14`.

The test called “independent” is independent of the interface recurrence but
uses the existing `half_step` integer formula rather than a fresh cell-level
Rule 30 implementation.  This is adequate for the interface reduction because
that formula was established earlier, but the evidence should be described as
an independent route through the trusted half-line map, not an independent
truth-table implementation.

The completed standalone verifier now separately evolves ether and defect
triples through the Rule 30 truth table. Its post-arrival latch calculation
also uses a shrinking 18-cell truth-table cone. The larger direct-truncation
cross-check uses the established packed half-line identity. Thus the decisive
interface cycle and post-arrival boundary check have a separate cell-level
implementation.

This theorem excludes an unbounded positive-density family only.  It supplies
no bound for arbitrary finite halves and does not turn the period-7 ether into
a finite-support survivor or counterexample.

There is a further essential scope distinction.  Here `q(n_r)` starts the
alternating boundary at centre-zero phase on the original row `n_r`.  Failure
of that orbit proves that this row cannot begin an eternal alternating centre.
It does **not** exclude an unrelated Rule 30 orbit initially seeded by `n_r`
from entering an alternating centre later, because the left half at that onset
need not remain in the truncation family.  The result is therefore neither a
full period-two theorem nor an eventual-alternation exclusion, even for these
original integers.

### No wake-independent post-arrival horizon

The proposed obstruction to a naive generalization is valid.  Let
`A=101010...` and `B=01101010...` in low-index-first notation.  Directly,
`T_0(B)=A` and `T_0(A)=T_1(A)=A`, so the infinite `B` passes every latch.  For
any positive `H`, truncate `B` after its first `2H` bits.  The latch at
macrotime `k` depends only on initial indices through `2k+1`; consequently the
finite truncation agrees with `B` and passes for `k=0,...,H-1`.  Its first
failure time is therefore at least `H`.

On the other hand, `B` and the period-7 ether differ already at index zero:
`B_0=0` and `E_0=1`.  Their discrepancy-front arrival time is zero.  Since `H`
is arbitrary, no universal constant can bound survival time after front arrival
over all finite wakes relative to this ether.  Direct checks for `H=1,...,16`
agree with the dependency proof.

This obstruction does not refute a charge argument.  The truncated `B` wakes
carry growing support and growing initial weight.  A general interface theorem
must charge wake complexity, discrepancies, or initial ones; it cannot replace
the special 512-state period-7 orbit by all possible wakes while retaining the
same fixed post-arrival horizon.
