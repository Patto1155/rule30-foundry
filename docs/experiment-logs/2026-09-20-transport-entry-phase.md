# Transport, invariant entry, and arbitrary phase words

Date: 2026-09-20. Continues local commit `d2106d9` on
`research/slanted-cone-charge`, following the zero-tail and multi-route
sessions. The user authorized continuation with subagents and publication to
GitHub. No merge, paid model/JEV request, or claim-ledger change.

## Goal and setup

Work the three outstanding obligations: spatial charge on inherited cuts,
entry into the physical right renewal invariant, and generalisation beyond
alternation. A fourth derivation explored finite-support decreasing ranks;
a fifth agent independently audited the arguments and certificates. Agreement
is not evidence. Acceptance uses exact integer identities, complete finite
dependency cones, independent full-row replays, and the stated handwritten
arguments. None is a proof-assistant formalization.

The baseline repository gate passed eight stages with three expected skips:
the two absent canonical bitstreams and absent SAT/DRAT toolchain.

## An exact reduction of persistent nonentry

For the forced-centre right return `H=R_1 o R_0`, the sixteen-bit universal
invariant `K` contains 13,526 states. Its universal attraction basin contains
21,457 states, with a checked integer rank proving entry within twelve
macrosteps for every outer tail.

Outside `K`, the only cyclic components have sizes 9, 29, and 2,445. The two
small components disappear under exact predecessor lifts:

| Base component | Lift width | Lift states | Maximum consecutive observations |
|---:|---:|---:|---:|
| 9 | 22 | 576 | 10 |
| 29 | 18 | 116 | 16 |

Both lifted graphs are acyclic; all exterior choices are included. They
therefore exclude infinite physical residence even for infinite right tails.
The remaining condensation graph has a weighted path bound of 100. Thus:

> Every actual forced-right trajectory either enters `K`, or remains forever
> in the unique 2,445-state component `Q` from some macrotime at most 100.

This is not a bound on eventual `K` entry time. It does not exclude finite
right support persisting in `Q`. The projected `Q` graph has two distinct
equal-length output loops and hence aperiodic graph paths, so declaring all
nonentry output eventually periodic would be false. Those paths have not been
shown to share coherent infinite predecessors.

See [right-invariant-entry.md](../theory/right-invariant-entry.md).

## A stronger transport obstruction, with its family precisely limited

The proposed spatial inequality is

```text
A * slab_initial_weight - 7 >= Phi(next cut) - Phi(current cut).
```

Consider `Phi` consisting of nonnegative width-three temporal densities on
the two adjacent cut diagonals, retaining depth and time parity, plus an
arbitrary function of their bottom two and top eight cells. Forty-five actual
finite full rows provide a positive integer circulation refuting this family
for every finite `A`. All have zero slab weight and a right half initially
in a universal renewal entry cylinder. Their weighted endpoint features
cancel, while their density histogram drift is componentwise nonnegative.
Summing the hypothesized inequalities gives a contradiction. The verifier
uses exact integers and full Rule-30 replay; no optimizer is needed.

A telescoping gauge also excludes signed width-two densities whose global
bound is supplied by the specified local terminal flux. This corollary does
not cover arbitrary signed densities, order-sensitive cut functionals, or
potentials bounded using the full physical cone.

Full cuts need not concatenate for this family-level obstruction: every
individual physical edge already forces its inequality. They would need to
concatenate to claim an actual orbit. The distinction is retained throughout.

See [weighted-renewal-transport.md](../theory/weighted-renewal-transport.md).

## Local additive decreasing ranks also have a bulk obstruction

For range `r<=8`, consider

```text
E(n,z) = a*bit_length(n) + sum_i phi(n_i,...,n_(i+r-1)) + Psi(n,z),
phi(0)=0,   |Psi|<=C.
```

No such lower-bounded rank can decrease by a uniform positive amount on
every row passing a fixed finite physical alternation test. Signed `phi` and
arbitrary uniformly bounded corrections are allowed. Arbitrarily distant
periodic bulk insertions force all periodic bulk energy changes nonpositive;
local left-permutivity and Bernoulli averaging force them all to vanish.
Exact modular ranks, matched by an explicit kernel, then classify conserved
densities through range eight as constants plus spatial coboundaries. With
`phi(0)=0`, this reduces `E` to `a*bit_length+O(1)`. Lower boundedness forces
`a>=0`, contradicting uniform descent on arbitrarily long finite physical
alternating prefixes whose left length grows by two per macrostep.

The obstruction also holds conditional on the right renewal invariant:
fix right integer 167 and use triangular left inversion to construct arbitrary
finite alternating prefixes. It does not cover globally extendable-survivor
constraints, unbounded corrections, arbitrary range, or nonuniform descent.
The audit independently reproduced all modular ranks with a second prime.

See [zero-tail-rank-followup.md](../theory/zero-tail-rank-followup.md).

## An all-history phase encoding, and a period-three obstruction

Let `R_b` be the right-half map under centre bit `b`. For every `a>=0`,

```text
bit0(R_1 R_0^a(x)) = 1 - OR(x_0,...,x_(a+1)).
```

The proof follows the nearest one as it moves toward and then latches at the
boundary under centre zero. Sample immediately after each centre-one update.
The resulting event word avoids `11` for arbitrary centre histories with
infinitely many ones, including consecutive one events.

The event bit alone loses intermediate observations. If `a` is the number
of zeros before the next one event, use
`lambda=min(first_one_index(x),a+2)`. The right bit after `h<=a` zero steps
is `1[lambda<=h]`, and the next event bit is `1[lambda=a+2]`. These threshold
labels, cyclic event-gap order, and their seam condition give a finite
necessary input alphabet for every fixed nonconstant periodic centre word.
They recover the hidden nearest-right phases and the corresponding required
left bits. Shared predecessors still need to be retained across blocks.

The alternating bounded-zero-run rank does not generalise: for forced centre
`001`, the return `R_1 R_0^2` has the universally closed low-four-bit set
`{2,4,6,8,10,12,14}`. The finite right half zero has sampled states
`0 -> 1 -> 12` and sampled output zero forever thereafter. This is a forced
right-half example, not an alternating or period-three full finite orbit.

See [phase-word-transport.md](../theory/phase-word-transport.md).

## Interpretation and next obligation

Period two remains unexcluded. Even its exclusion would be an intermediate
step toward Prize 1, which requires excluding every eventual period for the
single-cell seed. No prize grade changes or submission claims are warranted.

The useful continuation is now more specific:

1. Construct a signed, order-sensitive or cone-conditioned potential on
   growing cuts, with a uniform lower bound on reachable terminal values
   and a uniform upper bound on permitted initial values. It may store
   unbounded intermediate credit. The refuted local density families do not
   exhaust this possibility.
2. Use the finite right tail to rule out persistent residence in `Q`, or prove
   left-weight growth for that exceptional boundary class separately. Entry
   into `K` alone still would not complete transport inside `K`.
3. For arbitrary periods, use ordered event gaps and threshold labels while
   preserving shared predecessors. No universal zero-run rank is available
   from event sampling alone.

A sound finite-state charge graph would imply a uniform bound on consecutive
empty slabs. A growing-cut potential with one-sided bounds need not imply
that stronger statement: earlier weighted slabs can supply large credit.
This is why another failed bounded feature search cannot settle the broader
transport question. See the
[independent scope audit](../theory/transport-entry-hostile-review.md).

## Reproduction and verification

```bash
python experiments/right_invariant_entry.py
python experiments/weighted_renewal_transport.py
python experiments/zero_tail_rank_followup.py
python experiments/phase_word_transport.py
python tools/verify_all.py
git diff --check
```

The lead reran all four focused reproducers successfully. The independent
audit separately checked the small-component lifts and condensation budget,
all 45 weighted physical witnesses, the rank matrices with a second prime,
the event identities, threshold equations, and period-three invariant.
The final serialized repository gate passed eight stages, skipped the same
three absent inputs/toolchain stages, and had zero failures. All four new
Python files compile. `git diff --check` passes. The complete claim ledger is
byte-for-byte unchanged from `d2106d9`; the skipped stages establish nothing
about the unavailable canonical bitstreams or DRAT toolchain.

## Publication method

The command-line GitHub push lacked credentials. Publication therefore uses
the connected GitHub app to create a snapshot commit on the research branch,
with remote main `b450406deace9f8adabc7b5b95d45b85e2c32432` as parent. It
includes the previously local research and this continuation. The original
local research history is retained; the snapshot commit message records the
source local commit. Before exposing the branch, require equality between
the uploaded Git tree and the final local commit tree. This checks every
tracked path, file mode, and byte, while allowing the two commit histories to
have different commit hashes. Main is not modified and no merge is requested.
