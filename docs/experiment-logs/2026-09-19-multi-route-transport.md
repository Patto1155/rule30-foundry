# Multi-route continuation: conditional renewal and exact transport barriers

Date: 2026-09-19. Starts at local `5874e33` on
`research/slanted-cone-charge`, following `f7812ab` and
`2026-09-19-zero-tail-composition.md`. Six subagents investigated separate
obligations; the lead read the arguments and reran every new reproducer.
No push, merge, paid model/JEV request, or claim-ledger grade change.

## Goal and setup

Pursue all concrete leads from the previous session: weighted phase-slip
transport, stronger physical right coherence, weaker initial-weight growth,
coefficient-independent potential falsification, and the arbitrary-period
bridge. A sixth agent audited the mathematical scopes and independently
replayed witnesses. Agent agreement is not evidence; local Boolean identities,
exact finite invariance, full-row replay, and the stated handwritten proofs are.

The baseline gate passed eight stages with three expected skips: the two
missing canonical bitstreams and the absent SAT/DRAT toolchain.

## Results

### A conditional right-boundary invariant supplies a narrower language

For the forced centre `01`, the three low-eight right prefixes with integers
`167,168,170` guarantee entry into a forward-invariant set, regardless of the
entire remaining right tail. The universal closure on sixteen right bits has
1,452 states and 5,808 edges, all inside the predicate
`right even OR right mod 8=7`. All 262,144 eighteen-bit dependency cones were
checked against a separate elementary-rule evaluator. Closure under every
possible next pair of outer bits proves forward invariance by induction.

After entry, successive ones in the sampled even-right word are exactly three
or five macrosteps apart. This is a necessary renewal relaxation, not a complete
trace language. Its first seven reconstructed initial-left cells have weight
at least two. **Neither entry for every finite right half nor subsequent
unbounded left weight has been proved.**

See `../theory/right-coherence-followup.md`.

### A second-slab identity is positive, but the third blocks simple transport

On the gap-three/five language, let `w=s_(k+2)..s_(k+7)` and let `F` be one
on `01000,10000,00010`, zero on the other permitted five-letter words. Then

```text
Delta14(k) = 1[w=100001] + F(w[1:]) - F(w[:-1]).
```

Thus every temporal prefix at depth fourteen has excess at least minus one,
with one unit of nonnegative production per complete five-gap in the displayed
range. The exhaustive local table proves an all-length telescoping identity.
Separately, the second seven-cell initial slab has weight at least one under
the weaker no-`11` assumption, with a four-case algebraic proof.

However, `s=(10000)^infinity` has depth-twenty-one excess minus two per five
temporal phases. The narrower language also admits an empty fourth initial
slab. These are exact obstructions to propagating temporal excess alone or
resetting the cut. **No spatial weighted-transport inequality follows.**

See `../theory/phase-slip-transport-followup.md`.

### Increasing the coefficient cannot repair two endpoint-only families

Two actual finite full rows give a zero-initial-weight cycle after projecting
each cut to depth parity plus the bottom two and top two bits of both
diagonals. Four actual rows give another cycle when the top window is enlarged
to eight bits. The inequalities would sum to `-14>=0` and `-28>=0`, respectively,
for every finite coefficient `A`. Independent full-row truth-table replay
verifies every centre trace, zero slab weight, and retained feature.

The complete cuts differ at cycle seams. These are exact obstructions to the
declared feature potentials, not infinite physical orbits or obstructions to
potentials carrying cut-wide information and accumulated credit.

See `../theory/coherent-potential-followup.md`.

### Arbitrary coefficients reduce to a precise finite-graph question

For any sound finite cut graph with nonnegative integer initial weights `w`
and demands `d`, a finite coefficient and bounded potential exist if and only
if there is no reachable cycle with zero total `w` and positive total `d`.
For `n` vertices, `A=n*dmax` and endpoint allowance `(n-1)*dmax` suffice when
the condition holds; replace `A` by at least one if needed. The proof uses
simple-cycle decomposition and shortest-path difference constraints.

This standard graph argument eliminates coefficient tuning inside a specified
finite abstraction. It does not supply a finite sound abstraction of the
growing cuts. A cycle of genuine projected edges already refutes that feature
family; absence of cycles in a sampled graph does not certify all-depth closure.

### Periodic right forcing has an explicit quantitative bound

If the sampled even-right word has period `p`, every `4p` reconstructed
initial-left cells contain a one. A hypothetical zero block of that length
would give two temporal columns zero for a full common period; zero-pair
propagation would then contradict the alternating centre.

Periodic extension and locality give the finite-prefix version
`W_(2M)>=floor(M/(2p))` whenever the first `M` right observations have period
`p`. Therefore an initial left weight `w` permits a repeated prefix `u^K` only
when `K<2(w+1)`. This is a quantitative periodic-boundary subclass, not a bound
on arbitrary aperiodic forcing. Repetitions at varying later start times do
not suffice because the restarted left weight can grow.

See `../theory/weak-weight-growth-followup.md`.

### The constant endpoint is audited; a local all-period reset is obstructed

A self-contained proof verifies the known adjacent-periodic-column exclusion
for nonzero finite configurations. Common-onset sideways periodicity and
finite support over one period yield a far-left zero pair, which propagates
zero throughout the row, contradicting persistence of its leftmost one.
An eventually zero centre makes its right neighbour Boolean nondecreasing
and eventually constant; an eventually one centre forces its left neighbour
zero. This supplies the period-one endpoint independently of the external
preprint. No novelty or classification/horizon audit is claimed.

For every nonconstant period word `u` of length `p`, an explicit five-cell
open-strip construction gives two distinct `3p`-step loops with the same
initial/terminal strip state. An aperiodic loop selector produces an aperiodic
left neighbour while preserving the periodic centre. This all-parameter
construction prevents using phase labels and free-boundary local resets as
the missing all-period bridge. It asserts no finite full-row counterexample.

See `../theory/all-period-bridge-followup.md` and
`../theory/multi-route-hostile-review.md`.

## Interpretation and next obligations

No nonconstant period has been excluded, and no prize result follows. The
strongest positive continuation is the conditional renewal invariant coupled
to the second-slab identity. To turn that into an exclusion requires two
separate steps, neither supplied here:

1. Prove weighted spatial transport on the renewal class while retaining the
   inherited cut and initial-row weight, with any finite coefficient and
   uniform one-sided endpoint bounds. Temporal excess and the tested small
   endpoint summaries are insufficient.
2. Prove eventual invariant entry for every relevant finite right half, or
   handle persistent nonentry by a separate finite-support obstruction.

The weaker no-`11` charge route remains open and avoids the additional entry
obligation if it can be proved directly. The new right invariant is therefore
a parallel simplification, not a replacement assumed to cover every case.
For arbitrary period words, retain whole phase and predecessor information;
use the zero-weight positive-demand cycle criterion only after sound graph
coverage is established. Do not spend further effort enlarging periodic-word
tables or the two endpoint-feature families already refuted.

## Reproduction and verification

```bash
python experiments/right_coherence_followup.py
python experiments/phase_slip_transport.py
python experiments/coherent_potential_followup.py
python experiments/weak_weight_growth.py
python experiments/all_period_bridge_followup.py
python tools/verify_all.py
git diff --check
```

The lead reran all five focused commands successfully. The weak-growth audit
covers 254 periodic words with independent forward cones and locality checks.
The all-period constructor audit covers 2,026 nonconstant words and 329,832
interior truth-table updates. Those finite checks validate the handwritten
constructions; their counts are not substituted for all-parameter proofs.
The independent review additionally replayed all closure edges, both feature
cycles, the second-slab identity, and the third-slab obstruction. These are
ordinary mathematical arguments with executable checks, not proof-assistant
formalizations.

The final serialized repository gate passed eight stages, with the same three
expected skips and zero failures. All five new Python files compile;
`git diff --check` passes. The full claim-ledger file was compared against
starting commit `5874e33` and is byte-for-byte unchanged. Work is retained on
the local research branch; no remote write was performed.
