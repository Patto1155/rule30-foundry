# Audit of spatial transport, invariant entry, and phase generalisation

This note separates checked implications from missing global arguments. It
does not establish a new period exclusion or change prize grades. Agreement
among agents is not evidence: algebraic arguments and independent elementary
rule evaluations are identified separately below.

## A bounded cut graph asks for more than the endpoint condition

Suppose a sound finite directed graph has `n` vertices. Every edge carries
nonnegative integer initial weight `w`, and positive slab demand `b`. The
existing zero-weight-cycle criterion is correct: a potential and some finite
coefficient `A` exist exactly when the zero-weight subgraph is acyclic.

This has an additional necessary consequence that is easy to overlook:
**every path contains at most `n-1` consecutive zero-weight edges**. A longer
path repeats a vertex, producing a forbidden zero-weight cycle. If each edge
represents seven fresh initial cells, the graph therefore imposes a uniform
bound on the number of consecutive empty aligned seven-cell slabs, regardless
of total initial weight or depth. Soundness means every physical witness must
obey that bound. Conversely, a genuine all-parameter family with arbitrarily
many consecutive empty slabs would refute every such finite graph certificate,
not merely one selected list of endpoint features. No such family is proved
here; the already recorded seven-zero-cell witnesses alone are insufficient.

The desired inequality on exact growing cuts is weaker:

```text
A*w - b >= Phi(next cut) - Phi(current cut).
```

It needs a uniform **lower** bound on reachable terminal values and a uniform
**upper** bound only on permitted initial values. It does not require an upper
bound on all intermediate values. A run of `m` empty slabs decreases `Phi`
by at least `mb`, so its incoming cut needs at least `mb` credit above the
lower bound. Earlier weighted slabs can supply arbitrarily large credit.
Thus long empty runs, if proved to exist, need not defeat this more general
certificate. Bounded endpoint summaries discard precisely that possible
accumulated credit. Failure of finite-state searches cannot be promoted to
failure of all transported potentials or well-founded ranks.

## Quantifiers that every continuation must preserve

* A graph whose states are all exact cuts at one finite depth is finite, but
  its cardinality can grow with depth. Coefficients or endpoint bounds that
  grow with this cardinality do not yield a uniform exclusion.
* Universal closure under all exterior inputs proves forward invariance.
  It does not prove attraction of all finite right halves. A projected cycle
  avoiding the invariant need not have a coherent infinite predecessor.
* Arbitrarily long finite-support witnesses need not converge to a
  finite-support witness: that class is not closed. A compactness argument
  must explicitly permit infinite-support limits or prove a uniform support
  bound separately.
* When concatenating projected edges to disprove a feature potential, the
  full witnesses need not concatenate: each individual inequality would
  already force a contradiction after summation. When using those same edges
  to claim an actual orbit or nonentry mechanism, shared predecessors and
  full seam agreement are indispensable.
* A periodic-word theorem must retain the whole phase word. Bounds may
  depend on the word, but checking every word up to a finite length does not
  establish a construction for arbitrary periods.

## Persistent nonentry has only one surviving width-sixteen cyclic component

For the existing right-return invariant `K`, reconstruct the entire
eighteen-bit dependency cone by the elementary Rule-30 table, then keep the
low sixteen cells. This gives all four possible outer-bit successors of each
sixteen-bit state. An independent NumPy implementation, without importing the
producer's packed map or graph routines, reproduced `|K|=13526`. Removing
`K` and taking strongly connected components leaves exactly three cyclic
components, of sizes `9`, `29`, and `2445`. Explicit examination of diagonal
edges finds no additional singleton self-loop component.

Independently refining the two small components by their low-bit output and
the set of successor equivalence classes gives deterministic labelled cycles:

| Component size | Quotient size | Cyclic output, up to phase |
|---:|---:|---|
| 9 | 4 | `0001` |
| 29 | 16 | `0000100001010001` |

Every internal edge increments the corresponding phase by one. This is a
finite exhaustive property of the overapproximating graph; no assumption
about the realizability of all graph paths is needed for its implication.

An actual infinite trajectory avoiding `K` eventually remains in one SCC:
the finite condensation graph is acyclic, so it changes components only
finitely many times. If it remains in either small component, its sampled
even-right word is eventually periodic. Under a genuinely alternating centre
and finite left support, the proved periodic-boundary weight lemma rules that
out after restarting at an even time past the transient. Consequently a
hypothetical finite-support eternally alternating orbit either enters `K` or
eventually stays in the `2445`-state component.

The conclusion is a necessary reduction, not attraction to `K`. The large
component may contain paths with periodic output too, but its unrestricted
output is not forced periodic by this argument. Conversely, no path in any
component is asserted to have a globally coherent, finite right predecessor.
The small SCCs themselves need not be forward invariant: exits cease along
the particular infinite path after its final component transition.

A stronger shared-predecessor check eliminates both small components without
using finite left support. Lift a component `C` to all width-`w` states whose
low sixteen bits lie in `C`, and retain only transitions whose targets remain
in that lifted set. An actual trajectory staying in `C` would give an
infinite path in the lift. A separate cell-list truth-table implementation and
topological sort reproduce:

| Base component | Width | Lift vertices | Distinct internal edges | Longest path, edges |
|---:|---:|---:|---:|---:|
| 9 | 22 | 576 | 672 | 9 |
| 29 | 18 | 116 | 160 | 15 |

Both lifts are acyclic. Thus the small components cannot contain an actual
right trajectory forever, even with an infinite initial right tail. The
stronger necessary conclusion is: every actual right trajectory either enters
`K` or eventually remains in the `2445`-state width-sixteen component. This
still gives neither attraction to `K` nor a coherent eternal path in the large
component. Reproduce with `python experiments/right_invariant_entry.py`.

The component elimination also gives a quantitative outside-region bound.
Remove `K` and the large component from the condensation DAG, assign weight
one to each singleton, ten to the size-nine component and sixteen to the
size-twenty-nine component. These weights count permitted consecutive
observations, one more than the lifted longest edge paths. An independent
weighted DAG calculation on 49,529 nodes gives maximum weight 100. Thus an
actual trajectory avoiding both sets can have at most 100 observations
there: it hits `K` or the large component by index 100. A trajectory that
never enters `K` must reach the large component by then and remain forever;
leaving could never return by the SCC DAG and would force subsequent entry
to `K`. This does not bound time spent in the large component before a later
entry into `K`.

## Event sampling extends no-11, but not the alternating zero-run rank

For the right-half update `R_b`, sample immediately after centre-one events.
Between consecutive such events the applied centre boundary is `0^a 1` for
some `a>=0`. Under `R_0`, the OR of the first `j` right cells equals the OR
of the first `j+1` old right cells. One way to see this without Boolean
expansion is to follow the first one: if its index is positive it moves one
place toward the boundary, and if it is zero the new boundary cell remains
one. Therefore

```text
bit0(R_1 R_0^a(x)) = NOT OR(x_0,...,x_(a+1)).
```

The next event-sampled bit is zero whenever the present bit is one. This
proves no consecutive ones for arbitrary event spacing, including all
periodic centre words with at least one one. The bit alone is not a complete
state: the length `a` and the full needed prefix OR both matter. Ordered
event gaps cannot be replaced by their counts or mean.

Independent cell-by-cell Rule-30 replays checked all 2,044 finite dependency
cones for `a=0,...,8`; those finite checks validate the conventions, while
the preceding first-one argument proves the arbitrary-`a` identity.

For the forced centre word `001`, the return `R_1 R_0 R_0` has a universal
even-valued low-four-bit invariant `S={2,4,6,8,10,12,14}`. Independent
truth-table evaluation of all 56 states plus three exterior bits reproduces

```text
2 -> {2,10,14}, 4 -> {4,12}, 6 -> {4,14}, 8 -> {2,10},
10 -> {14}, 12 -> {8,12}, 14 -> {4,12}.
```

The finite initial right half zero has sampled states `0 -> 1 -> 12`, hence
enters `S`. Its sampled output is identically zero thereafter. Consequently
the alternating-case bound on zero runs does not extend to arbitrary event
spacing, even for actual finite right initial halves under prescribed centre
forcing. This says nothing about compatibility with a finite left half: the
centre is an externally prescribed boundary in this example, not a proved
full finite Rule-30 orbit.

The threshold encoding in `phase-word-transport.md` correctly retains the
hidden phases: `lambda=min(first-one index,a+2)` gives the right bit after
`h<=a` zero steps as `1[lambda<=h]`, while the next event bit is
`1[lambda=a+2]`. The seam condition on successive labels is necessary but
does not supply shared right predecessors. The nearest-left equations in
that note follow directly from the centre equation, including the empty
zero-phase range when `a=0`; no alternating-specific mask is applied to a
general zero run.

## Local additive ranks: the bulk obstruction survives independent audit

The argument in `zero-tail-rank-followup.md` concerns

```text
E(n,z) = a*bit_length(n) + sum_(i>=0) phi(n_i,...,n_(i+r-1)) + Psi(n,z),
phi(0)=0,    |Psi| <= C,
```

with a uniform positive decrease on every finite row passing a fixed finite
physical alternation test. That last quantifier is essential. It permits
arbitrarily many copies of any chosen periodic bulk word to be inserted far
outside the test's dependency cone. Bulk energy change therefore cannot be
positive. For ring length at least `r+4`, each output `r`-block under two
Rule-30 updates is uniformly distributed under independent uniform input:
fix the four rightmost cone inputs and solve the remaining bits successively
by left permutivity. This is a local calculation, not an assertion that the
finite-ring global map is bijective. Averaging nonpositive changes gives zero,
forcing each periodic bulk change to vanish.

Independent cell-list updates and pure-Python modular elimination with the
different prime `1000003` reproduce conservation-matrix ranks
`1,2,4,8,16,32,64,128` at ranges one through eight, after respectively
`2,8,18,64,138,310,558,1132` ring equations. The explicit constant-plus-gradient
kernel has the complementary dimension, so these ranks certify the real
kernel as well. `phi(0)=0` removes the constant; the gradient sum telescopes
to a bounded endpoint term. Consequently `E=a*bit_length+O(1)`, with `a>=0`
forced by lower boundedness. Arbitrarily long physical ether truncations
contradict a uniform positive decrease because their left length grows by two
per macrostep.

This excludes the stated signed local-density rank family through range
eight. It does not cover a decrease valid only on globally extendable
survivors, an unbounded correction, arbitrary range, or a nonuniform decrease.
Reproduce with `python experiments/zero_tail_rank_followup.py`.

The same rank obstruction also applies after entry into the right renewal
invariant. Fix right integer `167`; for any finite horizon, initial left
bits can be chosen successively to enforce centre alternation. The extreme
initial input at position `-t` toggles centre time `t` by iterated left
permutivity and affects no earlier time. This constructs arbitrarily long
finite physical prefixes while the right half stays in its invariant.
Independent full-row replay confirms the displayed witnesses of lengths
8, 16, and 24 with left integers `86`, `33110`, and `7110998` respectively.
Arbitrary distant bulk still preserves any fixed finite test and the right
entry condition. The ether need not itself enter that invariant for the
conditional rank obstruction to hold.

## Weighted cut-wide densities: exact circulation, with signs stated

In the original convention `A*w-7 >= Phi(next)-Phi(current)`, consider `Phi`
equal to a sum of **nonnegative** local densities over adjacent-diagonal
temporal windows, plus an arbitrary function of the two bottom and eight top
bits of each diagonal and depth parity. The producer uses `Psi=-Phi`, so its
equivalent credit densities are nonpositive and the inequality is
`A*w-7+Psi(next)-Psi(current)>=0`.

Independent full finite-row evolution using a dictionary of physical
coordinates verified the initial 45-row width-three certificate. Every row
alternates through its end depth, has an empty seven-cell initial slab, and
has a right half in one of the universal entry cylinders. Recomputed exact
positive integer multipliers balance every endpoint feature. The 88
histogram coordinates have componentwise nonnegative total drift, with 41
strictly positive coordinates, and total drift equals seven times total edge
multiplicity. Thus weighted `Delta Phi>=0`, while the hypothesized zero-slab
inequalities require a strictly negative total. The coefficient `A` cancels
because all slab weights are zero. Physical witnesses need not concatenate
at their full cuts for this family-level contradiction.

There is a sound signed-density corollary. Write the pair-symbol word on a
cut as `v_0...v_(q-1)`, let `S_t=(v_t,v_(t+1))`, and suppose a signed width-two
credit density `F` has a local flux `chi` satisfying

```text
F(S_t) + chi(S_(t+1)) - chi(S_t) <= 0
```

on every actual three-symbol window, retaining the stated parity labels if
used. Summing converts the signed width-two density exactly into a
nonpositive width-three density plus
`F(S_last)+chi(S_first)-chi(S_last)`. Those extra terms are determined by the
declared endpoint feature. The width-three obstruction therefore also rules
out this locally bounded signed width-two family. It does not refute an
arbitrary signed density with a different proof of global endpoint bounds.
The permanent 45-row certificate is embedded in
`experiments/weighted_renewal_transport.py`; its default command replays the
full certificate without an optimizer. See `weighted-renewal-transport.md`.
