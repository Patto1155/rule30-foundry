# Right-invariant entry: a uniform exceptional-core reduction

## Scope

Use the forced-centre return `H=R_1 o R_0` and nearest-right-cell-first
encoding from `right-coherence-followup.md`. Write `v_k=H^k(x) mod 2^16`.
The initial right half may have finite or infinite support.

The new statement is a dichotomy, **not** universal entry for finite right
halves:

> Every actual forced-right trajectory either enters the existing invariant
> `K`, or there is a macrotime `k_0<=100` such that `v_k` belongs to one explicit
> 2,445-state exceptional component `Q` for every `k>=k_0`.

The second alternative has not been excluded for finite right halves. Nor is
it asserted that every abstract path in `Q` has a physical infinite extension.
The reduction does not use a finite-left hypothesis or prove left-weight
growth. It identifies exactly which recurrent part of this finite graph still
needs such an argument.

## An expanded, universally certified entry basin

The universal sixteen-cell graph has the four successors

```text
E(v,a) = H(v + 2^16 a) mod 2^16,        a=0,1,2,3.
```

Every actual right trajectory projects to a path, because two additional
right cells are precisely the dependency cone for two Rule-30 steps. The
existing `K` has 13,526 states and is closed under all four successors.

Set `B_0=K` and

```text
B_(r+1) = B_r union {v : E(v,a) in B_r for every a}.
```

This stabilizes after twelve rounds at a set `B` of 21,457 states. Give a
state its first-entry round `rho(v)`. Every edge out of `B\K` strictly
decreases `rho`; every edge from `K` stays in `K`. Thus every right half with
low sixteen bits in `B` reaches `K` in at most twelve macrosteps, regardless
of its entire outer tail. The verifier checks all these inequalities exactly.
This is a larger sufficient subclass than initial membership in `K`; it is
not an enumeration of successful finite trajectories.

## Three possible recurrent components before predecessor checks

Delete `K` and every incident edge. Exhaustive SCC decomposition of the
remaining graph gives exactly three cyclic components, of sizes

```text
9, 29, 2445.
```

There are no singleton self-loops. Here `Q` is the unique component of size
2,445, which makes its definition independent of implementation-specific
component numbering. A path that avoids `K` forever must eventually remain
inside one of these three components: the condensation graph is finite and
acyclic.

The small components have deterministic output phase quotients, even though
the underlying state graphs branch. Respectively, their output words would
be eventual repetitions, up to phase, of

```text
0001
0000100001010001.
```

For each component the verifier refines states by output and successor
classes. The resulting four- and sixteen-state quotients are directed
cycles, and every internal edge advances phase by one. This alone would let
the previous periodic-right-word lemma exclude these alternatives jointly
with finite left support. The following stronger argument eliminates them
without that hypothesis.

## Shared predecessors eliminate both small components

For a component `C` of sixteen-bit states and a larger width `L`, retain
**every** longer prefix whose low sixteen bits belong to `C`:

```text
V(C,L) = {v + 2^16 a : v in C, 0<=a<2^(L-16)}.
```

Connect `w` to each `H(w+2^L b) mod 2^L`, for all four `b`, when the successor
also belongs to `V(C,L)`. This stores the extra cells across time instead of
choosing them independently at each sixteen-cell transition. It still
contains every physical trajectory while that trajectory stays in `C`.

The exact certificates are:

| Sixteen-cell component | Lift width | Lift states | Longest internal path | Maximum consecutive observations |
|---|---:|---:|---:|---:|
| Size 9 | 22 | 576 | 9 edges | 10 |
| Size 29 | 18 | 116 | 15 edges | 16 |

Both lifted graphs are acyclic. A topological rank strictly increases on
every retained edge, with the displayed maximum path lengths. Therefore
neither small component can contain an actual right trajectory forever,
even when the outer initial support is infinite. Their apparent periodic
recurrence at width sixteen was caused by forgetting shared predecessors.

This is a finite exact graph proof with all exterior choices included, not a
claim inferred from observing finitely many trajectories. The lifted
transitions are separately evaluated from the elementary Rule-30 truth table
on all 2,304 and 464 dependency cones.

## Uniform time bound for the exceptional alternative

Remove `Q` as well as `K`, and work in the remaining condensation DAG. Give
every singleton component weight one; give the two small components weights
ten and sixteen, their proved maximum numbers of consecutive observations.
Every actual path can visit each condensation vertex at most once. The
longest weighted path has weight 100. The reproducer verifies the resulting
integer difference inequalities on all condensed edges.

It follows that an actual trajectory cannot spend more than 100 consecutive
observations outside `K union Q`. In particular it reaches `K union Q` by
macrotime 100. If it enters `Q` and later leaves without entering `K`, it
cannot return to `Q`, since that would contradict the SCC decomposition. It
then reaches `K` within the same outside-core budget. Consequently an actual
trajectory that never enters `K` must remain in `Q` forever from some time
at most 100, proving the stated dichotomy.

**This is not a bound of 100 on entry into `K`.** A trajectory can spend an
arbitrarily unspecified duration in `Q` before eventually leaving for `K`.
The earlier finite-right example `x=0`, whose first `K` entry is at macrotime
147, is fully consistent with the dichotomy.

## Why the exceptional component cannot simply be called periodic

The output bisimulation quotient of `Q` still has 2,092 classes and has
branching. More concretely, the verifier supplies two closed walks based at
state 4,128. One has length two and output `01`; the other has length 37 and
output

```text
0101010101010101010101010001000010101.
```

Repeating the first walk 37 times or the second walk twice gives different
74-letter words with the same starting and ending graph state. Concatenate
these two loops according to any non-eventually-periodic binary selector.
The resulting **graph path** has non-eventually-periodic output: if the
output were eventually periodic, its partition into fixed 74-letter blocks
would be eventually periodic, and the two distinct block values would make
the selector eventually periodic as well.

All walk edges are checked. This refutes the inference that every infinite
path in this sixteen-cell nonentry graph has eventually periodic output.
It does **not** construct a coherent infinite right half, a finite right
counterexample to entry, or a finite full row with alternating centre.
The exterior predecessors at successive steps are still free in this graph.

## Remaining obligation and reproduction

The entry route is now reduced to ruling out an ordinary finite integer
whose actual `H` trajectory stays forever in `Q` after some `k_0<=100`.
Alternatively, weighted left transport can treat eventual `K` and eventual
`Q` as separate boundary classes; only the first class is licensed to use the
three/five-gap renewal restriction. The zero tail has not yet been used to
exclude the exceptional class.

```bash
python experiments/right_invariant_entry.py
```

The script rebuilds the graph and `K`, certifies the entry rank, identifies
all cyclic SCCs, checks phase quotients, verifies both lifted DAGs with
independent elementary-rule cones, checks the weighted condensation budget,
and returns both exceptional closed walks. It requires NumPy and SciPy.
No prize exclusion or all-finite-right entry theorem follows from these
certificates alone.
