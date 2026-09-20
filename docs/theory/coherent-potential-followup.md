# Coherent cut potentials: an obstruction independent of the charge coefficient

This note excludes two finite endpoint-feature potential families. It does
not exclude a potential on full cuts, or establish period-two exclusion.
The witnesses are actual finite full rows with an alternating centre for
their stated horizons. Physical right coherence therefore does not remove
these particular feature-family obstructions.

## Arbitrary coefficients on a finite graph

Consider the proposed inequality on a directed graph of cut states,

```text
A w(edge) - 7 >= Phi(target) - Phi(source),
```

where `w` is a nonnegative integer. Some finite `A>0` and potential `Phi`
exist exactly when the subgraph of zero-weight edges is acyclic.

Necessity follows by summing around a zero-weight cycle of length `r`:
the result would be `-7r >= 0`. Conversely, if the graph has `V` vertices
and no zero-weight cycle, every simple cycle has at most `V` edges and total
weight at least one. Taking `A=7V` makes every cycle charge nonnegative.
The usual difference constraints then have a solution (shortest-path
distances after adjoining a zero-cost source give a potential).

This finite-graph statement does not establish that a sampled graph contains
every exact transition. A cycle of genuine sampled edges is already a valid
obstruction to the selected features; an acyclic zero-weight sampled graph
does not give an all-depth certificate. A finite *sound* overapproximation
without zero-weight cycles would give such a certificate, provided its
states and transitions cover the initial and terminal cuts correctly.

## Two physically coherent zero-weight edges

Use the exact characteristic cuts from `slanted-cone-geometry.md`,
`(D_(q-1),D_q)`, with each diagonal ordered from initial time toward the
centre boundary. Let the feature key be

```text
(q mod 2,
 bottom two bits of D_(q-1), top two bits of D_(q-1),
 bottom two bits of D_q,     top two bits of D_q).
```

Bit tuples are packed with their first temporal entry in the least
significant position. Two exact full-row witnesses give

| Slab | Source key | Target key | Initial slab weight |
|---|---|---|---:|
| `98 -> 105` | `(0,0,3,0,1)` | `(1,0,1,0,2)` | 0 |
| `133 -> 140` | `(1,0,1,0,2)` | `(0,0,3,0,1)` | 0 |

Their `(left,right)` initial integers, with nearest-neighbour bit first, are

```text
(16661046055632894622381385253,
 2587743454036826996634021153954)

(560767664687031763042728977378771580502,
 46182698749820787015413023666985299359229).
```

The centre initially equals zero. Set every cell not specified by those
finite integers to zero. Direct full-row Rule-30 truth-table evolution
gives centre bit `t mod 2` through time 105 and 140 respectively. In each
case the seven initial cells emitted by the displayed slab are zero.
The proposed inequalities would sum to `-14 >= 0`, regardless of `A`.

The entire cuts do **not** agree at either seam: their diagonal lengths are
`(105,106)` versus `(133,134)`, and `(140,141)` versus `(98,99)`.
This is an exact obstruction to a function of the displayed features, not
an infinite physical alternating orbit assembled from the witnesses.

## Keeping eight top bits still fails

A second certificate retains the same bottom two bits of each diagonal but
the top **eight** bits. Four physically coherent zero-weight slabs close:

```text
(1,0,85,0,170) -> (0,0,170,0,102) -> (1,0,85,0,213)
               -> (0,0,210,0,81) -> (1,0,85,0,170).
```

Their slab depths are `119->126`, `140->147`, `133->140`, and `126->133`.
Full initial rows are hardcoded in `REFINED_CYCLE` in the reproducer. Every
centre alternates through its edge's end depth. The cycle requires
`-28 >= 0`, again independently of `A`. Thus merely enlarging the retained
top boundary to eight bits cannot repair these bottom-two-bit features.

This does not rule out deeper initial-side predecessor information, a
population or phase-slip density along the complete cut, an accumulated
credit state, or a differently shaped cut. The new obstruction identifies
information that these particular endpoint-only potentials lose. It gives
no theorem that any one of those refinements succeeds.

## Reproduction and independence

```bash
python experiments/coherent_potential_followup.py
python experiments/coherent_potential_followup.py --discover
```

The default command independently evolves shrinking full-row cones using
the elementary Rule-30 truth table. It checks the actual centre trace,
zero slab weights, all retained feature bits, and exact cycle cancellation.
It does not use the sideways constructor, integer left-transfer map, or
prescribed-centre simulation during verification. A separate reviewer also
replayed the two-edge certificate with another direct truth-table loop.

The optional deterministic search follows physical forced-right histories,
constructs their left cones, and targets only zero-weight slab cycles. It
stops at a discovered cycle and verifies it by the independent full-row
routine. Discovery is not needed to reproduce either fixed certificate.
