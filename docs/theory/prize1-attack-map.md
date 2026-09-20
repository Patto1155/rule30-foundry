# Prize 1 attack map

Prize 1 asks whether the centre trace of the single-seed Rule 30 orbit is
eventually periodic.  A useful strengthening is to exclude eventual periodicity
for every nonzero finite configuration: restarting the orbit at the alleged
onset preserves finite support and removes the transient.

The central discipline is to distinguish a reformulation from a bridge.  “No
finite periodic fibre exists”, “every periodic fibre has infinite left
support”, and “the centre is not eventually periodic” have essentially the same
strength.  Progress needs either a proper subclass such as period two, or an
intermediate statement with an independently established endpoint.

## Known endpoint

Jen's Proposition 3 says that a finite Rule 30 orbit has at most one eventually
periodic spatial column.  Kopra's width-two theorem likewise excludes an
eventually periodic trace of two adjacent columns and records the width-one
case as open.  Therefore any theorem of the form

```text
periodic centre  =>  any second periodic column
```

finishes Prize 1.  It is a valid bridge only when its hypothesis is strengthened
or its conclusion is proved by a mechanism; stated without such a mechanism it
is vacuously equivalent to the prize if the prize is true.

## Period two: the clean termination theorem

At a centre-zero phase, encode the negative half-line as the ordinary integer

```text
n = sum_(j>=1) a(0,-j) 2^(j-1).
```

With prescribed centre boundary `b`, its autonomous update is

```text
T_b(n) = (n >> 1) XOR (n OR ((n << 1) OR b)).
```

For an alternating boundary `0101...`, put `G = T_1 o T_0`.  At every
intervening centre-one phase, the cell at `-1` must be one so that the centre
can return to zero.  Since

```text
bit_0(T_0(n)) = bit_1(n) XOR bit_0(n),
```

an alternating finite orbit must satisfy

```text
G^k(n) mod 4 in {1,2}  for every k >= 0.
```

This removes the right half completely.  Period two follows from one explicit
termination theorem:

> Every ordinary nonnegative integer eventually leaves `{n: n mod 4 in
> {1,2}}` under iteration by `G`.

Equivalently, the two-adic survivor set may exist, but it contains no binary
sequence that is eventually zero.  This distinction matters: proving the
survivor tree empty is probably false and is stronger than needed.

The original sharp **cone-charge lemma** is false. It proposed that passing
`m` left-only latches forced

```text
sum_(j=1..2m) e_j >= floor((m+3)/4) = ceil(m/4).
```

The weight-six integer `4329177217` passes 26 latches and refutes this bound.
Its inferred right word contains `11`, so it is not a physical alternating
orbit. See `alternating-right-boundary-coherence.md` and the
`2026-09-19-zero-tail-composition` experiment log.

A sufficient replacement is `A W_N-N >= -C` for every physically coherent
alternating cone, with any fixed finite `A>0,C`. The no-`11` right-word
relaxation is sufficient if it admits such a proof, but may be strengthened
when unphysical paths obstruct the certificate. Even a proved divergent
lower bound on `W_N` without a linear rate would exclude finite support.
Local discharge must retain the actual cut state and prove uniform endpoint
bounds; neither a finite exact encoder nor bounded two-sided cut credit is
required.

### Ranked proof mechanisms

| Route | Actual lemma | Exact falsifier / stop |
|---|---|---|
| Coherent cone charge / renewal | Prove `A W_N-N >= -C` for some finite constants on coherent cones, or another divergent initial-weight lower bound. | A coherent counterexample refutes a named bound; a feature-quotient negative cycle refutes only that potential family. The unrestricted sharp coefficient-eight claim is already false. |
| Zero-tail elimination | Assume all sufficiently high bits of `n` are zero; eliminate the triangular free bits only under that assumption until a contradiction repeats. | Satisfying zero-tail assignments at unbounded widths, or proof width growing with no shiftable clause. |
| Word/transducer rank | After stripping the universal new leading `11`, a finite-state or weighted potential strictly decreases along surviving transitions. | Two exact survivor words with the same proposed state and different required successors; a surviving rank cycle. |
| SAT proof-schema mining | Encode survival through the candidate horizon, generate checked UNSAT proofs, and identify clauses invariant under adding one spatial annulus. | A satisfying assignment refutes the horizon.  Stop if certificates grow without reusable shifted clauses. |
| ROBDD abstraction | Prove a width-independent residual semantics and show its accepting subgraph has no cycle. | Growing residual classes, or a spurious accepting cycle that refinement does not remove. |

The first two routes use finiteness directly and are the most promising.  A
small decision diagram is not enough: Rule 15 has a genuine finite alternating
fibre and its survivor function is constant true, with a zero-node ROBDD.

## All periods

The left-only map generalizes without restoring the right half.  For a period
word `u=u_0...u_(p-1)`, define

```text
G_u = T_(u_(p-1)) o ... o T_(u_0).
```

Let `n_j` be the left half after the first `j` prescribed boundary steps.  At
every phase with `u_j=1`, the OR latch removes the right neighbour and forces

```text
bit_0(n_j) = 1 XOR u_(j+1 mod p).
```

Every nonconstant cyclic word has a `1 -> 0` transition, so every such word
imposes at least one nontrivial latch constraint per period.  The uniform
left-only finish line is: no ordinary finite integer remains in these phase
constraints under every iterate of `G_u`, for any nonconstant primitive `u`.
This is a genuine framework for the whole prize, although proving that uniform
termination statement remains essentially the hard part.  Period-dependent
cone-charge constants are allowed.

Equivalently, for a primitive target word `u`, a fixed finite right half
determines a unique initial left half by left permutivity.  The fibre version is:

```text
for every primitive nonconstant u, every finite right word r, and every
support bound w, some forced initial bit strictly left of -w is one.
```

This is the full strengthened prize, so it is an organizing formulation rather
than progress by itself.  Three routes add genuine structure:

1. **Time-shift defect invasion.**  For finite `y`, study
   `delta_t(i)=F^(t+p)(y)_i XOR F^t(y)_i`.  A periodic centre is one permanently
   zero defect column, while the expanding finite-support edges seed defects.
   Seek a Rule-30-specific charge forcing one defect across the centre.  Rule
   90 is the mandatory negative control.
2. **Second-column regularisation.**  If `c_t` has period `p`, then exactly
   `l_(t+p) XOR l_t = (1 XOR c_t)(r_(t+p) XOR r_t)`.  It suffices to prove the
   right-hand observable eventually periodic, not necessarily zero; the left
   column then becomes periodic and Jen/Kopra closes the argument.  A three-bit
   local state already has exact collisions, so any quotient needs a proof.
3. **Period-word induction.**  Alternation is the unique primitive cyclic word
   with no equal adjacent bits.  Every other primitive word contains `00` or
   `11`.  An equal-pair shortening or renewal lemma would reduce all periods to
   the constant and alternating bases.  No valid shortening map is known;
   assuming a neighbouring column has the same period merely assumes the gap.

## Controls and rejection rules

- **Rule 15, alternating target:** `F(l,c,r)=1 XOR l`; an all-zero left half
  gives a real finite alternating fibre.  It rejects arguments using only left
  permutivity, triangular inversion, or low survivor complexity.
- **Rule 90, constant target:** a finite periodic centre exists for the same
  seed class.  It rejects generic permutivity and generic defect arguments.
- **Open strips:** periodic centre traces with nonperiodic neighbours exist
  when exterior boundaries are free.  A local strip lemma must explicitly use
  finite support.
- **No raw radius extension:** another finite horizon is justified only when it
  falsifies a stated rank, clause schema, or renewal lemma.
- **No automaticity detour:** nonautomaticity is strictly stronger than Prize
  1, while irrationality over `F_2(x)` is exactly the prize restated.

## Immediate program

1. Couple the explicit period-two left map `G` to necessary physical right
   coherence, keeping left-only and full-orbit conclusions distinct.
2. Transport phase-slip boundary terms and production across successive spatial
   slabs with their initial-row weights and shared predecessors retained.
3. Allow any finite charge coefficient. Prove every-cut validity and endpoint
   bounds; finite synthesis by itself is not the transport theorem.
4. Investigate weaker divergence/rank mechanisms and phase-dependent general
   period-word constraints when they supply a concrete, checkable obligation.
5. Keep time-shift defect invasion and equal-pair shortening conditional on
   an explicit invariant or exact repair rule, rather than enlarging scans.

The finish line for period two is Theorem A above.  The finish line for the
whole prize is either a uniform defect-invasion theorem or a justified
reduction of every primitive period word to the constant/alternating cases.
