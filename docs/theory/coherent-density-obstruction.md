# What the coherent density obstruction actually says

## Result

The two-update state fixes the first temporal resampling error, but the
resulting Farkas contradiction is not caused by another short false walk.
In the time-parity witness, all six bottom-to-top walks are exact growing
cuts of the same finite-seed cone.  Consequently, enlarging only the spatial
flux state `psi` cannot repair the certificate, even if `psi` remembers the
entire temporal strip.

What remains too small is the charge/terminal language.  The density
`F_(p,tau)(B,C)` forgets the temporal context in which `(B,C)` occurs, while
the terminal proof tests it on phase-labelled pair paths independently of the
growing-cone endpoint.  A density on two consecutive observations, for
example

```text
H_(p,tau)((B_t,C_t),(B_(t+1),C_(t+1))),
```

is therefore a mathematically meaningful next local repair.  It is the
minimal named new information exposed here: **temporal context of the charge
density**.  It requires new spatial-difference and terminal telescoping
formulas.  Adding more memory only to `psi` is ruled out by the argument
below.

## The 24-row identity as six genuine cuts

Use the cone determined through depth seven by the odd-diagonal endpoint bits
`(1,0,1,0)`.  Its relevant triple paths are

```text
q=4: 000 -> 110 -> 001                         (twice)
q=5: 000 -> 100 -> 011 -> 110                  (twice)
q=6: 000 -> 000 -> 110 -> 101 -> 010
q=7: 000 -> 000 -> 100 -> 011 -> 100 -> 111.
```

These are obtained by the exact diagonal recursion, with the correct start
phase and the exact top cap.  Thus every interior row, bottom row, and top row
in this decomposition belongs to a single liftable cut; overlapping
two-update windows do not resample a predecessor.  Summing each cut cancels
`psi` regardless of the size of its state.  All six bottom emitted bits are
zero, so their six spatial inequalities give

```text
2(Phi_4-Phi_3) + 2(Phi_5-Phi_4)
  + (Phi_6-Phi_5) + (Phi_7-Phi_6) >= 6,
```

or equivalently

```text
Phi_5 + Phi_7 - 2 Phi_3 >= 6.                 (1)
```

The bottom corrections cancel because the three odd cuts have the same
bottom pair.  Direct expansion of their exact pair lists gives

```text
Phi_5 + Phi_7 - 2 Phi_3
 = F_(1,0)(00) + 2 F_(1,1)(00) + F_(1,0)(11)
   + F_(1,1)(10) + F_(1,0)(01).              (2)
```

The six terminal-flow rows in the witness are three phase-consistent
two-cycles:

```text
(0,00) -> (1,00) -> (0,00),
(0,11) -> (1,00) -> (0,11),
(0,01) -> (1,10) -> (0,01).
```

Telescoping `chi` around them says that the right side of (2) is at most
zero, contradicting (1).  This is the exact `0 >= 6` Farkas identity.
Neither initialization nor a terminal-bound endpoint row occurs.

The cycles are not artifacts of independently composed one-step edges.  They
lift to infinite Rule 30 predecessor strips.  Write `01` and `10` for the
two alternating temporal words of a single diagonal bit.  For the
`01 <-> 10` pair cycle, assign the spatial diagonals alternating temporal
words

```text
..., 01, 10, 01, 10.
```

Every update sees one `1` among its two older inputs, so every bit flips and
the final pair alternates forever.  For `11 <-> 00`, give both final
diagonals the temporal word `10`, then give the older diagonals the alternating
words

```text
..., w=01, z=10, A=01, B=10, C=10.
```

The older words continue alternating to the left.  The OR of the two inputs
to each updated bit is identically one, including at the duplicated final
pair, so every bit flips.  The all-zero strip realizes the remaining fixed cycle.
Equivalently, these assignments obey the shrinking-word map

```text
T(x)_i = x_(i+2) XOR (x_(i+1) OR x_i),
```

at every site and time.  Thus merely replacing the one-step terminal graph
by any fixed number of overlapping local updates does not delete these modes.
They are genuine strips.  They are not asserted
to be complete centre-capped growing cuts, which is the distinction relevant
to a possible endpoint-conditioned terminal repair.

## The shorter identity

Without time parity, the nine-row witness has the same structure.  Its
spatial part consists of the exact cuts

```text
q=4: 000 -> 110 -> 001,
q=5: 000 -> 100 -> 011 -> 110,
```

and its terminal part is the liftable pair cycle `01 -> 10 -> 01`.  Their
bottom corrections and all flux variables cancel, giving `0 >= 2`.  Hence
the smaller witness also does not diagnose a need for a third update in the
spatial state.

## Scope and next implication

This proves an all-spatial-memory obstruction **for the present pair density
and terminal-flow requirement**: no refinement of `psi`, even to an exact
whole-strip spatial flux, can satisfy these inequalities.  It does not prove
that every terminal pair strip is reachable with the centre boundary, so it
does not rule out an exact cone-conditioned proof of `Phi_q <= K` that avoids
the three cycles.

There are therefore two coherent next moves.  The local one is to put the
missing temporal context into the density itself, using
`H((B_t,C_t),(B_(t+1),C_(t+1)))`, and derive its endpoint formulas.  The
alternative is to keep pair density but
replace `chi` by a terminal certificate that retains enough centre-endpoint
lineage to quantify only over growing-cone cuts.  A bounded-memory refinement
of `chi` justified only by longer free strips is not promising, because the
three obstructing modes lift through every finite horizon.  The cheaper
diagnostic is to audit cone reachability of those terminal modes before
building the larger temporal-pair density system.

## What survives after giving the density temporal context

The refinement tested in `temporal-pair-density.md` replaces `F(P_t)` by
`F(P_t,P_(t+1))`.  Its time-parity `0 >= 4` witness has a different and more
specific defect.  The spatial rows decompose into these four complete
bottom-to-top paths:

```text
q=6: 000 -> 000 -> 000 -> 110 -> 001
q=6: 000 -> 000 -> 110 -> 101 -> 010
q=7: 000 -> 000 -> 000 -> 100 -> 011 -> 110
q=7: 000 -> 000 -> 100 -> 011 -> 100 -> 111.
```

All four paths lift to single Rule 30 predecessor strips; thus they are not
made by resampling the overlapping two-update windows.  The second and fourth
are the exact growing cuts with latch prefixes `(1,0,1)` and `(1,0,1,0)`.
The first and third do not occur for any assignment of the three or four
free latch bits, despite satisfying the displayed top cap.  They lose the
correlation between the strip's older side and the centre boundary.  This
shows that an exact top state by itself is not an exact cone endpoint.

Their failure is already visible at the bottom.  For the first `q=6` path,
the initial `000` gives `e_4=e_5=e_6=0`; the next `000` and the diagonal
recurrence force `e_2=e_3=0`; the third `000`, propagated once more toward
the centre, forces `e_1=0`.  Thus `e_1 XOR e_2=0`, contrary to the exact
initial latch `e_1 XOR e_2=1`.  For the first `q=7` path, its three initial
zero triples similarly give first `e_5=e_6=e_7=0`, then `e_3=e_4=0`, and
then `e_1=e_2=0`, producing the same latch contradiction.  These paths are
valid free bulk strips whose bottom lineage starts from an invalid
centre-zero latch.

The terminal rows are weaker still.  In phase-expanded notation they form
the four-cycle

```text
(0,(00,00)) -> (1,(00,00)) -> (0,(00,11))
             -> (1,(11,00)) -> (0,(00,00)).
```

At the underlying pair level, repeating this cycle asks for

```text
00, 00, 00, 11, 00, 00, ... .                (3)
```

Every edge of the cycle is one of the 22 projections of a shared-seven-bit
coherent edge, but (3) is not a Rule 30 strip.  Exhausting the twelve-bit
original words gives a particularly small exact obstruction: every word
whose first five terminal pairs are

```text
00, 00, 00, 11, 00
```

has sixth pair `11`; none has sixth pair `00`.  Thus the `chi` cancellation
in this Farkas identity composes compatible projected edges whose hidden
older predecessors disagree.  The temporal-pair density escaped the earlier
infinite, genuinely liftable terminal cycles; the new contradiction is a
projection error rather than evidence for an intrinsic need for a still
larger charge table.

The smallest concrete missing dependency is now **common predecessor lineage
through the centre-capped terminal strip**.  A terminal state must at least
distinguish the forbidden six-pair word (3), and the spatial language must
exclude the two free strips that cannot share the centre boundary.  Merely
adding another unnamed window postpones the same obligation.  The justified
next lemma is a terminal upper bound formulated on exact growing cuts, or an
equivalent endpoint-credit relation carrying their latch lineage.  Until
that relation is derived, this witness gives no mathematical reason to add a
third temporal observation to the density.
