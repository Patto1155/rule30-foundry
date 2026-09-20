# A characteristic-front test for zero-tail elimination

The proposed four-latches-per-one target near the end of this note was later
refuted by a weight-six row with `q=26`.  The front lemma remains valid.
See `one-bit-delay-obstruction.md`, `triangular-interface-cocycle.md`, and
`golden-mean-phase-slip.md` for the exact composition obstructions and the
refined proof obligation.

## Result

There is an exact zero-tail propagation lemma which uses no fixed temporal
window.  It does **not** exclude finite integer survivors, because the only
parity on which it forces a failure is systematically avoided by two genuine
infinite survivor branches.

Write the bits of two left halves as `x_j,y_j`, with `j=0` nearest the
centre, and evolve both with the same prescribed boundary bits using

```text
(T_b(x))_j = x_(j+1) XOR (x_j OR x_(j-1)),   x_(-1)=b.
```

Let `d(x,y)` be the least index at which they differ.  If `d(x,y)=p>=1`,
then

```text
d(T_b(x),T_b(y)) = p-1.                         (1)
```

Indeed, every output below `p-1` sees identical inputs.  At output `p-1`,
the OR term is identical and the `x_p` term is different, so cancellation is
impossible.  This is a well-founded clock for the nearest discrepancy: it
drops by one at every half-step, regardless of all more distant bits.  Its
state is an integer distance rather than a successively refined finite
window.

Consequently, two left halves which pass every alternating latch cannot first
differ at an odd index `p`.  At time `p-1`, which is an even time, (1) says
that they agree at bit zero and differ at bit one.  One low pair is therefore
`01` or `10` and the other is `00` or `11`.  Exactly one passes the necessary
condition `G^k(n) mod 4 in {1,2}`, a contradiction.

The conditional finite-tail consequence is exact: if an eternal survivor
`y` and an eventually-zero half `x` first differ at an odd bit `p`, then `x`
fails no later than macrotime `(p-1)/2`.  This is a genuine use of the zero
tail, with no bounded-memory abstraction.

## Evidence and exact falsifier

The tempting strengthening, "a first discrepancy of either parity eventually
forces one of the two halves to fail", is false.  In the low-bit-first
notation used by `alternating_left_weight.py`, take the two infinite free
sideways words

```text
s = 000000...   ->   x = 1010101010...
s = 100000...   ->   y = 0110101010...
```

Every finite prefix on each line is produced by the exact sideways recurrence,
but the infinite assertion also has a direct algebraic check.  Call the two
halves `A` and `B`.  The alternating half `A_j=1` for even `j` and zero for
odd `j` obeys

```text
T_0(A)=T_1(A)=A.
```

At bit zero the one in `A_0` saturates the OR for either boundary; away from
the boundary, direct substitution of `101` and `010` gives the same
alternating word.  The word `B` has bits `011` followed by the tail of `A`.
Checking its first three output bits separately and then using the common
tail gives

```text
T_0(B)=A,   hence G(B)=T_1(A)=A.
```

Thus both halves pass every necessary alternating latch.  They first differ
at index zero, which is even.  This is an exact infinite counterexample, not
a conclusion extrapolated from finite prefixes.  The finite constructor is
only a reproducibility check:

```bash
python - <<'PY'
from experiments.alternating_left_weight import induced_left_prefix
for m in (5, 10, 20, 32):
    for s in (0, 1):
        print(m, s, ''.join(map(str, induced_left_prefix(s, m))))
PY
```

More generally, the triangular construction explains why the parity escape
is structural.  Extending a valid `2m`-bit initial prefix by one latch leaves
one free bit in the next pair.  The two branches first differ at the lower
bit of that pair, index `2m`, always even.  Thus comparison with an infinite
continuation of a finite zero tail cannot arrange the odd first discrepancy
needed by the lemma.

The same obstruction appears at finite scale in `n=129`.  Its low-bit-first
initial prefix is

```text
10000001 00000000
```

and it passes eight latches before failing the ninth.  Thus a zero block of
eight emitted initial bits can be crossed using credit carried from the first
block.  The characteristic clock alone neither bounds that carried credit nor
proves `q(n) <= 4 wt(n)`.

## Defect return map and a failed two-lane mass

The comparator `A` turns finite support into a particularly simple tail.
Put `d_j=n_j XOR A_j`.  Since `A` is fixed by both half-steps, direct Boolean
simplification gives, for `j>=1`,

```text
d'_j = d_(j+1) XOR ((NOT n_j)(NOT n_(j-1))),
```

while at the boundary

```text
d'_0 = d_1 XOR d_0(1-b).
```

The current macro-latch is exactly `d_0=d_1`.  Applying the two half-steps
shows that the next macro-latch is

```text
d_3 = d_2 d_0.                                  (2)
```

This is a small genuine return-map simplification: it replaces the next
low-bit test by one cubic-free relation among four current defect bits.

Pair the defect into lanes `X_j=d_(2j)`, `Y_j=d_(2j+1)`.  Away from the
boundary, one half-step is

```text
u_j = Y_j XOR X_j(1-Y_(j-1)),
v_j = X_(j+1) XOR X_j(1-Y_j).
```

For every ordinary integer `n`, the defect has the stationary tail
`(X_j,Y_j)=(1,0)`.  This suggests the named **stripped defect-mass lemma**:
if `M(d)` counts lanes different from `(1,0)`, perhaps a surviving macrostep,
after discarding its one newly exposed outer lane, obeys

```text
M(G(n))-1 <= M(n).                               (3)
```

Equation (3) is false.  The smallest integer counterexample is

```text
n=65          defect lanes: 00,10,10,00,10,...   M=2
G(n)=406      defect lanes: 11,00,00,11,00,10,... M=5.
```

Both states satisfy the latch (`65 mod 4=1`, `406 mod 4=2`), but
`M(G(n))-1=4>2`.  The same step has `q(65)=4`, so it is part of a genuine
four-latch finite survivor segment.  A companion rescue can therefore create
three additional non-tail lanes in one macrostep, even after paying for the
universal outer lane.  Plain two-lane defect population is not the missing
well-founded measure.

## Gap

At an even first discrepancy `p`, equation (1) brings the discrepancy to bit
zero at even time `p`.  It does not control bit one then.  If both halves are
still survivors, their two low pairs must be complements, so a second
discrepancy has arrived at bit one and repairs the latch.  Distant differences
can repeatedly supply this companion.  Counting only the nearest front loses
exactly the credit that made the one-bit and aligned-block deletion arguments
fail.

Therefore this route currently proves neither finite-integer exclusion nor a
uniform charge bound

```text
8 W_N - N >= -C.
```

In particular, it gives no finite `C`.  It should not be presented as a
period-two exclusion.

## Next implication to test

The next useful object is not a larger fixed window.  It is a **front ledger**:
retain the nearest discrepancy distance and charge every even-parity rescue
when the companion discrepancy at bit one is created.  A sufficient lemma
would say that each initial one supplies at most four such rescued macrosteps,
up to a fixed initial allowance.  Summed to the zero tail, that statement is
exactly `q(n) <= 4 wt(n)+O(1)` and hence yields the requested finite `C`.

This proposal has a sharp falsifier.  Follow two exact sideways branches and
mark an even rescue whenever their first discrepancy reaches the low pair
while both pairs remain unequal.  Any family with `r` initial ones and more
than `4r+C` rescued macrosteps refutes allowance `C`; a rescue that cannot be
assigned to a specific earlier initial one refutes the proposed local ledger
rule.  Until such an assignment and its no-reuse proof are supplied, calling
the front clock a growing-credit proof would only restate the charge target.
