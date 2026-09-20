# A zero-tail rank route and its bulk-conservation obstruction

Date: 2026-09-20. This tests a weaker-looking alternative to initial-density
transport: make a nonnegative rank decrease on the evolving finite left
integer. It does not repeat the compactness reduction or assume a fixed
insertion credit. The result closes an explicit rank family, not period two.

## The rank family and exact scope

Write `len(n)=bit_length(n)` and `G=T_1 o T_0`, with the established
nearest-cell-first convention. Consider

```text
E(n,z) = a len(n) + sum_(i>=0) phi(n_i,...,n_(i+r-1)) + Psi(n,z),
phi(0,...,0)=0,                  |Psi(n,z)| <= C.
```

Here `z` may include the right half and finite phase information. The local
coefficients `phi` may have either sign. The sum is finite because the initial
word has a zero tail. The correction `Psi` need not even be local; only its
uniform boundedness is assumed. This includes fixed-range pattern counts,
support length, and arbitrary bounded boundary corrections.

Suppose `E` is uniformly bounded below on the admissible finite rows, and
for some fixed `epsilon>0` it decreases by at least `epsilon` per forced
macrostep. The assumed decrease must be justified on **every finite row
passing a fixed finite physical alternation test**, for example alternation
through `2K` halfsteps for a fixed `K>=1`. A larger fixed test does not change
the argument. Actual infinite alternating orbits pass every such test.

**Result:** no rank of this form with `1<=r<=8` has those properties.
The bounded correction can contain arbitrarily many fixed boundary bits;
its size does not repair the obstruction. The bulk-conservation necessity
below holds for every finite `r`; the exact classification is established
here only through eight.

This scope matters. A decrease known only on globally extendable infinite
orbits is not covered: arbitrary distant bulk insertion need not preserve
infinite extendability. Spatially varying weights, unbounded cut-wide credit,
lexicographic/ordinal ranks, and decreases that have no uniform positive
lower bound are also outside this result.

## 1. Any such density must be conserved in the bulk

Let `F` be unforced Rule 30 on a spatially periodic bi-infinite row. For a
word `u` of spatial period `L`, define its two-step energy change

```text
D(u) = sum_(i mod L) [phi((F^2 u)_(i..i+r-1))
                     - phi(u_(i..i+r-1))].
```

Fix an actual finite row whose centre follows the alternating ether through
the required finite test, with a sufficiently large unchanged neighbourhood
around the centre. Far to its left append `m` copies of `u`, then a fixed
nonzero cap and a zero tail. All these finite rows pass the same test by
locality, whatever `m` is. The cap makes the highest occupied bit unambiguous.
Only a bounded number of windows touch the fixed interfaces. Consequently

```text
E(after)-E(before) = m D(u) + O(1),
```

where the bound is independent of `m`: the support grows by exactly two under
`G` for every positive finite left integer, and `Psi` is uniformly bounded.
The decrease assumption therefore implies `D(u)<=0` for every periodic word.

For every `L>=r+4`, average this inequality over all `2^L` equally weighted
periodic input words. Each input `r`-window is uniform. Each output `r`-window
under `F^2` is also uniform: its `r+4` distinct input sites are independent,
and fixing the four rightmost inputs makes the remaining `r` inputs map
bijectively to the `r` outputs, successively using left permutivity.
Thus the average of `D` is exactly zero. Since every summand is nonpositive,
`D(u)=0` for each word of length `L>=r+4`. Repeating any shorter word to reach
that length proves the same statement for every spatial period.

This does **not** assume that Rule 30 is bijective on a finite periodic ring;
it generally is not. Only the indicated local uniform-marginal calculation
is used. The conclusion is a necessary condition for the proposed decreasing
rank: its density must be an additive conserved density of unforced `F^2`.

## 2. Exact classification for ranges one through eight

A range-`r` density has `2^r` real coefficients. Every periodic-ring equation
`D(u)=0` is a homogeneous integer linear equation in those coefficients.
The obvious solutions are

```text
phi(b_0,...,b_(r-1)) = c
  + h(b_0,...,b_(r-2)) - h(b_1,...,b_(r-1)).
```

They are constants plus spatial coboundaries. Periodic sums telescope, so
these always solve every equation. Their vector space has dimension
`2^(r-1)`: the binary de Bruijn graph on `(r-1)`-words is connected, giving
`2^(r-1)-1` independent gradients, plus the independent constant. For `r=1`
the gradient space has dimension zero.

The reproducer builds enough exact ring equations to attain the complementary
rank modulo the prime 65521:

| Range `r` | Coefficients | Exact modular rank | Kernel dimension | Ring equations checked |
|---:|---:|---:|---:|---:|
| 1 | 2 | 1 | 1 | 2 |
| 2 | 4 | 2 | 2 | 8 |
| 3 | 8 | 4 | 4 | 18 |
| 4 | 16 | 8 | 8 | 64 |
| 5 | 32 | 16 | 16 | 138 |
| 6 | 64 | 32 | 32 | 310 |
| 7 | 128 | 64 | 64 | 558 |
| 8 | 256 | 128 | 128 | 1,132 |

Rank modulo a prime is a lower bound on rational, hence real, rank: a
nonzero modular minor is a nonzero integer minor. The explicit trivial
kernel is a matching upper bound on rank. Therefore these finite matrices
already prove that every density satisfying **all** ring equations is a
constant plus coboundary for the stated ranges. This is not extrapolation
from a numerical nullspace or a finite survival horizon.

## 3. Finite support finishes the obstruction to the rank

Because `phi(0,...,0)=0`, the constant `c` in the classified density is zero.
On a finite zero-tailed word the remaining sum telescopes:

```text
sum_(i>=0) phi(n_i,...,n_(i+r-1))
  = h(n_0,...,n_(r-2)) - h(0,...,0).
```

It is uniformly bounded. Thus `E(n,z)=a len(n)+O(1)`. Uniform lower boundedness
on finite rows passing the test forces `a>=0`, since arbitrary remote caps
make their left support length unbounded while preserving that finite test.

There are actual finite rows with arbitrarily long alternating prefixes:
truncate the spatial-period-seven ether on both sides,

```text
left n_R  = sum_(0<=j<R) 2^(7j),
right v_R = sum_(0<=j<R) 2^(5+7j),        centre = 0.
```

Their first centre mismatch is at time `7R+1`, by the already proved extreme
left-input argument. The ether itself has full temporal period four and centre
trace `0101...`. Taking `R` arbitrarily large gives arbitrarily many consecutive
macrosteps whose rows all pass any prescribed fixed finite alternation test.

For `k` such macrosteps, finite-left support length grows by `2k`. The proposed
rank would simultaneously satisfy

```text
E_k-E_0 = 2ak + O(1),             E_k-E_0 <= -epsilon k.
```

The bounded term is uniform in `R` and `k`. These are incompatible as `k`
grows, since `a>=0`. This proves the asserted impossibility for ranges 1..8.

## 4. Restricting to the entered right invariant does not rescue this family

The ether need not satisfy the conditional three/five-gap right invariant.
Nevertheless, the same rank obstruction holds when the finite alternation
test is supplemented by membership in the proved right invariant from
`right-coherence-followup.md`.

Fix the finite right integer `v=167`, one of its entry cylinders. For any
halfstep horizon `T`, a finite left row can be chosen to realize the prescribed
centre trace through `T`: at stage `t=1,...,T`, the centre at time `t` depends
on the new extreme initial-left bit at site `-t` with XOR coefficient one.
All other inputs in that cone are already fixed. Choose that bit to obtain
`c(t)=t mod 2`; causality ensures it cannot alter earlier centre times.
Set all still farther left bits to zero. This constructs a finite full row
for every finite `T`, without claiming that their limit has finite support.

During its alternating prefix the right half follows the forced return
exactly, so its membership in the right invariant persists. Thus the
construction supplies arbitrarily long segments of admissible rows for this
restricted class. Arbitrarily distant left-bulk modifications still preserve
the initial right invariant and any fixed finite alternation test. The
bulk-conservation, classification, and lower-bound arguments are unchanged;
the arbitrarily long constructed segments replace the ether in the final
contradiction. This extension uses the established invariant closure, not a
claim that every finite right half enters it.

The independent cone reproducer obtains `(left,right)=(86,167)`,
`(33110,167)`, and `(7110998,167)` for halfstep horizons 8, 16, and 24.
The all-horizon construction is the extreme-input induction above.

## Interpretation and remaining usable direction

A finite-support rank remains a legitimate sufficient route. This calculation
shows why a uniform sum of a few local pattern costs cannot simply turn the
existing zero tail into a decreasing integer: arbitrary distant bulk forces
conservation, and the checked conserved costs telescope away.

The missing useful rank must therefore escape at least one explicit
hypothesis, for example by retaining unbounded inherited interface credit or
using a genuinely hierarchical/well-founded state. That is a constraint on
mechanism design, not evidence that such a state exists. Enlarging finite
survival scans, tuning coefficients, or allowing signed coefficients inside
the refuted range does not repair the family.

The computation also suggests a sharply separated algebraic question:
classify finite-range additive conserved densities of `F^2` for arbitrary
range. An all-range triviality theorem would close this rank family at every
range; it still would not prove period-two exclusion. It should not displace
the more directly sufficient spatial-transport work.

## Reproduction and independent checks

```bash
python experiments/zero_tail_rank_followup.py
```

All modular elimination uses exact integers, with products safely within
signed 64-bit range. Each generated ring update is checked against a separate
eight-entry elementary-rule evaluator. Every matrix row is also checked to
annihilate the constant and all gradient basis densities. The four-phase
ether identity and four finite truncations are independently replayed as full
finite rows; their forced left halves agree with the integer map and grow by
two bits per macrostep. Three further cone-inversion witnesses check the
finite-prefix construction with right integer 167. A separate reviewer
recomputed every stated matrix rank using cell lists, pure-Python elimination,
and the different prime 1000003, obtaining the same ranks and stopping counts.

The finite computations verify the stated matrix ranks and local identities.
The arbitrary-bulk and averaging implications are the handwritten arguments
above. No theorem about all finite initial configurations having nonperiodic
centre traces, and no prize-status change, follows.
