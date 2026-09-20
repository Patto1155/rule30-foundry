# Period-7 ether truncations all escape

## Result

Let

```text
n_r = sum(2^(7j), j=0,...,r-1).
```

These left halves have arbitrarily large support, with ones separated by seven
sites and support density `r/(7r-6)`, tending to `1/7`.  Every one exits the
necessary survivor set when alternation is prescribed starting from that
half at a centre-zero phase.  More precisely, if
`q(n)` is the first macrotime at which
`G^q(n) mod 4` is `0` or `3`, then

```text
q(n_r) = (7r-1)/2          when r is odd,
q(n_r) = 7r/2 + D(m mod 512) when r=2m.
```

Here `D` is the exact 512-entry certificate table.  Its maximum is eight, so
`q(n_r) <= 7r/2+8`.  This implies `q(n_r) <= 4r` for `r>=16`; direct exact
evaluation of the seven smaller even cases proves the same bound there (with
equality at `r=2,4`).  The odd formula also satisfies it.  Thus the whole
family obeys `q(n_r) <= 4 wt(n_r)`.

This is an exact infinite-family theorem, not an extrapolation from bounded
values of `r`.

## The ether and the odd case

The infinite word `E_j=1` exactly when `j=0 mod 7` has the four-phase Rule 30
orbit below.  The strings are written in increasing half-line index:

```text
1000000 -> 1100001 -> 0110010 -> 1011111 -> 1000000.
```

The cell at index `-1` in these periodic words is `0,1,0,1`, so restriction to
the half-line is exactly compatible with the prescribed alternating boundary.
The low pair at both even phases is unequal.  Thus `E` is an eternal survivor
of the necessary left-latch condition.

The truncation `n_r` first differs from `E` at index `p=7r`.  Left permutivity
decreases the nearest-discrepancy index by exactly one per half-step.  Before
time `p-1`, the low pair of `n_r` therefore equals the ether low pair.  If `r`
is odd, `p` is odd, and at the even time `p-1` the two halves agree at bit zero
and differ at bit one.  Exactly one low pair is unequal.  The ether passes, so
the truncation fails, proving the stated equality.

## Exact moving-interface system for even `r`

For even `r`, the first discrepancy reaches bit zero at an even time and can
be repaired by its wake.  The wake is still a finite-state object at every
fixed lag from the moving front.

Write `delta_d(t)` for the difference from the ether at lag `d` behind the
front.  In half-line indices, the initial cut is `j=p`; input lag `d` at time
`t` is `j=p-t+d`, and output lag `d` at time `t+1` is
`j=p-t-1+d`.  Initially `delta_d(0)=1` exactly when `d=0 mod 7`, while
negative-lag bits are zero.  If `e_0,e_-` are the ether bits at the output
coordinate and its boundary-side neighbour, direct subtraction of the two
Rule 30 updates gives

```text
delta_d(t+1) = delta_d(t)
  XOR (((e_0 XOR delta_(d-1)) OR (e_- XOR delta_(d-2)))
       XOR (e_0 OR e_-)).
```

The recurrence for a prefix through lag `L-1` uses only that prefix.  It is
triangular, with each new `delta_d` equal to old `delta_d` XOR a function of
lower lags, and hence is a bijection.  Its coefficients have joint period 28
from ether periods seven in space and four in time.

For even `r=2m`, arrival occurs at time `14m`.  The exact 18-bit prefix,
together with time modulo 28, returns to its initial state after 512 such
samples.  Bijectivity makes this a pure cycle, so these 512 samples cover all
`m`, not merely the first 512.  At arrival, reconstructing the finite half as
`ether XOR delta` and applying the prescribed boundary gives this complete
failure-delay distribution:

| Delay after arrival | Number of cycle states |
|---:|---:|
| 0 | 256 |
| 1 | 128 |
| 2 | 64 |
| 4 | 32 |
| 5 | 24 |
| 6 | 4 |
| 7 | 2 |
| 8 | 2 |

There is a precise switch at arrival.  The first omitted initial one is at
half-line index `p`, a distance `p+1` from the centre, so it cannot change the
centre boundary before time `p+1`.  The moving interface therefore agrees with
the prescribed-boundary left trajectory through time `p`.  At that time the
certificate reconstructs the low 18 bits and stops the free interface
recurrence.  It then applies the forced boundary word `01` to those bits.  This
separation prevents the post-collision centre from being silently replaced by
the periodic ether continuation.

The JSON certificate records `D` in sample order under
`even_r_delay_by_m_mod_512_for_r_2m`, making the equality above directly
checkable without treating the distribution as a formula.

Eighteen arrival bits suffice through delay eight: a low-pair test after 16
half-steps depends only on initial indices 0 through 17.  Therefore every even
truncation fails by `7r/2+8`.

Reproduce the certificate with:

```bash
python experiments/period7_ether_interface.py \
  --out runs/theory/period7-ether-interface.json
python -m unittest tests.test_period7_ether_interface
```

The test independently evolves the original truncations to the arrival time
and compares their low 18 bits with the moving-interface recurrence.

## Scope and next implication

Subsequent work proves that no fixed post-arrival allowance works even for
one isolated added bit; see `triangular-interface-cocycle.md`.  The theorem
for the unmodified truncation family proved above is unchanged.

The theorem excludes an unbounded, positive-density restricted family.  It
does not exclude arbitrary finite left halves and does not establish a general
weight-to-lifetime bound.  It does show that an even-parity front rescue can be
handled without a growing spatial window: a fixed lag strip plus its exact
phase cycle is enough for this periodic tail.  A broader proof would need a
finite collection of such interface cycles for a class of eventually periodic
tails, or a common ranking that survives transitions between them.

The same delay-eight bound does not compose with even one modified initial
bit.  Take `n_1181` and add a one at index 8254.  Its first difference from the
ether is at the even index `p=8254`; it agrees again through the retained ether
one at index 8260, and its zero tail begins at 8267.  The exact interface state
at arrival has low 18-bit integer `120433`.  Under the forced boundary its low
pairs at delays zero through eight are

```text
1, 2, 2, 1, 1, 2, 1, 2, 1,
```

so every one passes the latch.  Direct integer evolution likewise finds no
failure through macrotime `8254/2+8=4135`.  This is a finite counterexample to
extending the `+8` interface allowance to all one-bit modifications; no claim
is made here about its later failure time.

There is also an onset distinction: an orbit initially having left half `n_r`
could, in principle, begin alternating later from a different left half. This
result does not exclude that possibility. The unrestricted termination lemma
is still needed to handle an arbitrary eventual-alternation onset.

Nor can the eight-step wake bound be extended to arbitrary finite wakes.
For any `H>=1`, truncate the infinite survivor `B=01101010...` to its first
`2H` bits. It passes the first `H` latch tests by locality, while its first
discrepancy from the period-seven ether is already at index zero. Thus there
is no universal constant delay from the first discrepancy for arbitrary
finite halves. (`T_0(B)=A=101010...`, and both half-steps fix `A`.) A useful
extension must charge the additional wake to the initial configuration.
