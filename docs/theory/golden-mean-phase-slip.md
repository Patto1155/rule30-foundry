# Golden-mean phase slips at spatial depth seven

## Scope

This note does not prove a cone-charge bound.  It isolates the exact state
lost by the most natural seven-cell renewal on an `11`-free right-boundary
word.  It also records a small fixed-weight check beyond the previously
reported weight-ten horizon.  The check is evidence about finite prefixes,
not an induction in the weight.

Write `s_k` for the cell immediately right of an alternating centre at even
time `2k`.  A physical Rule-30 right half has no block `11`, as proved in
`alternating-right-boundary-coherence.md`.  Let `X_j(t)` be the temporal
column at spatial depth `j` reconstructed sideways on the left, with
`X_0(2k)=0` and `X_0(2k+1)=1`.  The initial-left bit at depth `j` is
`e_j=X_j(0)`.

## The first eight depths

Direct substitution in the sideways Rule-30 recurrence gives, under the
sole hypothesis `s_k s_(k+1)=0`,

```text
depth 1: even 1-s_k,                         odd 1
depth 2: even s_k,                           odd s_(k+1)
depth 3: even 1-s_(k+1),                     odd 1-s_(k+1)
depth 4: even 0,                             odd s_(k+2)
depth 5: even 1-s_(k+1)-s_(k+2),             odd 1-s_(k+1)
depth 6: even s_(k+2),                       odd s_(k+1) XOR s_(k+2) XOR s_(k+3)
depth 7: even 1-s_(k+2)-s_(k+3),             odd 1-s_(k+3)+s_(k+1)s_(k+3)
depth 8: even s_(k+1)s_(k+3).
```

In particular the first seven initial bits have weight

```text
e_1+...+e_7 = 4 - 2s_1 - s_2 - s_3 >= 1.       (1)
```

The inequality follows by checking the five length-three golden-mean words.
For the sparse phase `s=010101...`, the initial row begins

```text
1000000 1000000 1000000 ...,
```

so the spatial period seven and the support pattern `0,7,14,...` in the
finite extremizers are explained by an exact return, rather than numerical
coincidence.  The other phase `101010...` also returns an alternating column
at depth seven, but its induced initial row is denser.

Equation (1) cannot simply be restarted at depth seven for a general
golden-mean word.  Define the zero-pair field

```text
d_i = 1 if s_i=s_(i+1)=0, and d_i=0 otherwise.
```

The depth-seven even/odd pair `(A_k,B_k)` is

```text
A_k = d_(k+2),
B_k = 1-s_(k+3)+s_(k+1)s_(k+3).                (2)
```

Thus it is the alternating pair `(0,1)` except for two explicit defects:

```text
s_(k+2)s_(k+3)=00       adds one: (0,1) -> (1,1),
s_(k+1)..s_(k+3)=001    removes one: (0,1) -> (0,0).
```

Equivalently,

```text
A_k+B_k-1
  = 1[s_(k+2)s_(k+3)=00] - 1[s_(k+1)..s_(k+3)=001].   (3)
```

There is a sharper coboundary form.  Put `Delta_k=A_k+B_k-1`.  On the
golden-mean shift,

```text
Delta_k = d_(k+2)-d_(k+1)+d_(k+1)d_(k+2),       (4)
```

where the product is exactly the indicator of
`s_(k+1)s_(k+2)s_(k+3)=000`.  Therefore

```text
sum_(0 <= k < M) Delta_k
  = d_(M+1)-d_1
    + #{i in {1,...,M}: s_i s_(i+1) s_(i+2)=000}.   (5)
```

The phase-slip field is an exact boundary coboundary plus nonnegative `000`
production.  In particular every prefix has excess at least `-1`.  This is a
real telescoping fact, but it is only a statement about the number of ones in
the new temporal column.  Rule 30 is not monotone in that column, so it does
not yet bound the number of ones deeper in the initial row.

The first-cut excess cannot simply be preserved at later cuts.  For the
golden-mean word `s=000...`, the reconstructed half is `101010...`, fixed in
time.  Column seven is constantly one, but column fourteen is constantly
zero.  Its excess over `M` temporal pairs is `-M`, despite the positive
first-cut excess.  This free word is excluded by full right coherence, but
it belongs to the weaker golden-mean lemma.  A transport inequality must
include the initial-row weight produced between cuts.

A sufficient remaining certificate is a functional `Phi` on exact adjacent-
column cuts, retaining their shared predecessors and centre-boundary lineage,
such that each seven-column slab obeys

```text
8 weight(the slab's seven initial cells) - 7
    >= Phi(next cut) - Phi(current cut),          (6)
```

with one uniform lower bound on `Phi` at all cuts and one uniform upper bound
at initial alternating-centre cuts, including every permitted free word.
Summing (6) through `R` slabs would give `8W_(7R)-7R >= -C`; the remaining
at most six initial cells cost at most six more units.  This is exactly the
weaker sufficient charge, without demanding the stronger slope seven.
Identity (5) supplies a possible component of `Phi`, not a complete definition
or a proof of these bounds.  The proposed mechanism retains its endpoint
field `d` and its nonnegative `000` production instead of resetting the cut.

Thus the remaining construction has the following precise local obligation:

> Transport the boundary term and production in (5) across a second slab,
> including its initial-row weight, and identify a cut functional satisfying
> (6) with the stated two uniform endpoint bounds.

Resetting the depth-seven column to the alternating column deletes precisely
the coboundary and production terms in (5).

## Exact obstruction to block reset

The `11`-free word

```text
s = 01010100
```

reconstructs the sixteen initial bits

```text
10000001 00000000.
```

The second aligned eight-cell block has weight zero.  Hence neither “every
eight-cell block contains a one” nor a renewal that discards the cut state is
valid even on the golden-mean input shift.  The first block carries exactly
the credit that the second block consumes.

This failure is not caused only by allowing an arbitrary golden-mean word.
There is an exact physical right half with finite support that produces an
aligned seven-zero block.  Take initial right value `17` (bit zero is nearest
the centre), centre zero, and the following 28-cell initial left prefix,
low depth first:

```text
0110010 0011010 0010110 0000000,
```

This is the integer `857638`.  The fourth aligned block, zero-based indices
`21..27` (spatial depths `22..28`), is all zero.  Setting all remaining initial
cells to zero gives

```text
01010101010101010101010101010
```

at the centre through time 28 by direct shrinking-cone truth-table evolution.
Its even-time right-neighbour word through time 26 is `10100001001001`, which
avoids `11` and comes from the stated physical right half.  Thus even exact
right coherence does not rescue an aligned block renewal that forgets its
cut state.  The identities and this full-row counterexample are checked by
`python tools/verify_golden_mean_phase_slip.py`.

## Fixed-weight evidence beyond weight ten

A locality-complete prefix enumeration retained only left prefixes which
both pass every Rule-30 alternating latch and whose forced even-time boundary
word avoids `11`.  For weight caps 11 through 14, the last jointly valid
prefix depths were:

| weight cap | last jointly valid depth | `4w-depth` |
|---:|---:|---:|
| 11 | 40 | 4 |
| 12 | 43 | 5 |
| 13 | 45 | 7 |
| 14 | 49 | 7 |

The next layer was empty in each case.  These numbers are not the unrestricted
first-failure times `q(n)`: a left prefix may continue satisfying latches
after its forced boundary word has already formed `11`.  For example, the
weight-four row supported at `0,7,14,21` has unrestricted `q=16`, while its
joint latch-plus-`11`-free depth is only 14.

Reproduce each row with
`python experiments/alternating_right_coherence.py --weight WEIGHT --bound DEPTH`.
The enumeration supplies no uniform argument.  Its role is limited to a
bounded falsifier check of the proposed `depth <= 4 weight` inequality; it
finds none at weights 11 through 14.  Equations (2)--(6), rather than a longer
scan or another state-free block rule, give the next concrete proof
obligation.
