# Right-boundary coherence for an alternating Rule-30 centre

This note separates a physical right boundary from the free sideways word in
the alternating left-cone construction.  The distinction removes the known
finite charge counterexample, but it does not yet prove a uniform charge
bound or exclude an alternating centre.

## Exact two-step constraint

At even time `2k`, write the cells immediately right of the centre as
`(s,u,v,...)`.  For a one-sided right half encoded with the nearest cell in
bit zero, one Rule-30 step with prescribed centre `b` is

```text
R_b(n) = ((n << 1) | b) XOR (n | (n >> 1)).
```

The alternating centre is `0,1`, so the even-time return map is
`H=R_1 o R_0`.  Two direct cell updates give

```text
s_next = (1-s)(1-u)(1-v).                       (1)
```

Thus `s_next=1` exactly when the current first three right cells are zero.
In particular `s_k s_(k+1)=0`: every physical boundary word avoids `11`.

There is a second local restriction.  Put `(a,b,c,d,e)` for the first five
right cells at an even time and assume `a=0`.  A truth-table simplification of
the first three cells under `H` gives the implications

```text
010 -> 000,
001 -> 010,
011 -> 00(d OR e).                              (2)
```

If the next boundary bit is also zero, `(b,c)` is not `00` by (1).  The cases
`10` and `01` in (2) reach `000` after one and two returns.  In the `11` case,
`d OR e=0` reaches it immediately; otherwise the forced chain is

```text
011 -> 001 -> 010 -> 000.
```

Equation (1) then emits a one.  Hence four consecutive zeros force the next
bit to be one, and `00000` is forbidden.

Equivalently, project an even-time right half to its low triple, encoded as an
integer.  Exhausting the two forgotten higher bits gives the exact necessary
factor graph

```text
0->{1,5,7}  1->{0,4}  2->{0}    3->{4,6}
4->{2}      5->{4}    6->{0,4}  7->{2,6}.
```

The zero-output states are `0,2,4,6`.  On them the rank
`rho(0,2,4,6)=0,1,2,3` strictly decreases along every zero-to-zero edge;
state zero must next enter an odd state.  This finite return ranking is the
same four-zero locking argument in closed factor form.  It also gives
`sum_(k<M) s_k >= floor(M/5)` for every physical boundary prefix.

These two forbidden blocks do not characterize the full trace language.
Exact enumeration of the finite dependency cone gives 17 length-six words,
and `101001` is absent even though it contains neither `11` nor `00000`.
Avoidance of `11` is therefore a deliberately weaker, sound relaxation.

## Coupling to the left side

The centre equation must also hold; a right trace alone is not a full orbit.
At even time the centre changes from zero to one, so Rule 30 gives

```text
1 = left XOR s,  hence left = 1-s.              (3)
```

At odd time the centre changes from one to zero, which forces `left=1`.
These identities show why the free sideways word used to reconstruct a left
cone must be checked for right coherence.  The weight-six counterexample has
free word

```text
01010100110101101101010011
```

and contains `11` starting at zero-based positions 8, 13, 16, and 24.  It is
an exact counterexample to the unrestricted left-only sharp charge, but it
cannot be the right boundary of an actual alternating Rule-30 orbit.

The recurrence and (3) are Rule-30 identities.  They must not be transferred
to generic left-permutive controls.  For Rule 15, whose output is `1-left`, an
alternating centre requires `left=centre` and places no analogous condition on
the right.  For Rule 90 the centre condition is
`left XOR right=next_centre`; again the Rule-30 return relation does not apply.

## Sufficient remaining lemma

Let `e_1,e_2,...` be the initial-left bits reconstructed from a free even-time
boundary word `s`, and let `W_N=sum_(j<=N)e_j`.  It would suffice to prove that
there is one constant `C` such that

```text
8 W_N - N >= -C                                (4)
```

for every `N` and every `s` avoiding `11`.  This is weaker than proving (4)
for all free words and stronger than needed for the actual right trace
language.  Every physical alternating orbit supplies an `11`-free word by
(1).  If its initial left half had finite weight `r`, then `W_N<=r`, while the
left side of (4) would tend to minus infinity, a contradiction.

Finite searches make this restricted lemma plausible, but they do not prove
it.  The sideways reconstruction retains an expanding cone rather than a
closed finite state, so the missing object remains a uniform induction or a
bounded local potential.  Proving (4) on the golden-mean input shift would
exclude an eventual alternating centre without requiring an exact symbolic
description of all coherent right traces.

For a finite left half `n`, define `ell(n)` as the number of initial passed
alternating latches before either a latch fails or the inferred word
`s_k=1-(G^k(n) mod 2)` first contains `11`.  Thus `ell(n)<=q(n)`.  The optional
CLI performs a locality-complete sparse-prefix enumeration at a specified
weight cap and proposed bound:

```bash
python experiments/alternating_right_coherence.py --weight 2 --bound 8
```

An empty layer at `bound+1` proves `ell(n)<=bound` for every ordinary integer
at that weight cap, without restricting its support width.  Each invocation
is only a fixed-weight finite certificate, not a uniform charge proof.

The focused verifier is:

```bash
python experiments/alternating_right_coherence.py
python -m unittest tests.test_alternating_right_coherence
```
