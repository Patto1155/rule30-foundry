# Triangular interface cocycles and a 2-adic obstruction

## Unitriangular return maps

Fix a cut position `p`, put `a=p mod 7`, and compare a half-line with the
period-seven ether in coordinates moving inward from that cut.  If `delta_d(t)`
is the defect at lag `d`, the exact recurrence is

```text
delta'_d = delta_d
  XOR (1 XOR e_-) delta_(d-1)
  XOR (1 XOR e_0) delta_(d-2)
  XOR delta_(d-1) delta_(d-2),                 (1)
```

where negative-lag defects are zero and `e_0,e_-` are the two ether
coefficients at that coordinate and time.  The coefficients have joint period
28.  On every prefix of width `L`, (1) is unitriangular: coordinate `d` is
`delta_d` XOR a function of lower coordinates.  It is therefore bijective,
and its inverse is obtained successively from `d=0` upward.

More strongly, every such map is a 2-adic isometry.  If two defect words first
differ at lag `k`, their lower inputs to the nonlinear term in (1) agree, so
their outputs still first differ at `k`.  Thus an added defect at lag `k`
never affects lower lags while both trajectories use the same free moving
interface.  An edit which becomes a new first defect instead requires a new
cut and changes both the phase and the represented wake; it is not a
composition on the old low interface state.

Let `Q_L` be the 28-halfstep return on `L` defect bits.  It is again
unitriangular and fixes bit zero.  Suppose the orbit of bits `0,...,i-1` has
period `P`.  During those `P` returns, bit `i` changes by the cocycle

```text
sigma_i = XOR_(j=0,...,P-1) phi_i(Q_L^j(delta)_(<i)).       (2)
```

If `sigma_i=0`, adjoining bit `i` preserves the period; if `sigma_i=1`, it
doubles it.  Induction gives

```text
period_Q(delta mod 2^L) = 2^s,  0 <= s <= L-1.             (3)
```

The phase-labelled halfstep orbit consequently has period `28*2^s`.
Sampling every 14 halfsteps, as in the period-seven truncation certificate,
gives period `2^(s+1) <= 2^L`.  This explains structurally why its exact
width-18 period 512 is a power of two.  The exponent eight in the associated
28-step return still comes from evaluating (2); triangularity alone does not
force that value.

## The 2-adic parameter and its dichotomy

For a fixed even residue `a` modulo 28, an isolated first defect has cut

```text
p = a + 28m
```

and initial moving-interface word `delta=1`.  On every finite prefix its
arrival state is `A_a Q_L^m(1)`, where `A_a` is the first `a` halfsteps of the
periodic interface.  By (3), this depends on `m` modulo a power of two and
therefore extends continuously from nonnegative integers to `m in Z_2`.

A latch condition through any fixed horizon reads only a finite arrival
prefix, so it is clopen in this 2-adic parameter.  Hence exactly one of the
following holds for the isolated-defect family:

1. some 2-adic parameter survives every latch, in which case ordinary
   nonnegative parameters approximate it for arbitrarily long horizons; or
2. no such parameter exists, in which case compactness supplies one uniform
   finite failure horizon, certifiable on a finite prefix.

This is a useful finite-orbit theorem, but it does not make escape automatic:
the first alternative actually occurs.

## An exact inverse-time survivor

Take residue `a=26`, whose spatial residue is five, and the 2-adic parameter
`m=-1`.  It represents the formal cut

```text
p = 26 + 28(-1) = -2.
```

Use (1) backward from the isolated word at time zero.  The only nonzero
defects are

```text
delta(0)  = 1000...
delta(-1) = 1000...
delta(-2) = 1100....                                  (4)
```

These identities are exact infinite-word identities: substituting in (1)
gives `U_-1(1)=1` and `U_-2(3)=1`, and unitriangularity makes each predecessor
unique.  At arrival, the actual half is the phase-two ether XOR `11`.  Apply
the prescribed boundary-zero halfstep.  Outputs at indices at least three are
unchanged by locality, while direct substitution at indices zero, one and two
gives the phase-three ether.  Thus

```text
T_0(E_phase2 XOR 11) = E_phase3.                       (5)
```

The ether then survives forever under the alternating compatible boundary.
The boundary discrepancy is harmless here because the changed low one
saturates the OR term; (5) checks the three possibly affected outputs rather
than assuming a bi-infinite continuation.

This formal survivor produces ordinary isolated wakes with arbitrarily long
post-arrival survival.  Set

```text
m_k = 2^k - 1,
p_k = 26 + 28m_k = 28*2^k - 2.                         (6)
```

For an arrival prefix of width `L`, choosing `k >= L-1` makes `m_k` congruent
to `-1` modulo `2^(L-1)`.  Equations (3)--(4) then make that entire prefix
identical to the formal survivor.  Testing `H` post-arrival latches needs only
`L=2H+2` arrival bits, so `k >= 2H+1` proves survival through that horizon.
There is therefore no constant post-arrival failure bound even for an
isolated single-bit defect.

The obstruction transfers to finite one-bit modifications of the ether
truncations.  Choose `r` so that

```text
7r > p_k + 2H + 1.
```

Then `n_r OR 2^p_k` agrees with the isolated periodic-tail configuration on
the complete dependency cone through the requested horizon.  Its first
defect is at `p_k` because `p_k=5 mod 7`, and it survives the same `H`
post-arrival latches.  The 2-adic point itself is not an ordinary negative-cut
configuration; it is an exact inverse-limit witness proving that its ordinary
finite approximants have no uniform horizon.

## A small exact one-edit family

There is also a much smaller counterexample to the old allowance of eight.
Flip ether bit 18 and evolve to its arrival.  The required 22-bit arrival word,
written low index first, is

```text
1010100001000001011010.
```

Under the forced macrostep `G=T_1 o T_0`, its low-pair residues through delay
ten are

```text
1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 0.
```

It therefore first fails at post-arrival delay ten.  That calculation reads
initial indices only through `18+2*10+1=39`.  For every `r>=6`, the first
truncation defect is at `7r>=42`, outside this cone, so locality proves the
exact all-parameter identity

```text
q(n_r OR 2^18) = 18/2 + 10 = 19,   r >= 6.             (7)
```

## Consequence for charge composition

The triangular structure preserves the discrepancy filtration and supplies
the parity cocycles (2), not a decreasing integer charge.  Those cocycles can
cancel, and the inverse-limit construction shows that one added defect can
carry an unbounded post-arrival delay as its phase parameter changes.  A
composition theorem must therefore retain the centre-boundary lineage and
permit unbounded local credit; assigning every edit one fixed delay allowance
cannot work.

There is a separate quantitative obstruction to the proposed zero-constant
bound.  The exact weight-six witness `4329177217` has `q=26>24=4 wt`, so even
`q(n)<=4 wt(n)` is false.  This does not refute `q(n)<=4 wt(n)+C` for a positive
constant, but it means any viable charge statement must relax both the fixed
per-interface allowance and the zero additive constant.
