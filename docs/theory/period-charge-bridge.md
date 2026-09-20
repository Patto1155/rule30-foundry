# Conditional bridge from cone charge to other periods

This note separates exact algebra from the missing charge construction. It
does not supply a period-two certificate or change a claim-ledger grade.

## Phase constraints and their exact cones

Write `x_j(t)=a(t,-j)` for `j>=1`, with prescribed centre `x_0(t)=u_t`.
Rule 30 is exactly

```text
x_j(t+1) = x_(j+1)(t) XOR (x_j(t) OR x_(j-1)(t)).
```

For `e_j=x_j(0)`, induction on time gives

```text
x_1(t) = e_(t+1) XOR H_t(e_1,...,e_t; u_0,...,u_(t-1)).
```

More generally, the coefficient of the deepest variable `e_(j+t)` in
`x_j(t)` is one: the deeper-neighbour summand carries that variable, while
both OR inputs depend on strictly shallower initial variables. This proves
the identity without assuming a right-half reconstruction.

The centre equation at every phase with `u_t=1` requires

```text
x_1(t) = 1 XOR u_(t+1).
```

Consequently each such phase fixes exactly one new initial bit `e_(t+1)`.
Each phase with `u_t=0` leaves that new bit free in this necessary left-only
system. For `N` initial bits and checks at times `0,...,N-1`, the exact number
of admissible left prefixes is therefore

```text
2^(number of zero phases among u_0,...,u_(N-1)).
```

For `u=01`, this is `2^ceil(N/2)`. This count is a useful completeness check
for cone enumeration, not a weight bound. These prefixes need not extend to
full Rule 30 orbits with the prescribed centre.

## The endpoint constant matters

The original sharp conjecture in `alternating_left_weight.py` was

```text
W_N >= floor((N+6)/8),  where W_N = sum_(j=1..N) e_j.
```

Since `W_N` is integral, its exact equivalent is

```text
8 W_N - N >= -1.
```

This unrestricted conjecture is refuted by the weight-six witness in
`../experiment-logs/2026-09-19-zero-tail-composition.md`. The equivalence
and endpoint arithmetic remain useful; the sharp inequality is not a live
unrestricted claim.

At `N=2m`, parity strengthens the right side to zero, giving
`W_(2m)>=ceil(m/4)`. The proposed relaxed bound `8 W_N-N>=-6` is not
equivalent: at `N=2`, it admits `W_N=0`. Nevertheless, any fixed lower
bound on `8 W_N-N` forces unbounded initial weight and suffices for
period-two exclusion. A certificate may prove this weaker conclusion
without claiming the sharp finite-prefix inequality.

## A conditional all-nonconstant-period lemma

Fix a nonconstant cyclic word `u` of period `p`, extended periodically.
Let `D_N` count indices `0<=t<N` with `(u_t,u_(t+1))=(1,0)`. Let `d`
be their number in a full period, so `d>=1` and
`D_N>=d floor(N/p)`.

Suppose a finite cut-state graph represents every admissible initial cone,
with paths `s_0,...,s_q` having these three properties:

1. Edge weights `w_i` count each initial one exactly once in total, so
   `sum w_i=W_N`. Edge demands `d_i` similarly sum to `D_N`.
2. Every legal edge satisfies, with an integer `A_u>0`,
   `A_u w_i-d_i+Phi(s_(i+1))-Phi(s_i)>=0`.
3. For all allowed initial and terminal cut states,
   `Phi(s_q)-Phi(s_0)<=C_u`, with `C_u` independent of `N`.

Summing the local inequalities gives

```text
A_u W_N - D_N >= Phi(s_0)-Phi(s_q) >= -C_u.
```

Thus `W_N >= (d floor(N/p)-C_u)/A_u`, which diverges with `N`.
No finite initial left half can meet every phase constraint. An actual
finite-configuration orbit is among these necessary left-only paths, so
its centre cannot have eventual period word `u`: restart at the alleged
onset and apply the inequality there. Rule 30 preserves finite support
at every finite time because its zero background is quiescent.

For a finite state set, one possible endpoint bound is
`C_u=max Phi(terminal states)-min Phi(initial states)`. A boundary credit
that grows with `N` does not meet the hypothesis. Phase information modulo
`p` must be retained whenever it is needed to make the edge rule sound.

This is a conditional proof template. The hard, unproved part is providing
the sound graph and potential. Constants may depend on `u`; a positive
charge coefficient uniform over all periods is unnecessary to exclude
each fixed period separately.

## Why latch gaps alone do not transfer the alternating proof

A `1 -> 0` event forces a one at a later spacetime cell, not a new one in
the initial row. Its backward cone overlaps earlier cones. One initial
one can influence many events, so counting latch events does not establish
an injective assignment or a bounded multiplicity charge.

Consecutive `1 -> 0` gaps also omit boundary information. The primitive
words `000001` and `000111` both have one such event per six steps, in
the same phase, but have different intervening `1 -> 1` constraints and
different transfer maps: `G_000001(0)=1`, while `G_000111(0)=6`.
These inputs fail the period checks; the map comparison only demonstrates
that identical event gaps do not define the same dynamics.

A generalization must retain the whole phase word, both zero-run and
one-run lengths, or explicitly prove a worst-case relaxation covering all
intervening boundary histories. Merely substituting a maximum latch gap
into an alternating constant has no established justification.

## Constant words require separate handling

An eventually constant-one centre is excluded directly by the local rule.
After restarting, its left neighbour must be the constant-zero column.
Sideways inversion then forces columns `-2,-3,-4,...` to be constant
`1,0,1,...`, respectively. The initial left half has infinitely many ones,
contradicting finite support.

For a constant-zero centre the left-only latch system imposes no checks.
Every ordinary integer remains admissible, and `T_0(0)=0`. Therefore an
all-nonconstant-word charge theorem still requires a separate constant-zero
argument. A self-contained audit is now given in
`all-period-bridge-followup.md`: a constant-zero centre makes its right
neighbour Boolean nondecreasing and hence eventually constant; two adjacent
eventually periodic columns are impossible for a nonzero finite Rule 30
configuration. The latter fact is proved there by common-onset sideways
periodicity, finite support over one period, and zero-pair propagation.
This supplies the needed endpoint without relying on the external preprint's
classification or finite-horizon claims. It does not change any ledger grade
or claim novelty for period one.

## Controls

- Rule 15 obeys `a(t,x)=a(0,x-t) XOR (t mod 2)`. A nonzero finite initial
  configuration supported strictly to the right has an alternating centre
  and an all-zero initial left half. Triangularity alone cannot imply a
  positive initial-weight charge. Its later odd-time configurations need
  not have finite support, so this control concerns finite *initial* data.
- Rule 90 preserves reflection symmetry. From the single-cell seed,
  the two neighbours of the centre are always equal, hence the centre
  is zero after the first step. Generic left-permutivity or defect claims
  cannot provide the separate constant-zero exclusion.

The next useful bridge obligation, after a period-two certificate exists,
is to prove a sound composition rule for its cut states across an arbitrary
specified zero-run/one-run pair, with bounded endpoint credit.
