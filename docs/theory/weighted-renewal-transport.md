# Weighted renewal transport: an exact local-credit obstruction

This note tests a cut-wide transport mechanism on the physical gap-three/five
branch. It obtains an obstruction to a specified family, not a spatial charge
bound or period-two exclusion. The result persists after the physical right
half has entered the previously proved invariant. It therefore cannot be
repaired merely by replacing the `11`-free relaxation with that invariant.

## Exact cuts and the sign convention

Use the characteristic cuts from `slanted-cone-geometry.md`. At depth `q`, put

```text
P_t = (D_(q-1)(t), D_q(t)),           0 <= t < q,
p = q mod 2,                        tau = t mod 2.
```

Both full diagonals are retained while constructing each witness. Define
`E_q` to contain `p` and the bottom two and top eight bits of **each**
diagonal. These endpoint words include their actual centre endpoints and
are not independently resampled.

Consider the credit functional

```text
Psi_q = sum_(t=0..q-3) F_(p,t mod 2)(P_t,P_(t+1),P_(t+2)) + g(E_q),
F_(p,tau)(window) <= 0.
```

Here `F` and `g` are arbitrary real tables on their finite alphabets. Thus
`Psi_q <= max g` uniformly in depth even though `Psi_q` can accumulate
unbounded negative credit. The proposed weighted transport inequality is

```text
A * weight(initial cells q+1,...,q+7) - 7
    + Psi_(q+7) - Psi_q >= 0.                         (1)
```

This uses the **upper-bounded credit** convention. In the earlier lower-bounded
potential convention `A*w-7 >= Phi(next)-Phi(current)`, set `Phi=-Psi`.
Consequently this note's nonpositive credit density corresponds to a
nonnegative density in `Phi`; these two signs must not be interchanged.

**Result.** No finite `A` and no such tables satisfy (1) at every finite
physical alternating-prefix cut, even restricted to initial right halves
whose low eight bits are `167`, `168`, or `170`. The same result covers
nonpositive credit densities on one or two consecutive pairs, because their
last one or two omitted summands can be absorbed into `g(E_q)`.

The named endpoint lengths and density width matter. This is not an
obstruction to arbitrary functionals of the entire ordered cut.

## The exact obstruction certificate

`experiments/weighted_renewal_transport.py` contains 45 finite full-row
witnesses, with positive integer multiplicities. Each gives a seven-cell
slab whose initial weight is zero. The verifier independently evolves the
full row with the eight-entry elementary Rule-30 truth table. It checks:

1. The centre is `t mod 2` through the slab's final depth.
2. Every initial cell in the slab is zero.
3. The right integer has low eight bits in `{167,168,170}`.
4. The weighted source and target endpoint-feature multisets agree exactly.
5. For every phase-labelled three-pair word, its weighted target count minus
   source count is nonnegative.

The third check invokes the universal-entry-cylinder theorem in
`right-coherence-followup.md`. No assertion that every finite right half
enters the invariant is needed. The forced right history of every witness
is in the gap-three/five relaxation indefinitely; its full alternating
centre is asserted only for the checked finite prefix.

Let `m_i` be the positive multiplicities, and let `h_i(q)` be the histogram
of `(p,tau,P_t,P_(t+1),P_(t+2))`. The exact certificate gives

```text
sum_i m_i [g(E_target_i)-g(E_source_i)] = 0,
Delta_h = sum_i m_i [h_i(target)-h_i(source)] >= 0 componentwise,
M = sum_i m_i = 6110652545858014158278360985185689401228.
```

There are 41 strictly positive histogram coordinates; all other coordinates
have zero drift. The sum of all histogram drifts is exactly `7M`, as required
because every slab adds seven temporal windows. Therefore

```text
sum_i m_i [Psi_target_i-Psi_source_i] = F dot Delta_h <= 0.
```

But summing (1), with every slab weight zero, requires the same quantity to
be at least

```text
7M = 42774567821006099107948526896299825808596 > 0.
```

This is a direct integer contradiction. Neither the coefficient `A` nor an
initialization allowance appears. Increasing either cannot repair this
family. The large multiplicities are exact integer data, not numerical
solver tolerances or repeated physical orbits.

The endpoint features balance in the certificate; the complete cuts do not.
This circulation cannot be assembled into an infinite physical alternating
orbit. Its purpose is to cancel the finite endpoint table while exposing the
information discarded by the local-density sum.

## Signed densities with a local terminal flux also fail

The width-three result excludes a useful signed enlargement without another
search. Suppose instead that

```text
S_t = (P_t,P_(t+1)),
Psi_q = sum_(t=0..q-2) J_(p,t mod 2)(S_t) + g(E_q),
```

where `J` may have either sign. Suppose its terminal upper bound is to be
certified by a local flux `chi_(p,tau)(S)` satisfying

```text
J_(p,tau)(S_t) + chi_(p,1-tau)(S_(t+1))
                       - chi_(p,tau)(S_t) <= 0        (2)
```

on every actual three-pair window at the cuts in question. Define the
nonpositive width-three density to be the left side of (2). Exact
telescoping gives the same `Psi_q` after changing its endpoint table to

```text
g'(E_q) = g(E_q) + J_(p,p)(S_(q-2))
                    + chi_(p,0)(S_0) - chi_(p,p)(S_(q-2)).
```

All extra quantities are determined by `E_q`: `S_0` needs the two bottom
bits of both diagonals; `S_(q-2)` needs the top two older-diagonal bits and
top three newer-diagonal bits. Thus this signed-density-plus-local-flux
family is exactly a subfamily of the refuted width-three credit language.
It fails even if the local terminal relation is restricted to actual windows
of the physical witnesses. The proof does not rely on spurious independent
predecessor strips.

This is narrower than excluding signed densities altogether. A signed
density whose total is bounded by a genuinely cone-conditioned argument,
which cannot be written as the local flux (2), remains possible.

## What this changes in the transport search

The right invariant is useful, but it does not by itself make this local
credit sum composable. The new test retains a histogram from the **whole**
cut and up to three consecutive observations, unlike the previous pure
endpoint obstructions. It still loses the ordering and centre-boundary
lineage connecting distant parts of that cut.

The precise remaining option is an order-sensitive or cone-conditioned
credit bound: for example, a signed accumulated quantity whose terminal
upper bound uses inherited centre lineage rather than a free local flux.
The certificate gives no justification for blindly adding a fourth temporal
observation. Nor does it establish that this next option succeeds.

The inequality tested here quantifies over every finite physical alternating
prefix. A proof using a smaller class would need an independently justified
condition that excludes the witnesses; it cannot silently assume that the
prefix has an infinite alternating extension.

## Reproduction and scope

```bash
python experiments/weighted_renewal_transport.py
```

The default uses only exact integer counts and full-row truth-table replay;
no optimizer is needed. Discovery is optional:

```bash
python experiments/weighted_renewal_transport.py --discover
```

Discovery uses the existing exact forced-right map and growing-diagonal
constructor, then linear programming proposes a circulation. Rational
Gaussian elimination recovers a candidate, and the default-style independent
replay and integer checks determine acceptance. Optimizer infeasibility or
absence of a sampled witness is not a theorem. The search stopped after the
named three-observation obstruction was found; no broader survival survey
was performed.
