# Temporal-pair cone density

This note tests the first refinement justified by the exact obstruction to the
predecessor-coherent pair density.  It changes the charge table, not the
predecessor memory: the charge sees two successive diagonal pairs, while the
spatial flux still runs on the same shared-seven-bit two-update relation.
The resulting finite language is infeasible.  This is a refutation of that
language, not of cone charge or of the desired Rule 30 density bound.

## Why this is the next representation

For the old density `F(B_t,C_t)`, genuine `q=4,...,7`, `n=1` cut inequalities
force

```text
Phi_5 + Phi_7 - 2 Phi_3 >= 6.
```

The left side splits into three closed walks for the terminal `chi` flow.
Consequently no enlargement of `psi` memory can repair the contradiction
while retaining the same pair-valued `F` and `chi`.  This is stronger than a
generic failed LP: it identifies the charge table as the missing dependency.

Put `P_t=(B_t,C_t)` and refine the density to

```text
U_t = (P_t,P_(t+1)),        F_(p,tau)(U_t).
```

The old combination no longer cancels.  In particular, its `q=3` bottom
context has `U_0=(00,11)`, whereas the `q=5,7` contexts have `U_0=(00,00)`.
A bottom correction `g_p(U_0)` retains the same justified temporal context.
No third row is added to `F`, and no longer predecessor word is added to the
relation.

## Shared two-update relation

For one seven-bit word `(u,v,w,z,A,B,C)`, define

```text
r  = (A,B,C),
r' = (A XOR (z OR w),
      B XOR (A OR z),
      C XOR (A OR B)),

z' = z XOR (w OR v),       w' = w XOR (v OR u),

r'' = (A' XOR (z' OR w'),
       B' XOR (A' OR z'),
       C' XOR (A' OR B')).
```

As before, `S=(r,r') -> S'=(r',r'')` only when one shared word produces all
three rows.  Enumeration gives 20 states and 44 edges.  Projection

```text
pi((r,r')) = ((B,C),(B',C')) = U
```

gives 10 temporal-pair states and 22 coherent projected edges `U->U'`.
These 22 edges, rather than a composition of independently witnessed one-row
updates, are used for the terminal flow.

## Exact spatial difference

For `p=q mod 2`, define

```text
Phi_q = sum_(t=0..q-2) F_(p,t mod 2)(U_t) + g_p(U_0).
```

For `S=(r,r')`, with `r=(A,B,C)` and `r'=(A',B',C')`, put

```text
L_(p,tau)(S)
  = F_(p,tau)(((B,C),(B',C')))
    - F_(1-p,tau)(((A,B),(A',B'))),

G_p(S)
  = g_p(((B,C),(B',C')))
    - g_(1-p)(((A,B),(A',B'))).
```

Then, for every `q>=3`,

```text
Phi_q-Phi_(q-1)
 = sum_(t=0..q-3) L_(p,t mod 2)(S_t)
   + F_(p,(q-2) mod 2)(U_(q-2)) + G_p(S_0).
```

Introduce `psi_(p,tau)(S)` and `b_p`.  On every one of the 44 coherent edges,
require

```text
L_(p,tau)(S)+psi_(p,1-tau)(S')-psi_(p,tau)(S) >= 0.
```

At the bottom require

```text
alpha*C-1 + psi_(p,0)(S) + G_p(S) >= b_p.
```

The interior edge sum is `t=0,...,q-4`.  The remaining `L` term and final
density are supplied by one exact top cap.

For even `q`, write `b in {0,1}`.  Here `t=q-3` is odd and

```text
r=(1,1-b,b),     r'=(0,b,1-b),
U_(q-2)=((b,1-b),(1,1)).
```

Require

```text
L_(0,1)((r,r')) + F_(0,0)(U_(q-2))
  - psi_(0,1)((r,r')) >= -b_0.
```

For odd `q`, with `u,v in {0,1}`, `t=q-3` is even and

```text
r=(u,1-u,v),     r'=(1,1,1-v),
U_(q-2)=((1,1-v),(0,v)).
```

Require

```text
L_(1,0)((r,r')) + F_(1,1)(U_(q-2))
  - psi_(1,0)((r,r')) >= -b_1.
```

These formulas also cover `q=3`: the interior range `0,...,-1` is empty,
`S_(q-3)=S_0`, and the bottom and top `psi` terms cancel directly.  Thus the
same summation proves

```text
alpha*e_q - 1 + Phi_q-Phi_(q-1) >= 0       (q>=3).
```

## Initialization and terminal constant

At `q=2`, the exact latch is

```text
U_0=((x,1-x),(1,1)),        x in {0,1},
Phi_2=F_(0,0)(U_0)+g_0(U_0).
```

For any finite tables, initialization therefore holds with the explicit
finite choice

```text
J = min_(x=0,1) [alpha-2+F_(0,0)(U_0)+g_0(U_0)].
```

It does not constrain synthesis when an arbitrary finite `J` is allowed.

For every coherent projected edge `U->U'`, impose

```text
F_(p,tau)(U)+chi_(p,1-tau)(U')-chi_(p,tau)(U) <= 0.
```

Summing for `t=0,...,q-3` bounds all but the final `F(U_(q-2))`.  Let
`T_p` be the finite top sets displayed above and let `U` range over the ten
projected states.  An explicit endpoint constant is

```text
K_p = max_U [g_p(U)+chi_(p,0)(U)]
      + max_(V in T_p) [F_(p,p)(V)-chi_(p,p)(V)],
K = max(K_0,K_1).
```

Hence `Phi_q<=K` for every `q>=2`.  As with `J`, arbitrary finite `K` adds no
LP constraint.  If the local system were feasible, summing the spatial
inequality would give

```text
alpha*sum_(q=1..N)e_q - N >= J-K.
```

## Exact result

`experiments/temporal_pair_density.py` tested phase-independent and
time-parity-dependent tables at `alpha=8`, then repeated both with `alpha` as
a variable subject only to `alpha>=0`.  All four systems are infeasible.
Exact nonnegative integer Farkas multipliers give

| time parity | alpha | variables | inequalities | witness rows | identity |
|---|---:|---:|---:|---:|---:|
| no | 8 | 102 | 178 | 19 | `0>=4` |
| no | variable | 103 | 179 | 19 | `0>=4` |
| yes | 8 | 182 | 310 | 24 | `0>=4` |
| yes | variable | 183 | 311 | 24 | `0>=4` |

The variable-alpha witnesses omit the `alpha>=0` row and use bottom rows with
`C=0`, so their alpha coefficients vanish.  No initialization or terminal
endpoint inequality exists in the matrices; `J` and `K` are the explicit
finite extrema above.  Changing either constant cannot repair the result.

The tests independently enumerate the shared seven-bit projection, verify the
spatial-difference identity for `q=3,...,7`, check the exact top caps, and
reassemble every contributing Farkas row directly from its saved label without
calling the matrix builder.  In each case all variable coefficients cancel
and the right-hand side is exactly four.

The temporal-pair charge does what motivated it: it distinguishes the old
three-cycle obstruction.  A different exact obstruction survives.  This is
the mandated stopping point for this representation.  Infeasibility alone
does not justify another memory expansion.
