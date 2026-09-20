# Two-update predecessor-coherent density language

This note specifies a finite sufficient certificate language for the slanted
cuts in `slanted-cone-geometry.md`.  It repairs one demonstrated relaxation
error: the predecessor bits used in two consecutive temporal updates are now
the same bits, evolved once by Rule 30.  The language remains a relaxation of
the reachable growing cuts.  Feasibility is not asserted here, and nothing in
this note is a period-two theorem or a period-exclusion theorem.

## The coherent two-update relation

For a fixed spatial step `q`, put

```text
r_t = (A_t,B_t,C_t)
    = (D_(q-2)(t),D_(q-1)(t),D_q(t)).
```

Let seven adjacent bits on the original row be ordered from older to newer
diagonals as

```text
(u,v,w,z,A,B,C).
```

For every one of the `2^7` assignments, define

```text
A' = A XOR (z OR w)
B' = B XOR (A OR z)
C' = C XOR (A OR B)

z' = z XOR (w OR v)
w' = w XOR (v OR u)

A'' = A' XOR (z' OR w')
B'' = B' XOR (A' OR z')
C'' = C' XOR (A' OR B').
```

Write `r=(A,B,C)`, `r'=(A',B',C')`, and
`r''=(A'',B'',C'')`.  The finite state alphabet is

```text
S = { (r,r') : some (u,v,w,z,A,B,C) produces r,r' }.
```

There is a phase-labelled edge

```text
(r,r') --tau--> (r',r'')
```

for either `tau in {0,1}` exactly when at least one *single* seven-bit
assignment produces all three rows by the displayed equations.  The phase
label records `tau=t mod 2`; the Boolean relation itself is phase independent.
This existential quantifier is local: it makes the graph a sound relaxation,
not an assertion that every graph walk is a reachable cone.

The last interior edge, starting at `t=q-4`, needs an explicit coverage check.
At that time the seven bits occupy physical offsets

```text
(u,v,w,z,A,B,C) at j=(-2,-1,0,1,2,3,4),
```

where negative `j` lies to the right of the centre.  Those two right-side bits
are not entries of the retained left diagonals, but they are actual cells of
the full Rule 30 spacetime.  Applying the local rule to this same seven-cell
patch gives exactly `z'=z XOR (w OR v)`, `w'=w XOR (v OR u)`, and the displayed
formulas for `r'` and `r''`.  Therefore every actual near-centre edge has a
witness in the enumerated relation.  Allowing all values of `u,v` can add
edges when the prescribed centre would restrict them; it cannot delete an
actual edge.  The subsequent two rows are handled by the exact top cap rather
than by pretending that these right-side cells are left-diagonal entries.

Enumeration gives 20 states, 44 coherent edges, and 48 two-edge walks if the
two one-update witnesses are instead chosen independently.  The four walks
removed by the shared seven-bit witness are

```text
000 -> 100 -> 111
001 -> 101 -> 110
010 -> 111 -> 100
011 -> 110 -> 101.
```

For the first walk, `000 -> 100` forces `z=0,w=1`.  The shared predecessor
then has `z'=z XOR (w OR v)=1` for both values of `v`.  Consequently the next
first component is `1 XOR (z' OR w')=0`, whereas `111` requires it to be one.
Thus the exclusion follows directly from the seven original bits; it does not
come from a reachability sample.

## Potential and coherent flux inequalities

Fix a coefficient `alpha>0`.  For `p=q mod 2` and `tau=t mod 2`, let
`F_(p,tau)(B,C)` be a finite signed density table and let `g_p(B,C)` be an
optional bottom correction.  Define

```text
Phi_q = sum_(t=0..q-1) F_(p,t mod 2)(D_(q-1)(t),D_q(t))
        + g_p(D_(q-1)(0),D_q(0)).
```

For a triple `r=(A,B,C)`, define

```text
L_(p,tau)(r) = F_(p,tau)(B,C)-F_(1-p,tau)(A,B),
G_p(r)       = g_p(B,C)-g_(1-p)(A,B).
```

Then the exact spatial difference is

```text
Phi_q-Phi_(q-1)
 = sum_(t=0..q-2) L_(p,t mod 2)(r_t)
   + F_(p,1-p)(D_(q-1)(q-1),D_q(q-1))
   + G_p(r_0).
```

Let `psi_(p,tau)(S)` be a flux on two-row states and `b_p` a scalar.  The
following finite inequalities are sufficient.  Every quantifier below ranges
over the explicitly finite sets stated here.

1. For each `p,tau in {0,1}` and each coherent edge
   `S=(r,r') -> S'=(r',r'')`, require

   ```text
   L_(p,tau)(r)+psi_(p,1-tau)(S')-psi_(p,tau)(S) >= 0.
   ```

2. For each `p in {0,1}` and every state `S=(r,r')` in the 20-state alphabet,
   with `r=(A,B,C)`, require the bottom inequality

   ```text
   alpha*C-1 + psi_(p,0)(S)+G_p(r) >= b_p.
   ```

   On an actual cut, `C=D_q(0)=e_q`; `S=(r_0,r_1)` is produced by its seven
   adjacent bottom bits.  Quantifying over all 20 locally produced states is a
   sound relaxation of the exact reachable bottom set.

3. The top states are fixed by the alternating centre boundary, rather than
   by a fresh existential predecessor.  For even `q` (`p=0`), for each
   `b in {0,1}` require

   ```text
   r  = (1,1-b,b),       r' = (0,b,1-b),       tau=1,
   L_(0,1)(r)+L_(0,0)(r')+F_(0,1)(1,1)-psi_(0,1)((r,r')) >= -b_0.
   ```

   For odd `q` (`p=1`), for each `u,v in {0,1}` require

   ```text
   r  = (u,1-u,v),       r' = (1,1,1-v),       tau=0,
   L_(1,0)(r)+L_(1,1)(r')+F_(1,0)(0,v)-psi_(1,0)((r,r')) >= -b_1.
   ```

The top formulas are exact local boundary conditions.  To derive them, put
`t=q-3`.  If `q` is even, the distance-one bit at time `t` is one; updating
through the centre bit one gives `(1,1-b,b)->(0,b,1-b)`.  If `q` is odd, call
the distance-one and distance-three bits `u` and `v`; updating through centre
bit zero gives `(u,1-u,v)->(1,1,1-v)`.  The final density pairs at time `q-1`
are respectively `(1,1)` and `(0,v)`.

For every `q>=3`, sum the interior inequality for `t=0,...,q-4`, the bottom
inequality, and the applicable top inequality.  The `psi` and `b_p` terms
cancel, including when `q=3` and the interior sum is empty.  The exact result
is

```text
alpha*e_q-1+Phi_q-Phi_(q-1) >= 0.
```

This explains why the state must contain two consecutive rows: a one-row flux
would again compose independently witnessed updates and would admit the four
false walks above.

## Initialization and a uniform terminal upper bound

At depth two the exact latch is `(e_1,e_2)=(x,1-x)` for `x in {0,1}`.  Hence

```text
Phi_2 = F_(0,0)(x,1-x)+F_(0,1)(1,1)+g_0(x,1-x).
```

Choose a fixed initialization constant `J` and require, for both `x=0,1`,

```text
alpha-2+Phi_2 >= J.
```

Signed densities need an independent certificate that `Phi_q` is bounded
above uniformly in the cut length.  Introduce a finite table
`chi_(p,tau)(B,C)`.  For every `p,tau`, every pair `P=(B,C)`, every
`a,z in {0,1}`, and

```text
P' = (B XOR (a OR z), C XOR (a OR B)),
```

require

```text
F_(p,tau)(P)+chi_(p,1-tau)(P')-chi_(p,tau)(P) <= 0.
```

This one-update relation contains every bulk pair transition and both special
final centre-touching transitions.  It is allowed to contain extra ones.
Finally choose a finite scalar `K` and impose the following endpoint rows:

* for `p=0`, every bottom pair `P_0 in {0,1}^2`:

  ```text
  g_0(P_0)+chi_(0,0)(P_0)+F_(0,1)(1,1)-chi_(0,1)(1,1) <= K;
  ```

* for `p=1`, every bottom pair `P_0 in {0,1}^2` and every `v in {0,1}`:

  ```text
  g_1(P_0)+chi_(1,0)(P_0)+F_(1,0)(0,v)-chi_(1,0)(0,v) <= K.
  ```

Summing the `chi` inequalities from `t=0` through `q-2` and then applying the
appropriate endpoint row proves `Phi_q<=K` for every `q>=2`.  The independent
bottom/top cross-product in the endpoint rows is a sufficient finite
relaxation; it makes no unproved claim about which endpoint pairs occur
together.

Combining initialization, the coherent spatial inequalities for
`q=3,...,N`, and the terminal bound yields, for every integer `N>=2`,

```text
alpha*sum_(q=1..N)e_q-N >= J-K.
```

For the sharp Rule 30 prefix target one may set `alpha=8`, `J=-1`, and `K=0`;
`N=1` is checked directly.  More generally, any feasible fixed
`alpha>0,J,K` gives a positive asymptotic lower density and rules out a
finite-support survivor.  Searching other finite positive values of `alpha`
is therefore meaningful if the sharp `alpha=8` system is infeasible.  A
failure of this finite system refutes only this certificate language.

The first justified enlargement, if an exact Farkas witness shows that the
pair density itself is the remaining bottleneck, is a density on the temporal
pair `S=(r_t,r_(t+1))`.  That enlargement would need new bottom, top, and
terminal telescoping formulas; it must not be inferred merely from
infeasibility of the present system.  The named need is dependence of the
charge density on a correlation that `F(B_t,C_t)` cannot distinguish.
