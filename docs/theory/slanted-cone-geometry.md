# Slanted characteristic cuts for alternating Rule 30

This note derives a sound geometry and sufficient certificate conditions. It
contains no cone-charge certificate and does not establish period exclusion.

## Exact growing cut

Write `a(t,j)` for the cell at physical coordinate `-j`, with centre boundary
`a(t,0)=t mod 2`. Rule 30 gives, for `j>=1`,

```
a(t,j+1) = a(t+1,j) XOR (a(t,j) OR a(t,j-1)).
```

The characteristic coordinate is `q=t+j`. Define

```
D_q(t) = a(t,q-t),                 0 <= t <= q.
D_q(t) XOR D_q(t+1)
          = D_(q-1)(t) OR D_(q-2)(t),   0 <= t <= q-2.
```

Both entries on the left lie on the same slanted characteristic. Its endpoints
are `D_q(q)=q mod 2` and `D_q(q-1)=a(q-1,1)`. The latter is forced to one for
even `q`; it is free for odd `q`. From this endpoint the recurrence determines
the entire new diagonal downwards. Its bottom value is the emitted initial
bit `e_q=D_q(0)`.

Thus `(D_(q-1),D_q)` is an exact growing cut. Retaining both complete diagonals
preserves the top latch constraint; replacing their far endpoint by fresh
bits repeats the forbidden temporal-window relaxation. The state has growing
length: this construction does not assert a finite exact automaton.

For comparison with `induced_left_prefix`, its input bit `s_k` gives the free
endpoint `D_(2k+1)(2k)=1 XOR s_k`. A direct implementation of these equations
matched that constructor for every boundary word at `m=1,...,6` (126 words).
This checks indexing only; exactness follows from the displayed local rule.

## Four-macrotime by eight-space tiles

At even time define `X_q(k)=a(2k,q-2k)`. Away from the centre boundary,
`q>=2k+4`, put `v_i=X_(q-i)(k)`. Then

```
X_q(k+1) = v_0 XOR (v_1 OR v_2)
  XOR ((v_1 XOR (v_2 OR v_3)) OR (v_2 XOR (v_3 OR v_4))).
```

An `8`-wide by `4`-high rectangle in `(q,k)` is a parallelogram in physical
spacetime. Its inputs are eight bottom bits and the four-column left halo at
each of four times: 24 input bits. Its outputs are eight top bits and the
four-column right halo at those four times. The interior transition is
exactly deterministic. The `2^24` assignments are local inputs, not a count of
reachable cone states. Adjacent tiles compose only when their entire shared
boundary words agree.

Tiles touching the centre need separate boundary relations:

```
X_(2k)(k)=0,
X_(2k+1)(k) XOR X_(2k+2)(k)=1.
```

Under the latter latch, the special distance-one update is
`X_(2k+3)(k+1)=X_(2k+3)(k)`. Applying the bulk formula across this boundary is
invalid. A whole cone has an increasing number of tile interfaces; a single
bounded state does not follow from this finite local alphabet.

## Charge and exact endpoint requirements

For `W_N=sum_(q=1..N)e_q`, assign spatial charge `c_q=8e_q-1`.
If a potential obeys

```
c_q + Phi_q - Phi_(q-1) >= 0,
```

then exactly

```
8 W_N-N >= Phi_0-Phi_N.
```

The desired terminal condition is therefore an **upper** bound on `Phi_N`.
A finite local-density alphabet on a growing cut can provide this without a
finite total state: take nonpositive local densities, whose sum may accumulate
arbitrarily negative credit, plus bounded endpoint terms. A potential bounded
on both sides would impose a stronger bound on every interior subpath; the
prefix conjecture alone does not justify that requirement.

There is an arithmetic correction to the supplied brief:

```
W_N >= floor((N+6)/8)  <=>  8 W_N-N >= -1.
```

At even `N=2m`, this is equivalently `8W_N-N>=0` or `4W_N-m>=0`.
The proposed `8W_N-N>=-6` is weaker, equivalent to
`W_N>=floor((N+1)/8)`. A uniform bound with *any* finite endpoint constant
would still exclude finite-support survivors; the sharp constant is optional
for period exclusion.

## A concrete local-density/flux certificate language

For a first small language let `p=q mod 2` and

```
Phi_q = sum_(t=0..q-1) F_p(D_(q-1)(t),D_q(t)),   F_p <= 0.
```

At `q>=3` write `r_t=(A_t,B_t,C_t)=(D_(q-2)(t),D_(q-1)(t),D_q(t))`, and

```
L_p(A,B,C) = F_p(B,C)-F_(1-p)(A,B).
```

The exact difference `Phi_q-Phi_(q-1)` is `sum_(t=0..q-2)L_p(r_t)` plus the
top contribution `F_p(D_(q-1)(q-1),D_q(q-1))`. Introduce a local flux
`psi_p(A,B,C)` and a scalar `b_p`. Sufficient finite inequalities are:

1. For every admissible interior pair `r,r'`,
   `L_p(r)+psi_p(r')-psi_p(r)>=0`.
2. For every admissible bottom triple `(A,B,C)`,
   `8C-1+psi_p(A,B,C)>=b_p`.
3. At every admissible top endpoint, `L_p(r)+F_top-psi_p(r)>=-b_p`.

Interior pairs must at least obey `C'=C XOR (A OR B)`. Allowing all pairs
with only that relation is a sound relaxation, but may be too weak. Older
adjacent diagonals or consecutive rows can retain additional dependencies.
Do not omit an actual transition merely because it is inconvenient.

The top assignments are explicit:

| `q` parity | Last triple `r_(q-2)` | `F_top` | Free bit |
|---|---|---|---|
| Even | `(0,b,1 XOR b)` | `F_0(1,1)` | `b` |
| Odd | `(1,1,1 XOR v)` | `F_1(0,v)` | `v` |

Sum interior inequalities for `t=0,...,q-3`, then add bottom and top
inequalities: all fluxes cancel, yielding `c_q+Phi_q-Phi_(q-1)>=0`.
Initialize at `q=2`: the latch gives `(e_1,e_2)=(x,1 XOR x)`, so the
cumulative charge is six and

```
Phi_2 = F_0(x,1 XOR x)+F_0(1,1).
```

It suffices to verify `6+Phi_2>=-1` for both `x`; telescoping from `q=3`
and using `Phi_N<=0` then gives the sharp full-prefix bound for `N>=2`.
`N=1` satisfies that bound directly. Requiring `6+Phi_2>=0` is a stronger
possible initialization. In contrast, imposing the transition inequality
from `Phi_0=0` at `q=1` with only nonpositive densities is impossible when
`e_1=0`: its charge is minus one. A verified interior table cannot erase
these endpoint costs. Phase-dependent endpoint corrections and
time-parity-dependent local densities can be added with their differences
explicitly included.

If the small relaxation has a negative cycle or an infeasible inequality
system, that refutes this certificate language. It does not refute the
cone-charge conjecture. The concrete next obligation is an integer solution
of a sound refined local-density/flux system, together with explicit finite
initialization and terminal upper bounds.

## Exact obstruction and the next required correlation

The synthesis experiment `experiments/slanted_density_certificate.py` produced
integer Farkas obstructions for fourteen specified languages, including signed
densities with a second terminal flux and bottom corrections. These are
refutations of certificate languages, not counterexamples to cone charge.

The smallest obstruction is particularly informative. Without time parity,
adding the even-phase bottom/top inequalities at `r=010` gives

```
F_0(10)-F_1(01)+F_0(11) >= 1.
```

Adding the odd-phase inequalities at `r=110` gives

```
F_1(10)-F_0(11)+F_1(01) >= 1.
```

Their sum requires `F_0(10)+F_1(10)>=2`, contradicting pointwise nonpositivity.
No interior or initialization inequality appears in this six-row obstruction.
Changing the initial endpoint allowance therefore cannot repair it. The
three-equation, time-parity version similarly forces

```
F_(0,0)(01)+F_(0,1)(10)+F_(1,0)(00)+F_(1,1)(10) >= 4,
```

contradicting its four corresponding nonpositivity inequalities.

Pointwise nonpositivity is unnecessarily strong: if the current diagonal pair
is `(B,C)=(1,0)`, the next `C` is necessarily one. For example the signed
local density

```
f(B,C)=B(1-C)-C
```

satisfies `f(B,C)<=C'-C` on every legal row transition. Including the exact
top endpoint gives `sum_t f<=-e_q<=0`. This is a concrete bounded terminal
sum containing positive local density. It is not a charge certificate: the
actual survivor prefix `100` has `Phi_2=0`, `Phi_3=-1`, and
`8e_3-1+Phi_3-Phi_2=-2`. Synthesis with a general signed two-bit density,
terminal flux, time parity, and bottom corrections also fails; its integer
Farkas witness uses 27 rows and yields `0>=6`. No initial or terminal-bound
inequality occurs in that witness, so relaxing the endpoint constant alone
still cannot repair this language.

A specific missing dependency is visible in this strongest witness. It uses
both consecutive row edges

```
000 -> 100 -> 111.
```

Each edge separately satisfies the three-diagonal existential rule, but the
two-edge walk cannot occur in real spacetime. To see this, let the next two
older diagonal bits be `z,w`. The first edge forces `z=0,w=1`. Their own
Rule 30 evolution then forces

```
z'=z XOR (w OR v)=1
```

regardless of the next older bit `v`. Consequently the first component at the
third row must be `1 XOR (z' OR w')=0`, whereas `111` asks for one. This is an
exact two-update obstruction, not a statistical observation. Enumeration of
all 128 seven-diagonal input words independently checked the three observed
rows and excludes that walk; four distinct two-edge walks accepted by the
rowwise relaxation fail this shared-predecessor check.

The sharply specified next refinement is therefore **two-update predecessor
coherence**: retain the predecessor obligation from one temporal transition
to the next, and enumerate two consecutive updates with the same older
boundary bits. In particular, a `000 -> 100` transition must carry the fact
`z'=1`, which forbids a subsequent first component one. Merely imposing more
single-row equations while resampling their existential predecessors repeats
the demonstrated error. It remains unproved that this particular refinement
admits a potential; the result identifies an actual spurious walk in the
strongest rejected language and the information needed to eliminate it.
