# Independent audit of the multi-route continuation

This note audits mathematical implications and concrete witnesses. It does not
change claim grades or assert a period-two exclusion. Agreement between agents
is not evidence; the checks below use explicit algebra or independent
cell-by-cell Rule-30 evolution.

## 1. A finite-graph criterion removes coefficient tuning

Let a finite directed cut abstraction have `n` states. Each edge has a
nonnegative integer initial-weight cost `w` and fixed positive slab demand
`b`. Seek a positive coefficient `A` and a state potential satisfying

```text
A w - b >= Phi(target) - Phi(source).
```

Such a certificate exists **if and only if the zero-weight edge subgraph has
no directed cycle**. Necessity follows by summing around a zero-weight cycle.
For sufficiency choose `A=b*n`. Every simple directed cycle has at most `n`
edges and at least one unit of total weight, so its total `A*w-b` cost is
nonnegative. All directed cycles decompose into simple cycles; hence the
standard difference constraints have a solution. More explicitly, attach a
zero-cost super-source and let `Phi` be shortest-path distance. Removing
nonnegative cycles leaves a simple path, so

```text
-(n-1)b <= Phi(state) <= 0.
```

Thus the local inequalities and uniform endpoint bounds hold on this finite
graph. This gives a concrete decision problem before coefficient optimisation.
For nonnegative integer demands `d` varying by edge, the corresponding
obstruction is a zero-total-weight cycle with positive total demand. If none
exists, `A=n*max_edge_d` suffices whenever the demand maximum is positive.

These statements require a **finite, sound graph covering every relevant
cone**, not a graph learned from a finite sample. Conversely, a cycle assembled
from genuine sample edges already refutes a potential that depends only on
the sampled features. It does not assert that the full cone witnesses can be
concatenated. Remove irrelevant states only with a proved reachability rule.

Potential signs differ across older notes: the convention above uses a lower
bound on terminal `Phi` and upper bound on initial `Phi`; negating `Phi`
converts the earlier `charge+Phi(next)-Phi(current)>=0` convention.

## 2. Two physical witnesses defeat the small endpoint abstraction for every A

The following witnesses were independently reconstructed by direct cell-by-cell
Rule-30 truth-table evolution. Left/right integers put their nearest cell in
bit zero; the centre starts at zero.

| Start/end depths | Initial left integer | Initial right integer |
|---|---:|---:|
| 98 / 105 | 16661046055632894622381385253 | 2587743454036826996634021153954 |
| 133 / 140 | 560767664687031763042728977378771580502 | 46182698749820787015413023666985299359229 |

Both actual full rows alternate at the centre through their respective end
depths. Each appended seven-cell initial-left slab is all zero. For
`D_q(t)=a(t,-q+t)`, retain depth parity and the first two and last two temporal
bits of each of `D_(q-1),D_q`, with each pair packed low-time bit first. The
first witness has feature transition

```text
(0, 0,3, 0,1) -> (1, 0,1, 0,2),
```

and the second has the reverse transition. Therefore the summed cost is
`-14`, independent of `A`, and the potential cancels. This refutes every
potential on these declared features and every positive charge coefficient.
It refutes neither a richer state nor a global weight lower bound. In
particular, the physical full cuts have different lengths and have not been
concatenated into a physical orbit.

## 3. A conditional right-return invariant survives independent replay

With the exact right return `H=R_1 o R_0`, project to the low sixteen bits
and quantify over all four values of the next two bits. The greatest universal
invariant in the predicate

```text
x even OR x mod 8 = 7
```

has 13,526 states. The 768 states in the three full low-eight cylinders
`x mod 256 in {167,168,170}` belong to it. Their universal forward closure
has 1,452 states. These counts were independently reproduced, and every one
of its `1452*4=5808` transitions was recomputed from the elementary-rule
truth table, rather than only the packed return formula.

The actual low-three projection of this closure is

```text
0 -> 7,  2 -> 0,  4 -> 2,  6 -> 4,  7 -> {2,6}.
```

Only state seven outputs a one. Starting at a one, the two possible return
paths therefore have exactly three or five steps. All right halves entering
one of the three cylinders subsequently have successive observed ones spaced
three or five macrosteps, including arbitrary infinite right halves.

This is conditional entry, not a proof that every finite right half enters.
A finite graph with arbitrary exterior inputs can prove forward invariance
without solving that separate attraction problem. The resulting gap language
can still be a strict relaxation of actual infinite right traces.

## 4. Periodic forcing gives a quantitative subclass result

The weak-growth route's proof is sound as follows. Prescribe centre `01` and
an even-time free right-neighbour word `s` of temporal period `p`. Choose
odd-time right-neighbour values zero; the centre-one phases mask this choice
in sideways reconstruction. Both seed columns then have common period
`P=2p`. Sideways inversion makes every reconstructed left column periodic
with this same period from time zero.

If the initial left row had any block of `2P` consecutive zeros entirely to
the left of the centre, the shrinking zero cone would keep its middle two
columns zero for times `0,...,P-1`. Periodicity would make both columns zero
forever. For two adjacent zero temporal columns, the update at the right
member is `0=0 XOR(0 OR r)`, forcing the next right column to be zero.
Iterating reaches the prescribed alternating centre, a contradiction.
Therefore every initial-left block of `4p` cells contains a one.

For a finite `s` prefix of length `M` with period `p`, take its periodic
extension. The first `2M` reconstructed initial bits are unchanged by locality,
so

```text
W_(2M) >= floor(M/(2p)).
```

For total finite left weight `w`, this forces `M<2p(w+1)`. No `11`-avoidance
or physical right-coherence assumption is needed. This is a uniform result
for each fixed forcing period; it does not bound an arbitrary nonperiodic
forcing word. In particular, it does not exclude an alternating centre whose
right-neighbour observations are aperiodic.

## 5. The eventual-constant exclusion has a self-contained local audit

If two adjacent columns are eventually periodic with common period `p` from
one onset `T`, sideways inversion produces every column to their left with
that same period and **the same onset T**. The temporal shift uses `t+1>=T`,
so the onset does not drift during the induction. The union of finite supports
at times `T,...,T+p-1` supplies a sufficiently far-left adjacent zero pair
for all subsequent times. The zero-pair implication in section 4 propagates
right and makes the entire row at `T` zero.

A nonzero finite Rule-30 row cannot become zero: its leftmost one produces
a one exactly one position farther left at the next time. Thus adjacent
eventually periodic columns are impossible for a nonzero finite orbit.

For an eventually zero centre, its immediate right bit obeys
`r(t+1)=r(t) OR r2(t)`, hence is binary nondecreasing and eventually constant.
This gives the forbidden adjacent periodic pair. For an eventually one centre,
its immediate left column is eventually zero, giving the same contradiction.

This is an independent derivation of a known exclusion, not a new period
result. The repository theory gate already attributes period-one exclusion
to Condrey's Corollary 5, with the constant-one part traced to Jen. The audit
does not promote either external evidence grades or prize status.

## 6. Arbitrary phase words still admit nonperiodic open-strip neighbours

The all-periods route's pulse construction is algebraically sound. Let
`c` be a nonconstant periodic centre word of period `p`, choose `j` with
`c_j=0`, put `tau=p+j`, and let `sigma` be the first index at least `tau`
with `c_sigma=1`. In a `3p`-step loop use either baseline `r=0`, or set
`r_t=1` exactly for `tau<=t<=sigma`. At `r_t=0` set
`z_t=c_t XOR r_(t+1)`; at `r_t=1` set `z_t=0`. The exact relation

```text
r_(t+1)=c_t XOR(r_t OR z_t)
```

holds: the pulse persists through centre-zero phases and dies at the next
centre-one phase. Then set

```text
l_t=c_(t+1) XOR(c_t OR r_t),
v_t=l_(t+1) XOR(l_t OR c_t).
```

The three interior columns of `[v,l,c,r,z]` satisfy Rule 30. The initial and
terminal five-cell rows agree in both loops: `tau>=p>=2` and
`sigma<=3p-2` leave sufficient margins for the one-time lookahead in `v,z`.
The pulse changes `l_tau`, since `c_tau=0`. An aperiodic sequence of loop
choices therefore yields an aperiodic left neighbour, as observed at the
fixed offsets `3pk+tau`, while the centre remains periodic.

This proves an obstruction only for this open-strip relaxation. The two
outer boundary columns are free; they are not asserted to arise from a
finite full initial configuration. The construction does not refute an
all-period exclusion theorem or an argument with globally coherent exterior.

## 7. Compactness is a reduction, not a new mechanism

For a closed prefix-compatible set of allowed infinite forcing words, if no
infinite reconstruction has left weight at most `w`, then the finitely
branching tree of weight-at-most-`w` prefixes has finite depth (Konig's lemma).
Equivalently, the minimum weight at depth `N` diverges with `N` exactly when
there is no bounded-weight infinite survivor. A proof of uniform termination
for every weight could suffice without a linear rate, but compactness itself
merely states that equivalence. It supplies no missing termination induction.

Finite-support right halves do not form a closed space. A compactness argument
that ranges over longer and longer finite right witnesses can converge to an
infinite-support right half; this must be permitted explicitly or excluded by
a separate argument. No stronger physical conclusion follows automatically.

## 8. Second-slab temporal transport improves, but the third defeats monotonicity

On the conditional right language with gaps three or five between ones, the
slab-transport route supplies an exact depth-fourteen identity. Let
`w=s_(k+2)...s_(k+7)`, and put `f=1` on the five-bit words
`01000,10000,00010`, zero on the other allowed five-bit words. Then

```text
Delta14(k) = 1[w=100001] + f(w[1:]) - f(w[:-1]).
```

The eight admissible six-bit words give this exhaustive claimed table:

| w | Delta14 |
|---|---:|
| 001000 | 1 |
| 001001 | 0 |
| 010000 | 0 |
| 010010 | 0 |
| 100001 | 0 |
| 100100 | 0 |
| 000010 | 1 |
| 000100 | -1 |

As an independent replay, direct sideways cell recurrence checked all 1,536
windows in the 64 cyclic words formed from six gaps chosen in `{3,5}`. These
include every local context required by depth fourteen. The identity gives
uniform temporal telescoping and nonnegative production from a five-gap.
The producer's exact dependency/factor coverage argument is still the reason
this finite check supports a uniform identity rather than only those cycles.

For periodic `s=(10000)^infinity`, the independent recurrence gives
`sum_(k=0..4) Delta21(k)=-2`. Thus repeated copies give an arbitrarily negative
temporal excess at the third seven-column slab. This is an obstruction to
monotone temporal-excess propagation even in the narrower gap-three/five
language. It is not a counterexample to initial-row weight growth, and it
confirms why a spatial transport proof must account for initial weight.

The later four-edge endpoint refinement in
`coherent-potential-followup.md` was also independently replayed, importing
only its four raw initial-row constants. Retaining bottom two and top eight
bits of each diagonal gives exactly

```text
(1,0,85,0,170) -> (0,0,170,0,102) -> (1,0,85,0,213)
                -> (0,0,210,0,81) -> (1,0,85,0,170).
```

The four physical centre traces and zero slab weights pass the separate
cell-by-cell check. Thus that richer endpoint family has total cycle cost
`-28` for every coefficient `A` as well.
