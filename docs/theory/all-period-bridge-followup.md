# All-period bridge: endpoint audit and exact certificate criteria

Date: 2026-09-19. Follow-up to `period-charge-bridge.md` and the zero-tail
composition session. This note records self-contained arguments and a
conditional certificate criterion. It does not exclude period two or any
new nonconstant period, claim literature novelty, or change ledger grades.

## 1. The constant-word gap can be audited locally

Use `a_i(t+1)=a_(i-1)(t) XOR (a_i(t) OR a_(i+1)(t))`.

**Adjacent-periodic-column lemma.** In the orbit of a nonzero finite-support
Rule 30 configuration, no two adjacent columns are both eventually periodic.
Their periods need not be equal.

Proof. Take a common onset `T` and a common period `p` (the least common
multiple suffices). Sideways inversion

```text
a_(i-1)(t) = a_i(t+1) XOR (a_i(t) OR a_(i+1)(t))
```

propagates `p`-periodicity to every column on their left, **at the same onset**
`T`: evaluating at `t+1` never calls a pre-onset value. The union of supports
of the finitely many rows at times `T,...,T+p-1` is finite. Choose two adjacent
columns sufficiently far left to vanish in every one of those rows. Their
periodicity makes them identically zero for all `t>=T`.

If columns `i-1` and `i` are identically zero for `t>=T`, the forward rule at
`i` reads `0=0 XOR (0 OR a_(i+1)(t))`, forcing column `i+1` identically zero.
Iterate rightwards. Columns farther left were already zero by the same finite
support and periodicity argument. Hence the entire row at `T` is zero.
But a nonzero finite row with leftmost one at `L` has a one at `L-1` in its
successor, since `F(0,0,1)=1`. It cannot evolve to zero. Contradiction.

This is a self-contained audit of the known width-two endpoint recorded in
the repository's theory gate, not a new theorem claim.

**Constant-zero consequence.** If column zero is zero for all `t>=T`, its
right neighbour satisfies

```text
a_1(t+1) = a_1(t) OR a_2(t).
```

Thus this Boolean sequence is nondecreasing and eventually constant. There
are then two adjacent eventually periodic columns, contradicting the lemma.
The time of stabilization need not have a uniform bound.

**Constant-one consequence.** If column zero is eventually one, its centre
equation directly forces column `-1` eventually zero. Apply the same lemma.

These arguments independently supply the period-one endpoint needed by an
all-nonconstant-word charge theorem. They do not audit the external preprint's
stronger fibre classification or sharp finite horizons, and do not assert
novelty over that preprint. They use the Rule-30-specific OR and finite
nonzero support; Rule 90's constant-centre example does not meet the monotone
neighbour step.

## 2. An existential coefficient is a zero-weight-cycle question

Fix any proposed **sound finite cut graph** for a prescribed phase word `u`.
Assume its edges count nonnegative integers `w(e)` (initial ones, each counted
exactly once) and `d(e)` (demand events). Discard vertices unreachable from
allowed starts. Let `n` be the number of remaining vertices and
`dmax=max_e d(e)` (zero when there are no edges).

The following statements are equivalent:

1. Some finite nonnegative integer `A` and vertex potential `Phi` satisfy
   `A*w(e)-d(e) >= Phi(head(e))-Phi(tail(e))` on every edge.
2. Every directed cycle with total initial-one weight zero also has total
   demand zero.

Necessity follows by summing the potential inequality around the cycle.
For sufficiency choose `A=n*dmax`. Any simple directed cycle has at most `n`
edges. If its total `w` is positive, integrality gives `w>=1`, so
`A*w-d>=n*dmax-d>=0`. If its `w` is zero, condition 2 gives `d=0`. Decomposing
arbitrary closed walks into simple cycles proves that the costs
`c(e)=A*w(e)-d(e)` have no negative cycle.

Add a supersource with zero-cost edges to every vertex and let `Phi(v)` be
its shortest-path distance to `v`. No negative cycle means these finite
integer distances exist. The shortest-path inequality is precisely
`c(e)>=Phi(head)-Phi(tail)`. Since there are finitely many vertices, endpoint
credits are uniformly bounded. One may replace `A` by `max(1,A)` if a
strictly positive coefficient is desired.

An explicit coarse endpoint bound is also available: a shortest path can be
chosen simple, and each original edge has cost at least `-dmax`, so

```text
-(n-1)*dmax <= Phi(v) <= 0.
A*W_N - D_N >= -(n-1)*dmax.
```

For a nonconstant periodic centre word with `d>=1` occurrences of `1->0`
per period `p`, use `D_N>=d*floor(N/p)`. This forces unbounded `W_N` and
excludes finite support, provided the sound graph and accounting really
exist. Phase modulo `p`, physical right coherence, and shared predecessors
must remain in whatever state makes the graph sound.

**Practical implication.** Coefficient eight is not the mathematical target.
Search the zero-`w` subgraph for an SCC containing a positive-demand edge on
a cycle. If absent, a finite coefficient and potential follow automatically.
If present, it refutes that graph's certificate, not necessarily the physical
problem: a projection can invent cycles that no exact predecessor history
realizes. A self-loop counts as a cycle. A positive-demand edge connecting
different SCCs does not.

The criterion is a standard finite-graph argument applied to this charge
setup. It supplies neither a bounded exact cut state nor a proof that a chosen
finite projection is sound enough. If exact cut states grow with depth, the
resulting `n`, `A`, and endpoint bound may grow too and provide no exclusion.

## 3. Why a phase-word-only local reset cannot be the all-period bridge

There is an explicit obstruction for **every nonconstant binary period word**
`u`, of any length `p`: a five-cell open Rule 30 strip with centre `u` admits
two equal-length closed walks at a common full strip state with different
left-neighbour output words. Thus it even admits a non-eventually-periodic
left neighbour. This construction is independent of a finite period table.

Write the strip row as `(v_t,l_t,c_t,r_t,z_t)` at sites `-2,...,2`, with
`c_t=u_(t mod p)`. Only its three interior cells obey the update rule; its
two exterior columns are free.

The right-neighbour update is

```text
r_(t+1) = c_t XOR (r_t OR z_t).
```

When `r_t=0`, either next bit is possible, by choosing
`z_t=c_t XOR r_(t+1)`. When `r_t=1`, the next bit is necessarily `1-c_t`.
Choose any phase `j` with `u_j=0`, set `tau=p+j`, and let `sigma` be the
first time at or after `tau` with `c_sigma=1`. Compare two right-neighbour
words: the all-zero word, and the single pulse

```text
r_t = 1 exactly for tau <= t <= sigma.
```

The pulse starts by a free choice at time `tau-1`, persists while the centre
is zero, and resets just after the next centre one. In particular it works
for isolated zero phases, including alternation; its birth is not restricted
to a centre-zero phase. Set `z_t` as above when `r_t=0`, and set `z_t=0`
when `r_t=1`. Define the other two strip columns by

```text
l_t = c_(t+1) XOR (c_t OR r_t),
v_t = l_(t+1) XOR (l_t OR c_t).
```

These definitions satisfy the centre and left-neighbour equations exactly.
Both choices give a closed walk of length `3p`. Their full rows agree at
times `0` and `3p` because the pulse and the one-step look-ahead used by `v`
are confined to the middle of the block. They differ at `l_tau`, since
`c_tau=0` and the two `r_tau` values differ.

Concatenate the pulse loop at block indices that are powers of two and the
zero loop elsewhere. The binary selector is not eventually periodic.
Neither is the resulting left output: any purported temporal period `q`
would also give period `3pq`, forcing the distinct length-`3p` output blocks,
and hence their selector, to repeat with block period `q`.

This is an all-parameter obstruction to five-cell free-boundary
regularisation, not a counterexample with finite support or a consistent
infinite exterior. The constant words escape the construction for a precise
reason: all-zero centres supply no reset, and all-one centres supply no
unmasked phase at which the left outputs can differ. Their monotonicity
proof therefore does not transfer to nonconstant words by phase labels alone.

## 4. Checks and remaining obligation

Run:

```bash
python experiments/all_period_bridge_followup.py
```

The producer is checked using the independent Wolfram rule-number truth
table on every interior update, for every nonconstant binary word of length
at most ten. This is a check of the symbolic constructor, not a bounded
substitute for its all-word proof. The same script verifies the local
sideways identity, monotone-neighbour identity, and nonzero-edge persistence.

The all-period bridge still needs a sound bounded cut graph without a
reachable zero-weight positive-demand cycle, or a different genuinely
well-founded invariant on unbounded cut states. For the prize seed alone,
restricting allowed starts to a proved invariant of actual seed snapshots
could suffice; merely sampling snapshots does not establish such an
invariant. The constant-word endpoint is no longer an algebraic mystery;
the nonconstant-word transport remains the substantive missing theorem.
