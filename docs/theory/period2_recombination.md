# Period two: the smallest remaining bridges

This is a recombination and bounded kill-test report, **not a period-two
proof**. The spatial-depth-seven mechanism is recovered work from
`44d11eebaab4e8486c195dee1e5c61a36860851f`,
`docs/theory/golden-mean-phase-slip.md`; it is not a new discovery.

**Rule 90 filter.** Every proposed proof below must use Rule30's OR in the
sideways inverse and NOR return. Left permutivity, finiteness, and time shifts
are infrastructure only. Several candidate predicates, including `prefix:7`,
also pass the paired finite Rule90 checks: this is an unresolved mechanism
filter, not evidence that their proofs are OR-specific. Rule90 has a finite
constant-zero fiber; that does not assert it has a finite alternating fiber.

**Zero filter.** The target is `c_t=t mod2` from time zero, so the zero orbit
does not qualify. The constant-zero control retains the zero row and its zero
completion. Results below concern each fixed finite right seed separately,
never a random initial-condition average or the single prize seed alone.

## 1. Exact setup and the complete conditional exclusion chain

Let `r` be a nonnegative integer, `h(r)=bitlength(r)` with `h(0)=0`, and
`H=R_1 o R_0`, where

`R_b(r)=((r<<1)|b) XOR (r|(r>>1))`.

Let `e_j(r)` be the unique forced initial negative cell at depth `j`, with
the **same** initial right integer at every reconstruction horizon, and set
`W_N(r)=sum_{j=1}^N e_j(r)`. Triangular inversion is
`X_(j+1)(t)=X_j(t+1) XOR (X_j(t) OR X_(j-1)(t))`.

| Link | Exact scope and current status |
|---|---|
| Onset reduction | An eventually alternating finite row can be advanced to a finite nonzero row with trace `t mod2`. Lean checked for the explicit alternating-tail hypothesis. |
| Forced completion | Every such row must have initial negative cells `e_j(r)`. The paper argument and independently checked finite instruments establish the construction; all-time triangular uniqueness is not yet Lean formalized. |
| Unbounded weight | **GAP U:** prove `forall r in Nat, W_N(r)->infinity`. This is equivalent to excluding finite rows with this trace, rather than an independently easier theorem. |
| Contradiction | A finite negative half has bounded total weight, contradicting U. This is an elementary conditional implication, not an obtained exclusion theorem. |

A sufficient stronger bridge is

`forall r in Nat, exists alpha(r)>0,C(r)<infinity, forall N>=1,
 W_N(r)>=alpha(r)*N-C(r)`.

It remains open. A finite violation of a chosen pair `alpha,C` cannot refute
the existential statement. A zero-density but infinite completion could
refute this density target while leaving exclusion true.

The 19 checked Lean lemmas supply onset, finite-support evolution, OR/NOR
identities, the constant-one construction, a four-step seven-cell ring, and the local temporal phase-slip identity.
They do not supply full Condrey uniqueness, constant-trace exclusion, U,
or the coupling of that ring to a finite right seed.

## 2. The sparse seed gives a small physical bridge, not a universal proof

For `s_k=bit0(H^k(r))`, the recovered exact depth-seven formulas require
only `s_k*s_(k+1)=0`. In particular

`e_1+...+e_7=4-2s_1-s_2-s_3>=1`.

If the **whole** physical boundary is `s_k=k mod2`, those same formulas
return the adjacent temporal columns `(X_7,X_6)` to `(X_0,X_-1)`.
Indeed `X_7(2k)=0`, `X_7(2k+1)=1`, while both phases of `X_6` equal `s_k`.
The initial block is `1000000`. Restarting sideways inversion then gives

`forall j>=1, e_j(r)=1 iff j=1 mod7`, and consequently `W_N(r)=ceil(N/7)`.

The right-neighbor odd-time value is also `s_k`: if `s_k=1`, OR forces it
to one; if `s_k=0`, the alternating next output `s_(k+1)=1` and the NOR
return force the first three right cells to zero. This explicitly uses OR,
and justifies the repeated adjacent-column pair, rather than assuming that
one returning column alone can be restarted.

**GAP S32:** prove `forall k>=0, bit0(H^k(32))=k mod2`, with ordinary finite
initial seed32. Seeds96 and160 have analogous open bounded survivors.
The finite ring orbit is checked; it does not prove S32. Measurements give
the exact sparse completion through100000 for seed32. D places its first64
returns in the2445-state Q, outside K; neither infinite Q residence nor
infinite alternating output follows. A finite-state graph permits fresh
exterior bits and is not the same as this one physical right orbit.

S32 would establish a useful exact extremal family and its support slope,
but would exclude only that fixed right seed. It would **not** establish U
for all finite seeds.

## 3. The sharper universal conjecture and a rounding audit

The most concrete surviving universal target is

> **P7:** `forall r in Nat, forall N>=1,
> 7*W_N(r)>=N-7*bitlength(r)`.

If proved by an OR-specific argument, P7 immediately gives the required
density with `alpha(r)=1/7` and `C(r)=bitlength(r)`, without first proving
S32. These constants are conjectural, stronger than the original obligation.

The candidate `W_N(r)>=W_N(32)-bitlength(r)` is not a weaker target after
S32: substituting `W_N(32)=ceil(N/7)` makes it exactly equivalent to P7 by
integer rounding. Its current kill-test implementation checks the endpoint
N, whereas `prefix:7` checks every prefix through the retained N. Without
S32, the baseline comparison and P7 are distinct conjectures.

Evidence: B reconstructed all65536 seeds at depth128,255 witness seeds at
4096, and18 selected finalists at100000. E generated32 candidates, killed18,
and retained14 bounded survivors. Survivors include predicates whose Rule90
mechanism filter is unresolved. Neither agreement nor these counts changes
an infinite claim's grade.

This recombination adds the focused check of **all prefixes N<=256 for
all65536 seeds r<2^16**, plus all prefixes N<=4096 for128 deterministic
96-bit samples. No violation of P7 was found. This extends the finite
prefix coverage; it does not prove P7 or the global minimum at longer N.

Cheapest useful next kill: coordinate/beam search minimizing
`min_(1<=N<=D) (7W_N(r)-N+7h(r))`, rather than just endpoint density.
One negative value exactly reconstructed from one finite `r` kills P7.
Use `D>7h(r)` so the bound is not vacuous; start with C's phase-normalized
plateau witnesses and B's slow rows. A kill rejects only the proposed slope
and width credit, not seed-dependent positive density or exclusion itself.

## 4. Why the recovered temporal production does not finish P7

At depth seven set `d_i=1[s_i=s_(i+1)=0]` and
`Delta_k=X_7(2k)+X_7(2k+1)-1`. The recovered identity is

`Delta_k=d_(k+2)-d_(k+1)+d_(k+1)*d_(k+2)`.

Summing gives boundary credit plus the nonnegative number of `000`
triples. This is exact **temporal** production. It is not spatial weight:
Rule30's sideways inverse is not monotone in its temporal input.
For a free all-zero right-output word, depth-seven temporal excess is
positive while depth-fourteen excess is negative. Thus repeating the
first-cut argument discards required state.

There is also an actual finite physical failure of aligned slab reset:
`r=17`, forced left integer857638 through depth28, has no ones at
depths22..28 while its centre alternates through time28. Therefore neither
"each seven-cell slab contains a one" nor a stateless restart can prove P7.

A distinct sufficient bridge, weaker in slope than P7, is an exact cut
functional `Phi` satisfying, along every finite-right lineage and every
seven-column slab m,

`8*slab_weight(m)-7 >= Phi(cut_(m+1))-Phi(cut_m)`.

**GAP T:** construct such a functional and prove endpoint bounds
`Phi(cut_m)>=-A(r)` for every m and `Phi(cut_0)<=B(r)` for finite A,B.
Telescoping would give `W_(7M)>=7M/8-(A+B)/8`, hence a positive density
at least1/8. The terminal bounds are part of the missing theorem, not an
assumption supplied by a finite cut table. All-time extension and the
remainder of at most six cells then give the original seed-dependent bound.

The width-three nonpositive local-density/endpoint ansatz is already
refuted by the replayed physical circulation. A new Phi must retain ordered
shared predecessors or additional nonlocal state; renaming that old local
ansatz does not evade its obstruction. The first-slab coboundary could be
a component of Phi; it is not a complete transport certificate.

## 5. Horizon and renewal information: useful restrictions, explicit gaps

C's two-phase staircase and retained DRAT proofs certify finite minima;
unknown solver results remain holes. The L96 bounds are censored brackets,
not an infinite recurrence. No tested low-order/low-degree recurrence
survived, and the sampled k-kernel probes do not exclude k-regularity.
The horizon curve supplies adversarial seeds and falsifiers for proposed
uniform onset-zero bounds. It supplies neither P7 nor S32.
The combined phase-zero exact block extends through L66: L65 follows by
monotonicity between certified R54 at L64 and L66. The opposite phase's
contiguous exact block runs through L63. This interpolation is a conditional
analytic consequence of those certificates, not a new L65 SAT/DRAT pair.

D reproduced K with13526 states and Q with2445 states. K trajectories
have one-gaps3 or5. **GAP K:** not every fixed finite seed is proved to
enter K; Q remains an alternative. **GAP C:** even conditional K event
counts need an ordered transfer to *initial left* support. Both gaps matter,
and proving gap5 charging alone would not resolve seed32 in Q.
The later reported740-state component was not recovered or replayed and
cannot be used in this chain.

## 6. Focused acceptance and what was not run

`python experiments/period2_recombination_check.py` reproduces the original
golden-mean checker (restored byte-for-byte), the finite physical slab-reset
counterexample, the bounded P7 coverage above, and controls. Retained output:
`runs/period2-recombination-2026-10-14/results.json`.

The counting-bound preflight was run before the final focused replay.
This check evaluates fixed exact predicates on65536 seeds and256 prefixes
each (16777216 inequalities), rather than searching a description class.
There is no random-word null, fit count, or complexity inference here.

Rule30 constant-zero target/right0 gives zero completion. Rule90 with
constant-zero target/right1 gives finite completion left1. For alternating
target at N256, Rule90 gives W(0)=85 and W(32)=86, so the Rule90 *baseline*
comparison fails at r0; P7 itself passes this case. Distinguish these outcomes.

Not run here: the full repository verifier, Lean, SAT/DRAT, infinite
physical-boundary proof, construction of Phi or endpoint bounds, all65536
seeds past256, complete universal kill search, GPU, or external-lineage
audit. No STATUS, claim ledger, or experiment log was edited by this worker.
