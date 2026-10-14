# Period two: audited period-one mechanism and the missing physical bridge

**RULE 90 TEST.** The argument below explicitly uses Rule 30's disjunction:
the constant-zero right-neighbor update is `s'=s OR u`. Replacing it with
Rule 90's XOR destroys this latch and the classified infinite left tail.
The independent checker keeps the Rule 90 single-seed centre `1,0,0,...`
and the two-cell seed at positions `-1,+1` centre `0,0,...` as countercontrols.
No argument from left permutivity alone is advanced as period exclusion.

**ZERO TEST.** The zero configuration belongs to the constant-zero fiber;
it is deliberately retained. The eventual-constant exclusion requires a
nonzero finite configuration, whose two extreme support cells expand by
one at every step. The period-two target is *nonconstant* `0101...`, so the
zero orbit cannot satisfy its hypothesis.

This is an analytical audit and a located failure, **not a proof excluding
period two**. Labels below are **proved here**, **cited**, or **open**.
The lead researcher owns ledger grades.

## 1. Conventions and exact scope

Rule 30 is `F(x)_i=x_{i-1} XOR (x_i OR x_{i+1})`. Write `R_j=x_j`,
`L_j=x_{-j}`, `c_t=F^t(x)_0`. A finite configuration is zero outside a finite
interval. Integer `r` encodes its initial positive half with `R_1` in bit
zero. Integer `n` encodes its negative half with `L_1` in bit zero.

The eventual-period-two target reduces to a trace starting `c_t=t mod 2`:
translate the distinguished column to zero, advance to the first zero
phase after the asserted onset, and take that whole row as the new initial
row. It is still finite and nonzero. **Proved here.** Advancing in time
usually increases support radius; a finite-prefix experiment with `T=0`
does not certify arbitrary onsets at the original radius.

Source audited: David Lee Condrey, *Finite Configurations Cannot Generate a
Constant Trace in Rule 30*, [arXiv:2609.09431v1](https://arxiv.org/abs/2609.09431),
8 September 2026. This is an unrefereed preprint. Its appendix explicitly
limits the accompanying Lean file; the main results are not claimed to be
machine checked. Jen's older eventually-one exclusion is cited through
Condrey, not independently read for this audit.

## 2. Period one: the proof can be reconstructed

**Triangular uniqueness — proved here, matching cited Lemma 1.** Fix the
whole initial right half including the centre. If two rows differ, choose
the rightmost differing negative coordinate `-j`. At each time `t<=j`,
the cells strictly right of `-j+t` agree by the light cone, whereas the
cell on that diagonal differs by left permutivity. At time `j` their centre
bits differ. Consequently identical traces through time `h` force identical
initial left cells through depth `h`. This establishes uniqueness, not any
support bound.

**Zero fiber existence — proved here, matching cited Theorem 2.** If the
first positive one is at `m`, prescribe

```
L_j=0 for j<m; L_m=1; L_j=j mod 2 for j>m.
```

The centre stays zero. For `m>1`, a step gives first positive and negative
ones at `m-1`; `L'_m=m mod 2`. For `j>m`, each neighboring left pair has
OR equal to one, including the special pair touching `L_m`; thus
`L'_j=L_{j+1} XOR 1=j mod 2`. This moves the class indexed by `m` to the
class indexed by `m-1`. At `m=1`, the left checkerboard is fixed, and
`R'_1=R_1 OR R_2=1` independently of the outer tail. Hence this class is
invariant. Triangular uniqueness makes it the complete fiber, not just one
construction. If the positive half is identically zero, uniqueness against
the zero row forces the left half to be zero as well.

**One fiber — proved here, matching cited Theorem 3.** Set `L_j=1` exactly
at positive even depths, and `L_0=1`. Every adjacent left pair has OR one,
so its update preserves this checkerboard. The centre update is
`0 XOR (1 OR R_1)=1`. Uniqueness again gives the entire fiber.

**Finite support exclusion — proved here, matching cited Corollaries 4–5.**
Every nonzero zero-fiber row has arbitrarily deep odd ones. Every one-fiber
row has arbitrarily deep even ones. Neither is finite. Moreover, a nonzero
finite row with support endpoints `a,b` has new ones at `a-1,b+1`, from
neighborhoods `001,100`; it cannot evolve to zero. Apply the fiber result
to a time-shifted finite row to exclude eventual constants. Alternatively,
an eventually-one centre forces its left neighbor eventually zero by the
OR latch, which the zero-fiber exclusion forbids.

**Sharp horizons — proved here, matching cited Theorems 6–7.** Let support
be contained in `[-R,R]`, `R>=1`. Finite-prefix uniqueness compares the row
to its infinite constant-trace completion. For the zero fiber the first
forced odd one beyond the support is at `q=2ceil(R/2)+1`; for the one fiber
the first forced even one is at `q=2floor(R/2)+2`. The discrepancy reaches
the centre exactly at time `q`. Truncating the corresponding completion
attains equality. Therefore the maximum *prefix lengths, including time 0*,
are

```
B_0(R)=2ceil(R/2)+1, with 2^R-1 extremizers;
B_1(R)=2floor(R/2)+2, with 2^R extremizers.
```

For zero traces, a zero positive half cannot attain the bound: any nonzero
left cell becomes visible by time `R`. For each nonzero positive word the
unique truncated completion attains it. For one traces every positive
word is admissible. These explain the counts. They do not apply to an
arbitrarily selected late interval with radius measured at time zero.

**Source caveat.** Section 5 asserts that a linear lower bound
`H(2,R)>=R` means no bounded law exists at period two. That inference is
unsupported: period one itself has `H(1,R)=R+2`. Linear growth as radius
increases does not rule out a finite maximum at each fixed radius. Also,
its phrase "eventually p-periodic central prefix" needs onset zero for the
stated period-one formula to apply. These wording/inference problems do
not enter the proofs of Theorems 2–3 or Corollaries 4–5.

## 3. Exactly where the OR latch stops working

Let `(s,u,v)` be the first three positive cells at a zero phase. Under a
constant-zero centre the first neighbor has update `s'=s OR u`: once one,
it remains one, giving the period-one invariant class.

For an alternating centre, take two steps, with prescribed centre zero
then one. The intermediate first two positive cells are

```
s_odd=s OR u; u_odd=s XOR (u OR v).
```

The next even-time first cell is therefore

```
s_next=1 XOR (s_odd OR u_odd)
      =(1-s)(1-u)(1-v).                         [proved here]
```

The equality follows by the three Boolean cases: `s=1` latches the inner
OR to one; if `s=0,u=1` its first input is one; if `s=u=0` its second input
is `v`. All eight triples are checked independently. Thus each one is
reset to zero at the next return. The *monotone* OR latch used to stabilize
`C_1` is gone, although the nonlinear disjunction still matters. This
refutes that particular invariant extension, not every possible proof.

Nor does the position of the first right one alone classify the period-two
fiber. Right integers `1` and `7` both have their first one at coordinate
one. Their unique negative completions for an alternating centre have
first eight cells `01101010`, but at depth nine they differ (`0` versus
`1`). Forward evolution with the elementary truth table checks this finite
witness. The first-one statistic sufficient for Theorem 2 is insufficient
already at this depth. No infinite-orbit conclusion follows from this witness.

## 4. Joint return formulation: known identities, not a discovery

The driven positive half obeys

```
R_b(r)=((r<<1)|b) XOR (r|(r>>1)); H=R_1 o R_0.
```

The negative half obeys

```
T_b(n)=(n>>1) XOR (n|(n<<1)|b); G=T_1 o T_0.
```

At every even time `2k`, centre consistency requires

```
G^k(n) mod 4 in {1,2};
(H^k(r) mod 2)=1-(G^k(n) mod 2).                [proved here]
```

The first condition says the odd-time nearest left cell is one:
`(T_0(n) mod 2)=n_1 XOR n_0=1`. The second is the zero-to-one centre
update. The following one-to-zero update uses that odd left one. Together
the conditions are necessary and sufficient for coupling the two driven
half-line evolutions to the alternating centre. Sufficiency is by induction
in time, not a claim that a finite pair can satisfy them forever.

These maps, the forbidden right word `11`, stronger `00000` restrictions,
and the distinction between a free sideways boundary and one fixed
physical right trajectory already occur in the recovered research branch.
See `origin/research/slanted-cone-charge` files:
`docs/theory/alternating-renewal.md`,
`docs/theory/alternating-right-boundary-coherence.md`,
`docs/theory/weighted-renewal-transport.md`, and
`docs/theory/right-invariant-entry.md`. They were read for overlap, not
independently replayed here. The recovery handover also reports later
740-state renewal and exceptional-component work without an independently
replayed support-growth proof; those reported components must not be
identified with the earlier 2,445-state component just by role.

## 5. An explicit additional lemma and its quantifiers

Fix an ordinary nonnegative integer `r`, so the whole initial positive
half has finite support. Let `e_j(r)` denote the unique initial negative
completion whose centre is `t mod 2`, constructed by triangular inversion
with this *same r at every horizon*. Define `W_N(r)=sum_{j=1}^N e_j(r)`.

A **sufficient, stronger-than-necessary open lemma** is:

> For every integer `r>=0` there exist constants `alpha(r)>0` and
> `C(r)<infinity` such that, for every `N>=1`,
> `W_N(r)>=alpha(r)*N-C(r)`.

If true, the negative half has infinite weight for every finite positive
half, so no finite row has an alternating centre; the time-shift reduction
then excludes every eventual period-two tail. Its proof must use the
ordered constraints emitted by the physical trajectory `H^k(r)`, not
independently chosen right bits at successive steps. Nothing here proves
this density statement. An eventual alternating fiber with infinite but
zero-density support could falsify this stronger target without refuting
period-two exclusion.

The weaker assertion `W_N(r)->infinity` for all finite `r` is **equivalent**
to period-two exclusion for finite rows, and is not a reduction in difficulty.
The hoped-for extra leverage would be an OR-reset/renewal lemma proving the
stronger bound, or some weaker unbounded growth with independently bounded
terminal credit. Requiring one universal `alpha` or assuming an `11`-free
word is sufficient are further unproved strengthenings. Known local-credit
obstructions in the recovered branch warn against claiming such a bound
from unordered cut histograms or a finite endpoint table.

Suggested parent-owned obligation id: `period2-physical-support-growth`.
Mechanism: derive a seed-conditioned lower bound for `W_N(r)` from ordered
OR-reset events along `H^k(r)`. Falsifier for a proposed numerical pair
`alpha,C`: one exactly reconstructed physical prefix violating the bound.
Such finite violations can reject a stated bound; an absence of violations
cannot prove it or the existential quantifiers above. The stronger density
lemma remains an obligation, not an accepted proof candidate.

## 6. Checks and exclusions

Run `python experiments/period2_identity_check.py`. It independently uses
Rule 30's eight-entry truth table to check 64 constant-fiber instances
through depth 32; every nonzero row of radius 1 through 6 for both exact
horizon indices and extremizer counts; all 32 five-bit right cones for the
return identity; 64 negative prefixes for the return map and modulo-four
latch; the depth-nine first-one counterexample; and Rule 90/zero controls.
`python tools/lint_ledger.py` checks documentation admission only.

Not run: full verifier, SAT, DRAT, GPU, Lean, outside-provider review,
recovered-branch renewal/core/local-credit reproducers, large searches,
or a proof of the open support-growth statement. No state files, experiment
logs, grades, or commits were changed by this worker.
