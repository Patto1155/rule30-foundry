# Hostile audit of the alternating cone-charge route

This audit checks the reduction and the obligations for a local certificate.
It does not prove the cone-charge inequality or exclude period two. The audit
was performed by a separately tasked agent of the same model lineage; it is
not outside-model review or independent corroboration.

## Quantifiers and rounding

Write `W_N = sum_(j=1..N) e_j`. The proposed inequalities are not all equivalent:

| Statement | Exact integer equivalent |
|---|---|
| `W_(2m) >= ceil(m/4)` | `8 W_(2m) - 2m >= 0` |
| `W_N >= floor((N+6)/8)` | `8 W_N - N >= -1` |
| `8 W_N - N >= -6` | `W_N >= ceil((N-6)/8)` |

For example, `N=2, W_N=0` satisfies the third numerical inequality and violates
the first two. This is a logical non-equivalence, not a Rule 30 survivor
counterexample. The first statement, for every `m`, implies the second for
every `N`: for odd `N=2m+1`, use `W_N >= W_(2m)`; even `N` is immediate.
The converse follows by restriction to even `N`.

The weaker constant `-6` would nevertheless suffice for exclusion. More
generally, any uniform bound `8 W_(2m) - 2m >= -C`, with `C` independent of
`m`, contradicts finite initial-left weight `r` once `2m > 8r+C`. A certificate
with a larger endpoint constant can therefore be useful without proving the
sharper conjecture.

## The left-half reduction

For `j>=1`, let bit `j-1` of `n` be the cell at coordinate `-j`. Rule 30 gives

```text
new_e_j = e_(j+1) XOR (e_j OR e_(j-1)),  e_0=b.
```

Consequently the stated integer map
`T_b(n)=(n>>1) XOR (n OR (n<<1) OR b)` has the correct orientation. At a
centre-zero phase, the next cell at `-1` is `e_2 XOR e_1`. At the following
centre-one phase, the next centre is `left XOR 1`, independently of the right
neighbour. Returning to zero therefore forces `e_2 XOR e_1=1`, exactly
`n mod 4 in {1,2}`. Repeating this argument gives the necessary latch condition
for every iterate of `G=T_1 o T_0`.

This does not prove that every surviving left trajectory has a right-half
completion. The exclusion implication needs no such completion: an actual
alternating orbit supplies a surviving left trajectory, so absence of all
finite left survivors excludes actual finite alternating orbits.

For eventual alternation, restart at a centre-zero time after its onset.
Rule 30 preserves finite support for every finite time because `F(0,0,0)=0`.
The restarted left half is therefore an ordinary nonnegative integer. A
nonzero finite row cannot disappear: its new leftmost cell, one position left
of the old leftmost one, is one. No bound on the onset time is required.

## Exact cone coverage

Set `A_j(k)=a(2k,-j)` and `B_j(k)=a(2k+1,-j)`. Sideways reconstruction reads

```text
A_(j+1)(k) = B_j(k) XOR (A_j(k) OR A_(j-1)(k)),
B_(j+1)(k) = A_j(k+1) XOR (B_j(k) OR B_(j-1)(k)).
```

At the centre, `A_0=0`, `B_0=1`, and with `s_k=A_(-1)(k)` one has
`A_1(k)=1 XOR s_k` and `B_1(k)=1`. The odd right-column values are immaterial
in this initialization because `B_0=1` saturates the OR. Thus setting those
irrelevant values to zero in the existing constructor does not impose an
extra compatibility assumption.

The first `2m` initial-left bits use only `s_0,...,s_(m-1)`. Values placed beyond
this triangular dependency cone are padding, not constraints that may be
reused by a tile. Coverage of every actual survivor is sufficient; claiming
every free `s` word comes from an actual finite right half is unnecessary and
has not been established here.

As an implementation cross-check, all induced prefixes for `m=1,...,8` were
compared as sets with all `2^(2m)` initial words filtered by exactly `m` direct
integer latch tests. The sets agree and contain `2^m` words at each size. This
checks finite constructor coverage; it is not an all-size proof or a new
survival horizon.

## What a local potential must certify

If every allowed edge satisfies `q(s,s')+Phi(s')-Phi(s)>=0`, summing along a
path gives

```text
sum q >= Phi(start)-Phi(end).
```

The proof must provide a uniform lower bound on the initial potential and a
uniform upper bound on the terminal potential, plus any uncancelled edge or
partial-tile charge. The signs matter. If `Phi` is a bounded function on a
finite total state set, its oscillation supplies an endpoint bound. If instead
`Phi` is a sum of local densities along a growing cut, the number of terms is
unbounded: a finite density alphabet does not itself bound the total
potential. For example, nonpositive densities can provide a uniform terminal
upper bound, but the start and any boundary correction still require proof.

A finite total state potential with `q=8*(new initial ones)-(new length)`
also bounds the negative charge of **every subpath**. In particular it forces
a uniform bound on internal runs of zero initial bits. This is stronger than
a lower bound on prefixes beginning at the centre. It must not be silently
assumed. A growing-cut sum can retain unbounded credit and is a different
certificate architecture.

A transition graph may safely over-approximate globally realizable tiles:
proving its inequalities then proves a stronger assertion. A negative cycle
in that relaxation does not refute the original cone-charge conjecture.
It refutes a potential on that graph only when the edge charges and state
semantics have been fixed. To call the cycle a Rule 30 counterexample requires
a compatible lift satisfying all shared boundaries and the alternating latch
at every required time.

For tiles advancing four macrotimes and eight initial cells, explicitly count
each initial bit and latch exactly once. A proof restricted to complete tiles
must bound the final incomplete tile. A residual spatial segment of length at
most seven contributes at worst `-7` to `8W-N`; this changes constants and
does not prevent asymptotic exclusion.

## Negative controls

Rule 15 is `F(l,c,r)=1 XOR l`, hence
`a(t,i)=(t mod 2) XOR a(0,i-t)`. Take a nonzero finite initial row supported
strictly to the right of the centre. Its centre is exactly `0,1,0,1,...`, and
its initial-left half is zero. This refutes any proposed exclusion derived
solely from left permutivity or triangular inversion. Rule 15 has a
nonquiescent zero background: odd-time rows are cofinite, while even-time rows
are shifted copies of the finite initial row. A control implementation must
handle that background and must not apply the Rule 30 OR latch formula to
Rule 15 unchanged.

For Rule 90, `F(l,c,r)=l XOR r`. A single cell at the centre remains spatially
symmetric, so the two neighbours of the centre agree at every time and its
centre is zero from time one onward. This rejects a generic left-permutive
finite-support periodic-trace exclusion. It does not refute a specifically
alternating Rule 30 argument using its OR latch.

## Acceptance boundary

Before accepting a certificate, check: exact state coverage; every local
transition including boundary phases; integer arithmetic; charge accounting;
both endpoints; incomplete tiles; and a separately implemented local rule
checker. A bounded enumeration, a floating-point LP solution, or a successful
selector ranking is not the missing all-size proof.

The all-period latch formula is necessary at every phase with centre one.
The existence of a `1 -> 0` transition in each nonconstant cyclic word does
not transfer an alternating proof across its intervening gaps. That requires
a new phase-aware charge or renewal lemma. Constant words require separate
treatment, and an all-period strengthened result must still be related
explicitly to the single-seed prize object.

## Review of the concrete companion constructions

The recurrence and sufficient conditions in `slanted-cone-geometry.md` were
checked algebraically, including the bulk two-time update, the special
distance-one update under the latch, both top-endpoint tables, flux signs,
and initialization at spatial depth two. Their implication is conditional:
an integer solution of all stated local inequalities and endpoint bounds
would prove the stated charge. The document does not provide that solution.

The `diagonal_cone` and separately implemented `forward_cut` routines in
`experiments/slanted_cone_potential.py` have consistent coordinates and latch
coverage, including the depth-zero cut. Their negative cycles concern
potentials on a quotient of exact growing cuts. A complete cycle can have
individually genuine edges while its consecutive full-cut witnesses fail to
match. Such a cycle really refutes the declared feature potential, since a
single feature potential must satisfy every witnessed edge, but it is not
an infinite alternating cone. The verifier must restrict witnesses to the
declared block length, depth phases, and rule; permitting arbitrary lengths
or mixed rules would establish a different obstruction.

Exact diagonal-popcount features take unbounded values as the depth grows.
They define a finite list of features, not a globally finite state set.
For every tested feature family, absence of a cycle in sampled edges remains
only absence of that sampled obstruction.

The conditional telescoping argument, triangular phase-count identity,
constant-one exclusion, and constant-zero gap in `period-charge-bridge.md`
were also checked. Its example maps on the same input differ exactly as
stated: `G_000001(0)=1` and `G_000111(0)=6`. These same-lineage checks audit
specific derivations; they do not constitute outside review.

## Density/flux obstruction audit

The 14 specified languages in `experiments/slanted_density_certificate.py`
were reviewed, including bottom-pair corrections and signed densities with a
second terminal flux. They are sufficient proof templates on local
relaxations. Refuting them does not rule out a potential on the exact
reachable growing cuts.

For an interior triple `(A,B,C)`, the strongest one-time transition refinement
allows

```text
(A',B',C') = (A XOR (z OR w), B XOR (A OR z), C XOR (A OR B)).
```

The existential older bits `z,w` contain every genuine transition. At the
centre-touching edge `t=q-3`, `z` is the centre value `1-p`, while `w` is
right of the centre and is not a defined older left diagonal. This does not
break coverage: for even `q`, the latch gives `A=1,z=1`, forcing `A'=0`;
for odd `q`, `z=0` and the centre-zero transition permits `w=1 XOR A`, forcing
`A'=1`. The implementation allows these values. As an indexing cross-check,
41,964 interior transitions from all boundary words at `m=2,...,8` passed
the strongest relation, including the centre-touching edges.

The time parity of the last triple is `p=q mod 2`; that of the final pair is
`1-p`. The builder uses these phases correctly. A bottom correction adds
`g_p(B_0,C_0)-g_(1-p)(A_0,B_0)` to the spatial potential difference, and
`g_0(x,1-x)` to its depth-two initialization. Both signs are correct.

For the signed-density refinement, the terminal-flow inequality is

```text
F_p(pair_t) <= chi_p(pair_t)-chi_p(pair_(t+1)).
```

Summing through time `q-2` leaves `chi(bottom)-chi(top)`. The terminal-bound
row requires this remainder plus the final density and the bottom correction
to be nonpositive. Hence the total cut potential is at most zero even though
individual densities may have either sign. The two-column successor relation
used here also contains the actual final centre-touching edge. These endpoint
constraints are sufficient; they may be stronger than bounds attainable on
the exact reachable cuts.

All 14 saved Farkas witnesses were independently re-summed from their labeled
mathematical constraints, without calling the inequality builder or its
Farkas checker. Every coefficient cancels exactly and the positive
right-hand sides agree with the artifacts. The strongest signed-density,
time-parity, bottom-corrected witness uses 27 nonzero rows and gives
`0>=6`. This verifies infeasibility of the stated finite inequality systems.
Regenerating a matrix with the same builder and checking integer arithmetic
alone would provide solver independence, but not an independent check of
the encoding; the formula review and separately assembled sums address that
additional distinction.

The simplest six-row obstruction has a useful algebraic explanation. With
time-independent densities, matching bottom and top triples gives

```text
even triple 010: F_0(1,0)-F_1(0,1)+F_0(1,1) >= 1,
odd  triple 110: F_1(1,0)-F_0(1,1)+F_1(0,1) >= 1.
```

Adding yields `F_0(1,0)+F_1(1,0)>=2`, contrary to the two nonpositivity
conditions. No interior-transition row occurs. Thus refining interior
transitions from `newest` to `two` to `three` cannot repair that language.
This is a six-row witness, not a claim of globally minimal support.

None of the 14 saved contradictions uses an initialization row. Consequently
loosening the initial charge allowance cannot repair these particular local
systems. The bottom-corrected strongest nonpositive witness also cancels its
`g` terms without using `g<=0`. A useful next refinement must change the
state/memory or the attainable local boundary relations; simply changing
the initial constant does not address the exhibited obstruction.
