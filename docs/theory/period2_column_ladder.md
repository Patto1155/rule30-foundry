# Column reduction, finite languages, and the remaining support bridge

This combines checked formal lemmas and finite certificates. It does not
exclude period two or decide asymptotic free-input density. Conventions are
`u_s=x_1(2s)`, `e_j=x_-j(0)`, `W_N=sum(e_1,...,e_N)` and centre `t mod 2`.

## Checked links and exact gaps

| Link | Evidence and scope |
| --- | --- |
| Normalize an eventual alternating tail | Lean `onset_reduction_evolve` exposes the actual evolved finite nonzero row and its onset. |
| The prescribed trace forces a unique left completion | Lean `triangular_uniqueness`, plus the exact inverse local rule. Complete traces are required; no finite-prefix theorem is asserted. |
| Only even right-column inputs matter | The reference triangular inversion erases odd right-column entries. Lean checks the odd **left** latch and physical no-11 even output. |
| Physical inputs obey local language constraints | Twelve minimal forbidden factors through length12 have exact-cone DRAT certificates; every allowed factor through12 has a retained realizing finite row. |
| Convert ordered constraints to unbounded initial-left support | **Missing.** Neither finite factor coverage, a transient table, nor the conditional ring return supplies an all-horizon spatial transport bound. |
| Contradict finite left support | Conditional on unbounded weight. The statement `W_N(r)->infinity` for every finite r is equivalent to the finite-row period-two exclusion, rather than an easier independent theorem. |

The separate period-one chain is now Lean checked: complete constant-zero
and constant-one fibres, finite nonzero evolution, and
`finite_nonzero_not_eventually_constant` for any onset. The zero row is
retained. These results use Rule30's OR and do not supply the missing
period-two link. The combined project audits 50 declarations against the
pinned Lean4.19 kernel, with only its standard axioms permitted.

## Finite outcomes

The free-input SAT curve certifies 20 exact minima at N=4,8,...,80.
At larger horizons the bounds remain wide: N300 is [9,43] and N500 is
[9,72]. The positive-density versus zero-density question is **undecided**;
there is no fit or inferred asymptotic law. At N81 the supplied weight9
witness and the certified N80 lower bound9 together give an exact derived
minimum9 by prefix consistency. The supplied no-11 N79 weight10 word,
combined with a separately DRAT-verified weight<=9 refutation, certifies
the second checkpoint minimum10.

The exact physical factor counts through length12 are
`2,3,5,8,12,17,25,36,50,68,91,119`. The forbidden words are

```
11  00000  101001  0100101
010010001  0101000101  0101010000
01010001001  10010001001
010010000101  010100010001  100100010000
```

Each local cone leaves its `2L-1` initial right cells independent; its
UNSAT certificate therefore excludes the factor at every even-time
snapshot. Exhaustive/sampled enumeration alone would not give that
conclusion. New minimal bans still appear at length12. This establishes
no all-length language characterization, regularity, or SFT property.

The proposed L1/L2/L3 ladder fails: `u=(01)^45 000101` has N191, W26 and
deficit9, while avoiding the first four bans. Its ones at176 and192 leave
15 intervening zeros. It contains the certified forbidden factor
`0101000101` at zero-based offset86, so it is not physically realizable.
The actual finite seed `r=0x98f9d8a0` instead has zeros177..190, next
one191, and deficit8 at N190. A post-prefix zero-run bound alone would
not control accumulated deficits at all later prefixes.

Call avoidance of all twelve listed bans **L4**. The finite alternating-
prefix/twelve-bit-tail table has maximum uncensored first transient run14
over L4; censored words and longer tails remain outside that maximum.
Targeted L4 deficit9 queries at N191 and N300 returned UNKNOWN. Physical
queries for some N<=300 with right seed widths64,128,256 also returned
UNKNOWN. No requested-width exclusion was obtained. The K0 UNSAT proof
is a one-seed regression control only.

## Sharp next conjecture and falsifier

For every N>=1 and every binary u of length ceil(N/2) avoiding the twelve
listed factors, conjecture `7*weight(completion_from_u(u,N)) >= N-8`.
This is a stronger **L4 relaxation** of the physical uniform-deficit
obligation, not a redefinition of its physical quantifiers. A single L4
SAT word with deficit>=9 kills the relaxation. A physical kill must also
satisfy the coupled right-half causal cone. The cheapest next experiment
is to resume the retained UNKNOWN L4 N191 query, then the N300 query,
with a larger explicit budget or a stronger encoding.

An all-horizon proof would still need ordered state and bounded endpoint
credit. The conditional `u=(01)^infinity` adjacent-column-pair return
proves `e_j=1 iff j=1 mod7`, on paper with exhaustive four-phase identity
checks; it does not establish that finite seeds32/96/160 realize that
infinite input. No finite influence horizon proves the absence of every
finite-state transducer.

## Controls, gates, and reproducible artifacts

The zero-target zero row remains admissible. Rule90 realizes the banned
right word11; the constant-zero Rule30 target realizes00000. The local
factor bans and column latch explicitly use OR and alternating centre.
Triangular uniqueness, time shifts, and finite support are infrastructure
rather than period-exclusion mechanisms on their own.

SAT purposes are exact-exclusion. A free completion has ceil(N/2)
independent bits before language restrictions; a physical K-bit seed has
K bits, or zero free bits when a regression control fixes that seed.
There is no complexity/randomness inference or statistical ensemble claim.
SAT witnesses are independently reconstructed, and finite physical ones
are forward simulated. Certificate replays regenerate semantic CNFs and
check hashes and external ASCII DRAT; UNKNOWN never counts as UNSAT.

See `docs/theory/period2_column_reduction.md`,
`docs/theory/period2_free_density.md`, `docs/theory/period2_u_language.md`,
`docs/theory/period2_transient.md`, and `docs/theory/period2_deficit_sat.md`
for commands and scopes. New certificate archives are ordinary compressed
Git files under `runs/p2-*-2026-10-15`, each below GitHub's size limit.
Inherited LFS proof objects from the base branch are a separate delivery
limitation; this session does not claim to replay unavailable old proofs.

Not established: period-two exclusion, all-r density, L4 all-horizon
deficit bound, exact free minima beyond the certified points, infinite
physical seed32 boundary, Lean no-00000, sharp prefix horizons, or a
different-model-family audit. Lean/DRAT grades apply only to their checked
statements, not to these open bridges.
