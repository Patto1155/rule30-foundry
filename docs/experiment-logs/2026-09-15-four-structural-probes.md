# 2026-09-15 — four structural probes: branch state, pruning, wedge, invariants

Four experiments, each designed to end in a recurrence, a verified cost, or a
counterexample rather than another plot. All four are single-seed where they
make a claim about Rule 30, all four ship a `--verify` that recomputes the
artifact, and all four were checked bit-exactly against direct simulation.

Manifests: `queue/{branch-summary-search,branch-dependency-trace,
wedge-hybrid-cost,seed-invariant-search}.json`. All four clear preflight.

**Theory gate, before any compute.** Ideas 1, 2 and 4 attack `STATUS.md` items
B1 and P1-local, both graded *Proof target needed*, and no row of
`docs/theory/README.md` §4 covers them. Idea 3 as originally posed — "does the
settled wedge reduce the cost of computing centre bits" — is **closed**: §3
argues the centre bit at time `T` lives on diagonal `d = T` and
`settle(T) ≈ 1.34T > T`, so the target is never settled. But that argument is
about the *target cell*, not about the target's *dependency cone*, whose
smaller-`d` cells settle long before the horizon. The cone fraction had never
been measured, so idea 3 was run in that form and in no other.

---

## 1. No compact summary is a branch state (`branch_summary_search.py`)

At a settled zero word the recursion `D_d(t+1) = D_{d-2}(t) ⊕ (D_{d-1}(t) ∨
D_d(t))` degenerates to `D_d(t+1) = D_{d-2}(t) ⊕ D_d(t)`, which has exactly two
period-`p` solutions. Seven candidate summaries were tested as states: the
predecessor word alone, and that word augmented with the last-reset phase (mod
`p` and mod `2p`), the transient parity, the settle phase, a tail boundary bit,
and phase-plus-parity.

**Arbitrary histories — all seven REFUTED**, over 2,509 constructed branch
events with explicit counterexamples:

| summary | distinct keys | colliding keys | events with a twin |
|---|---|---|---|
| `predecessor_word` | 726 | 180 | 2000 |
| `word+reset_phase` | 1318 | 137 | 1419 |
| `word+reset_phase_2p` | 1442 | 130 | 1278 |
| `word+transient_parity` | 906 | 202 | 1897 |
| `word+reset_phase+parity` | 1454 | 137 | 1258 |
| `word+settle_phase` | 1296 | 130 | 1434 |
| `word+tail_boundary_bit` | 726 | 180 | 2000 |

The "events with a twin" column is the control that makes the refutation mean
something: between 1,258 and 2,000 events shared a key with another event, so
every summary had ample opportunity to collide and every one took it.

**Seed orbit — UNTESTED, and that is the finding.** Only **3** zero-word branch
events are reachable at `depth 400, steps 8192`, and **no two share a summary
value under any of the seven candidates**. Zero collision opportunity, so the
seed population cannot distinguish a correct summary from a wrong one. This
bears directly on the existing B1 result: the "4 train, 3 held out" selector in
`transient_branch_selector.py` was fitted on a population of the same size, and
nothing in that population could have refuted it either.

**Two bugs found in this experiment's own first draft**, both recorded because
they are the shape of error a falsifier is most prone to. It recorded *both*
branches of each zero word as separate events, which guarantees a collision and
refutes every summary including a correct one. And its "arbitrary" population
was random columns, which never settle, so it produced zero events while
reporting a clean pass. Both are now pinned by tests.

## 2. Short-circuit pruning buys a constant factor (`branch_dependency_trace.py`)

`D_d(t)` appears only inside the OR, so whenever `D_{d-1}(t) = 1` the whole
sub-cone below `D_d(t)` is irrelevant. This is an exact algebraic cancellation,
and the pruned evaluation returns bit-identical values (7/7 checked against
simulation).

| `t` | rectangle | cone | pruned | saved |
|---|---|---|---|---|
| 16 | 289 | 161 | 116 | 28.0% |
| 64 | 4,225 | 2,177 | 1,876 | 13.8% |
| 256 | 66,049 | 33,281 | 27,674 | 16.8% |
| 1024 | 1,050,625 | 526,337 | 433,806 | 17.6% |

Growth exponent **1.969** for the pruned evaluation against **1.951** for the
unpruned cone. **Useful failure**: the cancellation is a 14–28% constant-factor
saving and does not touch the growth rate. Most intermediate work cannot be
skipped this way.

## 3. The wedge supplies a shrinking minority of the cone (`wedge_hybrid_cost.py`)

Exact cell accounting for the query `D_T(T)`, splitting its dependency cone into
cells the pattern map supplies in O(1) (`t ≥ settle(d)`) and cells that must be
evaluated.

| `T` | cone | free | work | free % | settle(T) | target settled |
|---|---|---|---|---|---|---|
| 64 | 3,169 | 889 | 2,280 | 28.1% | 88 | False |
| 128 | 12,481 | 2,630 | 9,851 | 21.1% | 171 | False |
| 256 | 49,537 | 10,180 | 39,357 | 20.6% | 319 | False |
| 1024 | 787,969 | 145,670 | 642,299 | 18.5% | — | False |

Work exponent **2.026** against a cone exponent of **1.990**; the free fraction
itself decays with exponent **−0.119**. **Useful failure, with numbers**: the
wedge is a constant-factor contribution that gets relatively *smaller* as `T`
grows, so further wedge optimisation is a weak route to centre bits.

Two independent confirmations fell out. The target cell is unsettled at every
`T` tested, as §3 requires. And the measured `settle(T)/T` — **1.375, 1.336,
1.246** at `T` = 64, 128, 256 — reproduces the ledger's fitted ≈1.34 constant
from a completely different computation.

*Caveat, stated because it bounds the result*: `settle(d)` is measured inside a
finite horizon, so it is an upper bound on the true settling time. That
direction is safe here — it can only make the wedge look **more** useful than it
is, so it does not weaken a negative.

## 4. No window invariant can exclude the loops (`seed_invariant_search.py`)

This is a characterisation, not a search. For window-based (subshift of finite
type) invariants `I(row) = "every k-window lies in W"`, there is a unique best
candidate at each `k`: the set of `k`-windows the seed actually shows. Any `I`
satisfying obligation (a) must contain it, and a larger `W` excludes less. So
testing that one `W` decides the whole family at that `k`, and the counting
bound does not apply because nothing is fitted.

**The seed's row language is FULL at every width tested, `k = 2 … 15`**
(`|W| = 2^k` exactly, at `steps = 1500`). So the best available window invariant
is "true of every row": preserved because it says nothing, and excluding
nothing. **No subshift-of-finite-type invariant of window width ≤ 15 can
exclude the alternating-centre loops**, at any of the seven saved widths 3–15.

Positive control, at a matched 160-step horizon, `k = 14`:

| rule | `|W(14)|` of 16384 |
|---|---|
| 30 | 10,478 |
| 45 | 11,149 |
| **90** | **87** |
| 110 | 2,958 |

Rule 90's language is restricted by a factor of 188, so the instrument detects a
restricted language when one exists. Rule 30's fullness is an absence of
structure, not an absence of sensitivity.

---

## What this closes and what it does not

Closed: the SFT/window family for P1-local at width ≤ 15. A wider window, or an
invariant that is not window-based (a counting or algebraic constraint), is
untouched by this.

Not closed, and sharpened: B1. A compact summary is not a state on arbitrary
histories, and the seed-reachable population is too small to test one either
way. Any route through the branch selector now needs either more reachable
branch events — which costs the transient the whole exercise was trying to
avoid — or an argument that never passes through a finite summary.

Neither 2 nor 3 changes a growth rate, so neither is progress on Problem 3, and
both now say so with an exact, reproducible number rather than an argument.
