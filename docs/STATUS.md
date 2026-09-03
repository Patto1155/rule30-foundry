# Status — what is in flight

**This is the only file in the repo that describes current state.** What the
repo *knows* lives in [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md); what is *next*
lives here. Overwrite this file in place; git history keeps the old versions.
No other file may carry a "current state as of" section —
`tools/lint_ledger.py` enforces it.

Updated: 2026-09-03 · Newest log: `docs/experiment-logs/2026-09-01-nersissian-end-to-end-audit.md`

## Where the three prize problems stand

| # | Problem | Best current result | Grade |
|---|---|---|---|
| 1 | Does the center column repeat? | No period `p <= 5,000,000` in the first 10M bits — **decided exactly**, all 9,999,936 candidates, 0 survivors. Cannot resolve the problem: eventual periodicity is asymptotic. | Certificate |
| 2 | Is there a shortcut for the nth bit? | None found. `s*(n)` minimal-DFAO curve certified to n=48 with DRAT proofs. ML routes (I/K/L) are **scoped down** — blind to long-lag XOR structure. | Certificate (s*(n)) |
| 3 | Are 0s and 1s equidistributed? | Bias < 0.05% over 10M bits. Uniform Bernoulli(1/2) is invariant (proved, left-permutivity) — which is *not* the same as the single seed's limiting frequency, the actual question. A published shortcut claim is now under audit, with the warm query separated from the cold `n -> c_n` cost. | Theorem + Observation |

## Open work, ranked

Sequence approved 2026-09-02: **A1 → A2 → C2**, with B3 as filler. Reasoning
and the full option analysis: [`handover/CURRENT.md`](handover/CURRENT.md).
A1 and A2 are done (see *Recently closed*); **C2 is next**.

| # | Work | Kind | Cost | Blocks |
|---|---|---|---|---|
| A3 | **Make bitstreams reachable** (Release asset or committed prefix) | Infra | ~½ day | B2 |
| C2 | **Algebraic annihilator search** — low-degree GF(2) relations over `w`-bit windows, via monomial-matrix rank | Research | Days | — |
| B3 | Extend `s*(n)` past n=48 — re-costed ~28× cheaper | Research | Hours | — |
| E1 | Write up the eight Theorem rows; `s*(n)` is citable | Writing | Days | — |
| B2 | Exact period search on 46M — no code change, extends to `p <= 2.3e7` | Research | Minutes | A3 |
| B1 | Item 14 pattern-map walk — palate cleanser, **not prize progress** | Research | ~26 min CPU | — |

**De-prioritised:** more neural experiments (the ceiling is partly the models'
— I/K/L are blind to long-lag XOR). Item 14 is worth closing but the ledger
grades left-edge structure as disjoint from the prize object.

## Chores

- **Delete 7 merged branches** — verified safe, blocked from the container by
  an egress `HTTP 403`. Command in [`handover/CURRENT.md`](handover/CURRENT.md).
  Then enable auto-delete head branches.

## In-flight branches

| PR | Branch | State |
|---|---|---|
| — | `claude/rule-30-foundry-env-tlbr9x` | Open, based on `main`. A2: the `verify` workflow. |

The #18 → #19 stack landed on 2026-09-02; #20 and #21 had already been merged
into #19. Nothing is stacked. The 7 merged branches listed in
[`BRANCHING.md`](BRANCHING.md) are still awaiting deletion (see *Chores*).

### Why #20 and #21 were merged rather than left open

They were siblings on the same base, and each was green on its own — but
**together they failed**, and nothing either branch could run would have shown
it. `lint_ledger`'s `STALE-STATUS` check (added by #21) requires this file to
name the newest dated experiment log; #20 added a newer one. Two green PRs, a
red merge.

That is a standing hazard, not a one-off: *any* PR adding an experiment log
will red-line *any* concurrent PR that touches this file. The check is right
to exist — it is what stops STATUS drifting behind results — so it was kept as
is and the merge was resolved by updating this file, which is what should have
happened anyway. See [`BRANCHING.md`](BRANCHING.md).

## Recently closed

- **A1**: the #18 → #19 stack landed on 2026-09-02. `main` carries Tier 0
  tooling, the Nersissian audit, and the agent context bootstrap.
- **A2**: CI added — `.github/workflows/verify.yml` runs
  `tools/verify_all.py` on every push to `main` and every PR. Two jobs: a
  CPU-only run, and a second that builds the SAT toolchain so the DRAT stage
  behind `s*(n)` genuinely runs (8 stages PASS there, against 7 without it).
  Both pass `--allow-skip` with an explicit stage list, so a stage that starts
  skipping because its input vanished fails the build instead of going green —
  the repo's "SKIP is not PASS" rule, enforced rather than documented.
  CI's dependencies are pinned in `requirements-ci.txt`: numpy alone, measured,
  not guessed.
- Golden reference closed byte-for-byte at 10M bits, three independent
  reproductions ([log](experiment-logs/2026-08-30-golden-reference-10M.md)).
- `s*(n)` promoted to Certificate, 207/207 DRAT-verified
  ([log](experiment-logs/2026-08-30-dfao-drat-certification.md)).
- Exact exhaustive period search
  ([log](experiment-logs/2026-08-30-period-search-exact.md)).
- I–L un-retracted but scoped down after a detection-power probe
  ([log](experiment-logs/2026-08-30-rerun-il-bitorder.md)).
- New `g(n)` smallest-grammar curve
  ([log](experiment-logs/2026-08-30-grammar-min-size-curve.md)).
- Nersissian shortcut claim audited end-to-end, separating the advertised warm
  `O(log n)` query from the cold-start `n -> c_n` cost that Problem 3 actually
  asks about ([log](experiment-logs/2026-09-01-nersissian-end-to-end-audit.md)).
  No `Omega(n)` established for the compressed representation yet — the next
  step is reconstructing it faithfully.
- `docs/COMPUTE_PLAN.md` D11 corrected: the fixed-`n` circuit definition was
  vacuous (a hard-coded constant), replaced by the whole index function `C(k)`.
