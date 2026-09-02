# Status — what is in flight

**This is the only file in the repo that describes current state.** What the
repo *knows* lives in [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md); what is *next*
lives here. Overwrite this file in place; git history keeps the old versions.
No other file may carry a "current state as of" section —
`tools/lint_ledger.py` enforces it.

Updated: 2026-09-02 · Newest log: `docs/experiment-logs/2026-08-30-rerun-il-bitorder.md`

## Where the three prize problems stand

| # | Problem | Best current result | Grade |
|---|---|---|---|
| 1 | Does the center column repeat? | No period `p <= 5,000,000` in the first 10M bits — **decided exactly**, all 9,999,936 candidates, 0 survivors. Cannot resolve the problem: eventual periodicity is asymptotic. | Certificate |
| 2 | Is there a shortcut for the nth bit? | None found. `s*(n)` minimal-DFAO curve certified to n=48 with DRAT proofs. ML routes (I/K/L) are **scoped down** — blind to long-lag XOR structure. | Certificate (s*(n)) |
| 3 | Are 0s and 1s equidistributed? | Bias < 0.05% over 10M bits. Uniform Bernoulli(1/2) is invariant (proved, left-permutivity). | Theorem + Observation |

## Open work, ranked

Sequence approved 2026-09-02: **A1 → A2 → C2**, with B3 as filler. Reasoning
and the full option analysis: [`handover/CURRENT.md`](handover/CURRENT.md).

| # | Work | Kind | Cost | Blocks |
|---|---|---|---|---|
| A1 | **Land the PR stack** (#18 first, then #19, #20, #21) | Infra | Review time | Everything |
| A2 | **Add CI running `tools/verify_all.py`** — there is no `.github/` at all | Infra | ~1 h | — |
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
| #18 | `fix/data-hygiene` | Open, based on `main`. |
| #19 | `claude/keen-sagan-21rzbr` | Open, based on #18. Tier 0 tooling + items 1, 7, 8, 11, 13. |
| #20 | `research/nersissian-audit` | Open, based on #19. |
| #21 | `claude/agent-context-bootstrap` | Open, based on #19 — **sibling of #20**, not a fourth layer. Agent-context bootstrap. |

This stack is 3 deep, which is over the limit set in
[`BRANCHING.md`](BRANCHING.md). Land #18 before opening anything further from
`main`.

## Recently closed

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
