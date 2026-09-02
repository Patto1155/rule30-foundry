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

1. **Finish the item-14 pattern-map walk** — ~26 min of CPU from a ledger row.
   `experiments/pattern_map_walk.py` is validated to `d = 5e7`, 0 branch
   points; not yet run to the prediction window and deliberately has no ledger
   row. Verify every surviving branch doubles in the same window; halt on full
   32-bit parity, not minimal-period parity. Not prize-facing.
2. **Extend `s*(n)` past n=48** — re-costed: the standalone-solver path is ~28×
   cheaper than the original plan assumed (207 solves in 133 s vs 105 in
   3747 s under the pysat harness).
3. **A parity-capable estimator** (Berlekamp–Massey, GF(2) rank) — the direct
   answer to the I/K/L blind spot. Do not add more neural experiments; the
   ceiling there is partly the models'.
4. **Extend the independent CPU reference to 46M** (~50 h CPU), or state the
   10M horizon explicitly in that ledger row rather than letting it inherit
   confidence from the 10M result.

## In-flight branches

| PR | Branch | State |
|---|---|---|
| #18 | `fix/data-hygiene` | Open, based on `main`. |
| #19 | `claude/keen-sagan-21rzbr` | Open, based on #18. Tier 0 tooling + items 1, 7, 8, 11, 13. |
| #20 | `research/nersissian-audit` | Open, based on #19. |

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
