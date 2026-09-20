# 2026-09-19 — Jev-guided local conjecture search

## Goal

Test a lateral use of Jev: one frontier agent defines a grammar of precise
claims, exact code rejects them, and Jev allocates symbolic-proof attention
among survivors. The test is whether this yields better targets than a frozen
ordering or random selection, not whether Jev probabilities constitute
mathematical evidence.

Nothing here settles eventual period two or any Wolfram prize.

## Setup

`tools/jev_conjecture_search.py` generated 4,728 implications about a five-bit
alternating centre window. Each selects centre phase, up to two simultaneous
bits in columns `-3..-1,1..3`, a target column, lag `0..2`, and target bit.
Claims repeating an antecedent as their conclusion were removed.

All nonzero rows of radius 3 supplied the preflight. The 925 supported claims
without a counterexample entered the full check. Radius 5 contains the complete
light cone of every claim in this grammar: centre alternation for five times
needs radius 4, and the furthest target is column 3 at lag 2. Exhausting every
radius-5 assignment therefore proves or refutes the finite-window implication;
it is not a sample from the single-seed stream.

The live pool was a frozen, stratified sample of 200 preflight survivors. One
OpenRouter Jev call per distinct claim asked four atomic questions in parallel:
likely direct recurrence consequence, proof value, next move, and proof route.
No candidate was voted on repeatedly. Published input price gives only an
estimate, not an invoice.

## Exact result

Across all 925 preflight survivors:

- 900 are exact local implications after complete light-cone enumeration;
- 25 have explicit counterexamples at radius 5;
- subset removal leaves 170 irredundant implications: 3 unconditional, 104
  one-premise, and 63 two-premise.

A compact consequence not present in the prior D1–D5 list is:

> If the centre follows `1,0,1,0`, then at the first time
> `a(t,-3) = 1 XOR a(t,-2)`.

It also follows by substitution. Write `l=a(t,-1)`, `m=a(t,-2)`,
`n=a(t,-3)`. Alternation gives `l=1`; at the next phase D2 and D3 give
`m=r(t+1)` and `l(t+1)=1 XOR r(t+1)`. Returning to a 1-phase gives
`l(t+2)=1`, hence `m(t+1)=r(t+1)`. But the update at column `-2` gives
`m(t+1)=n XOR (m OR l)=1 XOR n`, so `n=1 XOR m`.

This is structural but not a period-two exclusion: it forces one nearby left
cell to be one, not ones at unbounded left depth.

## Live Jev result

The campaign completed 200/200 distinct calls with no API failures:

- 182,286 input tokens;
- 135.9 seconds at concurrency 16 through OpenRouter;
- about `$0.0077` at the published `$0.042/M` input-token rate;
- top 40: 38 exact local implications and 2 counterexamples;
- entire pool: 192 implications and 8 counterexamples.

Twenty uniform-random 40-card baselines had a median of 2 counterexamples
(mean 1.7, range 0–4). Jev therefore matched random on finite validity. The
frozen structural ordering selected 0 counterexamples.

More importantly, only 9 of Jev's top 40 were irredundant after subset removal,
exactly the random median; the frozen ordering selected 27. Jev's top claim was
itself redundant and reduces to the ordinary OR update on column `+1` over two
phases.

The judgments were compressed rather than discriminating: 184/200 chose
`combine`; 135/200 chose `forced-tail`; `direct_rule` lay only in `0.46..0.71`
(median `0.56`), and proof value in `1.42..2.25` on a `0..4` rubric (median
`1.87`). The selector substantially reordered the pool, but not usefully.

## Following the selected route

Jev overwhelmingly selected the forced-tail route, so
`experiments/alternating_fiber_prefix.py` constructed the unique compatible
left prefix for every right half supported within 14 cells, through left depth
80, in both alternating phases. All 32,768 constructed traces verified exactly.

No constructed left prefix ended with 16 zeros. Phase 0 had maximum trailing
zero run 10; phase 1 had maximum 14. This is bounded evidence only. It is
consistent with every finite right half forcing infinitely many left-side ones,
but it gives no induction and therefore no period-two theorem. The mapping also
does not settle into the simple prefix-OR form used for constant traces.

## Verdict

The **workflow is operationally cheap but did not improve research allocation
in this form**. Hundreds of calls are affordable; affordability is not the
bottleneck. The candidate representation exposed only shallow local
implications, and Jev did not identify the irredundant ones better than random.

The useful output came from the verified grammar and complete light-cone
enumeration, not the selector. The next proof-facing target remains the same:
find a recursive invariant of the forced left half for an eventually-zero right
half. A future Jev experiment should rank explicit candidate recurrences or
induction invariants with mechanically computed features, rather than raw prose
descriptions of local implications.

## Reproduce

```bash
python -m unittest tests.test_jev_conjecture_search \
  tests.test_alternating_fiber_prefix

python tools/jev_conjecture_search.py \
  --policy fixed --preflight-radius 3 --radius 5 --steps 14 --window 5 \
  --max-cards 925 --max-tests 925 \
  --out runs/theory/jev-conjecture-all-local

python experiments/alternating_fiber_prefix.py \
  --right-support 14 --depth 80 \
  --out runs/theory/alternating-fiber-prefix-k14-d80.json
```

The live command and the interpretation rules are in
[`../JEV_LATERAL_SEARCH.md`](../JEV_LATERAL_SEARCH.md).
