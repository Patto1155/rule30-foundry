# Jev as a lateral research controller

Jev does not write mathematics. It maps structured state to typed probabilistic
decisions. The useful lateral move is therefore to put it **inside a verified
search process**, not to ask it the prize question and not to treat repeated
votes as independent evidence.

`tools/jev_conjecture_search.py` is the first bounded implementation. It asks a
frontier model or researcher to define a grammar once, then lets ordinary code
instantiate and test thousands of exact claims. Jev receives only candidates
that survived the first exact falsifier and estimates which deserve scarce
symbolic-proof attention.

## What the grammar contains

The current lane concerns a centre trace alternating for a finite window. A
candidate chooses:

- centre phase `0` or `1`;
- zero, one, or two simultaneous bits in columns `-3..-1,1..3`;
- a target column in the same range;
- a time lag `0..2`;
- the claimed target bit.

The default two-premise grammar has thousands of distinct implications. Claims
whose conclusion merely repeats an antecedent are removed mechanically.

## The compute graph

1. Generate the entire fixed grammar.
2. Exhaustively enumerate every nonzero finite row at the preflight radius.
3. Reject contradicted and unsupported claims. Keep exact witnesses.
4. Freeze and hash the surviving pool before any model call.
5. For each distinct candidate, make one Jev call containing several atomic
   questions evaluated in parallel: likely direct recurrence consequence,
   proof value, next action, and likely proof route.
6. Convert those answers into a transparent attention score.
7. Give the highest-ranked claims a stronger exact test and, only after that,
   expensive symbolic analysis by a frontier agent.

The Jev probability is never written into a theorem, certificate, or ledger
claim. It changes only the order in which human or frontier-agent attention is
spent.

## Why hundreds of calls can be legitimate

One call per **different claim** is a wide search over hypotheses. Hundreds of
calls on the same six cards would be repeated voting from one model prior and
would create no independence. This runner records a pool hash, one screening
record per candidate, token use, and the final exact trace so fixed, random,
and Jev policies can be compared on the same pool.

The relevant outcome is not call count. It is whether the Jev-ranked set gives
the reasoning agent more useful symbolic targets per hour than the frozen
ordering or a random seed. A live campaign that changes no attention decision
or promotes only direct restatements is evidence against the workflow.

## Commands

No model call, deterministic baseline:

```bash
python tools/jev_conjecture_search.py \
  --policy fixed --preflight-radius 3 --radius 5 \
  --steps 14 --window 5 --max-cards 200 --max-tests 30 \
  --out runs/theory/jev-conjecture-fixed
```

Live screening of 200 distinct candidates:

```bash
OPENROUTER_API_KEY=... python tools/jev_conjecture_search.py \
  --policy jev --jev-provider openrouter --concurrency 12 \
  --preflight-radius 3 --radius 5 --steps 14 --window 5 \
  --max-cards 200 --max-tests 30 \
  --out runs/theory/jev-conjecture-live
```

Every output directory must be new. Credentials are read only from the
environment and are never written into requests, traces, or repository files.

## Evidence ceiling

Every statement from this lane has a finite hypothesis, but exhaustive coverage
of its light cone matters. `PROVED_LOCAL_EXHAUSTIVE` means every assignment to
the complete finite light cone was checked, so the stated finite-window
implication is a theorem. `SURVIVED_BOUNDED` means only that no counterexample
occurred within the tested radius. Neither is an infinite alternating-fiber or
prize result. A useful local theorem becomes frontier research only when its
algebraic mechanism is derived and its bridge to finite-support exclusion is
explicit.

The grammar is also deliberately narrow. It can discover compact local
identities and combinations worth deriving; it cannot express a full forced
tail, quantify over every depth, or invent a proof language. A frontier agent
still owns those steps.

## First measured campaign

The 2026-09-19 campaign made 200 distinct calls over a frozen stratified pool.
It was cheap and operationally reliable, but Jev's top 40 matched the random
median for both exact counterexamples and irredundant implications; the frozen
structural ordering was much better. The model mostly returned `combine` and
`forced-tail`, with compressed scores. The detailed result and the Rule 30
constraints found by the exact layer are recorded in
[`experiment-logs/2026-09-19-jev-lateral-conjecture-search.md`](experiment-logs/2026-09-19-jev-lateral-conjecture-search.md).

That result is evidence against scaling this exact prompt/representation. The
next version should present candidate recursive invariants with computed
features and falsification depth, not more raw local implications.
