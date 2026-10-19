# The research portfolio loop

A layer above `docs/THEORY_FIRST.md`. That document says to write obligations
and rank them; this one says what an obligation has to state before it may
compete for the lead's hours, and records the pilot that produced the rules.

`docs/STATUS.md` remains the only file describing what is in flight, and
`docs/CLAIM_LEDGER.md` the only file recording what the repo knows. Nothing
here promotes a claim in either. Grades below are proposals.

## What a route card must state

Seven fields, and a card missing any of them is not rankable.

1. **Target** — a prize, or an honest `none`. A card with no prize bridge says
   `none` and ranks below prize-linked work rather than borrowing a label.
2. **Proposed theorem** — the lemma, stated so it could be false.
3. **Bridge** — the exact implication from lemma to prize, *for all lengths or
   all periods*, or an explicit statement of which step is missing. A partial
   period exclusion must name what remains to reach all periods.
4. **Smallest missing lemma** — what is actually unproved, not the whole goal.
5. **Cheapest decisive falsifier** — preferably one the repo already has.
6. **Cost** — Claude hours, compute, and how the result gets verified.
7. **Stop condition** — what makes the card dead, written before it is worked.

Machine-readable form: `queue/theory/portfolio.json`, validated by
`tools/theory_triage.py`. Fields 3–7 ride in the optional keys `bridge`,
`missing_lemma`, `falsifier_cost` and `stop`, plus `progress` for what each
cycle established. They exist because the required fields cannot change
between cycles — a lemma is fixed by what the card *is* — so a loop that
re-ranked after every finding was re-ranking an identical payload.

## The rule this pilot added

`tools/theory_triage.py` gained a seventh mechanism, `prize-restatement`, and
every obligation targeting Prize 1 must now address it or be refused.

> A lemma of the shape **P implies Q**, bridged by a theorem that forbids P and
> Q together, is **equivalent to not-P** the moment that theorem is admitted —
> because not-P implies the lemma vacuously. It is admissible as a
> reformulation and must be graded at equal strength. It is not a reduction and
> must not be ranked as though it were easier than the prize.

Prize 1 already has three equivalent forms on record: irrationality of
`sum c_n x^n` over `F_2(x)`, a right-special factor at every length, and now
this. The gate is scoped to Prize 1 because that is where the reformulations
are; `MANDATORY_BY_PRIZE` is where to widen it.

## The portfolio

Six cards, all clearing the mechanism gate on entry. Full text in
`queue/theory/portfolio.json`.

| Card | Target | Bridge | State after the pilot |
|---|---|---|---|
| `latch-descent` | Prize 1 | Complete, but **equal strength** | **Parked** — a reformulation, not a reduction |
| `nonconstant-periodic-trace` | Prize 1, partial | Incomplete: `p=1` done, `p>=3` open | **Upgraded** — method is live, law uncomputed |
| `or-latch-obstruction` | none (honest) | Explicitly missing | Open, cheap, feeds the two above |
| `christol-transcendence` | Prize 3 | Explicitly missing at the asymptotic step | Open, next rung needs a packed or GPU rank |
| `unsettled-core-lower-bound` | Prize 3 | Complete; no technique for the lemma | Parked |
| `centre-density-positive` | Prize 2, partial | Explicitly missing | Parked |

Rejected before entry, as restatements or closed routes: right-special factors
at every length (Prize 1 restated, `theory/README.md` §4); any route through
automaticity or the 2-kernel (strictly harder than Prize 1); automaticity
implies bounded diagonal period (closed); diagonal period growth (no bridge,
and the cone is disjoint from the centre column).

## Pilot: three cycles

Selector `~typesafe/jev-latest` through OpenRouter's Decisions API, via
`tools/theory_triage.py --policy jev`. The Claude-only control was frozen and
committed first: `FROZEN_CLAUDE_ORDERING.md`.

| Cycle | Jev choice | p | Frozen ordering | Action taken |
|---|---|---|---|---|
| 1 | `latch-descent` | 0.87 | rank 1 — same | Latch probe, 60k steps |
| 2 | `latch-descent` | 0.84 | rank 1 or 2 | **Lead overrode** → period-two horizon |
| 3 | `latch-descent` | 0.96 | — | Outside claim review |

**Did Jev change a decision? No — not once in three cycles.** It agreed with
the frozen ordering's rank 1 every time. It did respond to evidence in the
lower ranks: after cycle 2 closed the horizon sub-route,
`nonconstant-periodic-trace` fell 0.10 → 0.01 and `or-latch-obstruction` rose
to second. That is the optional-field extension working as intended, and it is
not a decision change.

**Cycle 2 override.** After cycle 1 the chosen card's computational falsifier
was spent and its remaining work was symbolic — which is not a bounded action
the loop can dispatch. Jev kept 0.84 on it anyway. The lead moved to the other
card's unspent exact falsifier. `THEORY_FIRST.md` says *rank with Jev, then
think*; the selector judges no mathematics, and this is what that sentence
costs when the two disagree.

**The finding that matters.** The card the selector ranked first three times,
and which the frozen Claude ordering also ranked first, is by this repo's own
standard a Prize 1 restatement. The mechanism gate passed it because no rule
covered that shape; the selector cannot catch it by construction. Only the
outside provider did. Call count and reported probability are not evidence of
benefit and are not used as such here.

## What the outside review changed

`tools/council.py` against `gpt-5.6-sol`, a different lineage. Two substantive
hits, both accepted.

- **It reversed a finding.** Cycle 2 concluded the sharp-horizon method cannot
  reach eventual period 2, following Condrey §5. The reason given there —
  left permutivity forces `H(2,w) >= w`, so no bounded law exists — does not
  separate the cases: the `p=1` law that paper proves, `H(1,w) = w+2`, is also
  linear in `w` and also at least `w`. Re-measured to radius 10 with no
  saturation, the alternating horizon is *finite at every radius*. The method
  is live; the card was upgraded, not closed. Condrey's conclusion that a
  structural account of period-two exclusion remains open is untouched.
- **It re-graded the top card** from reduction to reformulation, and caught two
  real defects in the probe: collision counts were reported without witnesses,
  and "no repeated window, so a clean answer is forced for any sequence" is
  false — a constant sequence repeats its windows at every width. Both fixed;
  the probe now emits checkable witnesses and calls those rows uninformative.

Scope note the review also insisted on, and it is right: the window exclusion
rules out one predictor shape measured on the *actual* orbit. A proof of the
lemma may assume the centre column is eventually periodic — a different object,
whose windows repeat constantly — so the measurement does not exclude local
proof strategies in general. The earlier wording did claim that; it does not
now.

## Results, each as theorem, bridge, missing lemma, falsifier, verdict

### 1. The pinning identity and the monotone latch

- **Theorem.** `l_t = c_{t+1} XOR (c_t OR r_t)`, so `c_t = 1` gives
  `l_t = 1 XOR c_{t+1}`: at every 1-phase, column -1 is determined by the
  centre alone. And from the rule at `i = 1`, `c_t = 0` gives
  `r_{t+1} = r_t OR a(t,2) >= r_t` — **column +1 is non-decreasing across every
  step where the centre is 0, and can fall only at a 1-phase.**
- **Bridge.** Supplies half of `latch-descent` for free and one constraint on
  the other half. Does not reach a prize on its own.
- **Missing lemma.** Whether the 0-phase residue is eventually periodic.
- **Falsifier.** Any violation of either identity.
- **Verdict.** Both hold. 0 violations in 60,000 positions; pinned fraction
  0.50155; 0 drops in 29,907 zero-phase steps; 0 drops over all 2,723 nonzero
  configurations of support radius ≤ 5. **Proposed grade: Theorem** (one-line
  derivations), with the counts as verification, not as the claim.

### 2. The 0-phase residue is not a bounded-window function of the centre

- **Theorem (bounded).** On the 60,000-step prefix, no function of a centre
  window of 1 to 25 bits computes `r_t` at the 0-phases.
- **Bridge.** None. It prices a proof strategy.
- **Missing lemma.** Unchanged.
- **Falsifier.** A width at which no collision occurs while windows still
  repeat.
- **Verdict.** Refuted constructively at every width to 25 bits, with
  witnesses. Example, independently re-derived through the repo's packed
  stepper: `t = 3189` and `t = 28768` share the 25-bit window
  `0110010101010101111101011`, both with `c_t = 0`, while `r` is 0 and 1.
  Controls calibrate: `r_t = c_t` collides nowhere; `r_t = c_{t+1}` collides at
  1 bit and nowhere from 3 bits on; a random null collides throughout. Above 25
  bits no window repeats and the test has no power — reported as uninformative,
  not clean. **Proposed grade: Certificate**, bounded to ≤ 25 bits on a 60,000
  -bit prefix, single seed, for that predictor shape. It does **not** show
  local prediction cannot prove the lemma.

### 3. The period-two horizon

- **Theorem (bounded), plus a reproduction.** The enumerator reproduces
  Condrey's published `p=1` table exactly for `w = 1..7` — both maximum horizon
  indices and both extremizer counts, `2^w - 1` and `2^w` — and extends it to
  `w = 8` (`H0 = 8`, 255; `H1 = 9`, 256). Exhaustively over every nonzero row
  of support radius `w ≤ 10`, the longest alternating centre prefix is
  `7,7,7,7,9,10,10,15,17,17`, none saturating the step budget.
- **Bridge.** Partial and stated: excluding `p = 2` leaves every `p >= 3`. The
  `n`-ladder is not an alternative — `RS(N)` failing implies an eventual period
  at most `2^N`, so it costs exponentially more than it returns.
- **Missing lemma.** A classification of the alternating-trace fiber, the
  analogue of Condrey's Theorems 2 and 3.
- **Falsifier.** A finite row whose alternating trace never breaks.
- **Verdict.** None found to radius 10. The horizon route is **not** closed.
  **Proposed grade: bounded finding** at `w <= 10`; the reproduction is a
  control on this implementation, not new evidence about period two.

### 4. UNKNOWN, explicitly

Whether the 0-phase residue is eventually periodic under the hypothesis that
the centre column is. Whether the alternating fiber classifies. Whether
`C*(N)` keeps tracking the null past `N = 1024`. None of these was measured and
none is claimed either way.

## Reproduce

```bash
python experiments/latch_descent_probe.py --self-test
python experiments/latch_descent_probe.py --steps 60000 --pretty
python experiments/period_two_horizon.py --self-test
python experiments/period_two_horizon.py --max-radius 10 --steps 60 --pretty
python tools/theory_triage.py queue/theory/portfolio.json --policy fixed --out runs/theory/x
```

`--policy fixed` makes no model call and is the baseline any claim of selector
benefit must beat. `--policy jev` needs `OPENROUTER_API_KEY` or
`TYPESAFE_API_KEY` in the environment; no key is ever written to a file, a log,
or a prompt.

## Sources

Read as primary text this session: Condrey, *Finite Configurations Cannot
Generate a Constant Trace in Rule 30*, arXiv:2609.09431v1, 8 Sep 2026 —
**preprint, not refereed**, whose own appendix calls its Lean file "supporting
verification, not a machine-checked proof of the main theorems". Cor. 5, §5
and the appendix table are quoted above.

Not consulted directly, and cited as second-hand: Kopra, *Theor. Comp. Sci.*
**946** (2023) 113668, Thm 3.5 and Problem 4.8; Jen, *Physica D* **45** (1990)
3–18, Prop. 3; Jen, *J. Stat. Phys.* **43** (1986) 219–242, Thms 2b and 7a.
`latch-descent`'s bridge needs only the weaker of the two statements attributed
to Jen — "no two *adjacent* columns are both eventually periodic" — because the
columns it produces are adjacent by construction.

## The alternating-trace hour

One focused hour on the card the pilot upgraded. **No period-two exclusion is
claimed and none was proved.**

### Constraints derived from the recurrence

For a centre trace with `c_{t+1} = 1 XOR c_t`, writing `l = a(.,-1)`,
`c = a(.,0)`, `r = a(.,1)`, `r2 = a(.,2)`:

| | Constraint | Checked |
|---|---|---|
| D1 | `c_t = 1  =>  l_t = 1` — the OR latches, independent of the right half | 0 violations |
| D2 | `c_t = 0  =>  l_t = 1 XOR r_t` | 0 violations |
| D3 | `a(t,-2) = r_{t+1}` if `c_t=1`, else `r_t` — column −2 mirrors column +1 | 0 violations |
| D4′ | `c_t = 1  =>  l_{t+1} = r_t OR r2_t` | 0 violations |
| D5 | `c_t = 1  =>  r_{t+1} = 1 XOR (r_t OR r2_t)` | 0 violations |

Each is a one-line substitution into the recurrence; the counts verify the
transcription, they do not establish the algebra. Checked over 699,029
configurations of support radius ≤ 9, inside each identity's own hypothesis.

**D1 and D4′ together** give the leverage: column −1 is identically 1 across an
alternating stretch **exactly when** `(r_t, r2_t)` is never `(0,0)` at a
1-phase inside it. Half of "column −1 is eventually constant" is free and does
not involve the right half at all.

Two earlier derivations, D6 and a first version of D4, were wrong and are not
above: both were index bookkeeping at the edge of the alternating window,
caught by the numerical check before anything was built on them.

### The four candidates and the ranking

| Candidate | Implication | Cheapest counterexample test |
|---|---|---|
| `left-column-constant` | column −1 eventually ≡1 ⇒ Condrey Cor. 5 ⇒ **p=2 excluded** for all nonzero finite rows | double-zero at a 1-phase among longest-prefix rows |
| `alternating-fiber-support` | forced left half has ones at arbitrary depth ⇒ no finite support ⇒ **p=2 excluded** | a forced left half that terminates |
| `mirror-collapse` | none directly; makes the left half explicit | `a(t,-3)` vs column 2 |
| `left-density-obstruction` | bridge explicitly missing (time density ≠ spatial support) | density collapsing with depth |

Frozen Claude ranking, committed before the call: 1 `left-column-constant`,
2 `alternating-fiber-support`, 3 `mirror-collapse`, 4 `left-density-obstruction`.

Jev: `left-column-constant` **0.79**, `mirror-collapse` 0.14,
`alternating-fiber-support` 0.04, `return-to-lead` 0.03,
`left-density-obstruction` 0.00.

**Same top choice — no decision changed, so nothing was learned that would not
have been.** Jev swapped ranks 2 and 3, preferring the cheap lever over the
heavy classification. That is defensible and, as it happens, agrees with the
hour's constraint against starting a days-long project — but it did not select
the action, so it changed nothing this hour.

### Theorem → bridge → missing lemma → falsifier → verdict

- **Theorem.** D1–D5 above.
- **Bridge.** Complete for `p = 2` **only**: lemma + Condrey Cor. 5 excludes an
  eventually alternating centre trace for every nonzero finite configuration.
  Residue to reach all periods: every `p >= 3`, untouched, no uniform argument
  over `p` known. The length ladder is not an alternative — `RS(N)` failing
  implies an eventual period at most `2^N`.
- **Missing lemma.** `(a(t,1), a(t,2)) != (0,0)` at every large 1-phase, given
  the centre alternates forever and the row is finitely supported.
- **Falsifier.** Does the double-zero event survive among the rows with the
  longest alternating prefixes?
- **Verdict.** It does, and universally. The event occurs in **100%** of rows
  achieving the maximal alternating prefix at every radius 1–9 (1/1, 1/1, 1/1,
  2/2, 7/7, 11/11, 67/67, 3/3, 1/1). The rows that alternate longest are
  exactly the rows where the sub-lemma fails. **No local or horizon-bounded
  argument can prove it.** Card parked. Maximal prefixes `7,7,7,7,9,10,10,15,17`
  agree exactly with `experiments/period_two_horizon.py`, an independent
  implementation. **Proposed grade: bounded finding**, radius ≤ 9.

This is not a refutation of the lemma. No finite row alternates forever, so
every row here leaves the hypothesis eventually. It prices a proof strategy and
kills a local one.

### Strongest counterexample search completed

Exhaustive over all 699,029 nonzero rows of support radius ≤ 9, 40 steps, every
alternating prefix and every 1-phase within it. No finite row with an unbounded
alternating trace exists at radius ≤ 9 — consistent with the lemma and, per the
line above, not evidence for it.
