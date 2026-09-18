# 2026-09-18 — Jev live over OpenRouter, and what actually limits a session

Two questions: does the Jev selector work in this environment, and can a single
session run meaningfully more experiments? Both are answered empirically below.
Nothing here is a prize result; every finding is a finite base-2 MSD DFAO
statement about a bounded prefix.

## 1. The OpenRouter adapter is live-verified

PR #44 shipped an OpenRouter adapter but could not reach the network, so it had
never made a real call. It does now.

- `POST https://openrouter.ai/api/alpha/decisions`, model `~typesafe/jev-latest`,
  `Authorization: Bearer $OPENROUTER_API_KEY` — HTTP 200, served by
  `typesafe/jev-1.13-20260917`.
- The `choice` response matches `validate_choice` exactly: `answers.next.type`,
  `choice`, a `probabilities` map summing to 1, and `usage.input_tokens`.
- End-to-end frontier run (`--policy jev --jev-provider openrouter`): 3
  decisions, 4254 input tokens, **$0.00018**, ~1.3 s per call, all three
  calibrations verified.

Behaviour worth recording. Given the n=64 frontier plan, Jev chose
`center64-s13` (p=0.63) and `center64-s14` (p=0.93) — the two genuinely open
cards — and then, with only matched random nulls left, returned `return-to-lead`
(p=0.56) rather than spending budget on controls. That is the right call, and
it independently picked the same open gap this log closes part of below. It is
one trace, not evidence that Jev beats the fixed selector; that needs a matched
comparison at equal budget, which has not been run.

## 2. What actually limits throughput

Measured on a 4-core container.

| stage | cost |
|---|---|
| Jev decision | ~1.3 s, ~$0.00006 |
| cheap card (n<=24) | ~5 ms solve + ~60 ms check |
| frontier card (n=52, s=11) | 18.5 s solve, **21.2 s check** |
| open-gap card (n=64, s=13) | >600 s, still UNKNOWN |

**`drat-trim` is the binding cost, not CaDiCaL.** The n=52 refutation solved in
18.5 s and took 21.2 s to check. Run with `--check-seconds 20`, two shards
halted `verification-failed` — short by **1.2 seconds** — and lost every
remaining card. The runner was right to refuse the unverified UNSAT; the budget
split was wrong. `--check-seconds` belongs several times above
`--solve-seconds`.

**Proofs exhaust the disk before the card space runs out.** One 5.6-minute
sweep wrote **3.7 GB** of DRAT. A single n=52 proof is 112 MB. CNF and DRAT are
99.8% of a run's footprint (3.7 GB -> 7.3 MB once pruned) and are regenerable
from the plan, so they are not worth retaining outside a ledger artifact.

**Sharding by `n` is free and loses no pruning.** The runner is sequential so
its verified-implication pruning stays sound, but shards are independent, and
cards sharing an `n` are exactly the cards that prune each other. Measured: 72
cards across 4 shards in **4.3 s wall**, 25 discharged by implication rather
than solved. `tools/jev_campaign.py` shards on `n`, prunes proofs and
aggregates.

The honest session shape is therefore *not* hundreds of Jev calls. It is
hundreds of cheap cards in seconds, dozens of mid-range cards, and a handful of
open cards that may each burn ten minutes and return UNKNOWN. An UNKNOWN costs
a full budget for no evidence — the expensive failure mode.

### The nulls cost more than the questions

Measured per card at n=64, in the order the fixed selector ran them:

| card | solve | check | total |
|---|---|---|---|
| `center64-s12` | 53.7 s | 69.3 s | 123 s |
| `random64-s12` | 186.3 s | 307.0 s | **493 s** |
| `center64-s13` | 134.8 s | 209.0 s | 344 s |
| `random64-s13` | 473.9 s | 364.3 s | **hit the wall budget** |

The matched random null is consistently **3-4x more expensive than the center
card it controls** at the same `(n, states)`. The consequence is operational
and sharp: the run halted `verification-failed` on `random64-s13` — a control —
having consumed its whole 1800 s budget, and **never reached `center64-s14`,
the one card that would settle `s*(64)`**. A control blocked the result.

This is a scheduling problem, and it is the clearest case for a selector that
exists in this repo. The fixed policy walks the plan in order and cannot know
which card is cheap. In the live run above, Jev put `center64-s13` and
`center64-s14` first and scored `random64-s13` at p=0.01 — the ordering that
would have reached the decisive card. That is a **mechanism and a testable
hypothesis, not a demonstrated win**: it is one trace, and the matched
fixed-vs-jev comparison at equal budget has not been run. It is the first thing
the next session should run, because there is now a concrete reason to expect a
difference rather than a hope for one.

Cheap mitigation available today without any model: order center cards ahead of
their nulls within a plan, and give nulls their own later shard. The null is
still required for admissibility — it just does not have to run first.

## 3. Results, and their standing

All are base-2 MSD, single-black-cell seed, exhaustive over the state class,
SAT witnesses independently evaluated and UNSAT proofs `drat-trim`-verified.
Counting bound clears every row (`log2|M|` at s=12 is 98.0, well above n=64), so
none is vacuous. Recorded under `purpose: exact-exclusion`.

| n | max verified UNSAT | min verified SAT | standing |
|---|---|---|---|
| 24 | 7 | 8 | reproduces certified curve |
| 28 | 8 | 9 | reproduces certified curve |
| 32, 36 | 9 | 10 | reproduces certified curve |
| 40, 44, 48 | 11 | 12 | `s*(48)=12` matches the certified anchor |
| **52** | 11 | 12 | **`s*(52)=12`** — new exact value |
| **56** | 12 | 13 | **reproduces the certified `s*(56)=13`** |
| **64** | **14** | 15 | **`s*(64) = 15`** — exact; closes STATUS item B3 |

Two things matter here. The n=56 row independently reproduces a value the
ledger already certifies by a different route — that is instrument validation,
not a new finding, and it is the reason to trust the other rows. The n=64 row
**closes STATUS.md item B3** ("benchmark MSD n=64 next"). `center64-s12`,
`center64-s13` and `center64-s14` are each verified UNSAT, so no base-2 MSD
DFAO with 14 or fewer states generates the first 64 center bits. The 15-state
witness was independently evaluated in the same session. Hence

> **`s*(64) = 15` exactly**, for base-2 MSD DFAOs on the single-seed center
> column.

The deciding instance, `center64-s14`, took **353 s** to solve and produced a
**2.26 GB** DRAT proof that `drat-trim` verified in **598 s** — a ~10x jump in
checking cost over `center64-s13` for one extra state, and a direct measure of
where this method runs out. The counting bound clears it comfortably: at
s=14 `log2|M|` is 120.6 against n=64, and the conservative threshold at n=64 is
11 states, so the exclusion is non-vacuous. Hashes and the 15-state witness are
retained in `runs/dfao-n64-msd-s14-2026-09-18.json`, following the convention of
the certified n=56 artifact; the 2.26 GB proof itself is not retained and
regenerates from the recorded CNF hash.

**Not claimed.** The single random null needed more states than the center
column at n=52 and n=56. This is *not* evidence of structure: the ledger's
certified separation rests on a 7-seed band, and one seed is not a band. Left
as an observation requiring the full null before it means anything.

## 3b. The matched comparison: Jev vs fixed vs random

The earlier sections said the fixed-vs-jev comparison had not been run. It has
now. One frozen plan, six runs, identical `--seconds 180 --solve-seconds 60
--check-seconds 120`, three processes at a time on four cores so no run was
starved.

The plan is n=56, center and matched random nulls at states 9..13, in
**interleaved ascending order** — the order a researcher writes by hand and the
order `queue/jev/frontier.json` already uses. It was not arranged for or
against any policy. Known costs from section 2 make the budget bind: the centre
cards total ~119 s, the nulls ~241 s, so 180 s buys about half the plan and
ordering decides which half.

| policy | center established | verified | inferred | nulls run | UNKNOWN | elapsed | stop |
|---|---|---|---|---|---|---|---|
| fixed | 3 | 3 | 0 | 3 | 0 | 180 s | `verification-failed` |
| random seed 30 | 4 | 2 | 2 | 1 | 1 | 180 s | wall-budget |
| random seed 7 | 5 | 4 | 1 | 2 | 1 | 180 s | wall-budget |
| random seed 99 | 4 | 1 | 3 | 1 | 0 | 180 s | `verification-failed` |
| **jev run 1** | **5** | **5** | 0 | 0 | 0 | **122 s** | `return-to-lead` |
| **jev run 2** | **5** | **5** | 0 | 0 | 0 | **132 s** | `return-to-lead` |

Both Jev runs took the five centre cards in ascending order, ran no nulls,
finished under budget, and then returned to lead rather than spending the
remainder on controls. On verified centre results per wall second that is
**0.041/s against 0.017/s for fixed**, about 2.4x, and it reproduced exactly
across two runs. Six Jev calls cost $0.00052.

Fixed lost twice over: it spent ~62 s on nulls before reaching the expensive
centre cards, and then the wall clock cut `center56-s12`'s proof check, so a
refutation it had already solved went unverified and halted the run. Random was
erratic, as expected — 4, 5 and 4 established, but much of it merely inferred,
and two of three seeds wasted a full 60 s budget on an UNKNOWN.

### What this does and does not show

It **does** show that on this plan, under a binding budget, the selector
delivered more verified centre results, faster, with no wasted budget, twice.

It **does not** show that the model is a better reasoner. Jev reads the plan's
`goal` and the state text telling it that random cards are matched nulls rather
than the prize orbit. The fixed and random policies have no channel to receive
that at all. The honest description is **a goal-aware policy against two
goal-blind ones**, which is what a selector is for, but it is not evidence of
mathematical insight.

Three further limits. This is one plan at one budget; generalisation is
untested. Jev **deferred** the null work rather than removing it — the five
exclusions are individually sound, each carrying its own verified refutation,
but the centre-versus-null comparison at these states rests on the separate
n=56 sweep in section 3, not on these runs. And the advantage exists only
because the budget binds; with enough wall clock every policy finishes the plan
and ordering stops mattering.

## 3c. LSD n=56 scoped for the next session

The ledger certifies base-2 LSD only through n=48, and `queue/jev/frontier.json`
still pointed at n=64 MSD cards that section 3 has now closed. That plan is
replaced, and `queue/jev/lsd56.json` is added and validated against
`load_plan`, targeting the real open frontier.

A fixed-policy pilot at `--solve-seconds 120` establishes bounds without
closing them:

| card | status | verified | solve | check |
|---|---|---|---|---|
| `center56-s11-lsd` | UNSAT | yes | 66.6 s | 93.2 s |
| `center56-s12-lsd` | UNKNOWN | — | >120 s | — |
| `center56-s13-lsd` | UNKNOWN | — | >120 s | — |
| `center56-s14-lsd` | SAT | yes | 0.0 s | — |

So **`s*(56)` in the LSD direction lies in `{12, 13, 14}`**: the 11-state
refutation is `drat-trim`-verified and the 14-state witness independently
evaluated. The two middle states are open, and closing them is a well-scoped
target for one session rather than a research programme.

Budget guidance for whoever takes it. LSD is **~4-5x more expensive than MSD**
at the same `(n, states)` — `center56-s11` cost 160 s in LSD against 34.5 s in
MSD — which is consistent with the ledger calling the LSD direction harder.
Extrapolating the MSD per-state growth, s=12 plausibly needs a few hundred
seconds of solve and comparable checking, so budget `--solve-seconds 600
--check-seconds 1200` and expect the s=13 card to be the one that may not
close. Matched random nulls at s=9..11 resolved in 1.7-40.9 s, so they are not
the constraint here that they were at n=64 MSD.

## 4. Reproduce

```bash
bash tools/build_sat_toolchain.sh
python tools/jev_campaign.py --out runs/jev/sweep --goal "..." \
    --n 40,44,48,52,56 --states 2,3,4,5,6,7,8,9,10,11,12,13,14 \
    --concurrency 4 --solve-seconds 60 --check-seconds 300
```

Each shard's `result.json` carries the CNF and proof hashes; replay any single
card with `python tools/jev_search.py --verify-result <card-dir>` after
regenerating its proof.
