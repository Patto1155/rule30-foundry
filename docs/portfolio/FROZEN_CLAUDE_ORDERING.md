# Frozen Claude-only ordering (written before any selector call)

This file is the control for the Jev comparison. It was written and committed
**before** `tools/theory_triage.py --policy jev` was run even once in this
session, so a later claim that the selector changed a decision can be checked
against something that could not have been adjusted afterwards.

It is a goal-aware ordering, not a coin flip: the ordering below is what the
lead agent would do with no selector at all, having read `CLAUDE.md`,
`docs/THEORY_FIRST.md`, `docs/theory/README.md` sections 4 and 5,
`docs/STATUS.md` and `docs/CLAIM_LEDGER.md`. The baseline the repo ships
(`--policy fixed`, first survivor) is a weaker control and is recorded
separately in the run directories.

## Ordering over the portfolio, cycle 1

| Rank | Route | Why here |
|---|---|---|
| 1 | `latch-descent` | The only card whose bridge closes with no residue: lemma + a published width-2 theorem gives Prize 1 for every period at once. Falsifier is exact, single-seed and already tooled. |
| 2 | `nonconstant-periodic-trace` | The literature's own next rung (Kari's suggestion, recorded in Condrey section 5). Genuine partial-exclusion ladder, but the obvious sub-route needs checking before any effort goes in. |
| 3 | `or-latch-obstruction` | Pure symbolic work, no compute, and it is an input to routes 1 and 2 rather than a competitor to them. Cheap enough that it does not have to win to be worth doing. |
| 4 | `christol-transcendence` | Instrument is calibrated and the curve is real, but the bridge to any prize is explicitly missing and the next measurement needs a packed-word or GPU rank. Value per Claude-hour is low right now. |
| 5 | `unsettled-core-lower-bound` | Bridge is complete, lemma is a circuit/description lower bound with no available technique. Park, do not spend thinking time. |
| 6 | `center-density-positive` | No bridge, no technique, and the finite evidence already sits where the lemma predicts. Park. |

## Predicted actions, cycle by cycle

1. **Cycle 1** — pick `latch-descent`; action: measure exactly how much of
   column -1 is pinned by column 0 alone on the real seed orbit.
2. **Cycle 2** — stay on `latch-descent` or move to
   `nonconstant-periodic-trace`; action: settle whether the sharp-horizon
   method extends from period 1 to period 2 before anyone builds on it.
3. **Cycle 3** — expect `return-to-lead` or `or-latch-obstruction`, because
   the two cheap exact falsifiers are spent by then and what remains is
   symbolic.

## What would count as the selector earning its place

Not agreement, and not a high probability. Only one of:

- it picks a card the ordering above ranks lower, and that card then produces
  a finding that changes a card (time saved, or a weak route exposed); or
- it refuses the whole portfolio (`return-to-lead`) at a point where the
  ordering above would still have spent an hour on rank 1.

Call count, token count and reported probabilities are not evidence of
benefit and are not used as such.
