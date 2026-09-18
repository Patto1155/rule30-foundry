# Running more useful experiments in one research session

This is the operating brief for a fresh lead agent. Read `CLAUDE.md`,
`docs/STATUS.md`, `docs/CLAIM_LEDGER.md`, `docs/theory/README.md`, and
`docs/JEV_SEARCH.md` before generating a plan. The implemented lane is a
**finite DFAO search**, not a general experiment executor. Jev chooses among
reviewed cards; CaDiCaL and `drat-trim` establish finite evidence.

## Set up once

From the repository root:

```bash
python tools/verify_all.py
bash tools/build_sat_toolchain.sh
python experiments/dfao_drat_proofs.py --self-test
python -m unittest tests.test_jev_search
python tools/jev_search.py queue/jev/smoke.json --policy fixed --out runs/jev/smoke-local
python tools/jev_search.py --verify-result runs/jev/smoke-local/control-center16-s4
```

Use a **new** `--out` path on each run. Check the exit code and
`summary.json`; an UNKNOWN, SKIP, or `stop_reason: error` is not a verified
result. `verify_all.py` must run again before committing. The two canonical
packed bitstreams may be absent and explicitly SKIP in a fresh clone.

## Credentials and live Jev calibration

**Verified live 2026-09-18 via OpenRouter.** The client speaks two endpoints,
selected by `--jev-provider` (`auto` picks OpenRouter when `OPENROUTER_API_KEY`
is set):

| provider | endpoint | model | key |
|---|---|---|---|
| `openrouter` | `POST https://openrouter.ai/api/alpha/decisions` | `~typesafe/jev-latest` | `OPENROUTER_API_KEY` |
| `typesafe` | `POST https://api.typesafe.ai/v1/systemone` | `jev-latest` | `TYPESAFE_API_KEY` |

The OpenRouter Decisions response was confirmed against `validate_choice`: it
returns `answers.<name>.type`, `choice`, a `probabilities` map summing to 1, and
`usage.input_tokens`. Observed cost is ~$0.00006 per decision at ~1.3 s latency,
so **the model is never the bottleneck** — see "Scale and score".

Export the key in the shell only. Do not put it in plans, commands, tracked
files, run logs, or agent transcripts. Mocked adapter tests are not a live
check; a live check means an authenticated answer with a valid choice
distribution, recorded usage, and a selected card followed by a *verified*
solver result. Model agreement or a high probability verifies no mathematics.

## The lead agent's research loop

1. **Choose one falsifiable question.** Write the exact claim, connection to a
   prize obligation, assumptions, searched interval or model class, and a
   result that would refute it. Check `STATUS.md` and the ledger to avoid
   reproducing a settled claim as a new finding.
2. **Prepare reviewed cards.** For the existing runner, use the JSON schema in
   `queue/jev/frontier.json`. Keep the required SAT and UNSAT calibrations,
   plus a random card matched in `n`, states, base and direction for every
   center frontier card. The plan accepts at most 200 distinct instances,
   `n <= 256` and `states <= 24`; these are guardrails, not throughput targets.
   A different research family needs an implemented runner and verifier first.
3. **Triage cheap cases.** Put likely quick counterexamples and informative
   boundary cases first in a frozen fixed-order plan. Run calibration and a
   small local pilot; inspect solver time, CNF size, checker time and UNKNOWNs.
   `--max-steps`, `--seconds`, `--solve-seconds`, and `--check-seconds` are
   explicit budgets. Raising `--max-steps` alone cannot beat the wall clock.
4. **Execute a matched campaign.** Run the same plan under fixed, seeded
   random and Jev policies with the same total limits. Keep calibration cost
   inside every policy's wall time. The runner is sequential and prunes only
   verified implications. Increase the plan and wall budget only when the
   pilot shows useful candidates and manageable solver time. Use separate
   plans for cheap cases and hard cases; retry UNKNOWNs with a larger budget
   in a new plan and retain the old trace.
5. **Audit the output.** Replay important SAT witnesses and UNSAT proofs with
   `--verify-result`. Check that each claimed improvement is new relative to
   the ledger and is on the single-seed center column; random and Thue-Morse
   results are controls. Preserve CNF, DRAT, witness, hashes, settings, tool
   versions and failure traces. Report direct results separately from sound
   implications, observations, and still open cases.
6. **Decide what follows.** Spend deeper solver time on survivors with a
   specific proof implication. When the queue is exhausted or Jev returns
   `return-to-lead`, formulate a new claim instead of looping over the same
   cards. Finish with a dated experiment log and the smallest justified update
   to `STATUS.md` or `CLAIM_LEDGER.md`.

## Scale and score

Measured on this 4-core container, 2026-09-18. These replace the earlier
conditional 100-300 estimate, which was never measured.

| stage | cost | note |
|---|---|---|
| Jev decision | ~1.3 s, ~$0.00006 | never the bottleneck |
| cheap card (n<=24) | ~5 ms solve + ~60 ms check | ~900/min sequential |
| frontier card (n=52, s=11) | 18 s solve + **21 s check** | checker dominates |
| open-gap card (n=64, s=13) | >600 s, UNKNOWN | buys nothing at 10 s |

Three findings govern any scale-up:

1. **`drat-trim` is the binding cost, not CaDiCaL.** An n=52 refutation solved
   in 18.5 s and needed 21.2 s to check. With `--check-seconds 20` the shard
   halted `verification-failed` — **1.2 s short**, and the whole shard's
   remaining cards were lost. Set `--check-seconds` to several times
   `--solve-seconds`. This is the opposite of the intuitive budget split.
2. **Proofs, not time, exhaust the session.** One 5.6-minute sweep wrote
   **3.7 GB** of DRAT; a single n=52 proof is 112 MB. CNF and DRAT are 99.8%
   of a run's footprint and are regenerable from the plan, so
   `tools/jev_campaign.py` prunes them by default and keeps `result.json`,
   witnesses and hashes. Pass `--keep-proofs` only for a retained artifact.
3. **Parallelism is free; sharding by `n` loses no pruning.** The runner is
   sequential so its verified-implication pruning stays sound, but shards are
   independent. Cards sharing an `n` are exactly the cards that prune each
   other, so splitting on `n` keeps every implication inside one shard.
   Measured: 72 cards across 4 shards in **4.3 s wall**, 25 of them discharged
   by implication rather than solved.

So the realistic session shape is **not** "hundreds of Jev calls". It is
hundreds of cheap cards resolved in seconds, a few dozen mid-range cards, and
a handful of genuinely open cards that may each consume ten minutes and still
return UNKNOWN. An UNKNOWN costs a full budget and yields no evidence; it is
the expensive failure mode, and pushing `--solve-seconds` up without evidence
of progress just buys more of them.

Score a session on **new verified center results per total wall time**, soundly
pruned solver calls, useful refutations and duplicate work avoided. Count
controls, random nulls, inferred answers and model calls separately.

**Jev has now met that bar once.** On a frozen n=56 plan at matched budgets it
returned 0.041 verified center results per second against 0.017 for fixed
(~2.4x), reproducibly across two runs, for $0.00052 — and it stopped early
rather than spending the remainder on controls. The caveats matter: Jev reads
the plan's `goal` and the baselines cannot, it is one plan at one budget, and
the edge exists only while the budget binds. See section 3b of
`docs/experiment-logs/2026-09-18-jev-openrouter-live-and-campaign-scaling.md`.

## Prompt to give the next SOL agent

> Work in `Patto1155/rule30-foundry`, starting from PR #44's branch. Read
> `CLAUDE.md`, `docs/STATUS.md`, `docs/CLAIM_LEDGER.md`, `docs/theory/README.md`,
> `docs/JEV_SEARCH.md`, and this guide. Run baseline verification. Build and
> self-test the SAT toolchain. Replay the smoke plan, including one independent
> SAT and UNSAT check. If an OpenRouter key and network route are available,
> make a real Jev decision and compare a frozen plan with fixed and random
> policies at equal budgets through the Decisions API. Propose
> distinct prize-relevant claim cards, pilot cheap cases, then scale only
> where verification throughput supports it. Keep model decisions separate
> from solver proof. Log exact results, timeouts, budgets, costs, novelty
> against the ledger, and a reproducible next step. Do not claim a prize
> result from a finite DFAO exclusion.
