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

The client supports OpenRouter's `POST /api/alpha/decisions` with model
`~typesafe/jev-latest` and `OPENROUTER_API_KEY`, or TypeSafe's direct
`POST /v1/systemone` with `TYPESAFE_API_KEY`. Select with `--jev-provider`; the
default selects OpenRouter when its key is present. Do not put either key in
plans, commands, tracked files, run logs, or agent transcripts. Supply the
appropriate key as an environment variable in a terminal with network
access. Mocked adapter tests do not constitute a live model check.

When Jev access is available, freeze one plan and run a small
comparison:

```bash
python tools/jev_search.py queue/jev/frontier.json --policy fixed --seconds 180 --out runs/jev/fixed-001
python tools/jev_search.py queue/jev/frontier.json --policy random --seed 30 --seconds 180 --out runs/jev/random-001
python tools/jev_search.py queue/jev/frontier.json --policy jev --jev-provider openrouter --seconds 180 --out runs/jev/jev-001
```

Inspect `decision-*/request.json`, `response.json`, `decision.json`, and
`summary.json`. A real Jev check requires an authenticated answer with a valid
choice distribution, recorded usage, and a selected card followed by a
verified solver result. Model agreement or a high probability does not verify
a mathematical claim. If the route cannot be reached, report the network
failure and continue the exact fixed/random experiments.

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

The old 12-attempt/120-second default is a safety budget. A target of
100–300 cheap checks in a session is **conditional**, not measured: the
current `jev_search.py` can attempt at most 200 cards per plan, and hard
`n=64` SAT cases timed out at ten seconds in the pilot. For the current
frontier, 5–10 deeper checks may consume most of a session. Hundreds of Jev
calls are useful only if there are hundreds of distinct, worthwhile decisions;
otherwise the additional calls buy no new mathematical evidence.

Measure **new verified center results per total wall time**, soundly pruned
solver calls, useful refutations, duplicate work avoided, and full cost.
Compare these with the fixed selector. Count controls, random cases, inferred
answers, and model calls separately. If Jev does not improve verified research
yield, keep the deterministic selector.

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
