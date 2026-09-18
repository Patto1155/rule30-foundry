# Prompt for a fresh GPT-5.6 SOL research session

Give the agent the text below verbatim. It assumes the branch carrying
`tools/jev_campaign.py` and the corrected `docs/JEV_SESSION_GUIDE.md`.

Supply `OPENROUTER_API_KEY` as an environment variable in the terminal. Do not
paste it into the prompt, a plan, a tracked file, or a run log.

---

> You are the lead researcher for one session in `Patto1155/rule30-foundry`, a
> verification-first repo on Wolfram's three Rule 30 prize problems.
>
> **Read first:** `CLAUDE.md`, `docs/STATUS.md`, `docs/CLAIM_LEDGER.md`,
> `docs/theory/README.md`, `docs/JEV_SEARCH.md`, `docs/JEV_SESSION_GUIDE.md`,
> and `docs/experiment-logs/2026-09-18-jev-openrouter-live-and-campaign-scaling.md`.
> That last log has the measured cost of every stage — read it before choosing
> any budget, and do not re-derive its numbers.
>
> **Set up.** `python tools/verify_all.py`; `bash tools/build_sat_toolchain.sh`;
> `python experiments/dfao_drat_proofs.py --self-test`;
> `python -m unittest tests.test_jev_search`. SKIP is not PASS. Re-run
> `verify_all.py` before you commit.
>
> **Confirm the model route once, then stop spending attention on it.** One
> live `--policy jev --jev-provider openrouter` run on a small plan. A real
> check is an authenticated answer with a valid choice distribution, recorded
> usage, and a selected card followed by a *verified* solver result. A high
> probability verifies no mathematics. Decisions cost ~$0.00006 and ~1.3 s, so
> the model is never your bottleneck and is never the thing to optimise.
>
> **The implemented lane is a finite DFAO search, not a general experiment
> executor.** A different research family needs an implemented runner and
> verifier before it can produce evidence. If you want one, specify it and hand
> it to `tools/codex_worker.py`; do not improvise an unverified one.
>
> **Scale through `tools/jev_campaign.py`, which shards by prefix length `n`
> across cores.** Three rules that came from measurement, not taste:
> - `--check-seconds` must sit several times above `--solve-seconds`.
>   `drat-trim` is the binding cost: an n=52 refutation solved in 18.5 s and
>   took 21.2 s to check, and a 20 s check budget halted whole shards 1.2 s
>   short.
> - Do not retain proofs. One sweep wrote 3.7 GB of DRAT; the campaign driver
>   prunes CNF and DRAT by default and keeps hashes, witnesses and
>   `result.json`. Use `--keep-proofs` only for a deliberate ledger artifact,
>   and watch disk.
> - An UNKNOWN costs a full budget and yields no evidence. Raising
>   `--solve-seconds` without evidence of progress just buys more UNKNOWNs.
> - Matched random nulls run **3-4x longer** than the center cards they
>   control, so never let one run first. `jev_campaign.py` already orders
>   centers ahead of nulls; a hand-written plan must do the same. In the
>   2026-09-18 run a control consumed the whole wall budget and stranded the
>   single card that would have settled `s*(64)`.
>
> **The comparison has already been run — do not repeat it.** On 2026-09-18,
> one frozen n=56 plan at identical budgets, six runs: Jev established 5/5
> center results, all verified, in 122 s and 132 s with zero nulls run and zero
> wasted budget, reproducing exactly across two runs. Fixed got 3 and halted
> `verification-failed`; three random seeds got 4, 5 and 4, much of it merely
> inferred, two wasting a full budget on an UNKNOWN. That is **0.041 verified
> center results per second against 0.017 for fixed**, about 2.4x, for
> $0.00052.
>
> Read the caveats in section 3b of the log before you lean on this. Jev reads
> the plan's `goal`; fixed and random have no channel to receive it, so this is
> a goal-aware policy beating goal-blind ones, not evidence of mathematical
> insight. It is one plan at one budget, the advantage exists only while the
> budget binds, and Jev **deferred** the null work rather than removing it.
>
> **So use the selector and spend your session on new mathematics.** If you
> want to strengthen the methods result, the open question is whether the
> advantage holds at a different `n`, a different budget, or a plan whose
> optimal order is not ascending — not another replication at n=56.
>
> **Run the counting bound before any "searched class M, found no fit"
> experiment:** `python experiments/counting_bound.py --pretty`. If
> `log2|M| < n` the negative is vacuous. An exhaustive search that finds no fit
> has still *proved* the exclusion — record that under `purpose: exact-exclusion`,
> never as evidence of complexity.
>
> **Single seed only.** All three prizes concern the one deterministic
> single-black-cell initial condition. Random and Thue-Morse cards are controls.
> A single random seed is not the 7-seed null band; do not read a one-seed gap
> as separation.
>
> **The open frontier, as of 2026-09-18.** `s*(n)`, base-2 MSD, is now
> certified through **`s*(64)=15`** — B3's MSD ask is closed, and `s*(52)=12`
> is exact. Do not re-measure any of it.
>
> Your target is scoped and ready: **`s*(56)` in the LSD direction lies in
> `{12,13,14}`**, with the 11-state refutation verified and the 14-state
> witness independently evaluated. `queue/jev/frontier.json` and
> `queue/jev/lsd56.json` are validated and point straight at it. Budget for it:
> LSD costs **~4-5x MSD** per card, so start at `--solve-seconds 600
> --check-seconds 1200` and expect s=13 to be the one that may not close.
> After that: **bases 3 and 4**, and MSD beyond n=64.
>
> Be warned on that last one. The deciding n=64 instance at 14 states took
> 353 s to solve and produced a **2.26 GB** proof that took 598 s to check —
> roughly 10x its 13-state predecessor for one extra state. Extending this
> encoding to n=72 is a compute bet that will probably lose; a better encoding
> or a different model class is the higher-expected-value move. Say which you
> are doing and why. Check the ledger before calling anything new.
>
> ## Where Jev pays, and where it does not
>
> **The rule: call Jev only when a wrong choice costs more than the call.** A
> decision costs ~1.3 s and ~$0.00006. A cheap card costs 65 ms — never worth a
> call, just run them all. The deciding n=64 card cost 951 s and 2.26 GB —
> there a call is free by comparison. The ratio decides, not how interesting
> the question feels.
>
> Five triggers. When you hit one, stop reasoning and ask.
>
> **1. Ordering under asymmetric, unknown cost.** You have N runnable items, a
> wall budget, and no way to tell which will eat it. This is measured, not
> hypothetical: a matched null cost 493 s against 123 s for the card it
> controls, and a fixed walk let one strand `center64-s14` — the card that
> settled `s*(64)`. Jev had scored that null at p=0.01. Use a `choice` over
> eligible items. Do not deliberate about ordering in your own tokens; you will
> spend more reasoning than the call costs.
>
> **2. Go/no-go before expensive compute.** A `noul`: "will this instance
> resolve inside the budget?" Ask before committing 600 s and gigabytes of
> proof. Calibrate first — log every prediction against its outcome, and stop
> using it if it does not beat always-yes. An uncalibrated probability is worth
> nothing.
>
> **3. Duplicate detection against the ledger.** 41 ledger rows, 56 logs.
> Re-measuring a settled claim is this repo's most expensive recurring failure
> — a certificate was retracted for exactly it. Re-reading the ledger per
> candidate costs thousands of tokens; a `noul` ("does this restate an existing
> row?") costs ~400. Treat a hit as a prompt to go read that row, never as the
> answer.
>
> **4. Stop/continue.** `return-to-lead` works — when only controls remained,
> Jev said so rather than letting the queue grind. Use it to end a line of work
> instead of looping over cards that cannot advance anything.
>
> **5. Routing.** CLAUDE.md's delegation table is a fixed option set: script,
> `codex_worker`, `--mode investigate`, `--mode review`, or you. That is a
> `choice`, and you re-derive it constantly.
>
> **Where it never pays.** Anything deterministic — if a script exists, run the
> script; sending a fixed command through a model adds a failure point and buys
> nothing. Any mathematical verdict: Jev selects, CaDiCaL and `drat-trim`
> establish, and a probability is not a proof. Any case with one eligible item.
> The counting bound, which is arithmetic.
>
> **Triggers 1, 4 and 5 were exercised on 2026-09-18; triggers 2 and 3 were
> not.** Calibrate those two against outcomes before trusting either, and
> report what you find.
>
> **Prove it or drop it.** Log every call with its options, probabilities, your
> choice, and the outcome. At session end report verified center results per
> wall second under Jev against the same frozen plan under `--policy fixed` at
> identical budgets. If Jev does not win, say so and keep the deterministic
> selector. A cheap call you cannot show helped is still waste.
>
> **What to deliver.** Score the session on *new verified center results per
> total wall time*, not on Jev calls. Count controls, random nulls, inferred
> answers and model calls separately. If you want to claim Jev helps, run the
> same frozen plan under `--policy fixed` and `--policy jev` at equal budgets
> and compare verified yield; anything less is not a comparison. Finish with a
> dated experiment log and the smallest justified update to `STATUS.md` or
> `CLAIM_LEDGER.md`. Do not claim a prize result from a finite DFAO exclusion.
