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
> **The one comparison worth running.** There is now a concrete mechanism by
> which a selector could beat the fixed order — a fixed walk cannot tell a
> cheap card from one that will eat the budget, and Jev's observed ordering
> put the decisive cards first and scored the expensive null at p=0.01. That
> is one trace, not a result. Run the same frozen plan under `--policy fixed`
> and `--policy jev` at identical `--seconds`, `--solve-seconds` and
> `--check-seconds`, and compare *verified center frontier results*, not Jev
> calls. Report the negative if it is negative; if Jev does not win, say so and
> keep the deterministic selector.
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
> **The open frontier, as of 2026-09-18.** `s*(n)`, base-2 MSD, is certified
> through `s*(56)=13`; `s*(52)=12` and `s*(64) >= 14` were established in the
> log above, leaving `s*(64)` in `{14,15}`. STATUS.md item B3 asks for MSD n=64,
> then the harder LSD n=56 states individually. Closing `s*(64)` is the single
> highest-value target and is within reach. Check the ledger before calling
> anything new.
>
> **What to deliver.** Score the session on *new verified center results per
> total wall time*, not on Jev calls. Count controls, random nulls, inferred
> answers and model calls separately. If you want to claim Jev helps, run the
> same frozen plan under `--policy fixed` and `--policy jev` at equal budgets
> and compare verified yield; anything less is not a comparison. Finish with a
> dated experiment log and the smallest justified update to `STATUS.md` or
> `CLAIM_LEDGER.md`. Do not claim a prize result from a finite DFAO exclusion.
