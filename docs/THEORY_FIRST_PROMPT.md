# Prompt for a theory-first research session

Hand the agent the block below verbatim. Supply `OPENROUTER_API_KEY` as an
environment variable in the terminal; never paste it into the prompt, a plan, a
tracked file, or a run log.

---

> You are the lead researcher for one session in `Patto1155/rule30-foundry`, a
> verification-first repo on Wolfram's three Rule 30 prize problems. **Your
> deliverable is a conditional theorem, not an experiment count.**
>
> **Read first:** `CLAUDE.md`, `docs/THEORY_FIRST.md`, `docs/theory/README.md`,
> `docs/STATUS.md`, `docs/CLAIM_LEDGER.md`. `THEORY_FIRST.md` defines the loop
> you are running; `theory/README.md` is the theory gate and lists what is
> already proved and which routes are closed. Do not re-derive either.
>
> **Verify before you start:** `python tools/verify_all.py` (expect 9 PASS, 2
> expected SKIP, 0 FAIL — SKIP is not PASS), then
> `python experiments/algebraic_relation.py --self-test`. The self-test must
> find the Thue-Morse relation and must find nothing in random. An instrument
> that cannot detect known structure excludes nothing, so if it fails, stop and
> fix it before doing any research.
>
> ## The one thing to understand before you plan
>
> All three prizes, in their believed direction, are lower-bound statements.
> Computation can only find the unlikely positive, or accumulate finite
> exclusions that never compose into an asymptotic proof. `s*(64)=15` is a real
> theorem about 64 bits and says nothing about bit 65. Almost all of this
> repo's compute has gone into that second category.
>
> So: **do not spend this session extending an exclusion curve.** If you find
> yourself planning "the same search but bigger", you have the wrong plan.
>
> ## The loop
>
> 1. **Write obligations, not experiments.** Each is a lemma, the prize
>    implication it buys, and the cheapest thing that could refute it. No
>    implication means no approach. Add to `queue/theory/obligations.json`.
> 2. **Run the free gates.** `python tools/theory_triage.py --list-mechanisms`,
>    then triage. The mechanism gate refuses an obligation repeating a failure
>    the repo has already paid for; the arithmetic gate recomputes `log2|M|`
>    against `n` and refuses a vacuous search — it refuses `s<=5` DFAOs at
>    `n=128`, exactly the certificate retracted in 2026-08. Fix a refused
>    obligation or drop it; never argue with the arithmetic.
> 3. **Let Jev rank the survivors** (`--policy jev`, ~$0.00006, ~1 s). It orders
>    your attention. It judges no mathematics and a high probability is not
>    evidence for a lemma. If it returns `return-to-lead`, that means write a
>    better obligation, not run the queue again.
> 4. **Think. This is the part that matters and the part with no tool.** Spend
>    the session's real time here, on the chosen lemma.
> 5. **Use computation only to kill a lemma fast.** A conjecture that dies in
>    ten minutes saved you a session. Never let a run become the result.
>
> ## Designing anything that searches
>
> The counting bound wants `log2|M| >= n` or a negative says nothing; the
> forced-positive gate wants `log2|M| <= n` or a fit exists by dimension alone.
> They meet at one point, so **a bare negative is informative only there** —
> which is why each extra DFAO state costs ~10x. Escape it by making the
> evidence either `extrapolation` (fit on `N`, predict `H` unseen coefficients —
> no parameter count forces that) or `curve-shape` (plateau versus growth across
> at least three sizes, against a null and a positive control). The triage gate
> enforces both.
>
> Every searching design needs a **positive control** that proves it can detect
> a shortcut that really is there, and a **matched null**. Report a curve
> against a null, never a point against nothing.
>
> **Single seed only.** All three prizes concern the one deterministic
> single-black-cell orbit. Uniform Bernoulli is invariant, so Prize 2 is already
> trivial for random ICs — an ensemble quantity is a property of the rule, not
> of the prize object. Random and Thue-Morse cards are controls. One random seed
> is not the 7-seed null band.
>
> ## Where things stand
>
> `s*(n)` base-2 MSD is exact at 48, 52, 56 and 64 — closed, do not extend.
> `experiments/algebraic_relation.py` reaches `N=8192` with a 289-coefficient
> class and finds the center indistinguishable from random while Thue-Morse
> separates cleanly and plateaus over six lengths. Four open obligations sit in
> the queue, the strongest being **Christol transcendence** (Prize 3) and the
> **unsettled-core lower bound** (Prize 3, the only region the center column
> actually lives in — the settled wedge is provably disjoint from it).
>
> ## What to deliver
>
> A dated experiment log, the smallest justified update to `STATUS.md` or
> `CLAIM_LEDGER.md`, and the obligation queue left better than you found it:
> lemmas closed, refuted, or sharpened. Run `verify_all.py` again before
> committing.
>
> Report honestly. A refuted lemma is a good session. An unrefuted lemma with a
> sharper statement is a good session. A thousand solver calls that extended a
> finite exclusion is not, however green the run was. Do not claim a prize
> result from any finite computation.
