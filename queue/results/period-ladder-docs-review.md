# Period ladder: reference-doc realignment review

Date: 2026-10-14. Base: `b450406deace9f8adabc7b5b95d45b85e2c32432`.
This is a review draft for the lead, not an alternate state file. No grade
change is proposed by these reference edits.

## Reference edits made

- `CLAUDE.md` and `AGENTS.md`: replace the obsolete fresh-clone ~2-second
  verifier estimate with about 1–2 minutes on a CPU-only checkout, explicitly
  qualified by inputs and hardware. Final acceptance timing is recorded below.
- `CLAUDE.md`: add the requested delegation fallback when no outside provider is
  configured or reachable; lead review, acceptance replay and grade ownership
  remain mandatory, and subagent agreement is never corroboration. Qualify the
  outside-provider routing rule so the fallback does not contradict it.
- `docs/theory/README.md`: retain the annihilator gate as §5 and renumber
  standing proposal rules to §6. Its internal §5 references correctly point at
  the annihilator gate and are unchanged.
- `.claude/agents/theory-gate.md`: both references to standing rules become §6.
- `docs/center-col-46M-benchmark.md`: both nonexistent GPU-script paths now point
  at `gpu/rule30_sim.py`, which exists. No GPU benchmark was rerun.

One historical cross-reference requires lead ownership under the instruction
that subagents never edit experiment logs:
`docs/experiment-logs/2026-08-15-structured-wedge-decomposition.md:370` refers
to the promotion-path standing rule as `docs/theory/README.md` §5; replace that
reference with §6. The 2026-09-03 annihilator log's §5 reference remains correct.
The archived 2026-08-30 handover's ~2-second timing is historical, not a live
instruction; it was not modified.

## Exact STATUS draft for lead-owned application

1. Replace the header with:

   `Updated: 2026-10-14 · Newest log: docs/experiment-logs/2026-09-18-theory-first-loop-and-algebraic-relations.md`

   Keep the path backticked in the actual STATUS line. The newest filename under
   the ledger lint's date/name ordering is the theory-first log. If this session
   adds a newer experiment log, cite that instead. The original header's 09-17
   date is stale even though the body already cites both 09-18 logs, and this is
   why baseline `lint_ledger.py` passes rather than detecting header drift.

2. In B3, replace the parenthesised plan reference with:

   `queue/jev/frontier.json is the focused default plan; queue/jev/lsd56.json is the wider sweep`

   Preserve backticks around paths in the actual table. Evidence: commit
   `6ae8e5b54361a188b23876150511d14197df29a8` changed the default frontier from
   closed MSD n=64 cards to open LSD n=56 cards and added the wider plan. Reading
   the checked-out JSON confirms its goal and direction, calibrations at n=56
   s=11 and open cards s=12,13, plus matched random controls. The existing bound
   remains `{12,13,14}`; this task did not rerun its certificates.

3. Replace the Jev selector pilot paragraph with:

   **Jev selector pilot and live follow-up:** `tools/jev_search.py` selects
   bounded DFAO instances using fixed, random or Jev policies, verifies
   witnesses/DRAT, and prunes only checked implications. The 2026-09-17 CPU
   pilot was followed by a live OpenRouter adapter check on 2026-09-18 (HTTP
   200, three validated decisions), and a matched six-run comparison at one
   frozen n=56 plan and one 180-second budget. The two Jev runs established
   all five centre results in 122 and 132 seconds; fixed established three
   before its proof check was cut off. This is a scoped selector-throughput
   result, about 2.4x verified centre results/second: a goal-aware policy
   compared with goal-blind fixed/random policies, not mathematical insight or
   demonstrated generalisation. Jev deferred null work. No live calls were
   repeated in the present session because `OPENROUTER_API_KEY` and
   `CODEX_COUNCIL_*` are unset.
   `docs/JEV_SEARCH.md`; `docs/experiment-logs/2026-09-17-jev-selector-pilot.md`;
   `docs/experiment-logs/2026-09-18-jev-openrouter-live-and-campaign-scaling.md`
   §§1, 3b.

   Evidence: commits `91435570cc7815129d332483fc4a10dba2e4df0a` and
   `b40c6d9d291ad53b76d8f76461a987348afe8cc6` plus the named checked-in log.
   The log's early §3 text saying the comparison was not run is explicitly
   superseded by §3b. These historical claims were read, not live-reproduced.
   Environment presence check found OPENROUTER_API_KEY false and no
   CODEX_COUNCIL_* variable names; no values were printed.

4. Resolve the competing priorities explicitly; this agent does not choose.

   - B3 is called next in the introductory paragraph and its row: close LSD
     n=56 s=12/13 using `queue/jev/frontier.json`, then bases 3/4.
   - T1's row calls Christol-transcendence and unsettled-core obligations next:
     `queue/theory/obligations.json`, backed by
     `docs/experiment-logs/2026-09-18-theory-first-loop-and-algebraic-relations.md`.

   The approved session handover assigns period-2 and nonlinear-obstruction
   work; the lead should replace the obsolete unconditional B3-next sentence
   with that session's chosen order, keeping B3 listed as optional bounded work.
   No priority or mathematical grade was changed by the subagent.

## Acceptance and exclusions

Rule 90 test: N/A, infrastructure/reference documentation only.
Zero test: N/A, infrastructure/reference documentation only.

Acceptance:

- `python tools/lint_ledger.py`: PASS, 45 ledger rows / 57 experiment logs.
- `python tools/verify_all.py`: FAIL, elapsed 80.373 seconds, 7 stages PASS,
  3 stages SKIP and 1 stage FAIL. Unittest stage took 78.5 seconds, 747 tests
  (74.415-second unittest runner), 10 individual skips, one failing test:
  `test_worker_pool.TestPool.test_concurrent_worktrees_do_not_corrupt_the_main_checkout`
  counted 7 vs 8 worktrees while another suite/worker changed shared git
  metadata. The lead observed the same concurrent-creation race in its baseline
  and requested no further concurrent suite runs; final suite replay is lead-owned.
  The 3 stage SKIPs are absent `data/center_col_10M.bin`, absent
  `data/center_col_46M.bin`, and absent SAT toolchain in this isolated worktree.
  The bitstream/toolchain owner is another agent; these artifacts were not copied.
- Final `git diff --check`, no obsolete GPU path in reference files, unique
  numbered second-level theory sections, and theory-gate §6-reference checks:
  PASS.

No commits made; manifest not regenerated; STATUS, CLAIM_LEDGER, and experiment
logs untouched.

Not run / not verified: no historical Jev/DRAT/SAT campaign replays; no live
outside-provider calls; no GPU or 46M/500M simulation; no new proof attempt or
search; no counting-bound search needed; no Lean/Mathlib/network installation;
no commit/verifier gate agent (lead-owned); no CI execution; no PR or merge.
