# 2026-09-15 — the delegation pipeline, driven end to end

**Goal.** `docs/STATUS.md` has recorded since 2026-09-08 that the delegation
machinery is "argued for rather than demonstrated": `workhorse.py --agent
codex` had never implemented an experiment, and `queue/b1-pattern-map-walk.json`
had never been run. This is the first real task through
`tools/codex_worker.py --backend agent`, chosen to exercise the pipeline
rather than to advance a prize — including an interrupted run and the
recovery of its evidence from a fresh checkout.

**Setup.** Backend `agent` (the tool-using loop), provider OpenRouter,
`deepseek/deepseek-v4-flash-0731`. Base `HEAD` of the purpose-taxonomy branch.
Verify level `fast`. Acceptance command:
`python tools/gates.py preflight queue/halo-repin-random-ic.json`.

Task: create the manifest for the open-boundary halo re-pin check that
`docs/WORKFLOW.md` mandates under *Correctness lessons already paid for* and
that had no manifest, because until this branch the seed gate refused the
random IC the check requires.

## Run 1 — interrupted

`--max-turns 14 --max-tool-calls 25`. The model spent every turn reading and
hit the turn cap before writing the file.

| | |
|---|---|
| `stop_reason` | `max-turns` |
| `budget.turns` | **14** against `max_turns` 14 |
| `budget.tool_calls` | 23 against 25 |
| cost | $0.0154, 223 s |
| `verification.ok` | **false** — acceptance exited 2 |
| `verdict` | `BLOCKED` |

Three things worth recording.

**The turn count is exactly the limit, not one over.** Before this branch the
working phase ran to `max_turns` and then billed a wrap-up turn on top, so
`--max-turns 14` cost 15. The report is now budgeted rather than added.

**The agent's report and the harness agreed, and only one of them counts.**
The report named the blocker honestly — *"Stopped at the max-turns limit
before writing queue/halo-repin-random-ic.json … the harness will see it
missing"* — and the harness independently ran the acceptance command and got
exit 2. The verdict came from the second.

**Multi-call batches are normal, not hypothetical.** Turns 1–3 each returned
three tool calls. Before this branch the budget was checked only between
turns and the whole batch then executed, so a limit of *n* calls could be
overrun by the width of a batch.

## Recovery from a fresh checkout

`git clone --single-branch` of the task branch, with no `runs/` tree present:

```
runs/codex present? NO
queue/results/c4655882.json          <- the agent's report   (testimony)
queue/results/c4655882.review.json   <- the harness's result (evidence)
```

The review record recovered `verification.ok = false`, all three checks with
their exit codes, the inlined log, the verdict and the stop reason. Its
`commit_tested` is `b13364d…`, and `git rev-parse HEAD^` is the same commit —
so the claim was checked against the graph, not merely read from the field.
`git cat-file -e HEAD^:…review.json` returns absent: the tested commit does
not contain the record, so a report-only commit made after verification
cannot pass as the state that was verified.

## Run 2 — completed

`--max-turns 30 --max-tool-calls 45`. Wrote the manifest at turn 17, then
iterated against the gate (`run` at 18, `edit_file` at 22, `run` at 24–26).

| | |
|---|---|
| `stop_reason` | `done` |
| `budget` | 27 turns, 43 tool calls, $0.0141, 635 s |
| `verification.ok` | **true** — all three checks exit 0 |
| `verdict` | `READY-FOR-REVIEW` |

## Review — what the lead checked, and what the harness could not

`READY-FOR-REVIEW` is not a merge. The manifest was read and its claims
checked independently:

- `purpose: correctness-check` with `seed: random-ic` is the combination the
  unscoped seed gate used to refuse. Correct here: the run is a statement
  about the instrument.
- `script: experiments/eca_sim.py` with `argv: []` — checked by running it.
  Its default action is the verification itself: packed-CPU vs naive across
  seven rules, and the rule-30 CPU path vs the `rule30_open_utils` reference.
- The random IC is real, not just declared: `verify()` builds its rows with
  `rng.integers(0, 2, …)` (line 257). A lone spike leaves the edges at 0 and
  hides the boundary leak, which is why `WORKFLOW.md` requires the random IC.
- `simulation.cells` 2048 / `steps` 512 clear the light-cone gate but are
  **not** the sizes `eca_sim.py` picks internally. Declarative for the gate,
  inaccurate as a description. Left in place and named rather than quietly
  fixed.

**The acceptance command was too weak, and that is the lead's error, not the
worker's.** "Preflight exits 0" is satisfied by any gate-passing manifest,
including one whose `script` never runs the described check. The worker
happened to pick a script that does; nothing in the harness would have caught
one that did not. An acceptance command that actually ran the script would
have.

## Verdict

The pipeline works, on a real task, with a real model, including the failure
path. The claim `STATUS.md` has carried since 2026-09-08 — that the machinery
is argued for rather than demonstrated — no longer holds for
`codex_worker.py --backend agent`. It still holds for `workhorse.py --agent
codex` and for `queue/b1-pattern-map-walk.json`, neither of which this run
touched.

Total spend: **$0.0295** across both runs.
