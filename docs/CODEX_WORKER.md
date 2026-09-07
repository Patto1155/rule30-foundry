# Codex as a callable worker

Claude is the lead researcher in this repository. It decides what is worth
doing, judges whether a result is real, and owns every grade in
`CLAIM_LEDGER.md`. This document is about how it stops doing its own grunt
work.

`tools/codex_worker.py` makes Codex a **worker Claude can call**: coding,
debugging, writing tests, refactoring, tracing something through the tree,
implementing an experiment that does not exist yet. What comes back is a
reviewable branch plus a structured report. Nothing is merged, and nothing
reaches the lead's working tree.

## Why not the alternatives

**Why not a Claude subagent?** `CLAUDE.md` already answers this: subagents
give throughput, not independence. They share a model lineage with whoever
spawned them and therefore share its blind spots. For grunt work that is
often fine, but it means a subagent's "I checked it" is worth exactly what
the parent's would be, and this repo's two expensive failures — the 2026-08
counting-bound retraction, the bit-order bug that survived five months —
were both cases of an agent checking its own work and agreeing with itself.
Codex is the only worker here whose agreement is evidence.

**Why not `tools/council.py`?** The council asks a question and reads an
answer. It is read-only by design and has no repository, so it cannot fix
anything. Review is one useful thing to ask an outside model for. It is not
the only one, and building the whole integration around it was the drift this
tool corrects: `review` is now one of six task modes.

**Why not delegate everything?** Because most of the backlog is a script that
already exists and has simply not been run. Sending a deterministic command
through an LLM buys nothing and adds a failure point. `tools/workhorse.py
--agent script` runs those, and that stays the default. Delegate the work that
needs judgement; let the gated runner execute what is already written.

| The work is… | Route |
|---|---|
| an existing script, with a manifest | `workhorse.py --agent script` (default) |
| an experiment that must be *written* first | `codex_worker.py --mode implement --manifest …` |
| a bug with a known symptom and no known cause | `codex_worker.py --mode debug` |
| "does the repo do X, and where" | `codex_worker.py --mode investigate` |
| a claim that needs an outside opinion | `codex_worker.py --mode review`, or `council.py` |
| a decision about what any of it means | **Claude. Not delegated.** |

## Shape

```
┌──────────┐  task spec   ┌───────────────┐  isolated worktree
│  Claude  │ ───────────► │ codex_worker  │ ─── git worktree add ──┐
│  (lead)  │              │               │                        ▼
└────▲─────┘              └───────┬───────┘                codex/<branch>
     │                            │ backend                        │
     │  structured result         ├─ local  : codex exec            │
     │  + branch + patch          │           --sandbox             │
     │                            │           workspace-write       │
     │                            └─ remote : dispatcher /ask,      │
     │                                        reply carries a patch │
     │                                                              ▼
     │              ┌────────────────────────────────────────────────┐
     └──────────────┤ verification the WORKER ran itself:            │
                    │ verify_all · gates.postflight · the task's own │
                    │ acceptance commands                            │
                    └────────────────────────────────────────────────┘
```

### Two properties, and each is the reason for a design choice

**Isolation is structural, not promised.** Every task runs on its own branch
in its own `git worktree`. The lead's checkout is never touched, a failed task
leaves nothing to clean up, and two tasks cannot interleave edits. A worker
that edited the working tree would make "review before integrating" a matter
of discipline; a worktree makes it a matter of fact.

**Claimed and verified are different fields.** `commands_run` and `tests` are
what Codex *says* it did. `verification` is what the worker ran, in the
worktree, afterwards, without consulting the report. They are kept apart on
purpose: an agent reporting its own green tests is the oldest way to get a
wrong answer past a review. Trust `verification`; read the rest as testimony.
`tests/test_codex_worker_e2e.py` pins this down with a fixture that reports a
passing suite it never ran — the verdict follows the acceptance command, not
the claim.

## Backends

| | `local` | `remote` |
|---|---|---|
| Where | wherever `codex` is installed and logged in | the dispatcher VM, over HTTPS |
| Codex has | a writable checkout, and can run things | the quoted files, and nothing else |
| Edits travel as | edits | a unified diff, applied here |
| `commands_run` | real | required to be empty |

`--backend auto` (the default) picks `local` when `codex` is on `PATH`, else
`remote`. In the Claude container there is no `codex`, so it is `remote`; on
the dispatcher VM it is `local`. The same task spec works either way.

Remote mode is honest about its limits rather than papering over them: Codex
cannot run anything, so it is told to return empty `commands_run` and `tests`,
and every test in the result is one this process ran after applying the patch.

### Applying a hand-written diff

A model writing a diff from quoted file contents gets the hunk *content* right
and the `@@` line counts wrong — it has no line numbers to count from. The
first live remote run rejected a perfectly good patch as `corrupt patch at
line 31`. So `apply_patch` tries strategies strictest-first — `strict`,
`recount`, `recount+fuzz` — and records which one worked in
`patch_apply_strategy`. The ladder loosens *counting*, never *matching*: a
hunk whose context is not in the file still fails. Anything past `strict` is
surfaced rather than absorbed, because a loosened apply can land a hunk
somewhere the author did not mean.

## The gates stay around everything

Delegation must not be the way around a gate. `tools/gates.py` runs on both
sides of a delegated task that carries a manifest:

- **preflight, before dispatch.** A manifest that cannot produce information
  is refused with verdict `REFUSED`, before a worktree exists and before a
  token is spent. The 2026-08 retraction was an experiment that ran when it
  should have been refused; a delegation path that skipped preflight would
  reintroduce exactly that.
- **postflight, on the report.** A conclusion that states more than the run
  measured — an unqualified "never", a metric with no baseline — fails, and
  the verdict cannot be `READY-FOR-REVIEW`.

On top of that, `verification` runs `verify_all` **in strict mode**
(`--allow-skip` naming only the bitstreams and the SAT toolchain, which a
worktree legitimately does not have). "SKIP is not PASS" has to survive being
run by a robot, or it only ever applied to humans.

## Verdicts

| Verdict | Means |
|---|---|
| `READY-FOR-REVIEW` | contract intact, verification green, no blockers. **Still not merged.** |
| `NEEDS-ATTENTION` | verification failed, the report is malformed, or codex exited nonzero |
| `BLOCKED` | the agent declared a blocker, or its patch would not apply |
| `REFUSED` | preflight rejected the manifest; nothing was dispatched |

`READY-FOR-REVIEW` is a statement about the harness, not about the science.
The lead still reads the diff.

## The result contract

Every mode returns the same seven keys, enforced by `validate_result`:
`summary`, `changes`, `commands_run`, `tests`, `artifacts`, `uncertainties`,
`blockers`. `uncertainties` and `blockers` are **required keys**: an empty
list asserts there were none, and a missing key is a worker that was never
asked. Those two must not look alike.

The worker adds what the agent cannot be trusted to report about itself:
`files_changed` (from `git diff --name-status`), `patch`, `branch`, `commit`,
`verification`, and the verdict.

## Usage

```bash
python tools/codex_worker.py modes          # the six task modes
python tools/codex_worker.py check          # which backend is available, and why

python tools/codex_worker.py submit \
    --mode implement \
    --task "Add a --json flag to tools/lint_ledger.py" \
    --context tools/lint_ledger.py \
    --acceptance "python tools/lint_ledger.py" \
    --pretty
```

`--context` matters only for the remote backend, which has no checkout: name
the files Codex needs to see. `--acceptance` matters for both — those commands
are run by the worker against the result regardless of what the report says,
and the task prompt tells Codex so up front.

`--dry-run` prints the assembled prompt and dispatches nothing. `--verify
none` disables verification and says so in the result, in words: a run with
`verification.ok = null` carries no evidence beyond the agent's own testimony.

Artifacts land in `runs/codex/<task_id>/` (gitignored): `prompt.txt`,
`raw.txt`, `changes.patch`, `verification.txt`, `result.json`.

## The end-to-end run

The previous integration's own documentation had to record that `--agent
codex` was "written but never exercised". A code path that has never run is
not an integration, so this one was landed only after a real task was put
through it.

**2026-09-07, remote backend, `gpt-5.6-sol`, `implement` mode.** Task: add a
`--list-roles` flag to `tools/council.py`, with two acceptance commands.

```
READY-FOR-REVIEW  [implement · remote · 60.1s]
  branch      codex/implement-add-a-list-roles-flag-to-tools-council-p-28cffb56
  verify ok   python tools/verify_all.py --allow-skip=bitstream:* --allow-skip=drat-toolchain
  verify ok   python tools/council.py --list-roles
  verify ok   python -m unittest tests.test_council
  uncertainty Only tools/council.py was supplied, so the test was added as a
              separate unittest module rather than modifying the unseen
              tests/test_council.py.
```

It changed two files — `tools/council.py` and a new
`tests/test_council_list_roles.py` — applied under the `recount` strategy, and
passed all three checks. The branch was reviewed and not merged: the feature
was a vehicle for exercising the pipeline, not something the repo needed.

Two things that run went into this tool rather than into a changelog:

1. The **first** attempt returned `BLOCKED`. Codex's patch was correct and
   `git apply` rejected it as corrupt over the hunk counts — hence the apply
   ladder above. Worth noting that the report said `"blockers": []` and read
   as a success; the harness caught it because the acceptance command exited
   2. That is the claimed-versus-verified split doing its job on its first
   real outing.
2. Codex's `uncertainties` were substantive both times — it flagged that it
   could not safely patch a file it had not been shown. That is the field
   earning its place in the contract.

## What this does not do

- **It does not merge.** Ever. The output is a branch and a report.
- **It does not schedule.** One task per invocation.
- **It does not make Codex's agreement into corroboration by itself.** A
  worker that was told what to build and then built it has not independently
  confirmed anything. Use `--mode review`, or `council.py`, when what you want
  is disagreement.
- **Local backend is unexercised against a real model.** `codex` is not
  installed in the Claude container, so the live run above went through
  `remote`. The local path is exercised end to end by
  `tests/test_codex_worker_e2e.py` against a stand-in binary — every stage
  real except the model — and is the path to use on the dispatcher VM.
