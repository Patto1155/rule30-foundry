# The agent loop — a worker that can actually look things up

`WORKER_POOL.md` describes workers that answer once. This describes the one
that iterates: it reads, greps, runs commands, fetches pages, and keeps going
until it has an answer or its budget is spent.

## Why it exists

Measured on 2026-09-08, before this: a three-task pool spent 12k–22k tokens
per task and made **zero tool calls**. It could not make one — a completion
endpoint has no mechanism for it. An `investigate` task "read" exactly the one
file that had been pasted into its prompt, and 21,866 of its 21,909 completion
tokens were internal reasoning.

`tools/agent_loop.py` drives an OpenAI-shaped tool-calling conversation
instead. Same worktree, same result contract, same gates — the difference is
that the worker can find things out.

## Tools

| tool | reaches |
|---|---|
| `list_dir`, `read_file`, `grep` | the worktree, and only the worktree |
| `write_file`, `edit_file` | the worktree, and only the worktree |
| `run` | a shell, cwd the worktree |
| `fetch_url` | the network, through whatever egress policy applies |

The first five resolve their path argument — both sides resolved, so a symlink
pointing out of the tree is caught — and refuse anything landing outside the
worktree.

`run` is a shell, and a shell is not confined by that check. **The boundary
around `run` is the container, not this module.** What the module provides
instead is budgets, a full transcript, and a diff a human reads before
anything merges. Run the pool somewhere you would run a contributor's PR
script.

`read_file` numbers its lines, `grep` returns `path:line: text`, `edit_file`
refuses an ambiguous anchor, and every result is truncated in the middle:
context in a loop is quadratic in what you let into it.

## Fetched pages are data, never instructions

A worker that can `run` a shell and then reads a page saying "ignore your
instructions" is the whole prompt-injection problem in one process. Three
things push back, and none is a guarantee:

- every fetch is wrapped in `<untrusted_content source="...">`,
- the system prompt says, before any task text, that such content is evidence
  to quote and never an instruction to follow,
- the transcript records every call, so a surprising diff can be traced to
  what the worker read just before it changed behaviour.

Treat a branch from a task that fetched anything the way you would treat a
patch from a stranger, because in part it is one.

## Budgets

Five, enforced rather than hoped for: **turns** (40), **tool calls** (50),
**wall clock** (the task's own budget), **US dollars** (1.00, from the
provider's own usage accounting rather than an estimate), and per-result
output size. The first to trip ends the loop, and `stop_reason` names it.

A budget stop is a blocker on the result, so the verdict cannot be
`READY-FOR-REVIEW`. The partial work still reaches a branch — an agent stopped
at its budget has usually done something worth reading.

**When a budget trips, the loop takes the tools away and asks for a report
anyway.** That is not politeness. The first live research run hit its turn cap
after 36 tool calls and six minutes and returned *nothing*: the work was done
and unreadable. Tools are withheld rather than discouraged, because the budget
is spent and a model that can call one will.

## Live: the first real research task

`investigate` mode, `deepseek/deepseek-v4-flash-0731`, 2026-09-08. Task: work
out which `verify_all` stages run and which SKIP, *by actually running it*,
then explain the `--allow-skip` semantics with line numbers.

```
READY-FOR-REVIEW  [investigate · deepseek/deepseek-v4-flash-0731 · 561.4s]
  tools       grepx2, list_dirx5, read_filex7, runx23, write_filex1
  budget      29 turns, 38 calls, $0.014, 561.4s  [done]
```

**38 tool calls, 23 of them shell commands, for 1.4 cents.** The answer cited
`build_stages()` lines 63–104, `run_stage()` 121–123 and `skip_permitted()`
107–118, and correctly distinguished `allowed is None` from `allowed == ()`.

Two things it produced are worth more than the answer:

**It found a real bug in this repo's own test suite.** `verify_all` came back
with one failure — `test_dry_run_dispatches_nothing_and_prints_the_prompt`
asserts that no `codex-wt-*` worktree exists, which is false whenever the
suite runs *inside* one, as a delegated worker's does. It correctly called
this an environment collision rather than a repo regression. The test now
counts before and after.

**It disclosed that it had broken its own checkout.** To test a hypothesis it
ran `git worktree remove` on the worktree it was living in, then re-created
it. The files were identical, so `git diff` would have reported no changes and
been believed — the harness would have said "no changes" about a task that had
made some. Two fixes came from that: the system prompt now forbids `git
worktree`, `git branch` and anything else that changes what the checkout *is*,
and the harness records HEAD before and after and raises a blocker if the
worktree moved or vanished.

Neither would have surfaced from a completion worker, because a completion
worker cannot run anything. The capability that makes it useful is the same
one that makes it dangerous, which is the honest summary of this whole file.

## Usage

```bash
python tools/agent_loop.py tools --pretty        # the schemas the model sees

python tools/codex_worker.py submit --backend agent --mode investigate \
    --task "..." --max-turns 45 --max-cost 0.75

python tools/worker_pool.py run --backend agent --concurrency 10
```

`--backend auto` prefers `local` (the codex CLI, if installed), then `agent`,
then `remote`. A completion backend is the fallback for when no tool-capable
provider is configured, not the default: both reach an outside model, and only
one can read a file it was not handed.

`--tools grep,read_file,run` restricts the set. Artifacts land beside every
other run's, with `transcript.jsonl` — every turn, every call, every result —
which is the first thing to read when a result surprises you.
