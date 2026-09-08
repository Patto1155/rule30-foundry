# Task queue for the worker pool

One JSON file per delegated task. `tools/worker_pool.py run` with no
arguments takes every `*.json` here and runs them concurrently, each on its
own branch in its own `git worktree`.

Committed like code, and reviewed like code: a task spec is an instruction to
an outside model that will edit this repository, so it goes through a PR the
same way a script would. Do not drop a spec here and run it in the same
breath.

Schema is `tools/codex_worker.py`'s task spec; `EXAMPLE.json.template` is a
filled-in one. The template is deliberately not a `.json` file, so the pool
does not pick it up as work.

    python tools/worker_pool.py list                     # what is queued
    python tools/worker_pool.py run --dry-run            # what would be sent
    python tools/worker_pool.py run --concurrency 15     # send it

Nothing here merges anything. The pool leaves N branches and a summary; the
lead reads them.

## What is queued

Fifteen specs, chosen to be **file-disjoint** so the branches they produce can
be reviewed and merged in any order:

- **Twelve `test` tasks**, one per module under `experiments/` that has no test
  today. Each creates exactly one new `tests/test_<module>.py` and is
  forbidden from editing the module it tests. Two workers therefore cannot
  touch the same file.
- **Three `investigate` audits** — the bitorder hazard, the single-seed rule,
  and whether CI's `--allow-skip` list still matches the stages that skip.
  These change nothing and are given `["list_dir", "read_file", "grep",
  "run"]` and no writing tools at all.

Disjointness is a property of *these* specs, not of the pool. The pool will
run two colliding tasks quite happily and leave the conflict to the lead.

## Writing one

`tools/codex_worker.py`'s task spec is the schema, and
`EXAMPLE.json.template` is a filled-in one; the template is deliberately not
`.json`, so the pool does not pick it up as work. Beyond the required `mode`
and `task`:

| Key | Why you would set it |
|---|---|
| `tools` | Restrict what the worker can do. A read-only task should be given no `write_file`/`edit_file` — the tool list is enforcement; prose is not. |
| `limits` | Per-task `max_turns`, `max_tool_calls`, `max_cost_usd`, `wall_clock_s`. Set these; fifteen unbounded workers on a metered provider is a bill. |
| `acceptance` | Commands that must exit 0. They check what you thought to check — `verify_all` checks the repository, and is what catches the rest. |
| `context_files` | Files pasted into the prompt. Under `--backend agent` this is a *hint*, not a limit: the worker can read anything in its worktree. |

`python tools/worker_pool.py list` validates every spec — including unknown
tool names and unknown `limits` keys — without dispatching anything. An
unknown budget key would otherwise be dropped in silence, leaving a cap you
believe is in force and is not. `tests/test_worker_pool.py` asserts that every
spec committed here validates.
