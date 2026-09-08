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
    python tools/worker_pool.py run --concurrency 10     # send it

Nothing here merges anything. The pool leaves N branches and a summary; the
lead reads them.
