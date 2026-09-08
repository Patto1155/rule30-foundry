# The worker pool — many cheap workers, one lead

`docs/CODEX_WORKER.md` covers delegating **one** task. This covers doing ten
at once, and switching who does them.

The shape the repo now supports:

```
Claude Code (web or CLI)          ← the lead. Decides, judges, commits.
    │  loads this repository, which holds the instructions and the queue
    ▼
tools/worker_pool.py              ← the orchestrator, versioned here
    │  N at a time, each in its own git worktree
    ▼
tools/providers.py                ← who actually answers
    ├── openrouter          any model, ten at once, metered
    └── codex-dispatcher    your VM, one careful worker
    │
    ▼
N branches + one summary          ← nothing merged, ever
```

Claude does not live in the repository. Its *instructions*, its *task queue*
and its *orchestrator* do; a Claude Code session supplies the running lead
that loads them. That is why the orchestration is a script here rather than a
prompt: a prompt dies with the session.

## Concurrency buys throughput, not independence

Ten workers clear ten independent chores in the time one would take. They do
not check each other. Ten calls to one model share that model's blind spots
exactly as ten Claude subagents share Claude's — `CLAUDE.md` makes this point
about subagents and it is no less true here. **Agreement between two workers
in the same pool is one prior sampled twice, not corroboration.** For
disagreement, send the claim to a *different* provider, or to
`tools/council.py`, and read `verification` rather than either model's account
of itself.

Independence *between tasks* is structural: each gets its own worktree, so two
workers editing the same file are editing different checkouts of it. What they
share is one `.git`, whose index and worktree registry are not
concurrency-safe — so every operation against the main repository takes a lock
and everything inside a worktree runs free. The slow parts (the model call,
`verify_all`) are the unlocked ones, which is what makes the fan-out worth
having. `tests/test_worker_pool.py` runs four tasks against one `.git` and
asserts both the overlap and, as a negative control, that `--concurrency 1`
does *not* overlap.

Two tasks that edit the same file will both succeed and conflict at
integration time. That is the lead's problem to resolve, not a reason to
serialise the pool on a guess about overlap.

## Providers

| | `openrouter` | `codex-dispatcher` |
|---|---|---|
| Reaches | `openrouter.ai`, any model it serves | your VM over HTTPS |
| Concurrency | as many as you pay for | one process per request |
| Cost | metered per token | the VM you already rent |
| Use for | the pool: breadth, chores, ten at once | one careful worker; reviews comparable with past reviews |

Both are **completion** providers: prompt in, text out. Neither can run
anything, so an edit comes back as a unified diff which the worker applies and
then tests itself. The agentic path — `codex exec` with a writable checkout —
is `--backend local` and does not fan out.

```bash
python tools/providers.py check                   # what is usable, and why not
python tools/providers.py models --grep deepseek  # live catalogue, needs egress
```

### The default model is a guess until you check it

`DEFAULT_OPENROUTER_MODEL` in `tools/providers.py` is `deepseek/deepseek-chat`
and **it was not verified**: `openrouter.ai` was not reachable from the
environment where this was written, so the catalogue could not be queried. Run
`python tools/providers.py models --grep deepseek` from a host that can reach
it and set `OPENROUTER_MODEL` to whatever actually exists. A wrong slug is a
400 per task rather than one clear failure.

## Configuring the key

Two things are needed, and neither can be done from inside a session.

**1. Let the environment reach OpenRouter.** Claude Code's remote environments
use a default-deny egress allowlist. An unlisted host fails the CONNECT with
`403` before any request is sent, which is why a provider error and a policy
denial are told apart explicitly in `providers.explain_url_error`. Add
`openrouter.ai` to the environment's network policy:

> claude.ai → Settings → **Code** → your environment → **Network access** →
> add `openrouter.ai` to the allowed domains.

The policy is read when a session starts, so **start a new session** after
changing it. Confirm from inside one with:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://openrouter.ai/api/v1/models
# 401 → reachable (no key sent). 000 with "CONNECT tunnel failed" → still blocked.
```

**2. Put the key in the environment as a secret**, in the same environment
settings, named exactly:

```
OPENROUTER_API_KEY = sk-or-v1-...
```

Optionally `OPENROUTER_MODEL` alongside it, once you have checked the slug.

Never paste the key into a task spec, a commit, a PR comment, or a shell
command — a key in argv is a key in `ps` output. Every tool here reads it from
the environment and nothing prints it. If it does end up somewhere it should
not, rotate it at openrouter.ai/keys; that is cheaper than auditing where it
went.

For local (non-web) use, `export OPENROUTER_API_KEY=...` in the shell that
runs the pool works the same way.

## Running it

```bash
python tools/worker_pool.py list                          # what is queued
python tools/worker_pool.py run --dry-run                 # what would be sent
python tools/worker_pool.py run --concurrency 10 --provider openrouter
python tools/worker_pool.py run queue/tasks/one.json --verify fast
```

Task specs live in `queue/tasks/`, one JSON file each, committed and reviewed
like code — a spec is an instruction to an outside model that will edit this
repository. `queue/tasks/EXAMPLE.json.template` is a filled-in one.

`--max-tasks` caps a run, which with a metered provider is the difference
between a test and a bill. `--dry-run` prints exactly what would be submitted
and creates no worktrees.

Artifacts land in `runs/pool/<stamp>/` (gitignored); each task's own
`prompt.txt`, `raw.txt`, `changes.patch` and `verification.txt` are under it.

## Keep `--verify full`

The two live runs on 2026-09-07 make the case better than an argument would.
Two tasks, concurrent, through the `codex-dispatcher` provider:

```
  ATTN  add-quiet-flag.json       63.8s  codex/implement-add-a-quiet-flag-...
  ready investigate-gates.json    64.6s  codex/investigate-which-gates-...
  1 needs-attention, 1 ready-for-review
```

64s wall for two 60s calls — they genuinely overlapped. The `implement` task
applied cleanly under the `strict` strategy, returned a well-formed report
with `"blockers": []`, and **passed both of its acceptance commands**. It was
still wrong: the test file it added had a syntax error, and only the
`unittest` stage of `verify_all` saw it.

Acceptance commands check the thing you thought to check. `verify_all` checks
the repository. A worker that breaks something you did not think to name is
the normal case, not the exotic one.

## What this does not do

- **It does not merge.** N branches and a summary; the lead reads them.
- **It does not survive the session.** Everything runs inside the Claude Code
  session that started it. Batches that must continue after the session ends
  need a host that outlives it — a Codespace or a VM — running the same
  script. Nothing in the design assumes one; that is the *only* thing renting
  one would buy here.
- **It does not schedule or retry a whole task.** A provider's 429 is retried
  inside `providers.py`; a task that comes back `NEEDS-ATTENTION` is a
  judgement call, and judgement is the lead's job.
