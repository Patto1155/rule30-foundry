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
anything, so under `--backend remote` an edit comes back as a unified diff
which the worker applies and then tests itself.

## Choosing a backend

The provider says *who answers*; the backend says *what the worker can do*.

| `--backend` | What the worker can do | Fans out |
|---|---|---|
| `agent` **(pool default)** | reads, greps, writes, edits, runs a shell — inside its own worktree | yes |
| `remote` | one completion; the diff it writes is applied for it | yes |
| `local` | `codex exec` with a writable checkout | **no** — one process per task |
| `auto` | prefers `local`, then `agent`, then `remote` | only if `local` is absent |

`agent` is the default because it is the only backend that can *investigate*:
a completion worker answers from the prompt and whatever `context_files` were
pasted into it, so it cannot discover the file nobody thought to attach. The
loop is `tools/agent_loop.py` and needs no CLI installed anywhere — only
`OPENROUTER_API_KEY`.

`auto` is a poor default *for the pool* specifically, because it prefers
`local`, and `local` is one `codex` process per task. It is the right default
for `codex_worker.py` running a single task, which is why the two differ.

Read `docs/AGENT_LOOP.md` before enabling `fetch_url` on a task: `run` is a
shell, the boundary is the container rather than the worktree, and a worker
that reads an attacker-controlled page and then acts on it is the hazard that
combination creates.

```bash
python tools/providers.py check                   # what is usable, and why not
python tools/providers.py models --grep deepseek  # live catalogue, needs egress
```

### The default model, and what it costs

`DEFAULT_OPENROUTER_MODEL` is `deepseek/deepseek-v4-flash-0731`, checked
against the live catalogue and exercised end to end on 2026-09-08. Pinned to
the dated build rather than the floating `deepseek/deepseek-v4-flash` alias:
a repo whose whole argument is reproducibility should not have its worker
change underneath it. The alias is about 40% cheaper if you would rather have
that than the pin.

Measured over five live tasks: **$0.0015–$0.018 each, 2 to 8 minutes each**.
It is a reasoning model, and the time is reasoning — on the first task 14,332
of 14,741 completion tokens were reasoning tokens. That is the whole argument
for the pool: ten tasks at 5 minutes each is 5 minutes concurrent and 50
serial, for the same few cents.

Re-check with `python tools/providers.py models --grep deepseek` before
assuming that slug is still current. Set `OPENROUTER_MODEL` to override it per
environment, or `--model` per run. A slug remembered rather than checked gets
you ten tasks that each fail identically.

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
A `--backend agent` task also leaves `transcript.jsonl` — every tool call and
result, which is the only place the *work* is legible rather than the claim.

### Budgets belong in the spec

`limits` in a task spec overrides `agent_loop`'s module defaults per task:

```json
"limits": {"max_turns": 40, "max_tool_calls": 80,
           "max_cost_usd": 0.35, "wall_clock_s": 1500}
```

Fifteen unbounded workers against a metered provider is how a fan-out becomes
a bill, so the shipped specs carry their caps explicitly rather than
inheriting them — the number that matters should be readable in the diff.
`worker_pool.py` validates both `limits` and `tools` before it creates a
worktree: an unknown budget key would otherwise be silently dropped, leaving
the author believing a cap is in force that is not.

## A fifteen-task fan-out

`queue/tasks/` ships fifteen specs — twelve `test` tasks, one per untested
module under `experiments/`, and three read-only `investigate` audits:

```bash
python tools/worker_pool.py run --concurrency 15
```

They are deliberately **file-disjoint**: each `test` task creates exactly one
new `tests/test_<module>.py` and is forbidden from editing the module it
tests, so fifteen branches can be read and merged in any order without
conflict. That is a property of these specs, not of the pool — the pool will
happily run two tasks that collide and leave the lead to resolve it.

The three `investigate` specs are handed `["list_dir", "read_file", "grep",
"run"]` and nothing else. A read-only task that *cannot* write is enforced by
the tool list; prose telling a model not to write is not enforcement.

**On this machine (4 cores), `--verify full` at fifteen is affordable.** The
suite is wait-bound rather than CPU-bound — measured 61s alone, 72s with four
concurrent, 108s with eight — so verification is not what makes the run long.
The model is: reasoning time dominates, which is the whole reason to overlap.

**Fifteen deepseek workers agreeing about something is not evidence.** It is
one prior sampled fifteen times, exactly as `CLAUDE.md` says of Claude
subagents. Corroboration needs a differently trained model — that is what
`tools/council.py` is for.

## Live: what three concurrent OpenRouter tasks actually did

2026-09-08, `deepseek/deepseek-v4-flash-0731`, `--concurrency 3`. Two
`implement` tasks (a `--count` flag for `lint_ledger.py`, a `--json` flag for
`lint_bitorder.py`) and one `investigate`. **311s wall against 658s of serial
work**, so the fan-out is real.

The first pass scored 2 of 3 as `NEEDS-ATTENTION`, and both failures were
*reporting* failures on code that was correct:

1. **Two tasks returned a verified patch and no JSON report at all.** The
   change applied, `verify_all` passed, both acceptance commands passed — and
   the contract check found nothing to read. The output contract had been
   sitting in the middle of a long prompt, above the task and the quoted
   files. It now comes last, where the model still has it in view when it
   starts writing, and says explicitly that a diff without a report is
   incomplete.
2. **One task reported "no repository checkout was provided" as a blocker** —
   a restatement of the setup the prompt had just given it. Any blocker means
   `BLOCKED`, so a finished change was held up by the mode's own definition.
   The contract now says what a blocker is not: no checkout, no shell, and no
   sight of a file you did not need are `uncertainties`, not blockers.

After both fixes the same three tasks came back `READY-FOR-REVIEW`, with
substantive uncertainties — *"I assumed `--count` is a short-circuit mode that
exits before running the lint checks"* is exactly the kind of thing a lead
needs to see and a confident summary would have hidden.

Neither fix is about the model being bad. Both are about a prompt that asked
for two things and got the one it emphasised, which is what prompts do. They
are recorded here because the next provider will need the same treatment and
the failure will not look like this one.

## Keep `--verify full`

Two live runs on 2026-09-07 make the case better than an argument would.
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
