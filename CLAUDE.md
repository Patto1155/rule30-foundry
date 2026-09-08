# rule30-foundry — read this first

GPU-backed empirical research on **Wolfram's three Rule 30 prize problems**.
This is a verification-first research repo, not a notebook dump. Results here
are graded, and the grade is enforced by tooling.

## Health check — run before you start and before you commit

```bash
python tools/verify_all.py
```

~2 s on a fresh clone. Prints PASS / FAIL / SKIP per stage. **`SKIP` is not
`PASS`** — the canonical bitstreams are gitignored, so on a machine that has
not regenerated them those stages check nothing.

CI (`.github/workflows/verify.yml`) runs the same command on every PR, passing
`--allow-skip` with an explicit list of the stages whose inputs are genuinely
absent. Any *other* SKIP fails the build. So if you add a stage, add it to that
list or make it runnable — CI will not accept a stage that quietly checks
nothing.

## Where state lives — exactly two files

| Question | File | Never elsewhere |
|---|---|---|
| What does the repo **know**? | `docs/CLAIM_LEDGER.md` | Do not restate claim levels in READMEs or logs. |
| What is **in flight / next**? | `docs/STATUS.md` | Do not add a "current state" section to any other file. |

Everything else — `AGENTS.md`, `docs/WORKFLOW.md`, `docs/theory/README.md` — is
**stable reference**, not status. If those two files disagree with anything
else in the repo, they win. `tools/lint_ledger.py` fails the build if a second
file starts tracking current state.

## Three rules that have each cost this repo months

1. **Run the counting bound before any "we searched class `M`, found no fit"
   experiment.** `python experiments/counting_bound.py --pretty`. If
   `log2|M| < n` the negative is guaranteed and the run is worthless. A
   certificate was retracted in 2026-08 for exactly this.
2. **`bitorder='little'` for every `data/center_col_*.bin`.** A bare
   `np.unpackbits` reverses each 8-bit block — 49.95% of positions differ while
   the bit mean is *identical*, so no aggregate check catches it.
   `tools/lint_bitorder.py` rejects bare calls.
3. **Single seed only.** All three prizes concern the one deterministic
   single-black-cell initial condition. An ensemble or random-IC quantity is
   not prize progress. See `docs/theory/README.md` §0.

A ~50% bit difference between two streams is **never** a kernel bug — it means
a packing or seed mismatch. Real kernel bugs diverge *late*.

## Deeper reference, in the order worth reading

- `docs/STATUS.md` — what is in flight right now. **Start here.**
- `docs/theory/README.md` — the theory gate: what is already proved (do not
  re-measure it) and which routes are closed.
- `AGENTS.md` — naming, logging standard, implementation guardrails.
- `docs/WORKFLOW.md` — the operating loop.
- `docs/AGENT_QUICKSTART.md` — tool map and prize-facing triage.

## Delegation — who does what

You are the lead researcher. You decide what is worth doing, judge whether a
result is real, and own every grade in `docs/CLAIM_LEDGER.md`. That does not
mean you type everything.

**Grunt work goes to an outside model, not to a Claude subagent.** Coding,
debugging, writing tests, refactoring, tracing something through the tree,
implementing an experiment that does not exist yet — delegate it and review
what comes back. Each task runs on an isolated branch in its own `git
worktree`, the harness re-runs every test the worker claims to have passed,
and the output is a reviewable branch plus a structured report. Nothing
merges.

```bash
python tools/codex_worker.py modes            # implement debug test refactor
                                              # investigate review
python tools/codex_worker.py submit --mode debug \
    --task "..." --context path/to/file.py --acceptance "python -m unittest ..."

python tools/worker_pool.py run --concurrency 10   # a queue of them at once
python tools/providers.py check                    # who can answer right now
```

One task: `docs/CODEX_WORKER.md`. Many at once, and which provider answers
(OpenRouter, the Codex dispatcher): `docs/WORKER_POOL.md`.

An outside model is not a Claude subagent, and that is the point. **Subagents
give throughput, not independence** — they share a model lineage with whoever
spawned them, and therefore share its blind spots. Never cite agreement among
subagents as corroboration. **Nor between workers in one pool**: ten calls to
one model is one prior sampled ten times. A fan-out buys throughput; only a
*different* provider buys disagreement, which is why `--mode review` and
`tools/council.py` exist. And delegating a build and then citing the worker's
own "it works" is the same mistake in a third costume: read the `verification`
field, which the harness produced, not the `tests` field, which the worker
wrote.

| The work is… | Route |
|---|---|
| an experiment that already has a script | `tools/workhorse.py --agent script` — **not an agent** |
| code that needs writing, fixing, or testing | `codex_worker.py` |
| ten independent chores | `worker_pool.py run --concurrency 10` |
| "does the repo do X, and where" | `codex_worker.py --mode investigate` |
| a claim that needs an outside opinion | `codex_worker.py --mode review`, or `tools/council.py` |
| deciding what any of it means | you |

Sending a deterministic command through an LLM buys nothing and adds a failure
point. If the script exists, run the script.

### The three gate subagents

Spawning subagents is **authorised** — you do not need to ask first. Three are
defined in `.claude/agents/`, and each enforces a gate this repo mandates in
prose but would otherwise never enforce mechanically. They are gates, not
workers; delegate work to Codex and use these to check it.

| Agent | Gate it enforces | Run it before |
|---|---|---|
| `counting-bound` | Rule 1 above — `log2\|M\| >= n`, or the negative is vacuous | any searched-and-found-no-fit experiment |
| `theory-gate` | `AGENTS.md` "Before Proposing Experiments" | proposing any experiment |
| `verifier` | "SKIP is not PASS" | an experiment, and a commit |

Each starts cold and re-derives context, so use one where the isolation is
worth that cost. Do not fan out across subagents for work that is faster done
directly, and do not use one where `codex_worker.py` would give you an
outside-lineage answer for the same effort.

## Branches

Short version: **branch from `main`, one PR deep, delete after merge.**
Full policy, including when stacking is allowed: `docs/BRANCHING.md`.
