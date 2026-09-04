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

## Subagents

Spawning subagents is **authorised** in this repo — you do not need to ask
first. Three are defined in `.claude/agents/`, each enforcing a gate this repo
already mandates in prose but has never enforced mechanically:

| Agent | Gate it enforces | Run it before |
|---|---|---|
| `counting-bound` | Rule 1 above — `log2\|M\| >= n`, or the negative is vacuous | any searched-and-found-no-fit experiment |
| `theory-gate` | `AGENTS.md` "Before Proposing Experiments" | proposing any experiment |
| `verifier` | "SKIP is not PASS" | an experiment, and a commit |

Each starts cold and re-derives context, so use one where the isolation is
worth that cost — an independent check, or a search wide enough that the
findings matter more than the transcript. Do not fan out across subagents for
work that is faster done directly.

**Subagents give throughput, not independence.** They share a model lineage
with whoever spawned them, and therefore share its blind spots. They are not a
substitute for `tools/council.py`, whose whole purpose is to put a differently
trained model on the same claim. Never cite agreement among subagents as
corroboration.

## Branches

Short version: **branch from `main`, one PR deep, delete after merge.**
Full policy, including when stacking is allowed: `docs/BRANCHING.md`.
