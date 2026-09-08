# The workhorse — gates as code, and a runner that obeys them

The council (`tools/council.py`) reviews claims. This runs experiments. The
difference matters more than it sounds, and the design follows from it.

A reviewer's output is advice: if it is wrong, a human reads it and disagrees.
A runner's output is an artifact that enters the repo. So the rules in
`CLAUDE.md` cannot stay prose for a runner the way they can for a reviewer.
They are `tools/gates.py`, and `tools/workhorse.py` cannot execute anything
that has not passed them.

## Why this shape

Asked to red-team "promote Codex from reviewer to workhorse", the council
answered (2026-09-04):

> The specific failure is a misdiagnosis of the bottleneck: it proposes buying
> autonomous execution capacity when the repository's limiting factor is
> experimental validity and follow-through, not compute. […] Carrying
> guardrails as prompt text is not verification.

That is consistent with this repo's own history. Neither expensive failure was
a shortage of cycles:

- a certificate retracted in 2026-08 for a search class below the counting
  bound — the experiment ran when it should have been refused;
- experiments I–L invalidated by bit order, where 49.95% of positions differed
  while the bit mean stayed identical, so no aggregate check could see it.

Both would recur under autonomy, faster. Hence: gates first, autonomy second,
and hardware last — see `COMPUTE_PLAN.md` §1, *"The premise for renting was
wrong"*, and §2, *"Most of the best work costs nothing."*

## The pipeline

```
queue/<name>.json      a manifest, reviewed and committed like code
    │
    ▼
gates.py preflight     refuses what cannot produce information
    │                  counting bound · seed · theory gate · light cone
    ▼
execute                --agent script  runs manifest.script  (default)
    │                  --agent codex   asks Codex to implement and run
    ▼
gates.py postflight    rejects results that state more than they measured
    │                  censoring · noise floor · divergence · ~50% rule
    ▼
verify_all.py          the repo's own integrity check, after the run
    │
    ▼
branch → commit → PR   never main; CI gates it, a human merges
```

## Three deliberate choices

**Pull, not push.** The queue is a directory in this repo. There is no inbound
endpoint, so there is no token whose leak becomes code execution on the box
that runs experiments. The council endpoint stays `--sandbox read-only`; making
*it* writable would have turned a leaked bearer token into RCE on a public IP.

**The default agent is no agent.** `--agent script` runs the manifest's script
with its argv and nothing else. Most of the backlog in `STATUS.md` is an
existing script that has simply not been run — B1 is ~26 min of CPU, B2 is
"minutes". An LLM is for experiments that do not exist yet; `--agent codex` is
opt-in.

**Refusals are recorded.** A refused manifest lands in `queue/refused/` with
the full gate report. Not in `docs/experiment-logs/`: `lint_ledger`'s
`STALE-STATUS` check requires `STATUS.md` to cite the newest dated log there,
so an automated writer to that directory would red-line every concurrent PR
(`BRANCHING.md` §7).

## The gates

Each delegates to the tool that already owns the rule, rather than
reimplementing it — a gate that re-derived the arithmetic could drift from the
tool it claims to enforce.

| Gate | Rule | Authority |
|---|---|---|
| `counting-bound` | `log2\|M\| >= n`, or a negative is vacuous | `experiments/counting_bound.verdict` |
| `seed` | single black cell only | CLAUDE.md rule 3 |
| `theory-gate` | must be `OPEN` | AGENTS.md, declared in the manifest |
| `light-cone` | tape long enough for the step count | `gpu/tape_geometry.check` |
| `bitorder-lint` | no bare `packbits`/`unpackbits` | `tools/lint_bitorder.py` |
| `golden-self-test` | naive == packed, OEIS prefix | `gen_golden_reference --self-test` |
| `censoring` | no unqualified "never" | AGENTS.md |
| `noise-floor` | every metric carries a baseline | AGENTS.md |
| `divergence-invariant` | `first_divergence >= distance` | AGENTS.md |
| `fifty-percent` | ~50% differing is a packing/seed mismatch | CLAUDE.md |

Equality passes the counting bound. `counting_bound.py` marks `margin >= 0`
informative and Experiment S sits at that boundary; a strict `>` would reject
boundary-case evidence for a mathematically false reason.

## The trap, and why `verify_all` runs it

`queue/trap-vacuous-dfao.json` is a 24-state DFAO class tested against 10,000
bits — `log2|M| = 244.078`, the shape of the 2026-08 retraction. The
`gates-trap` stage in `verify_all` asserts that preflight still **refuses** it:

```bash
python tools/gates.py preflight queue/trap-vacuous-dfao.json --expect-fail
```

A gate that silently stops gating is worse than no gate, because the runner
would then report PASS on exactly the run that must not happen. Verified by
negative control: weakening the trap to `prefix_bits: 200` turns the build red.

## Usage

```bash
python tools/workhorse.py list --pretty                       # queue + verdicts
python tools/workhorse.py run queue/b1-pattern-map-walk.json --dry-run
python tools/workhorse.py run queue/b1-pattern-map-walk.json  # branch, commit, PR
```

`--dry-run` gates and executes into a temp directory, touching no branch.
It implies `--no-external`, and that is load-bearing rather than an
optimisation: preflight's external gates shell out to `verify_all`, whose
`unittest` stage runs `tests/test_workhorse.py`, which invokes the runner —
the external path recurses without bound. It hung the suite once.
`verify_all.py` documents the same hazard for its own `skip_permitted` tests.

## Writing a manifest

Schema is documented at the top of `tools/gates.py`. The minimum:

```json
{
  "name": "b1-pattern-map-walk-32",
  "kind": "measurement",
  "seed": "single-black-cell",
  "theory_gate": "OPEN",
  "script": "experiments/pattern_map_walk.py",
  "argv": ["--max-d", "12000000000"],
  "outputs": ["data/wedge/pattern_map_walk32.json"],
  "budget": {"minutes": 30, "device": "cpu"}
}
```

`outputs` names files the run produces outside `runs/`. They are copied into
`runs/<name>/outputs/` and hashed, so the PR carries the result it reports.
Anything a run leaves untracked fails the run and is named on stderr: an
untracked file blocks the *next* run, which refuses to start on a dirty tree,
and the remedy (a `.gitignore` `!` exception plus `make_manifest`) is a
decision for a person rather than something to `add -f` past.

`kind: "search"` with `"claims": ["negative"]` additionally requires `search`
with either DFAO parameters or a declared `log2_size` — a guess there defeats
the gate, so a non-DFAO class without a defensible bound is refused rather than
run.

## What this does not yet do

- **No scheduler.** One manifest per invocation, by hand or by cron. A queue
  daemon is not warranted until the queue is longer than the backlog.
- **`--agent codex` is unexercised.** The flag is written and its failure path
  is tested, but no experiment has been implemented by Codex through it. The
  pilot the council proposed — run B1 and B2 with traps planted, and check
  that the validation layer catches every invalid result — is the thing that
  would justify going further.
- **No rented hardware, and none justified yet.** `COMPUTE_PLAN.md` §5 puts
  the whole of §3 under about $20, and the GPU simulator still does not
  checkpoint, which makes spot instances actively wrong: preemption discards
  the run.
