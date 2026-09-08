#!/usr/bin/env python
"""Fan a queue of delegated tasks out across N concurrent workers.

`tools/codex_worker.py` runs one task. This runs a queue of them at once,
each on its own branch in its own `git worktree`, and hands back one table the
lead agent can read in a glance. Nothing merges; the output is N branches and
a summary.

    queue/tasks/*.json          task specs, reviewed and committed like code
        │
        ▼
    worker_pool run --concurrency 10
        │
        ├── worktree ──► provider ──► patch ──► verify ──► codex/<task-1>
        ├── worktree ──► provider ──► patch ──► verify ──► codex/<task-2>
        │   … up to --concurrency at a time …
        └── worktree ──► provider ──► patch ──► verify ──► codex/<task-N>
        │
        ▼
    runs/pool/<stamp>/summary.json   + a table on stderr

## What concurrency here does and does not buy

It buys **throughput**. Ten cheap workers clear ten independent chores in the
time one would take, and the chores this repo has are genuinely independent:
a lint fix, a missing test, a docstring, an investigation.

It does **not** buy independence, and the distinction is the one CLAUDE.md
keeps making about Claude subagents. Ten calls to one model share that model's
blind spots exactly as ten Claude subagents share Claude's. A fan-out is not a
panel, and agreement between two workers in the same pool is not corroboration
-- it is the same prior, sampled twice. If what you want is disagreement, send
the claim to a *different* provider (`--provider codex-dispatcher`, or
`tools/council.py`), and read `verification` rather than either model's
account of itself.

## Independence between tasks is structural

Each task gets its own worktree, so two workers editing the same file cannot
collide: they are editing different checkouts of it. What they *do* share is
one `.git`, whose index and worktree registry are not concurrency-safe, so
every operation against the main repository takes a lock and everything inside
a worktree runs free. That is the whole reason the fan-out is worth having --
the slow parts (the model call, verify_all) are the unlocked ones.

Two tasks that edit the same file will both succeed here and conflict at
integration time. That is correct and is the lead's problem to resolve, not
something to serialise the pool over: the alternative is a pool that refuses
work on a guess about overlap.

## Usage

    python tools/worker_pool.py list
    python tools/worker_pool.py run --concurrency 4 --dry-run
    python tools/worker_pool.py run --concurrency 10 --provider openrouter
    python tools/worker_pool.py run queue/tasks/fix-lint.json --verify fast

With no paths, `run` takes the whole of `queue/tasks/`. A task spec is what
codex_worker.submit takes; see its docstring for the schema.

Budget is a real consideration with ten workers and a metered provider, so
`--max-tasks` caps the run and `--dry-run` prints exactly what would be sent
without sending it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_QUEUE = REPO_ROOT / "queue" / "tasks"
RUNS = REPO_ROOT / "runs" / "pool"

DEFAULT_CONCURRENCY = 4
# Above this, the bottleneck stops being the provider and starts being this
# machine: every task runs verify_all in its own worktree, which is a full
# test suite. Ten is the number the design was asked for and it is fine; the
# cap exists so a typo in --concurrency does not fork a hundred test suites.
MAX_CONCURRENCY = 16


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tasks(paths: list[Path]) -> list[tuple[Path, dict]]:
    """Read task specs, in a stable order.

    Sorted by path so two runs of the same queue submit in the same order:
    with a metered provider, a nondeterministic order makes two runs
    impossible to compare on cost.
    """
    out = []
    for p in sorted(paths):
        out.append((p, json.loads(p.read_text(encoding="utf-8"))))
    return out


def queue_paths(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    return sorted(TASK_QUEUE.glob("*.json")) if TASK_QUEUE.is_dir() else []


# Keys agent_loop.Budget actually accepts. A spec naming anything else is a
# budget the run will silently ignore -- which on a metered fan-out means a
# cap the author believes is in force and is not.
LIMIT_KEYS = {"max_turns", "max_tool_calls", "max_cost_usd", "wall_clock_s"}


def validate(spec: dict, worker) -> list[str]:
    """Cheap checks before any worktree or token is spent.

    Everything here is checked against the modules that will consume it, not
    against a copy of their rules: a mistyped tool name or budget key is worth
    catching now rather than fifteen worktrees later, and both are the kind of
    typo that otherwise fails silently -- an unknown budget key is dropped, and
    an unknown tool name used to select nothing.
    """
    problems = []
    if spec.get("mode") not in worker.MODES:
        problems.append(f"mode must be one of {sorted(worker.MODES)}, "
                        f"got {spec.get('mode')!r}")
    if not str(spec.get("task", "")).strip():
        problems.append("task is empty")

    tools = spec.get("tools")
    if tools is not None:
        try:
            _load("agent_loop", "tools/agent_loop.py").tool_schemas(tools)
        except (ValueError, TypeError) as exc:
            problems.append(f"tools: {exc}")

    limits = spec.get("limits")
    if limits is not None:
        if not isinstance(limits, dict):
            problems.append("limits must be an object")
        else:
            unknown = sorted(set(limits) - LIMIT_KEYS)
            if unknown:
                problems.append(
                    f"limits: unknown key(s) {', '.join(unknown)}; "
                    f"expected {', '.join(sorted(LIMIT_KEYS))}")
            for k, v in limits.items():
                if k in LIMIT_KEYS and (isinstance(v, bool)
                                        or not isinstance(v, (int, float))
                                        or v <= 0):
                    problems.append(f"limits.{k} must be a positive number, "
                                    f"got {v!r}")
    return problems


def run_pool(specs: list[tuple[Path, dict]], *, concurrency: int,
             provider: str, model: str | None, backend: str,
             verify_level: str, base: str | None, out_root: Path,
             on_done=None) -> list[dict]:
    """Submit every task, at most `concurrency` at a time.

    One lock, shared by every worker and passed down into submit(), guards the
    operations that touch the main repository's `.git`. Everything else --
    the provider call, applying the patch, running verify_all in the worktree
    -- runs unserialised, which is where the time actually goes.
    """
    worker = _load("codex_worker", "tools/codex_worker.py")
    git_lock = threading.Lock()
    results: list[dict] = []
    results_lock = threading.Lock()

    def one(path: Path, spec: dict) -> dict:
        started = time.time()
        problems = validate(spec, worker)
        if problems:
            return {"task": path.name, "verdict": "REFUSED",
                    "reason": "; ".join(problems),
                    "duration_s": round(time.time() - started, 2)}
        try:
            r = worker.submit(spec, backend=backend, base=base,
                              verify_level=verify_level,
                              out_dir=out_root / path.stem,
                              provider=provider, model=model,
                              git_lock=git_lock)
        except Exception as exc:  # noqa: BLE001
            # One task's failure must not take the pool down. A provider
            # outage on task 3 should still leave tasks 1, 2 and 4 reviewable,
            # so the exception becomes this task's verdict and the run
            # continues. The class name is kept: "ProviderError" and
            # "RuntimeError" send the reader to different places.
            return {"task": path.name, "verdict": "ERROR",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "duration_s": round(time.time() - started, 2)}
        return {"task": path.name, "verdict": r["verdict"],
                # The id links this row to runs/<id>/ and to the branch's own
                # queue/results/<id>.json, so a summary row is traceable to
                # the report even after the run directory is gone.
                "task_id": r.get("task_id"),
                "branch": r.get("branch"), "patch": r.get("patch"),
                "model": r.get("model"), "provider": r.get("provider"),
                "files_changed": r.get("files_changed") or [],
                "verification_ok": (r.get("verification") or {}).get("ok"),
                "blockers": (r.get("report") or {}).get("blockers") or [],
                "uncertainties": (r.get("report") or {}).get("uncertainties") or [],
                "out": r.get("out"),
                "duration_s": round(time.time() - started, 2)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, p, s): p for p, s in specs}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            with results_lock:
                results.append(r)
            if on_done:
                on_done(r)
    # Sorted by task name, not by completion order, so the summary is
    # reproducible and diffable between runs.
    return sorted(results, key=lambda r: r["task"])


def table(results: list[dict]) -> str:
    width = max((len(r["task"]) for r in results), default=4)
    lines = []
    for r in results:
        mark = {"READY-FOR-REVIEW": "ready", "NEEDS-ATTENTION": "ATTN",
                "BLOCKED": "BLOCK", "REFUSED": "REFUS",
                "ERROR": "ERROR"}.get(r["verdict"], r["verdict"][:5])
        extra = r.get("branch") or r.get("reason", "")
        lines.append(f"  {mark:<5} {r['task']:<{width}}  "
                     f"{r['duration_s']:>6.1f}s  {extra}")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines.append("  " + ", ".join(f"{v} {k.lower()}"
                                  for k, v in sorted(counts.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="show the task queue")
    ls.add_argument("paths", nargs="*")

    rn = sub.add_parser("run", help="run the queue across N workers")
    rn.add_argument("paths", nargs="*",
                    help=f"task specs (default: all of {TASK_QUEUE.name}/)")
    rn.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    metavar="N", help=f"default {DEFAULT_CONCURRENCY}, "
                                      f"max {MAX_CONCURRENCY}")
    rn.add_argument("--provider", default="auto",
                    help="openrouter, codex-dispatcher, or auto")
    rn.add_argument("--model", help="override the provider's default model")
    rn.add_argument("--backend",
                    choices=("auto", "local", "remote", "agent"),
                    default="agent",
                    help="agent (default) = the tool-using loop "
                         "(tools/agent_loop.py), which reads, greps and runs "
                         "inside its own worktree -- the only backend that can "
                         "do research rather than answer from the prompt. "
                         "remote = one completion, no tools, and works with "
                         "the codex-dispatcher too. local = the codex CLI, one "
                         "process per task, which does not fan out; auto "
                         "prefers it, so auto is a poor pool default.")
    rn.add_argument("--verify", choices=("full", "fast", "none"),
                    default="full", dest="verify_level")
    rn.add_argument("--base", help="branch tasks from this instead of origin/main")
    rn.add_argument("--max-tasks", type=int, metavar="N",
                    help="cap the run; with a metered provider this is the "
                         "difference between a test and a bill")
    rn.add_argument("--out", help="artifact root (default runs/pool/<stamp>)")
    rn.add_argument("--dry-run", action="store_true",
                    help="print what would be submitted and exit, sending "
                         "nothing and creating no worktrees")

    args = ap.parse_args(argv)
    paths = queue_paths(args.paths)
    if not paths:
        print(f"worker_pool: no task specs. Put them in {TASK_QUEUE.relative_to(REPO_ROOT)}/ "
              "or name them on the command line.", file=sys.stderr)
        return 2

    try:
        specs = load_tasks(paths)
    except (OSError, ValueError) as exc:
        print(f"worker_pool: {exc}", file=sys.stderr)
        return 2

    worker = _load("codex_worker", "tools/codex_worker.py")

    if args.cmd == "list":
        rows = [{"path": str(p), "mode": s.get("mode"),
                 "task": str(s.get("task", ""))[:70],
                 "problems": validate(s, worker)} for p, s in specs]
        print(json.dumps(rows, indent=2))
        return 0

    if args.max_tasks:
        specs = specs[:args.max_tasks]
    concurrency = max(1, min(args.concurrency, MAX_CONCURRENCY))

    if args.dry_run:
        print(json.dumps({
            "would_run": [str(p) for p, _ in specs],
            "concurrency": concurrency, "provider": args.provider,
            "model": args.model, "backend": args.backend,
            "verify": args.verify_level,
        }, indent=2))
        return 0

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out) if args.out else RUNS / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"worker_pool: {len(specs)} task(s), {concurrency} at a time, "
          f"provider={args.provider}", file=sys.stderr)
    started = time.time()
    results = run_pool(
        specs, concurrency=concurrency, provider=args.provider,
        model=args.model, backend=args.backend,
        verify_level=args.verify_level, base=args.base, out_root=out_root,
        # Printed as each finishes rather than only at the end: a pool of ten
        # full verifications takes minutes, and a run that prints nothing
        # until it is over is one nobody can tell from a hang.
        on_done=lambda r: print(f"  · {r['verdict']:<16} {r['task']}",
                                file=sys.stderr, flush=True))

    summary = {"started": stamp, "duration_s": round(time.time() - started, 2),
               "concurrency": concurrency, "provider": args.provider,
               "model": args.model, "verify": args.verify_level,
               "results": results}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\n" + table(results), file=sys.stderr)
    print(f"  artifacts: {out_root}", file=sys.stderr)
    print("  Nothing was merged. Review each branch before integrating.",
          file=sys.stderr)
    return 0 if all(r["verdict"] == "READY-FOR-REVIEW" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
