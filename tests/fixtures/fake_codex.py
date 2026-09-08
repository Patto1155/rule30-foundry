#!/usr/bin/env python
"""A stand-in for the `codex` CLI, so the worker pipeline can be run for real.

`tools/codex_worker.py` invokes the local backend as

    codex exec --skip-git-repo-check --sandbox workspace-write -C <dir> <prompt>

and reads a JSON report out of its stdout. This program accepts exactly that
argv, makes real edits on the real filesystem in <dir>, and prints a real
report. Point CODEX_BIN at it and every other part of the pipeline -- the
worktree, `git apply`/`git diff`, the commit, the verification subprocesses,
the verdict -- is the production code path, running for real.

That matters because the alternative is mocking `subprocess.run`, which tests
the mock. The one thing not exercised here is the model itself, and a test
could not exercise that deterministically anyway: what a test can pin down is
that a real edit made by a real subprocess reaches a reviewable branch, and
that a *dishonest* report is caught by verification rather than believed.
Hence the directives below, which exist to produce bad workers on demand.

Directives are read from the task text in the prompt, one per line:

    FAKE-WRITE <relpath> :: <text>   write <text> to <relpath> under -C
    FAKE-APPEND <relpath> :: <text>  append instead
    FAKE-DELETE <relpath>            delete a file
    FAKE-OMIT <key>                  leave <key> out of the JSON report
    FAKE-CLAIM-PASS                  report a test suite as passed, run nothing
    FAKE-BLOCKER <text>              report a blocker
    FAKE-NO-JSON                     print prose only, no report at all
    FAKE-EXIT <n>                    exit with status <n>
    FAKE-SLEEP <seconds>             take that long, so a pool test can show
                                     that tasks actually overlap
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def directives(prompt: str) -> list[tuple[str, str]]:
    out = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("FAKE-"):
            head, _, rest = line.partition(" ")
            out.append((head, rest.strip()))
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] != "exec":
        print("fake_codex: expected `exec`", file=sys.stderr)
        return 64
    workdir = Path.cwd()
    if "-C" in argv:
        workdir = Path(argv[argv.index("-C") + 1])
    prompt = argv[-1]

    report = {
        "summary": "fake_codex ran and did what the task's directives said.",
        "changes": [], "commands_run": [], "tests": [],
        "artifacts": [], "uncertainties": [], "blockers": [],
    }
    no_json = False
    exit_code = 0

    for name, rest in directives(prompt):
        if name in ("FAKE-WRITE", "FAKE-APPEND"):
            rel, _, text = rest.partition("::")
            target = workdir / rel.strip()
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            body = text.strip() + "\n"
            if name == "FAKE-APPEND" and existed:
                target.write_text(target.read_text(encoding="utf-8") + body,
                                  encoding="utf-8")
            else:
                target.write_text(body, encoding="utf-8")
            action = "modified" if existed else "added"
            report["changes"].append({"path": rel.strip(), "action": action,
                                      "why": "directed by the task"})
        elif name == "FAKE-DELETE":
            (workdir / rest).unlink(missing_ok=True)
            report["changes"].append({"path": rest, "action": "deleted",
                                      "why": "directed by the task"})
        elif name == "FAKE-OMIT":
            report.pop(rest, None)
        elif name == "FAKE-CLAIM-PASS":
            # Deliberately dishonest: says a suite passed without running it.
            report["tests"].append({"cmd": "python -m unittest", "exit_code": 0,
                                    "passed": True})
            report["commands_run"].append({"cmd": "python -m unittest",
                                           "exit_code": 0, "note": "claimed"})
        elif name == "FAKE-BLOCKER":
            report["blockers"].append(rest)
        elif name == "FAKE-NO-JSON":
            no_json = True
        elif name == "FAKE-SLEEP":
            time.sleep(float(rest))
        elif name == "FAKE-EXIT":
            exit_code = int(rest)

    print("fake_codex: working in", workdir)
    if no_json:
        print("I did some things but I am not going to tell you in JSON.")
    else:
        print("```json")
        print(json.dumps(report, indent=2))
        print("```")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
