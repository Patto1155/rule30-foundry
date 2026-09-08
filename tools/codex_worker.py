#!/usr/bin/env python
"""One delegated task: an outside model, an isolated worktree, and gates.

Claude is the lead researcher here: it decides what is worth doing, judges
whether a result is real, and owns every grade in `docs/CLAIM_LEDGER.md`.
This module is how it stops doing its own grunt work. Coding, debugging,
writing tests, refactoring, tracing a bug through the tree -- that is work
delegated to a model outside Claude's lineage, and what comes back is a
**reviewable branch plus a structured report**, never a merged change.

*Which* outside model is a configuration detail. `tools/providers.py`
supplies the text: a DeepSeek worker through OpenRouter, Codex on the
dispatcher VM, or `codex exec` locally with a writable checkout. Everything
here is the same either way -- the same isolation, contract, and gates.

Why not a Claude subagent? CLAUDE.md already says it: subagents give
throughput, not independence -- they share a model lineage with whoever
spawned them, and therefore share its blind spots. For grunt work that is
usually fine, but it means a Claude subagent's "I checked it" is worth
exactly what the parent's would be. An outside model is the only worker here
whose agreement is evidence -- and ten copies of one outside model is still
one model's blind spots, so a fan-out buys throughput, not a panel.

Why not the council client? `tools/council.py` asks a question and reads an
answer. It is read-only by design and it has no repository, so it cannot fix
anything. Review is one useful thing to ask an outside model for; it is not
the only one. Here it is one mode out of six.

What this is NOT for: running an experiment that already has a script.
`tools/workhorse.py --agent script` runs those directly, and that stays the
default. Sending a deterministic command through an LLM buys nothing and adds
a failure point. Delegate the work that needs judgement -- writing the
experiment that does not exist yet, finding why one is wrong -- and let the
gated runner execute what is already written.

    ┌──────────┐  task spec   ┌───────────────┐  isolated worktree
    │  Claude  │ ───────────► │ codex_worker  │ ─── git worktree add ──┐
    │  (lead)  │              │  (this file)  │                        │
    └────▲─────┘              └───────┬───────┘                        ▼
         │                            │                        codex/<branch>
         │  structured result         │ backend                        │
         │  + branch + patch          ├─ local  : codex exec --sandbox │
         │                            │           workspace-write -C   │
         │                            └─ remote : dispatcher /ask,     │
         │                                        reply carries a patch│
         │                                                             ▼
         │                       ┌──────────────────────────────────────┐
         └───────────────────────┤ verification the WORKER ran itself:  │
                                 │ verify_all · gates.postflight · the  │
                                 │ task's own acceptance commands       │
                                 └──────────────────────────────────────┘

Two properties are the whole point of the shape.

**Isolation is structural, not promised.** Every task runs on its own branch
in its own `git worktree`. The lead's checkout is never touched, a failed
task leaves nothing to clean up, and two tasks cannot interleave edits. A
worker that edited the working tree would make "review before integrating"
a matter of discipline; a worktree makes it a matter of fact.

**Claimed and verified are different fields.** `commands_run` and `tests` are
what Codex *says* it did. `verification` is what this process ran, in the
worktree, after the fact. They are kept apart on purpose: an agent reporting
its own green tests is the oldest way to get a wrong answer past a review,
and merging the two fields would erase exactly the distinction a lead needs.
Trust `verification`. Read the rest as testimony.

Task spec (JSON; `mode` and `task` are the only required fields):

    {
      "mode":          "implement" | "debug" | "test" | "refactor"
                       | "investigate" | "review",
      "task":          "what to do, in prose, with enough detail to act on",
      "context_files": ["tools/gates.py", "docs/theory/README.md"],
      "acceptance":    ["python -m unittest tests.test_gates"],
      "manifest":      {...},        -- an experiment manifest, if this task
                                        implements one: preflight runs on it
                                        BEFORE dispatch, postflight after
      "budget_minutes": 30
    }

Result contract, which every mode returns and `validate_result` enforces:
`summary`, `changes`, `commands_run`, `tests`, `artifacts`, `uncertainties`,
`blockers`. A worker that cannot say what it is unsure about has not been
asked properly; `uncertainties` and `blockers` are required keys, and an
empty list is an assertion rather than an omission.

Usage:
    python tools/codex_worker.py modes
    python tools/codex_worker.py check
    python tools/codex_worker.py submit --mode implement \
        --task "Add a --json flag to tools/lint_ledger.py" \
        --context tools/lint_ledger.py --acceptance "python tools/lint_ledger.py"
    python tools/codex_worker.py submit --spec task.json --pretty

Configuration (environment, never a flag -- a key in argv is a key in `ps`
output). Which provider answers the remote backend is `tools/providers.py`'s
business, not this file's:

    OPENROUTER_API_KEY    openrouter provider
    OPENROUTER_MODEL      its default model slug
    CODEX_COUNCIL_URL     codex-dispatcher provider
    CODEX_COUNCIL_TOKEN   its bearer token
    CODEX_BIN             codex executable for the local backend (default
                          `codex`). Substitutable so the end-to-end test
                          drives the real pipeline with a stand-in binary.

One task per invocation. To run many at once -- ten cheap workers over a
queue -- use `tools/worker_pool.py`, which calls submit() concurrently and
holds the git lock this module accepts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "runs" / "codex"

READY, ATTENTION, BLOCKED, REFUSED = (
    "READY-FOR-REVIEW", "NEEDS-ATTENTION", "BLOCKED", "REFUSED")

DEFAULT_BUDGET_MIN = 30

# Client-side cap on what remote mode ships as context. The dispatcher has its
# own, lower or equal; this one exists so an accidental
# `--context data/center_col_10M.bin` fails here rather than after a 40 MB
# upload through the proxy.
MAX_CONTEXT_BYTES = 60_000

# Set while the worker runs verification inside a worktree. verify_all's
# unittest stage discovers this module's own tests, which invoke the worker:
# without a guard the verification step re-enters itself and never returns.
# tests/test_codex_worker_e2e.py skips when it sees this. workhorse.py
# documents the same hazard for --dry-run, and verify_all.py for its
# skip_permitted tests -- it is a property of running the checker from inside
# the thing being checked, not a quirk of this file.
NESTED_ENV = "CODEX_WORKER_NESTED"

# Stages whose inputs are legitimately absent in a worktree: the canonical
# bitstreams are gitignored, so a worktree never has them. Naming them keeps
# verify_all in strict mode, where any OTHER skip fails -- "SKIP is not PASS"
# has to survive being run by a robot, or it only ever applied to humans.
WORKTREE_ALLOW_SKIP = ("bitstream:*", "drat-toolchain")


# --------------------------------------------------------------------------
# task modes
# --------------------------------------------------------------------------
#
# A mode is a prompt framing plus an expectation about the output, not a
# different model. What actually changes the answer is the preamble and
# whether a patch is required, so that is what a mode is. `review` is the
# council role kept as one mode among six rather than as the architecture.

class Mode:
    __slots__ = ("name", "preamble", "expects_patch")

    def __init__(self, name: str, preamble: str, expects_patch: bool):
        self.name, self.preamble, self.expects_patch = name, preamble, expects_patch


MODES = {m.name: m for m in [
    Mode("implement", """\
You are implementing a change in this repository. Write the code, run it, and \
leave the tree in a state a reviewer can read. Prefer the smallest change that \
does the job; match the conventions of the files you are editing rather than \
importing your own. Add or extend a test for anything you claim works.""", True),

    Mode("debug", """\
You are diagnosing a defect. Reproduce it first and say exactly how -- a fix \
for a bug you never reproduced is a guess. Then find the root cause, state it \
in one sentence, and fix that rather than the symptom. If the root cause is \
outside the scope you were given, fix nothing, and report it as a blocker with \
the evidence that identifies it.""", True),

    Mode("test", """\
You are writing tests. Each test must fail if the behaviour it describes \
breaks -- write it, then break the code deliberately and confirm the test goes \
red, then restore. A test that passes against a broken implementation is worse \
than no test, because it is cited as evidence. State in the summary which \
tests you confirmed this way.""", True),

    Mode("refactor", """\
You are restructuring code without changing its behaviour. No observable \
behaviour may change: same outputs, same exit codes, same file formats. Run \
the existing tests before and after and report both. If you find a bug while \
refactoring, do not fix it here -- report it under `uncertainties` and leave \
the behaviour as it is, so the diff stays reviewable.""", True),

    Mode("investigate", """\
You are answering a question about this repository by reading and running it, \
not by guessing. Cite file paths and line numbers for every claim about the \
code, and paste the command output for every claim about behaviour. You may \
change files only if the task explicitly asks for it; ordinarily return an \
empty `changes` list and put the findings in `summary`. Say plainly what you \
could not determine.""", False),

    Mode("review", """\
You are an independent reviewer. You did not write this work and you have no \
stake in its conclusions. Assess whether the claim is supported by the \
evidence offered, at the strength claimed. Distinguish clearly between proved, \
empirically supported on the stated range, and asserted. Where the grade is \
overstated, say what the honest grade would be. Change nothing: return an \
empty `changes` list.""", False),
]}


RESULT_INSTRUCTIONS = """\
Return your report as a single JSON object, and nothing else after it, with
exactly these keys:

  "summary"       one paragraph: what you did and what a reviewer should look
                  at first.
  "changes"       [{"path": "...", "action": "modified"|"added"|"deleted",
                    "why": "..."}] -- one entry per file you touched. Empty
                  list if you changed nothing.
  "commands_run"  [{"cmd": "...", "exit_code": 0, "note": "..."}] -- every
                  command you actually ran. Do not list commands you intended
                  to run.
  "tests"         [{"cmd": "...", "exit_code": 0, "passed": true}] -- the
                  subset of commands_run that were tests.
  "artifacts"     ["path", ...] -- files you produced that are not source
                  changes (logs, data, plots). Empty list if none.
  "uncertainties" ["...", ...] -- what you are not sure about, including
                  anything you assumed because the task did not say. An empty
                  list asserts there were none; do not use it as a default.
  "blockers"      ["...", ...] -- what stopped you finishing. Empty list if
                  you finished. A blocker is something that prevented the
                  task from being done. Not having a checkout, not being able
                  to run commands, and not having been shown a file you did
                  not need are NOT blockers -- they are how this mode works,
                  the harness already knows, and reporting them holds up a
                  finished change. If those are your only reservations, put
                  them in `uncertainties` and leave `blockers` empty.

Report what happened, not what should have happened. A partial result with an
honest blocker is more useful here than a complete-looking one that is wrong:
the lead reads this to decide what to trust, and re-runs every test you claim
to have passed.

The report is not optional and it is not the part to drop when you are nearly
done. Put it in a ```json fenced block. Finish your reply with it."""


REQUIRED_RESULT_KEYS = ("summary", "changes", "commands_run", "tests",
                        "artifacts", "uncertainties", "blockers")
LIST_RESULT_KEYS = REQUIRED_RESULT_KEYS[1:]


def _council():
    spec = importlib.util.spec_from_file_location(
        "council", REPO_ROOT / "tools" / "council.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo_guardrails() -> str:
    """Reuse the council's failure-mode list rather than keeping a second copy.

    Under a lead-in that says "doing" rather than "reviewing": the list is the
    same either way, but telling a worker it is a reviewer is how you get a
    review back instead of a patch.
    """
    return _council().guardrails("doing")


def _gates():
    spec = importlib.util.spec_from_file_location(
        "gates", REPO_ROOT / "tools" / "gates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _agent_loop():
    spec = importlib.util.spec_from_file_location(
        "agent_loop", REPO_ROOT / "tools" / "agent_loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _providers():
    spec = importlib.util.spec_from_file_location(
        "providers", REPO_ROOT / "tools" / "providers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# prompt assembly -- pure, so the wording is tested without a VM
# --------------------------------------------------------------------------

def gather_context(paths: list[str], root: Path = REPO_ROOT,
                   cap: int = MAX_CONTEXT_BYTES) -> tuple[str, list[str]]:
    """Read the named files for a backend that has no repository of its own.

    Returns (text, problems). A missing file is a problem, not an exception:
    the lead asked for it by name, so silence would be the wrong answer, but
    the remaining files are still worth sending.
    """
    chunks, problems, used = [], [], 0
    for rel in paths:
        p = (root / rel).resolve()
        try:
            p.relative_to(root.resolve())
        except ValueError:
            problems.append(f"{rel}: outside the repository; not sent")
            continue
        if not p.is_file():
            problems.append(f"{rel}: not a file")
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            problems.append(f"{rel}: unreadable ({exc.__class__.__name__})")
            continue
        if used + len(body.encode("utf-8")) > cap:
            problems.append(f"{rel}: omitted, context cap of {cap} bytes reached")
            continue
        used += len(body.encode("utf-8"))
        chunks.append(f"----- {rel} -----\n{body}")
    return "\n\n".join(chunks), problems


def render_prompt(spec: dict, context: str = "", *, has_repo: bool) -> str:
    """Assemble the full instruction sent to Codex.

    `has_repo` is the one real difference between the backends. With a
    checkout, Codex is told to work in it and to run things; without one, it
    is told to answer with a unified diff, because a patch is the only way an
    edit can travel over a read-only endpoint.
    """
    mode = MODES[spec["mode"]]
    parts = [
        f"You are working as a delegated worker for the rule30-foundry "
        f"repository, in `{mode.name}` mode.\n\n{mode.preamble}",
        _repo_guardrails(),
    ]
    if has_repo:
        parts.append(
            "You have a writable checkout of the repository at your working "
            "directory, on a scratch branch of its own. Read CLAUDE.md and "
            "AGENTS.md before you start. Work only in this checkout: it is a "
            "git worktree, and nothing you do here reaches anyone's working "
            "tree until a human reviews the branch. Do not run `git commit`, "
            "`git push`, or any command that rewrites history -- the harness "
            "commits your work for you, and a commit of your own splits the "
            "diff a reviewer has to read.")
    else:
        parts.append(
            "You do NOT have a checkout. The files you need are quoted below. "
            "If the task requires changing code, put a unified diff -- `git "
            "diff` format, paths relative to the repository root with a/ and "
            "b/ prefixes -- in a ```diff fenced block, and describe each file "
            "in `changes`. Do not worry about the `@@` line counts being "
            "exact; the harness recounts them. You cannot run anything, so "
            "`commands_run` and `tests` must be empty lists: the harness runs "
            "the tests and does not want your prediction of the result.\n\n"
            "Your reply must contain BOTH the diff and the JSON report, in "
            "that order. A reply with a diff and no report is incomplete: the "
            "code may be perfect and it is still rejected, because the lead "
            "has no way to read what you were unsure about. Write the diff, "
            "then write the report.")
    if spec.get("acceptance"):
        parts.append(
            "The harness will run these acceptance commands against your work "
            "regardless of what you report:\n"
            + "\n".join(f"  $ {c}" for c in spec["acceptance"]))
    if spec.get("manifest"):
        parts.append(
            "This task implements the experiment manifest below. It has "
            "already passed preflight; its result will be run through "
            "postflight (tools/gates.py), which rejects a conclusion that "
            "states more than the run measured.\n\nMANIFEST\n"
            + json.dumps(spec["manifest"], indent=2))
    parts.append(f"--- task ---\n\n{spec['task'].strip()}")
    if context:
        parts.append(f"--- files ---\n\n{context}")
    # Last, deliberately. The output contract is the instruction most often
    # dropped -- measured on 2026-09-08, two of three pool tasks returned a
    # correct, verified patch and no report at all, which scores a clean
    # change as NEEDS-ATTENTION. Instructions nearest the end of a long prompt
    # are the ones a model still has in view when it starts writing.
    parts.append(RESULT_INSTRUCTIONS)
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# parsing the reply
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", re.S)
_DIFF_FENCE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.S)


def extract_json(text: str) -> dict | None:
    """Find the report object in free-form model output.

    Tried in order: a fenced ```json block, then the last balanced object in
    the text. Returns None rather than guessing -- a caller that gets None
    records a blocker, which is the honest outcome, whereas a caller handed a
    half-parsed object reports success on a result nobody produced.
    """
    for m in reversed(_FENCE.findall(text)):
        try:
            obj = json.loads(m)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    depth, start = 0, None
    best = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                except ValueError:
                    pass
                else:
                    if isinstance(obj, dict) and "summary" in obj:
                        best = obj
    return best


def extract_patch(text: str) -> str | None:
    """Pull a unified diff out of the reply, for the backend with no checkout.

    The fenced form is preferred because it is unambiguous. Falling back to
    "the text from the first `diff --git` onwards" catches the common case of
    a model that emitted a correct patch and forgot the fence -- `git apply`
    is the real judge either way, and a patch that does not apply is recorded
    as a blocker rather than being partially applied.
    """
    fenced = _DIFF_FENCE.findall(text)
    if fenced:
        body = max(fenced, key=len).strip("\n")
        return body + "\n" if body else None
    idx = text.find("diff --git ")
    if idx == -1:
        return None
    return text[idx:].rstrip() + "\n"


def validate_result(obj: object) -> list[str]:
    """Check the structured report against the contract. Returns problems.

    An empty list means the shape is right; it says nothing about whether the
    content is true. That is what `verification` is for.
    """
    if not isinstance(obj, dict):
        return ["report is not a JSON object"]
    problems = []
    for key in REQUIRED_RESULT_KEYS:
        if key not in obj:
            problems.append(f"missing required key {key!r}")
    if not isinstance(obj.get("summary", ""), str) or not obj.get("summary", "").strip():
        problems.append("summary is empty")
    for key in LIST_RESULT_KEYS:
        if key in obj and not isinstance(obj[key], list):
            problems.append(f"{key!r} must be a list, got {type(obj[key]).__name__}")
    for i, c in enumerate(obj.get("changes") or []):
        if not isinstance(c, dict) or "path" not in c:
            problems.append(f"changes[{i}] has no 'path'")
    return problems


# --------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------

class _NullLock:
    """Stand-in for the pool's lock when a task runs alone. Cheaper to read
    than `if lock:` at four call sites."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _git(*args: str, cwd: Path = REPO_ROOT, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def branch_name(spec: dict, task_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", spec["task"].lower())[:40].strip("-")
    return f"codex/{spec['mode']}-{slug or 'task'}-{task_id}"


def add_worktree(branch: str, base: str, root: Path = REPO_ROOT) -> Path:
    """A branch and a checkout of its own, per task.

    Isolation the lead does not have to trust anyone for: the worker cannot
    reach the main checkout because it is not in it.
    """
    path = Path(tempfile.mkdtemp(prefix="codex-wt-"))
    path.rmdir()  # git wants to create it
    _git("worktree", "add", "-b", branch, str(path), base, cwd=root)
    return path


def remove_worktree(path: Path, root: Path = REPO_ROOT) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                   cwd=root, capture_output=True, text=True)


def resolve_base(base: str | None, root: Path = REPO_ROOT) -> str:
    """Which commit the task branches from.

    origin/main by default, and fetched first: branching from whatever is
    checked out stacks each task on the last one, which carries unrelated
    commits into the review (BRANCHING.md §1). A checkout with no reachable
    remote falls back to local main and says so through the returned name, so
    the result records what it actually branched from.
    """
    if base:
        return base
    fetched = subprocess.run(["git", "fetch", "origin", "main"], cwd=root,
                             capture_output=True, text=True)
    if fetched.returncode == 0:
        return "origin/main"
    for candidate in ("main", "HEAD"):
        r = subprocess.run(["git", "rev-parse", "--verify", candidate],
                           cwd=root, capture_output=True, text=True)
        if r.returncode == 0:
            return candidate
    return "HEAD"


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def codex_bin() -> str:
    return os.environ.get("CODEX_BIN", "codex").strip() or "codex"


def choose_backend(requested: str) -> str:
    """Which backend to use, checked rather than assumed.

    An explicit `--backend local` on a host with no codex must fail here, with
    a sentence saying so, rather than as a FileNotFoundError from deep inside
    the run after a worktree has already been created.
    """
    if requested == "local" and not shutil.which(codex_bin()):
        raise RuntimeError(
            f"--backend local, but {codex_bin()!r} is not on PATH. Run this "
            "where codex is installed and logged in, or use --backend agent "
            "(the tool-using loop, which needs only OPENROUTER_API_KEY).")
    if requested != "auto":
        return requested
    if shutil.which(codex_bin()):
        return "local"
    # agent before remote: both reach an outside model, and only one of them
    # can read a file it was not handed. A completion backend is the fallback
    # for when no tool-capable provider is configured, not the default.
    if _providers().OpenRouter().available()[0]:
        return "agent"
    return "remote"


def codex_argv(prompt: str, workdir: Path) -> list[str]:
    """The one place that decides how codex is invoked with a writable tree.

    tools/workhorse.py runs codex too, for an experiment manifest rather than
    a free task. It calls this rather than assembling its own command line:
    two argv builders is two sandbox decisions, and the one that drifts is
    always the one that stops passing --sandbox.

    A list, never a shell string. The prompt is an argv element, so nothing in
    it can become a command however it is worded.
    """
    return [codex_bin(), "exec", "--skip-git-repo-check", "--sandbox",
            "workspace-write", "-C", str(workdir), prompt]


def codex_exec(prompt: str, workdir: Path, timeout: int, out: Path
               ) -> tuple[int, str]:
    """Run codex once against a writable directory. Returns (exit_code, raw).

    Writes prompt.txt and raw.txt into `out`: a delegated run whose transcript
    was not kept cannot be reviewed afterwards, and "what did you actually ask
    it?" is the first question about any surprising result.
    """
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    p = subprocess.run(codex_argv(prompt, workdir), cwd=workdir,
                       capture_output=True, text=True, timeout=timeout)
    raw = (p.stdout or "") + (("\n--- stderr ---\n" + p.stderr) if p.stderr else "")
    (out / "raw.txt").write_text(raw, encoding="utf-8")
    return p.returncode, raw


def run_local(spec: dict, worktree: Path, timeout: int, out: Path) -> dict:
    """codex with a writable checkout: the full agentic loop.

    This is the mode that runs where codex is installed and logged in -- the
    dispatcher VM, or any host with the CLI. `-C worktree` and
    `--sandbox workspace-write` together mean the agent can edit and run
    anything inside this task's checkout and nothing outside it.
    """
    started = time.time()
    code, raw = codex_exec(render_prompt(spec, has_repo=True), worktree,
                           timeout, out)
    return {"backend": "local", "exit_code": code, "raw": raw,
            "duration_s": round(time.time() - started, 2),
            "report": extract_json(raw), "patch_applied": None}


def run_agent(spec: dict, worktree: Path, timeout: int, out: Path,
              model: str | None = None, limits: dict | None = None) -> dict:
    """The tool-using loop: the worker reads, greps, runs and fetches.

    Unlike the completion backends this one edits the worktree directly, so
    there is no patch to apply -- the diff comes from git afterwards, exactly
    as it does for `codex exec`. What it adds over `local` is that it needs no
    CLI installed anywhere: the loop is tools/agent_loop.py and the model is
    whatever OpenRouter serves.
    """
    al = _agent_loop()
    prov = _providers().OpenRouter(model=model)
    ok, why = prov.available()
    if not ok:
        raise RuntimeError(f"agent backend needs OpenRouter: {why}")

    lim = dict(limits or {})
    budget = al.Budget(
        max_turns=lim.get("max_turns", al.DEFAULT_MAX_TURNS),
        max_tool_calls=lim.get("max_tool_calls", al.DEFAULT_MAX_TOOL_CALLS),
        max_cost_usd=lim.get("max_cost_usd", al.DEFAULT_MAX_COST_USD),
        # The task's own budget is the wall clock unless something tighter was
        # asked for: a loop with no clock is the failure mode that costs money
        # while nobody is watching.
        wall_clock_s=lim.get("wall_clock_s", timeout))
    prompt = render_prompt(spec, has_repo=True)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")

    started = time.time()
    head_before = _git("rev-parse", "HEAD", cwd=worktree)
    r = al.run_loop(prompt, worktree, prov, budget=budget,
                    tools=spec.get("tools"),
                    transcript=out / "transcript.jsonl")
    (out / "raw.txt").write_text(r["final"], encoding="utf-8")

    # The worker has a shell, so the worktree is not beyond its reach. On the
    # first live research run a worker ran `git worktree remove` on its own
    # checkout to test a hypothesis and then re-created it -- honestly
    # reported, and invisible to everything downstream, because the files were
    # identical. Downstream is `git diff`, and a detached or re-created
    # worktree silently produces an empty one. Check rather than trust: a
    # missing checkout or a moved HEAD is a blocker on the result, not a
    # surprise in the review.
    disturbed = None
    if not (worktree / ".git").exists():
        disturbed = "the worker removed or replaced its own checkout"
    else:
        head_after = _git("rev-parse", "HEAD", cwd=worktree, check=False)
        if head_after and head_after != head_before:
            disturbed = (f"HEAD moved during the run, {head_before[:8]} -> "
                         f"{head_after[:8]}; the worker changed git state")

    # A loop that stopped at a budget did not finish, and that must not read
    # as a worker who chose to stop. It becomes a blocker on the result, which
    # is what keeps the verdict off READY-FOR-REVIEW.
    stopped_early = r["stop_reason"] != al.STOP_DONE
    return {"backend": "agent", "provider": "openrouter",
            "exit_code": 0 if not stopped_early else 1,
            "raw": r["final"], "duration_s": round(time.time() - started, 2),
            "report": extract_json(r["final"]), "patch_applied": None,
            "model": prov.model, "stop_reason": r["stop_reason"],
            "budget": r["budget"], "tool_calls": r["tool_calls"],
            "worktree_disturbed": disturbed,
            "usage": {"cost": r["budget"]["cost_usd"]}}


# `git apply` strategies, strictest first. A model writing a diff by hand from
# quoted file contents gets the hunk *content* right and the `@@` line counts
# wrong -- it has no line numbers to count from. That is what `--recount` is
# for, and observed on the first live remote run (2026-09-07): a correct patch
# rejected as "corrupt patch at line 31", which applied cleanly on a recount.
#
# Anything past the first strategy is surfaced in the result as
# `patch_apply_strategy`, not silently absorbed: a loosened apply can place a
# hunk somewhere the author did not mean, and the reviewer should know the
# patch needed help. Verification is what catches it landing wrong; saying so
# is what stops a reviewer assuming it landed right.
APPLY_STRATEGIES = (
    ("strict", ["--index", "--whitespace=nowarn"]),
    ("recount", ["--index", "--recount", "--whitespace=nowarn"]),
    ("recount+fuzz", ["--index", "--recount", "-C1", "--whitespace=nowarn"]),
)


def apply_patch(patch_path: Path, worktree: Path
                ) -> tuple[bool, str | None, list[str]]:
    """Try each strategy in turn. Returns (applied, strategy, errors)."""
    errors = []
    for name, flags in APPLY_STRATEGIES:
        r = subprocess.run(["git", "apply", *flags, str(patch_path)],
                           cwd=worktree, capture_output=True, text=True)
        if r.returncode == 0:
            return True, name, errors
        errors.append(f"[{name}] {r.stderr.strip()}")
    return False, None, errors


def run_remote(spec: dict, worktree: Path, timeout: int, out: Path,
               provider: str = "auto", model: str | None = None) -> dict:
    """A completion provider: no checkout, so edits travel as a patch.

    Whoever answers -- a DeepSeek worker through OpenRouter, Codex on the
    dispatcher VM -- cannot run anything, because a completion endpoint has no
    shell and no copy of this tree. That is a real limitation and it is
    recorded rather than papered over: `commands_run` comes back empty by
    instruction, and every test in the result is one this process ran itself,
    after applying the patch here.
    """
    prov = _providers().get(provider, model=model)
    ok, why = prov.available()
    if not ok:
        raise RuntimeError(f"provider {prov.name!r} is not usable: {why}")

    context, ctx_problems = gather_context(spec.get("context_files") or [])
    prompt = render_prompt(spec, context, has_repo=False)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")

    kw = {"role": spec["mode"]} if prov.name == "codex-dispatcher" else {}
    completion = prov.complete(prompt, timeout, **kw)
    answer = completion.text
    (out / "raw.txt").write_text(answer, encoding="utf-8")

    patch_applied, strategy = None, None
    patch = extract_patch(answer)
    if patch:
        (out / "changes.patch").write_text(patch, encoding="utf-8")
        patch_applied, strategy, errors = apply_patch(
            out / "changes.patch", worktree)
        if not patch_applied:
            (out / "patch-apply-error.txt").write_text(
                "\n\n".join(errors), encoding="utf-8")
    return {"backend": "remote", "provider": prov.name, "exit_code": 0,
            "raw": answer, "duration_s": round(completion.duration_s, 2),
            "report": extract_json(answer), "patch_applied": patch_applied,
            "patch_apply_strategy": strategy,
            "context_problems": ctx_problems,
            "model": completion.model, "usage": completion.usage}


# --------------------------------------------------------------------------
# verification -- what the worker ran, as distinct from what Codex claimed
# --------------------------------------------------------------------------

def verification_commands(spec: dict, level: str) -> list[list[str]]:
    py = sys.executable
    cmds: list[list[str]] = []
    if level == "full":
        cmds.append([py, "tools/verify_all.py",
                     *sum(([f"--allow-skip={g}"] for g in WORKTREE_ALLOW_SKIP), [])])
    elif level == "fast":
        cmds.append([py, "tools/lint_bitorder.py"])
        cmds.append([py, "tools/lint_ledger.py"])
    for c in spec.get("acceptance") or []:
        cmds.append(["/bin/sh", "-c", c])
    return cmds


def verify(spec: dict, worktree: Path, level: str, out: Path) -> dict:
    """Run the checks ourselves, in the worktree, and report them honestly.

    Nothing here reads the agent's report. That is deliberate: the value of
    this field is precisely that it was produced without consulting the party
    being checked.
    """
    if level == "none":
        return {"level": "none", "ok": None, "checks": [],
                "note": "verification disabled with --verify none; this result "
                        "carries no evidence beyond the agent's own testimony"}
    env = dict(os.environ, **{NESTED_ENV: "1"})
    # Verification checks the repository; it must not inherit this process's
    # delegation config. Leaving CODEX_BIN set leaks the caller's worker into
    # the verification subprocess -- which is how a nested tests/test_workhorse
    # case that asserts "codex is not on PATH" found a stand-in binary on it
    # and failed. A checker that can itself delegate is not a checker.
    env.pop("CODEX_BIN", None)
    checks = []
    log = []
    for argv in verification_commands(spec, level):
        started = time.time()
        p = subprocess.run(argv, cwd=worktree, capture_output=True, text=True,
                           env=env)
        checks.append({"cmd": " ".join(argv[-1:] if argv[0] == "/bin/sh" else argv),
                       "exit_code": p.returncode,
                       "passed": p.returncode == 0,
                       "duration_s": round(time.time() - started, 2)})
        log.append(f"$ {' '.join(argv)}\n{p.stdout}{p.stderr}\n")
    (out / "verification.txt").write_text("\n".join(log), encoding="utf-8")
    return {"level": level, "ok": all(c["passed"] for c in checks),
            "checks": checks, "log": "verification.txt"}


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def changed_files(worktree: Path) -> list[str]:
    """What actually changed, from git, not from the agent's list.

    The two disagree more often than one would like -- an agent that edited a
    file it forgot to mention, or mentioned one it never touched. git is the
    record; `changes` in the report is the explanation.
    """
    _git("add", "-A", cwd=worktree)
    out = _git("diff", "--cached", "--name-status", cwd=worktree)
    return [line for line in out.splitlines() if line.strip()]


def decide(report_problems: list[str], verification: dict, blockers: list,
           patch_applied: bool | None, exit_code: int | None,
           timed_out: bool = False) -> str:
    """One verdict, from facts the lead can re-derive from the fields.

    `exit_code` may be None -- a run killed by the budget never produced one.
    None is not zero here: treating a missing exit code as success is how a
    timed-out task reports READY-FOR-REVIEW.
    """
    if patch_applied is False:
        return BLOCKED
    if blockers:
        return BLOCKED
    if timed_out or exit_code != 0:
        return ATTENTION
    if report_problems:
        return ATTENTION
    if verification.get("ok") is False:
        return ATTENTION
    return READY


def submit(spec: dict, *, backend: str = "auto", base: str | None = None,
           verify_level: str = "full", out_dir: Path | None = None,
           keep_worktree: bool = False, timeout: int | None = None,
           provider: str = "auto", model: str | None = None,
           git_lock=None) -> dict:
    """Run one delegated task and return the structured result."""
    gates = _gates()
    if spec.get("mode") not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}, "
                         f"got {spec.get('mode')!r}")
    if not str(spec.get("task", "")).strip():
        raise ValueError("task is empty; a worker cannot infer the job")

    task_id = uuid.uuid4().hex[:8]
    out = Path(out_dir) if out_dir else RUNS / task_id
    out.mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result: dict = {
        "task_id": task_id, "mode": spec["mode"], "started": started_at,
        "out": str(out), "spec": spec,
    }

    # Gate before dispatch. An experiment that cannot produce information is
    # refused here, before a worktree exists and before a token is spent --
    # the 2026-08 retraction was an experiment that ran when it should have
    # been refused, and the delegation path must not be the way around that.
    if spec.get("manifest"):
        pre = gates.preflight(spec["manifest"], run_external=False)
        result["preflight"] = pre
        if pre["verdict"] == gates.FAIL:
            result.update(verdict=REFUSED, branch=None,
                          summary="refused by preflight; nothing was dispatched")
            (out / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result

    chosen = choose_backend(backend)
    result["backend"] = chosen
    seconds = (timeout or int(spec.get("budget_minutes") or DEFAULT_BUDGET_MIN) * 60)
    branch = branch_name(spec, task_id)
    # Resolved once: resolve_base fetches, and calling it twice would fetch
    # twice and could straddle a push, branching from one commit while the
    # result recorded another.
    lock = git_lock or _NullLock()
    with lock:
        # git serialises through .git/index.lock and the worktree registry.
        # Concurrent `worktree add` calls race there, so the pool passes a
        # lock and everything that touches the MAIN repo takes it. Work inside
        # a worktree has its own index and needs no lock, which is what makes
        # the fan-out worth having.
        base_ref = resolve_base(base)
        worktree = add_worktree(branch, base_ref)
    result["branch"] = branch
    result["base"] = base_ref

    try:
        try:
            if chosen == "local":
                raw = run_local(spec, worktree, seconds, out)
            elif chosen == "agent":
                raw = run_agent(spec, worktree, seconds, out, model,
                                spec.get("limits"))
            else:
                raw = run_remote(spec, worktree, seconds, out, provider, model)
        except subprocess.TimeoutExpired:
            raw = {"backend": chosen, "exit_code": None, "raw": "",
                   "duration_s": seconds, "report": None, "patch_applied": None,
                   "timed_out": True}
        report = raw.pop("report", None)
        result.update(raw)
        result["report"] = report
        problems = validate_result(report) if report is not None else [
            "no JSON report found in the reply"]
        result["contract_problems"] = problems

        # The report is committed into the branch itself, not left in the
        # gitignored `runs/` tree. `investigate` mode is contractually
        # forbidden from touching the files it audits, which used to mean
        # no diff at all -- and with nothing staged, submit() never made a
        # commit, so the branch was indistinguishable from its base and the
        # audit's findings existed only in this session's container. A
        # 15-task pool run in 2026-09 lost three investigate reports this
        # way. Every task now gets one, whether or not it changed anything
        # else, so `changed_files`/the patch check below always has
        # something to commit.
        results_dir = worktree / "queue" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"{task_id}.json").write_text(
            json.dumps({"task_id": task_id, "mode": spec["mode"],
                        "task": spec["task"], "report": report,
                        "contract_problems": problems}, indent=2) + "\n",
            encoding="utf-8")

        # git's account of the diff, alongside the agent's.
        result["files_changed"] = changed_files(worktree)
        patch = _git("diff", "--cached", cwd=worktree)
        if patch:
            (out / "changes.patch").write_text(patch + "\n", encoding="utf-8")
            result["patch"] = str(out / "changes.patch")
            _git("-c", "user.email=codex-worker@local",
                 "-c", "user.name=codex-worker",
                 "commit", "-q", "-m",
                 f"codex[{spec['mode']}]: {spec['task'].strip().splitlines()[0][:60]}\n\n"
                 f"task_id={task_id} backend={chosen}\n"
                 f"Delegated by the lead agent; not reviewed.",
                 cwd=worktree)
            result["commit"] = _git("rev-parse", "--short", "HEAD", cwd=worktree)
        else:
            result["patch"] = None

        result["verification"] = verify(spec, worktree, verify_level, out)

        if spec.get("manifest") and isinstance(report, dict):
            post = gates.postflight({**report, "manifest": spec["manifest"]})
            result["postflight"] = post
            if post["verdict"] == gates.FAIL:
                result.setdefault("gate_failures", []).append("postflight")

        blockers = list((report or {}).get("blockers") or [])
        if result.get("gate_failures"):
            blockers.append("postflight rejected the result")
        if result.get("worktree_disturbed"):
            blockers.append(
                f"the worker changed its own git state: "
                f"{result['worktree_disturbed']}. Any diff below may be "
                "incomplete; read the transcript before trusting it.")
        if result.get("stop_reason") not in (None, "done"):
            blockers.append(
                f"the agent loop stopped at its {result['stop_reason']} limit "
                "rather than finishing; the work on this branch is partial")
        result["verdict"] = decide(problems, result["verification"], blockers,
                                   result.get("patch_applied"),
                                   result.get("exit_code"),
                                   bool(result.get("timed_out")))
    finally:
        if keep_worktree:
            result["worktree"] = str(worktree)
        else:
            with lock:
                remove_worktree(worktree)

    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n",
                                     encoding="utf-8")
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def summarise(result: dict) -> str:
    r = result.get("report") or {}
    who = result.get("model") or result.get("provider") or result.get("backend")
    lines = [f"{result['verdict']}  [{result['mode']} · {who} · "
             f"{result.get('duration_s', 0)}s]"]
    if result.get("branch"):
        lines.append(f"  branch      {result['branch']}"
                     + ("" if result.get("patch") else "  (no changes)"))
    if result.get("patch"):
        lines.append(f"  patch       {result['patch']}")
    if result.get("tool_calls"):
        used = ", ".join(f"{n}x{c}" for n, c in sorted(result["tool_calls"].items()))
        b = result.get("budget") or {}
        lines.append(f"  tools       {used}")
        lines.append(f"  budget      {b.get('turns')} turns, "
                     f"{b.get('tool_calls')} calls, ${b.get('cost_usd')}, "
                     f"{b.get('elapsed_s')}s  [{result.get('stop_reason')}]")
    ver = result.get("verification") or {}
    if ver.get("checks"):
        for c in ver["checks"]:
            mark = "ok  " if c["passed"] else "FAIL"
            lines.append(f"  verify {mark} {c['cmd']}")
    elif ver.get("level") == "none":
        lines.append("  verify      DISABLED -- nothing was checked")
    for key in ("uncertainties", "blockers"):
        for item in (r.get(key) or []):
            lines.append(f"  {key[:-1]:<11} {item}")
    for p in result.get("contract_problems") or []:
        lines.append(f"  contract    {p}")
    if r.get("summary"):
        lines.append("")
        lines.append(r["summary"].strip())
    return "\n".join(lines)


def load_spec(args) -> dict:
    if args.spec:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    else:
        spec = {}
    if args.mode:
        spec["mode"] = args.mode
    if args.task:
        spec["task"] = args.task
    if args.task_file:
        spec["task"] = Path(args.task_file).read_text(encoding="utf-8")
    if args.context:
        spec["context_files"] = list(args.context)
    if args.acceptance:
        spec["acceptance"] = list(args.acceptance)
    if args.manifest:
        spec["manifest"] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.budget:
        spec["budget_minutes"] = args.budget
    if getattr(args, "tools", None):
        spec["tools"] = args.tools.split(",")
    limits = {k: v for k, v in (
        ("max_turns", getattr(args, "max_turns", None)),
        ("max_tool_calls", getattr(args, "max_tool_calls", None)),
        ("max_cost_usd", getattr(args, "max_cost", None))) if v is not None}
    if limits:
        spec["limits"] = {**(spec.get("limits") or {}), **limits}
    spec.setdefault("mode", "implement")
    return spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("modes", help="list task modes and what each expects")
    sub.add_parser("check", help="report which backend is available and why")

    s = sub.add_parser("submit", help="delegate one task and return a branch")
    s.add_argument("--spec", help="task spec JSON; flags below override it")
    s.add_argument("--mode", choices=sorted(MODES))
    s.add_argument("--task", help="what to do")
    s.add_argument("--task-file", help="read the task text from a file")
    s.add_argument("--context", action="append", metavar="PATH",
                   help="file to quote for a backend with no checkout. Repeatable.")
    s.add_argument("--acceptance", action="append", metavar="CMD",
                   help="command the WORKER runs against the result. Repeatable.")
    s.add_argument("--manifest", help="experiment manifest; gates run around it")
    s.add_argument("--budget", type=int, metavar="MIN")
    s.add_argument("--backend", choices=("auto", "local", "remote", "agent"),
                   default="auto",
                   help="agent = the tool-using loop (tools/agent_loop.py); "
                        "remote = one completion, no tools; local = the codex "
                        "CLI. auto prefers local, then agent, then remote.")
    s.add_argument("--tools", help="comma-separated subset for --backend agent "
                                   "(default: all of them)")
    s.add_argument("--max-turns", type=int)
    s.add_argument("--max-tool-calls", type=int)
    s.add_argument("--max-cost", type=float, metavar="USD")
    s.add_argument("--provider", default="auto",
                   help="remote backend only: openrouter, codex-dispatcher, "
                        "or auto (first one configured). See tools/providers.py.")
    s.add_argument("--model", help="override the provider's default model")
    s.add_argument("--base", help="branch from this instead of origin/main")
    s.add_argument("--verify", choices=("full", "fast", "none"), default="full",
                   dest="verify_level")
    s.add_argument("--out", help="directory for artifacts (default runs/codex/<id>)")
    s.add_argument("--keep-worktree", action="store_true",
                   help="leave the worktree in place for inspection")
    s.add_argument("--timeout", type=int, metavar="S")
    s.add_argument("--pretty", action="store_true",
                   help="human summary on stderr as well as JSON on stdout")
    s.add_argument("--dry-run", action="store_true",
                   help="print the assembled prompt and exit, dispatching nothing")

    args = ap.parse_args(argv)

    if args.cmd == "modes":
        for name in sorted(MODES):
            m = MODES[name]
            patch = "edits code" if m.expects_patch else "reports only"
            print(f"{name:<12} {patch:<12} {m.preamble.splitlines()[0]}")
        return 0

    if args.cmd == "check":
        local = shutil.which(codex_bin())
        provs = _providers().status()
        state = {
            "codex_bin": codex_bin(),
            "local_available": bool(local),
            "local_path": local,
            "providers": provs,
            "chosen_backend": choose_backend("auto"),
        }
        print(json.dumps(state, indent=2))
        return 0 if (local or any(p["available"] for p in provs)) else 1

    spec = load_spec(args)
    if not str(spec.get("task", "")).strip():
        ap.error("a task is required: --task, --task-file, or --spec")
    if spec["mode"] not in MODES:
        ap.error(f"unknown mode {spec['mode']!r}")

    if args.dry_run:
        has_repo = choose_backend(args.backend) == "local"
        context = ""
        if not has_repo:
            context, _ = gather_context(spec.get("context_files") or [])
        print(render_prompt(spec, context, has_repo=has_repo))
        return 0

    try:
        result = submit(spec, backend=args.backend, base=args.base,
                        verify_level=args.verify_level,
                        out_dir=Path(args.out) if args.out else None,
                        keep_worktree=args.keep_worktree, timeout=args.timeout,
                        provider=args.provider, model=args.model)
    except (RuntimeError, ValueError) as exc:
        print(f"codex_worker: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    if args.pretty:
        print(summarise(result), file=sys.stderr)
    return 0 if result["verdict"] == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
