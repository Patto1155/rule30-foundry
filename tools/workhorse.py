#!/usr/bin/env python
"""Pull-based experiment runner: gate, run, gate again, open a PR.

The council review of the "make Codex a workhorse" plan (2026-09-04) put the
requirement plainly: more autonomy does not make results more trustworthy,
and carrying the repo's rules as prompt text is not verification. So this
runner is a harness around executable gates, not around an agent.

    queue/<name>.json          a manifest, reviewed and committed like code
        |
        v
    tools/gates.py preflight   refuses what cannot produce information
        |                      (counting bound, seed, theory gate, light cone)
        v
    execute                    --agent script  runs manifest.script directly
        |                      --agent codex   asks Codex to implement & run
        v
    tools/gates.py postflight  rejects results that state more than measured
        |                      (censoring, noise floor, divergence, ~50% rule)
        v
    tools/verify_all.py        the repo's own integrity check, post-run
        |
        v
    branch + commit + push     a PR. Never main. CI gates it; a human merges.

Three properties are the point, and each is a choice rather than an accident:

- **Pull, not push.** The queue is a directory in the repo. There is no
  inbound endpoint, so there is no token whose leak becomes code execution
  on the box that runs experiments. The council endpoint stays read-only.
- **The default agent is no agent.** `--agent script` runs the manifest's
  script with its argv and nothing else -- most of the backlog in STATUS.md
  is an existing script that has simply not been run. An LLM is for writing
  experiments that do not exist yet, and `--agent codex` is opt-in for that.
- **Refusals are recorded, not swallowed.** A manifest preflight rejects is
  written to queue/refused/ with the full gate report. Deliberately not to
  docs/experiment-logs/: lint_ledger's STALE-STATUS check requires STATUS.md
  to cite the newest dated log there, so an automated writer to that
  directory would red-line every concurrent PR.

Usage:
    python tools/workhorse.py list [--pretty]
    python tools/workhorse.py run queue/b1-pattern-map-walk.json --dry-run
    python tools/workhorse.py run queue/b1-pattern-map-walk.json
    python tools/workhorse.py run queue/x.json --agent codex --no-push

--dry-run executes and gates but touches no branch and pushes nothing;
outputs go to a temporary directory. It is what the tests use.

`--dry-run` also implies `--no-external`, and that is load-bearing rather
than an optimisation. preflight's external gates shell out to verify_all,
whose `unittest` stage discovers tests/test_workhorse.py, which invokes this
runner -- so a dry run that took the external path would re-enter verify_all
and never terminate. verify_all.py documents the same hazard for its own
`skip_permitted` tests. The repo-state gates are verify_all's job anyway; a
dry run is a smoke test of the pipeline, not of the checkout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE = REPO_ROOT / "queue"
REFUSED = QUEUE / "refused"
RUNS = REPO_ROOT / "runs"

DEFAULT_BUDGET_MIN = 30


def _gates():
    spec = importlib.util.spec_from_file_location("gates", REPO_ROOT / "tools" / "gates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    A manifest need not live under queue/ -- an ad-hoc one in /tmp is a
    legitimate way to try something before committing it. A bare
    relative_to() raises on those, which turned a refusal (the gate working)
    into a traceback (the gate crashing)."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def in_repo(p: Path) -> bool:
    try:
        Path(p).resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def budget_seconds(m: dict) -> int:
    return int((m.get("budget") or {}).get("minutes", DEFAULT_BUDGET_MIN)) * 60


# --------------------------------------------------------------------------
# agents
# --------------------------------------------------------------------------

def run_script(m: dict, out: Path, timeout: int) -> dict:
    """The default: run the manifest's script. No LLM anywhere."""
    argv = [sys.executable, str(REPO_ROOT / m["script"]), *map(str, m.get("argv") or [])]
    started = time.time()
    try:
        p = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=timeout)
        code, so, se, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, timed_out = None, True
        so = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        se = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    (out / "stdout.txt").write_text(so, encoding="utf-8")
    (out / "stderr.txt").write_text(se, encoding="utf-8")
    try:
        stdout_json = json.loads(so)
    except ValueError:
        stdout_json = None
    return {
        "manifest": m,
        "agent": "script",
        "argv": argv[1:],
        "exit_code": code,
        "timed_out": timed_out,
        "duration_s": round(time.time() - started, 2),
        "stdout_json": stdout_json,
        # Filled by whoever writes the result up. A script that does not
        # emit conclusions has made no claims, so postflight has nothing to
        # reject; the claims come later, in the PR, where they are reviewed.
        "conclusions": [],
        "metrics": {},
    }


CODEX_PROMPT = """\
You are executing one experiment for rule30-foundry under an executable
contract. Read CLAUDE.md and AGENTS.md first.

The manifest below has already passed preflight (tools/gates.py). Implement
and run exactly what it describes -- no more. Then write
{result_path}
as JSON in the postflight schema documented at the top of tools/gates.py:
`manifest` (echo it), `horizon`, `metrics` (each with a measured `baseline`),
`conclusions` (every one horizon-qualified: "through N", never "never"),
and `divergence` / `stream_comparison` if you measured them.

Do not touch docs/STATUS.md, docs/CLAIM_LEDGER.md, or docs/experiment-logs/.
Do not add files under data/ without an explicit .gitignore exception and a
`python tools/make_manifest.py` regeneration. Do not write a conclusion the
result does not support.

MANIFEST
{manifest}

PREFLIGHT
{preflight}
"""


def run_codex(m: dict, out: Path, timeout: int, preflight_report: dict) -> dict:
    """Opt-in: ask Codex to implement and run the experiment, in a writable
    sandbox scoped to this checkout. Runs where codex is installed and logged
    in -- the dispatcher VM -- not through the read-only council endpoint."""
    if not shutil.which("codex"):
        raise RuntimeError("codex is not on PATH; --agent codex runs on the VM")
    result_path = out / "result.json"
    prompt = CODEX_PROMPT.format(result_path=result_path,
                                 manifest=json.dumps(m, indent=2),
                                 preflight=json.dumps(preflight_report, indent=2))
    argv = ["codex", "exec", "--skip-git-repo-check", "--sandbox",
            "workspace-write", "-C", str(REPO_ROOT), prompt]
    started = time.time()
    p = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True,
                       timeout=timeout)
    (out / "stdout.txt").write_text(p.stdout, encoding="utf-8")
    (out / "stderr.txt").write_text(p.stderr, encoding="utf-8")
    if not result_path.exists():
        raise RuntimeError("agent did not write result.json; nothing to gate")
    r = json.loads(result_path.read_text(encoding="utf-8"))
    r.setdefault("manifest", m)
    r.update(agent="codex", exit_code=p.returncode,
             duration_s=round(time.time() - started, 2))
    return r


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def refuse(m: dict, report: dict, path: Path) -> Path:
    REFUSED.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REFUSED / f"{m.get('name', path.stem)}-{stamp}.json"
    dest.write_text(json.dumps({"refused_at": _now(), "manifest_path": _rel(path),
                                "report": report}, indent=2) + "\n", encoding="utf-8")
    return dest


def run(path: Path, agent: str, dry_run: bool, no_branch: bool, no_push: bool,
        timeout: int | None, pretty: bool, no_external: bool = False) -> int:
    gates = _gates()
    m = load_manifest(path)
    name = m.get("name", path.stem)

    # 1. preflight -- before a branch, before any compute.
    #
    # External gates are off under --dry-run because they shell out to
    # verify_all, whose unittest stage runs tests/test_workhorse.py, which
    # runs this function: the external path would recurse without bound.
    external = not (dry_run or no_external)
    pre = gates.preflight(m, run_external=external)
    if pretty:
        print(gates.pretty(pre), file=sys.stderr)
    if pre["verdict"] == gates.FAIL:
        dest = refuse(m, pre, path)
        print(json.dumps({"status": "refused", "name": name,
                          "refusal": _rel(dest),
                          "preflight": pre}, indent=2))
        return 1

    # 2. workspace.
    if dry_run:
        out = Path(tempfile.mkdtemp(prefix=f"workhorse-{name}-"))
    else:
        if _git("status", "--porcelain"):
            print("workhorse: working tree is dirty; commit or stash first. "
                  "Provenance requires a known starting commit.", file=sys.stderr)
            return 2
        if not no_branch:
            _git("checkout", "-b", f"feat/{name}")
        out = RUNS / name
        out.mkdir(parents=True, exist_ok=True)
    head = _git("rev-parse", "HEAD")
    started = _now()

    # 3. execute.
    seconds = timeout or budget_seconds(m)
    try:
        result = (run_codex(m, out, seconds, pre) if agent == "codex"
                  else run_script(m, out, seconds))
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"workhorse: {exc}", file=sys.stderr)
        return 1
    result["provenance"] = {
        "head": head, "started": started, "finished": _now(),
        "python": platform.python_version(), "agent": agent,
        "budget_s": seconds,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # 4. postflight.
    post = gates.postflight(result)
    (out / "postflight.json").write_text(json.dumps(post, indent=2) + "\n", encoding="utf-8")
    if pretty:
        print(gates.pretty(post), file=sys.stderr)

    # 5. hashes of everything the run left behind, so the PR can be checked
    #    against what was actually produced.
    hashes = {p.name: _sha256(p) for p in sorted(out.iterdir())
              if p.is_file() and p.name != "hashes.json"}
    (out / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    ok = (post["verdict"] == gates.PASS and result.get("exit_code") == 0
          and not result.get("timed_out"))
    summary = {"status": "ok" if ok else "needs-attention", "name": name,
               "out": str(out), "exit_code": result.get("exit_code"),
               "timed_out": result.get("timed_out", False),
               "duration_s": result.get("duration_s"),
               "postflight": post["verdict"]}

    if dry_run:
        print(json.dumps(summary, indent=2))
        return 0 if ok else 1

    # 6. verify_all on the post-run tree. This is where a script that wrote
    #    under data/ without a manifest regen gets caught (manifest-current).
    va = subprocess.run([sys.executable, "tools/verify_all.py"], cwd=REPO_ROOT,
                        capture_output=True, text=True)
    (out / "verify_all.txt").write_text(va.stdout + va.stderr, encoding="utf-8")
    summary["verify_all"] = "PASS" if va.returncode == 0 else "FAIL"
    ok = ok and va.returncode == 0

    # 7. commit. The manifest moves out of the queue and into the run, so
    #    merging the PR drains the queue.
    if in_repo(path):
        _git("mv", _rel(path), _rel(out / "manifest.json"))
    else:
        (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    _git("add", "-A", _rel(out))
    tag = "" if ok else " [needs attention]"
    _git("commit", "-q", "-m",
         f"workhorse: {name}{tag}\n\n"
         f"agent={agent} exit={result.get('exit_code')} "
         f"postflight={post['verdict']} verify_all={summary['verify_all']}\n"
         f"head={head}")
    summary["commit"] = _git("rev-parse", "--short", "HEAD")

    # 8. push, and say where the PR goes.
    if not no_push:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        _git("push", "-u", "origin", branch)
        remote = _git("remote", "get-url", "origin")
        slug = remote.rstrip("/").removesuffix(".git").split("github.com")[-1].lstrip(":/")
        summary["pr_url"] = f"https://github.com/{slug}/compare/main...{branch}?expand=1"
        if shutil.which("gh"):
            title = f"workhorse: {name}{tag}"
            body = (f"Automated run of `{path.name}`.\n\n"
                    f"- agent: `{agent}`\n- exit: `{result.get('exit_code')}`\n"
                    f"- postflight: **{post['verdict']}**\n"
                    f"- verify_all: **{summary['verify_all']}**\n\n"
                    f"Gate reports and hashes are in `{_rel(out)}/`.")
            gh = subprocess.run(["gh", "pr", "create", "--title", title, "--body", body],
                                cwd=REPO_ROOT, capture_output=True, text=True)
            if gh.returncode == 0:
                summary["pr_url"] = gh.stdout.strip()

    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


def list_queue(pretty: bool) -> int:
    gates = _gates()
    rows = []
    for p in sorted(QUEUE.glob("*.json")):
        try:
            m = load_manifest(p)
            pre = gates.preflight(m, run_external=False)
            failing = [g["gate"] for g in pre["gates"] if g["status"] == gates.FAIL]
            rows.append({"path": _rel(p), "name": m.get("name"),
                         "kind": m.get("kind"), "verdict": pre["verdict"],
                         "failing": failing,
                         "budget_min": (m.get("budget") or {}).get("minutes")})
        except (OSError, ValueError) as exc:
            rows.append({"path": _rel(p), "error": str(exc)})
    print(json.dumps(rows, indent=2))
    if pretty:
        for r in rows:
            if "error" in r:
                print(f"  ??    {r['path']}  {r['error']}", file=sys.stderr)
            else:
                extra = f"  refused by: {', '.join(r['failing'])}" if r["failing"] else ""
                print(f"  {r['verdict']:<4}  {r['path']}  [{r['kind']}, "
                      f"{r['budget_min']} min]{extra}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="show the queue with pure-gate verdicts")
    ls.add_argument("--pretty", action="store_true")

    rn = sub.add_parser("run", help="gate, execute, gate, commit, push")
    rn.add_argument("manifest", type=Path)
    rn.add_argument("--agent", choices=("script", "codex"), default="script")
    rn.add_argument("--dry-run", action="store_true",
                    help="execute and gate into a temp dir; no branch, no push")
    rn.add_argument("--no-branch", action="store_true",
                    help="commit on the current branch instead of feat/<name>")
    rn.add_argument("--no-push", action="store_true")
    rn.add_argument("--timeout", type=int, metavar="S",
                    help="override budget.minutes")
    rn.add_argument("--no-external", action="store_true",
                    help="skip preflight's subprocess gates (lint, golden, "
                         "verify_all). Implied by --dry-run, which would "
                         "otherwise recurse through verify_all's test stage.")
    rn.add_argument("--pretty", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        return list_queue(args.pretty)
    return run(args.manifest, args.agent, args.dry_run, args.no_branch,
               args.no_push, args.timeout, args.pretty, args.no_external)


if __name__ == "__main__":
    raise SystemExit(main())
