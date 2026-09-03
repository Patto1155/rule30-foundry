#!/usr/bin/env python
"""Dispatch a self-contained brief to an external model and capture the reply.

Why this exists
---------------
The expensive failures in this repo have all been *silent*: a byte-reversed
bitstream, a vacuous counting bound, a ledger row citing a template. Every one
of them was caught by a second reader who did not share the first reader's
assumptions. This tool buys that second reader on demand, from a model that is
not Claude, and writes the reply to a file so the reasoning is reviewable
rather than lost in a chat scrollback.

Design constraints, all deliberate:

  * **Stdlib only.** CI installs numpy and nothing else (requirements-ci.txt);
    `tests/test_council_config.py` must import this module there. The MCP
    server path imports `mcp` lazily, inside `serve()`.
  * **Read-only reviewers.** Codex runs with `--sandbox read-only`. External
    models report; Claude applies. Two agents writing one tree is a lost
    afternoon, and an unreviewed patch from a model that cannot run
    `tools/verify_all.py` is worse than no patch.
  * **One config block.** Everything you would want to change lives in ROLES.

Usage
-----
    python tools/council.py roles
    python tools/council.py ask math briefs/c2c-lag-sets.md --repo
    python tools/council.py ask lit - < brief.md --out briefs/out/lit.md
    python tools/council.py doctor          # is this environment wired up?
    python tools/council.py serve           # optional MCP server (needs `mcp`)

Environment setup for a Claude Code cloud session: docs/COUNCIL.md.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Configuration. This block is the whole knob panel - edit it, not the code.
#
# backend   "codex"      -> OpenAI Codex CLI signed in with a ChatGPT account.
#                           Can read the repo (--repo). Draws on the ChatGPT
#                           subscription's rate limits, shared with your own
#                           interactive Codex use, so spend it on jobs that
#                           need repo access or heavy reasoning.
#           "openrouter" -> one HTTPS call, no repo access, pennies. Right for
#                           literature sweeps and cheap fan-out.
# model     Backend-specific ID. These drift; `codex --help` and
#           openrouter.ai/models are the authorities, not this comment.
# reasoning Codex only: minimal | low | medium | high | xhigh.
# timeout   Seconds. An xhigh Codex review over a repo can genuinely take 20 min.
# --------------------------------------------------------------------------
ROLES: dict[str, dict] = {
    # Code and diff review, with the repo in front of it.
    "review": dict(backend="codex", model="gpt-5.2", reasoning="high", timeout=1800),
    # Proofs, counting bounds, "is this negative result vacuous".
    "math": dict(backend="codex", model="gpt-5.2", reasoning="xhigh", timeout=2400),
    # Literature: prior art on Rule 30, algebraic immunity, CA cryptanalysis.
    "lit": dict(backend="openrouter", model="deepseek/deepseek-r1", timeout=900),
    # Adversarial reader: what would make this claim wrong?
    "redteam": dict(backend="openrouter", model="qwen/qwen3-235b-a22b", timeout=900),
}

DEFAULT_TIMEOUT = 1800
CODEX_HOME = pathlib.Path(os.environ.get("CODEX_HOME", pathlib.Path.home() / ".codex"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class CouncilError(RuntimeError):
    """A dispatch failed for a reason the caller can act on."""


# ---------------------------------------------------------------- codex ----
def ensure_codex_auth() -> None:
    """Seed ~/.codex/auth.json from CODEX_AUTH_B64 if it is not already there.

    Cloud sandboxes are ephemeral: the environment's setup script normally does
    this before Claude starts, so this is a fallback for a plain local shell.
    Base64 because auth.json is full of double quotes and the environment
    variable editor is not a shell - a raw paste breaks on the first quote.
    """
    auth = CODEX_HOME / "auth.json"
    if auth.exists():
        return
    blob = os.environ.get("CODEX_AUTH_B64", "").strip()
    if not blob:
        raise CouncilError(
            "codex backend: no ~/.codex/auth.json and CODEX_AUTH_B64 is unset.\n"
            "On a trusted machine: `codex login`, then\n"
            "  base64 -w0 ~/.codex/auth.json    (macOS: base64 -i ~/.codex/auth.json)\n"
            "and store that single line as CODEX_AUTH_B64. See docs/COUNCIL.md."
        )
    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    CODEX_HOME.chmod(0o700)
    try:
        auth.write_bytes(base64.b64decode(blob, validate=True))
    except Exception as exc:  # noqa: BLE001 - message matters more than type
        raise CouncilError(f"CODEX_AUTH_B64 is not valid base64: {exc}") from exc
    auth.chmod(0o600)


def run_codex(cfg: dict, brief: str, repo: bool) -> str:
    ensure_codex_auth()
    if not shutil.which("codex"):
        raise CouncilError("codex is not installed: npm install -g @openai/codex")
    # Without --repo the reviewer gets an empty scratch cwd, so a brief that
    # forgot to inline its evidence fails loudly instead of quietly reading
    # whatever happens to be in the working directory.
    cwd = str(REPO_ROOT) if repo else tempfile.mkdtemp(prefix="council-")
    out = pathlib.Path(tempfile.mkstemp(prefix="council-", suffix=".md")[1])
    cmd = [
        "codex", "exec", "--skip-git-repo-check",
        "--sandbox", "read-only",
        "-C", cwd,
        "-m", cfg["model"],
        "-c", f'model_reasoning_effort="{cfg.get("reasoning", "medium")}"',
        "-o", str(out),
        "-",  # prompt on stdin: no argv length limit, no shell quoting
    ]
    proc = subprocess.run(
        cmd, input=brief, text=True, capture_output=True,
        timeout=cfg.get("timeout", DEFAULT_TIMEOUT),
    )
    answer = out.read_text(encoding="utf-8", errors="replace")
    out.unlink(missing_ok=True)
    if not answer.strip():
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        hint = ""
        if "401" in tail or "login" in tail.lower():
            hint = ("\nHint: the seeded token has expired. Run `codex exec \"ok\"` on the "
                    "machine you logged in from (that refreshes it), re-base64 "
                    "~/.codex/auth.json, and update CODEX_AUTH_B64.")
        raise CouncilError(f"codex exited {proc.returncode} with no answer:\n{tail}{hint}")
    return answer


# ----------------------------------------------------------- openrouter ----
def run_openrouter(cfg: dict, brief: str, repo: bool) -> str:
    if repo:
        raise CouncilError(
            "the openrouter backend has no repo access; inline the evidence in the "
            "brief, or use a codex-backed role."
        )
    body = {
        "model": cfg["model"],
        "max_tokens": cfg.get("max_tokens", 8000),
        "messages": [{"role": "user", "content": brief}],
    }
    headers = {"Content-Type": "application/json"}
    # Empty is fine and expected in a cloud session: the agent proxy injects
    # the Authorization header from the environment's API credential, so the
    # key is never visible to this process or to the model driving it.
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=cfg.get("timeout", DEFAULT_TIMEOUT)) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise CouncilError(f"openrouter HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise CouncilError(
            f"openrouter unreachable ({exc.reason}). In a cloud session, check the "
            "environment's network allowlist covers openrouter.ai - see docs/COUNCIL.md."
        ) from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise CouncilError(f"unexpected openrouter response: {json.dumps(payload)[:2000]}") from exc


BACKENDS = {"codex": run_codex, "openrouter": run_openrouter}


# ---------------------------------------------------------------- core -----
def ask(role: str, brief: str, repo: bool = False) -> str:
    """Send `brief` to `role`'s model and return the reply, provenance-stamped."""
    if role not in ROLES:
        raise CouncilError(f"unknown role {role!r}; known roles: {', '.join(sorted(ROLES))}")
    if not brief.strip():
        raise CouncilError("refusing to dispatch an empty brief")
    cfg = ROLES[role]
    started = time.time()
    answer = BACKENDS[cfg["backend"]](cfg, brief, repo)
    stamp = (
        f"<!-- council role={role} backend={cfg['backend']} model={cfg['model']} "
        f"repo_access={repo} elapsed={time.time() - started:.0f}s "
        f"utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} -->\n\n"
    )
    return stamp + answer


def doctor() -> int:
    """Report whether this machine can actually reach each configured backend."""
    used = {cfg["backend"] for cfg in ROLES.values()}
    failures = 0
    if "codex" in used:
        ok_bin = shutil.which("codex")
        print(f"codex binary        : {'FOUND ' + ok_bin if ok_bin else 'MISSING (npm i -g @openai/codex)'}")
        auth = CODEX_HOME / "auth.json"
        seeded = auth.exists() or bool(os.environ.get("CODEX_AUTH_B64", "").strip())
        print(f"codex credentials   : {'present' if seeded else 'MISSING (CODEX_AUTH_B64 unset)'}")
        if auth.exists():
            try:
                mode = json.loads(auth.read_text()).get("auth_mode", "?")
            except Exception:  # noqa: BLE001
                mode = "unreadable"
            print(f"codex auth_mode     : {mode}  (expected: chatgpt)")
        failures += 0 if (ok_bin and seeded) else 1
    if "openrouter" in used:
        proxied = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
        print(f"openrouter auth     : {'env key set' if proxied else 'none in env (fine if the agent proxy injects it)'}")
    print(f"roles configured    : {', '.join(sorted(ROLES))}")
    return failures


# ----------------------------------------------------------------- mcp -----
def serve() -> None:
    """Expose ask/roles as MCP tools, so a dispatch is a typed tool call."""
    from mcp.server.fastmcp import FastMCP  # lazy: `pip install mcp`, not a CI dep

    mcp = FastMCP("council")

    @mcp.tool()
    def council_ask(role: str, brief: str, repo_access: bool = False) -> str:
        """Send a self-contained brief to an external reviewer model.

        The reviewer sees ONLY this brief (plus read-only repo files when
        repo_access=True and the role is codex-backed). Do not include your own
        conclusion - independence is the thing being bought. Call council_roles
        for the available roles.
        """
        return ask(role, brief, repo_access)

    @mcp.tool()
    def council_roles() -> str:
        """List reviewer roles and the model behind each."""
        return json.dumps(ROLES, indent=2)

    mcp.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("roles", help="print the role table as JSON")
    sub.add_parser("doctor", help="check this machine can reach each backend")
    sub.add_parser("serve", help="run as an MCP server (requires `mcp`)")
    a = sub.add_parser("ask", help="dispatch a brief")
    a.add_argument("role", choices=sorted(ROLES))
    a.add_argument("brief", help="path to a brief .md, or '-' for stdin")
    a.add_argument("--repo", action="store_true",
                   help="let the reviewer read this repo, read-only (codex roles only)")
    a.add_argument("--out", help="write the reply here instead of stdout")
    args = parser.parse_args(argv)

    if args.cmd == "roles":
        print(json.dumps(ROLES, indent=2))
        return 0
    if args.cmd == "doctor":
        return doctor()
    if args.cmd == "serve":
        serve()
        return 0

    brief = sys.stdin.read() if args.brief == "-" else pathlib.Path(args.brief).read_text(encoding="utf-8")
    try:
        result = ask(args.role, brief, args.repo)
    except CouncilError as exc:
        print(f"council: {exc}", file=sys.stderr)
        return 2
    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result, encoding="utf-8", newline="")
        print(f"wrote {dest} ({len(result)} chars)")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
