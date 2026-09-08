#!/usr/bin/env python3
"""Dispatcher that runs `codex exec` on behalf of an authenticated caller.

Runs on the VM, not in the Claude container. It exists so `tools/council.py`
can get a second opinion over ordinary HTTPS instead of shelling into another
machine over ssh.

Deployment shape, and every part of it is load-bearing:

    internet --443--> nginx (TLS, rate limit) --> 127.0.0.1:8080 (this)

This binds to loopback by design. It speaks plain HTTP and does no rate
limiting, because nginx in front does both. Do not "simplify" it by binding
0.0.0.0 and exposing it directly -- the endpoint runs an agent with your
ChatGPT credentials, so an unauthenticated one is an open door to both your
OpenAI account and this VM.

Security properties, in the order they matter:

- Bearer token, compared with hmac.compare_digest. Absent or wrong -> 401/403
  carrying `X-Dispatcher: codex`, which is how the client tells "your token is
  wrong" apart from "the egress proxy denied this host", two failures that
  otherwise look identical.
- The prompt is passed to codex as an argv element via a list, never through a
  shell. There is no interpolation anywhere in this file, so prompt content
  cannot become a command.
- The model is checked against an allowlist. Not for injection -- argv makes
  that impossible -- but so a leaked token cannot run up an unbounded bill on
  a model you did not choose.
- codex runs sandboxed read-only where the installed version supports it. The
  flag is probed at startup rather than assumed, because flag names move
  between codex releases and a hard-coded one fails closed at the worst moment.
- The token is never logged. Prompts are logged by length and SHA-256 prefix,
  never by content: this box would otherwise accumulate a plaintext archive of
  everything ever reviewed.

Configuration (environment):

    CODEX_COUNCIL_TOKEN   required; the bearer token. No default, no fallback.
    CODEX_BIND            default 127.0.0.1
    CODEX_PORT            default 8080
    CODEX_DEFAULT_MODEL   default gpt-5.6-sol
    CODEX_ALLOWED_MODELS  comma-separated; default is the default model alone
    CODEX_TIMEOUT_S       default 900
    CODEX_MAX_PROMPT      default 100000 bytes
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("codex-dispatcher")

TOKEN = os.environ.get("CODEX_COUNCIL_TOKEN", "")
BIND = os.environ.get("CODEX_BIND", "127.0.0.1")
PORT = int(os.environ.get("CODEX_PORT", "8080"))
DEFAULT_MODEL = os.environ.get("CODEX_DEFAULT_MODEL", "gpt-5.6-sol")
ALLOWED_MODELS = {
    m.strip() for m in
    os.environ.get("CODEX_ALLOWED_MODELS", DEFAULT_MODEL).split(",")
    if m.strip()
}
TIMEOUT_S = int(os.environ.get("CODEX_TIMEOUT_S", "900"))
MAX_PROMPT = int(os.environ.get("CODEX_MAX_PROMPT", "100000"))

VALID_ROLE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

# Set by probe_codex_flags() at startup.
CODEX_FLAGS: dict[str, bool] = {"sandbox": False, "output_last_message": False}


def probe_codex_flags(codex: str = "codex") -> dict[str, bool]:
    """Ask the installed codex which flags it actually has.

    Flag names move between releases. Hard-coding `--sandbox read-only` and
    having a future version reject it would take the dispatcher down at the
    moment it is least convenient; worse, silently dropping the flag would run
    the agent unsandboxed without anyone noticing. Probing makes the answer a
    fact about this machine rather than an assumption baked in months earlier.
    """
    flags = {"sandbox": False, "output_last_message": False}
    try:
        out = subprocess.run([codex, "exec", "--help"], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not probe codex flags (%s); assuming none", exc)
        return flags
    text = (out.stdout or "") + (out.stderr or "")
    flags["sandbox"] = "--sandbox" in text
    flags["output_last_message"] = "--output-last-message" in text
    return flags


def build_codex_argv(prompt: str, model: str, last_message_path: str | None,
                     flags: dict[str, bool]) -> list[str]:
    """Assemble the codex command line.

    A list, always. Pure, so the exact argv is asserted in tests rather than
    discovered in production.
    """
    argv = ["codex", "exec", "--skip-git-repo-check", "-m", model]
    if flags.get("sandbox"):
        argv += ["--sandbox", "read-only"]
    if last_message_path and flags.get("output_last_message"):
        argv += ["--output-last-message", last_message_path]
    argv.append(prompt)
    return argv


def extract_answer(stdout: str) -> str:
    """Pull the assistant's reply out of `codex exec` console output.

    Only used when the installed codex lacks --output-last-message; when it has
    it, the file is exact and this never runs. The format is a banner, the
    echoed prompt after a `user` line, the reply after a `codex` line, then a
    `tokens used N` trailer.

    Deliberately conservative: if the shape is not recognised, return the whole
    output rather than a confident slice of the wrong thing. A caller reading
    an over-long answer notices; one reading a silently truncated answer does
    not.
    """
    lines = stdout.splitlines()
    marker = None
    for i, line in enumerate(lines):
        if line.strip() == "codex":
            marker = i
    if marker is None:
        return stdout.strip()
    body = lines[marker + 1:]
    for i, line in enumerate(body):
        if re.match(r"^\s*tokens used\b", line):
            body = body[:i]
            break
    answer = "\n".join(body).strip()
    return answer or stdout.strip()


def run_codex(prompt: str, model: str, timeout: int) -> tuple[bool, str, str]:
    """Run codex once. Returns (ok, answer, raw_stdout)."""
    tmp_path = None
    if CODEX_FLAGS.get("output_last_message"):
        fd, tmp_path = tempfile.mkstemp(prefix="codex-last-", suffix=".txt")
        os.close(fd)
    try:
        argv = build_codex_argv(prompt, model, tmp_path, CODEX_FLAGS)
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
        raw = proc.stdout or ""
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip() or raw.strip()
            return False, detail[:4000], raw
        if tmp_path:
            content = Path(tmp_path).read_text(encoding="utf-8").strip()
            if content:
                return True, content, raw
        return True, extract_answer(raw), raw
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def token_ok(header: str | None) -> bool:
    """Constant-time bearer check.

    compare_digest rather than ==, so response timing does not leak the token
    a character at a time. An unset server token rejects everything: failing
    closed is the only safe reading of "no credential configured".
    """
    if not TOKEN or not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[len("Bearer "):], TOKEN)


def prompt_fingerprint(prompt: str) -> str:
    """A loggable identifier for a prompt that is not the prompt.

    This VM would otherwise accumulate a plaintext archive of every claim ever
    reviewed, in the logs, forever.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


class Handler(BaseHTTPRequestHandler):
    server_version = "codex-dispatcher/1.0"

    def log_message(self, fmt, *args):  # noqa: A003
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, payload: dict, extra: dict | None = None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _deny(self, code: int, reason: str):
        # X-Dispatcher is what lets the client distinguish a token problem here
        # from a 403 issued by the caller's egress proxy. Without it the two
        # are indistinguishable and the operator debugs the wrong layer.
        self._send(code, {"ok": False, "error": reason},
                   {"X-Dispatcher": "codex"})

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send(200, {"ok": True, "service": "codex-dispatcher",
                             "codex_flags": CODEX_FLAGS,
                             "models": sorted(ALLOWED_MODELS)})
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/ask":
            self._send(404, {"ok": False, "error": "not found"})
            return
        if not token_ok(self.headers.get("Authorization")):
            self._deny(401, "missing or invalid bearer token")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"ok": False, "error": "bad Content-Length"})
            return
        if length > MAX_PROMPT + 4096:
            self._send(413, {"ok": False, "error": "request too large"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send(400, {"ok": False, "error": "body is not valid JSON"})
            return

        role = str(payload.get("role", "")).strip()
        prompt = payload.get("prompt", "")
        model = str(payload.get("model") or DEFAULT_MODEL).strip()

        if not VALID_ROLE.match(role):
            self._send(400, {"ok": False, "error": "invalid role"})
            return
        if not isinstance(prompt, str) or not prompt.strip():
            self._send(400, {"ok": False, "error": "empty prompt"})
            return
        if len(prompt.encode("utf-8")) > MAX_PROMPT:
            self._send(413, {"ok": False, "error": "prompt over limit"})
            return
        if model not in ALLOWED_MODELS:
            self._send(400, {"ok": False,
                             "error": f"model not allowed: {model}"})
            return

        started = time.time()
        log.info("ask role=%s model=%s bytes=%d sha=%s", role, model,
                 len(prompt.encode("utf-8")), prompt_fingerprint(prompt))
        try:
            ok, answer, _raw = run_codex(prompt, model, TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.warning("codex timed out after %ss", TIMEOUT_S)
            self._send(504, {"ok": False, "error": f"timeout after {TIMEOUT_S}s"})
            return
        except OSError as exc:
            log.error("codex failed to start: %s", exc)
            self._send(500, {"ok": False, "error": "codex not runnable"})
            return

        duration = time.time() - started
        log.info("done role=%s ok=%s %.1fs", role, ok, duration)
        self._send(200 if ok else 502, {
            "ok": ok, "role": role, "model": model,
            "answer": answer, "duration_s": round(duration, 2),
        })


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not TOKEN:
        log.error("CODEX_COUNCIL_TOKEN is unset. Refusing to start: an "
                  "unauthenticated dispatcher is an open shell on this VM.")
        return 2
    if len(TOKEN) < 32:
        log.error("CODEX_COUNCIL_TOKEN is under 32 characters. Refusing to "
                  "start; generate one with: openssl rand -hex 32")
        return 2
    if not shutil.which("codex"):
        log.error("codex is not on PATH for this user. Install it and check "
                  "that this service runs as the user that ran `codex login`.")
        return 2

    CODEX_FLAGS.update(probe_codex_flags())
    log.info("codex flags: %s", CODEX_FLAGS)
    if not CODEX_FLAGS["sandbox"]:
        log.warning("this codex has no --sandbox flag; the agent will run "
                    "with whatever default it ships with. Verify that default "
                    "before exposing this service.")
    log.info("models allowed: %s", sorted(ALLOWED_MODELS))
    log.info("listening on %s:%s", BIND, PORT)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
