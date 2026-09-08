#!/usr/bin/env python
"""Where a delegated task's text comes from. One interface, several suppliers.

`tools/codex_worker.py` turns a task into a prompt, applies whatever comes
back, and verifies it. None of that cares which model answered. This module is
the seam: a provider takes a prompt and returns text, and everything above it
is provider-agnostic.

That seam is the point rather than an abstraction for its own sake. The repo
had exactly one supplier -- `codex exec` on a VM you rent and maintain -- which
made "delegate this" and "use Codex" the same sentence. They are not. A pool of
ten cheap workers and one careful reviewer are different jobs, and the reviewer
being a different model from the workers is the only reason its agreement is
worth anything (CLAUDE.md, *Delegation*).

    provider          runs on                    good for
    ----------------  -------------------------  ---------------------------
    openrouter        openrouter.ai, any model   breadth and concurrency:
                                                 ten workers at once, cheap
    codex-dispatcher  your VM, via HTTPS         one careful worker; the
                                                 model the council already
                                                 uses, so its review is
                                                 comparable to past reviews

Both are *completion* providers: prompt in, text out. They cannot run anything,
so an edit travels back as a unified diff which the worker applies and then
tests itself. The agentic path -- `codex exec` with a writable checkout, which
runs its own commands -- is not a provider and lives in codex_worker.run_local,
because "returns text" is the wrong shape for it.

Configuration, all from the environment. A key in argv is a key in `ps`
output, so none of these is a flag:

    OPENROUTER_API_KEY     required for the openrouter provider
    OPENROUTER_MODEL       default model slug (see `models` below)
    OPENROUTER_BASE_URL    default https://openrouter.ai/api/v1
    CODEX_COUNCIL_URL      dispatcher /ask endpoint
    CODEX_COUNCIL_TOKEN    its bearer token

Usage:
    python tools/providers.py check                  # what is usable, and why not
    python tools/providers.py models --grep deepseek # live catalogue, needs egress
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# The default worker model. Env-overridable rather than hard-coded in the
# pool: model names move, and a wrong one is a 400 per task rather than one
# clear failure.
#
# Verified against the live catalogue on 2026-09-08 and exercised end to end
# the same day (docs/WORKER_POOL.md, *Live*). Pinned to the dated build rather
# than the floating `deepseek/deepseek-v4-flash` alias: a research repo whose
# whole argument is reproducibility should not have its worker silently
# change underneath it. The alias is about 40% cheaper if you would rather
# have that than the pin.
#
# It is a reasoning model: on the first live task 14,332 of 14,741 completion
# tokens were reasoning, which is why one task takes minutes and why running
# them concurrently is the point. Cost was $0.0026 for that task.
#
# Re-check with `python tools/providers.py models --grep deepseek` before
# assuming this line is still true; a slug remembered rather than checked is
# how you get ten tasks that each fail identically.
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731"

# Retries are for the server saying "not now", never for the server saying
# "no". A 400 retried three times is three times the wrong request.
RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0


def _council():
    """The council module, for its transport only.

    _opener() there refuses every redirect rather than letting urllib rebuild
    the request and forward the Authorization header to whatever answers. That
    hazard is identical for an OpenRouter key, so the same opener is used here
    rather than a second, laxer one.
    """
    spec = importlib.util.spec_from_file_location(
        "council", REPO_ROOT / "tools" / "council.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Completion:
    """What a provider returns. `ok` false means `error` says why."""

    __slots__ = ("text", "model", "provider", "duration_s", "usage", "ok",
                 "error", "attempts")

    def __init__(self, text: str = "", model: str = "", provider: str = "",
                 duration_s: float = 0.0, usage: dict | None = None,
                 ok: bool = True, error: str = "", attempts: int = 1):
        self.text, self.model, self.provider = text, model, provider
        self.duration_s, self.usage = duration_s, usage or {}
        self.ok, self.error, self.attempts = ok, error, attempts

    def as_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model, "ok": self.ok,
                "error": self.error, "duration_s": round(self.duration_s, 2),
                "usage": self.usage, "attempts": self.attempts}


class ProviderError(RuntimeError):
    """A provider could not answer, and retrying will not help."""


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={**headers,
                                          "Content-Type": "application/json",
                                          "Accept": "application/json"})
    with _council()._opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, method="GET", headers=headers)
    with _council()._opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def explain_url_error(err: urllib.error.URLError) -> str:
    """Tell an egress denial apart from a provider outage.

    They look alike from inside a container with a default-deny allowlist, and
    the remedies are opposite: one is a policy change by a human, the other is
    waiting. A CONNECT tunnel refused by the proxy never reaches the provider
    at all, so there is no HTTP status to read -- it surfaces here.
    """
    reason = str(getattr(err, "reason", err))
    if "tunnel" in reason.lower() or "403" in reason or "proxy" in reason.lower():
        return (f"{reason} -- this looks like the egress proxy, not the "
                "provider: the host is not on this environment's network "
                "allowlist. That is a policy denial, not a transient error. "
                "Add it to the environment's network policy and start a new "
                "session; do not route around it. Confirm with: "
                "curl -sS \"$HTTPS_PROXY/__agentproxy/status\"")
    return reason


class Provider:
    name = "abstract"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def complete(self, prompt: str, timeout: int) -> Completion:
        raise NotImplementedError


class OpenRouter(Provider):
    """Any model OpenRouter serves, over one OpenAI-shaped endpoint.

    This is the concurrency supplier: cheap enough that ten workers at once is
    a sensible thing to do, which is what tools/worker_pool.py does with it.
    Ten workers on the same model are ten times the throughput and exactly one
    model's blind spots -- see the pool's own docstring; that is a fan-out, not
    a panel.
    """

    name = "openrouter"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None):
        self.base = (base_url or os.environ.get("OPENROUTER_BASE_URL")
                     or OPENROUTER_BASE).rstrip("/")
        self.model = (model or os.environ.get("OPENROUTER_MODEL")
                      or DEFAULT_OPENROUTER_MODEL)
        self._key = api_key if api_key is not None else os.environ.get(
            "OPENROUTER_API_KEY", "")

    def available(self) -> tuple[bool, str]:
        if not self._key.strip():
            return False, ("OPENROUTER_API_KEY is unset. See "
                           "docs/WORKER_POOL.md, *Configuring the key*.")
        return True, f"key present, model {self.model}"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._key}",
            # OpenRouter attributes traffic by these; they are not credentials
            # and carry nothing about the repository's contents.
            "HTTP-Referer": "https://github.com/Patto1155/rule30-foundry",
            "X-Title": "rule30-foundry",
        }

    def complete(self, prompt: str, timeout: int) -> Completion:
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}]}
        started = time.time()
        last = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                data = _post_json(f"{self.base}/chat/completions", payload,
                                  self._headers(), timeout)
            except urllib.error.HTTPError as err:
                detail = _read_error(err)
                if err.code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                    last = f"HTTP {err.code}: {detail}"
                    time.sleep(BACKOFF_BASE_S ** attempt)
                    continue
                raise ProviderError(explain_http(err.code, detail)) from None
            except urllib.error.URLError as err:
                raise ProviderError(explain_url_error(err)) from None

            if data.get("error"):
                # OpenRouter reports some upstream failures as HTTP 200 with an
                # error object. Reading only the status code would hand the
                # worker an empty completion and let it report "no JSON report
                # found" -- a provider outage disguised as a bad worker.
                msg = data["error"].get("message", json.dumps(data["error"]))
                raise ProviderError(f"provider returned an error: {msg}")
            choices = data.get("choices") or []
            if not choices:
                raise ProviderError(f"no choices in response: {json.dumps(data)[:400]}")
            text = (choices[0].get("message") or {}).get("content") or ""
            return Completion(text=text, model=data.get("model", self.model),
                              provider=self.name,
                              duration_s=time.time() - started,
                              usage=data.get("usage") or {}, attempts=attempt)
        raise ProviderError(f"gave up after {MAX_ATTEMPTS} attempts: {last}")

    def models(self, timeout: int = 60) -> list[dict]:
        return (_get_json(f"{self.base}/models", self._headers(), timeout)
                .get("data") or [])


def _read_error(err: urllib.error.HTTPError) -> str:
    try:
        body = err.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - the error path must not raise
        return err.reason or ""
    try:
        return (json.loads(body).get("error") or {}).get("message") or body[:400]
    except ValueError:
        return body[:400]


def explain_http(code: int, detail: str) -> str:
    """Say which of the several things a status could mean it actually is."""
    if code == 401:
        return (f"HTTP 401: the key was rejected. {detail} Check "
                "OPENROUTER_API_KEY is the whole key, unquoted.")
    if code == 402:
        return f"HTTP 402: the account is out of credit. {detail}"
    if code == 403:
        return (f"HTTP 403 from the provider (not the egress proxy -- a proxy "
                f"denial fails the CONNECT and never gets a status). {detail}")
    if code == 404:
        return (f"HTTP 404: no such model, most likely. {detail} List what "
                "exists with: python tools/providers.py models --grep <name>")
    return f"HTTP {code}: {detail}"


class CodexDispatcher(Provider):
    """`codex exec` on your VM, reached over HTTPS. One careful worker.

    Same endpoint the council uses, so a review from here is comparable with
    every review already in the logs. Sandboxed read-only on the VM, hence a
    completion provider rather than an agentic one.
    """

    name = "codex-dispatcher"

    def __init__(self, url: str | None = None, token: str | None = None,
                 model: str | None = None):
        # `if url is not None`, not `url or ...`: an explicitly empty url
        # means "not configured" and must not silently fall back to the
        # environment, or a caller disabling the provider gets the ambient one.
        self.url = (url if url is not None
                    else os.environ.get("CODEX_COUNCIL_URL", "")).strip()
        self._token = (token if token is not None
                       else os.environ.get("CODEX_COUNCIL_TOKEN", "")).strip()
        self.model = model

    def available(self) -> tuple[bool, str]:
        missing = [n for n, v in (("CODEX_COUNCIL_URL", self.url),
                                  ("CODEX_COUNCIL_TOKEN", self._token))
                   if not v]
        if missing:
            return False, f"{', '.join(missing)} unset. See docs/CODEX_COUNCIL.md."
        return True, "configured"

    def complete(self, prompt: str, timeout: int, role: str = "implement"
                 ) -> Completion:
        ok, why = self.available()
        if not ok:
            raise ProviderError(why)
        payload: dict[str, object] = {"role": role, "prompt": prompt}
        if self.model:
            payload["model"] = self.model
        started = time.time()
        try:
            data = _post_json(self.url, payload,
                              {"Authorization": f"Bearer {self._token}"}, timeout)
        except urllib.error.HTTPError as err:
            if err.code in (401, 403) and err.headers.get("X-Dispatcher") == "codex":
                raise ProviderError(
                    f"dispatcher rejected the token (HTTP {err.code}); "
                    "CODEX_COUNCIL_TOKEN does not match how it was installed"
                ) from None
            raise ProviderError(explain_http(err.code, _read_error(err))) from None
        except urllib.error.URLError as err:
            raise ProviderError(explain_url_error(err)) from None
        if not data.get("ok"):
            raise ProviderError(f"dispatcher: {data.get('error', 'unknown error')}")
        return Completion(text=data.get("answer", ""),
                          model=data.get("model", ""), provider=self.name,
                          duration_s=time.time() - started)


PROVIDERS = {"openrouter": OpenRouter, "codex-dispatcher": CodexDispatcher}


def get(name: str, **kw) -> Provider:
    """Build a provider by name. `auto` picks the first configured one.

    Order matters and is deliberate: openrouter first, because the pool is the
    normal case and the dispatcher is one VM answering one request at a time.
    """
    if name == "auto":
        for candidate in ("openrouter", "codex-dispatcher"):
            p = PROVIDERS[candidate](**_kw_for(candidate, kw))
            if p.available()[0]:
                return p
        raise ProviderError(
            "no provider is configured. Set OPENROUTER_API_KEY, or "
            "CODEX_COUNCIL_URL and CODEX_COUNCIL_TOKEN. See "
            "docs/WORKER_POOL.md.")
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider {name!r}; "
                            f"choose from {sorted(PROVIDERS)} or 'auto'")
    return PROVIDERS[name](**_kw_for(name, kw))


def _kw_for(name: str, kw: dict) -> dict:
    """Only pass a provider the arguments it takes: `model` is common to both,
    the rest are not, and a stray kwarg would be a TypeError at dispatch time
    rather than a clear message here."""
    allowed = {"openrouter": ("model", "base_url", "api_key"),
               "codex-dispatcher": ("url", "token", "model")}[name]
    return {k: v for k, v in kw.items() if k in allowed and v is not None}


def status() -> list[dict]:
    rows = []
    for name in sorted(PROVIDERS):
        p = PROVIDERS[name]()
        ok, why = p.available()
        rows.append({"provider": name, "available": ok, "reason": why,
                     "model": getattr(p, "model", None)})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="which providers are usable, and why not")
    m = sub.add_parser("models", help="list the live OpenRouter catalogue")
    m.add_argument("--grep", help="case-insensitive substring of the model id")
    m.add_argument("--limit", type=int, default=40)
    args = ap.parse_args(argv)

    if args.cmd == "check":
        rows = status()
        print(json.dumps(rows, indent=2))
        return 0 if any(r["available"] for r in rows) else 1

    try:
        catalogue = OpenRouter().models()
    except ProviderError as exc:
        print(f"providers: {exc}", file=sys.stderr)
        return 1
    rows = [m for m in catalogue
            if not args.grep or args.grep.lower() in m.get("id", "").lower()]
    for row in rows[:args.limit]:
        pricing = row.get("pricing") or {}
        print(f"{row.get('id'):<50} in={pricing.get('prompt')} "
              f"out={pricing.get('completion')} ctx={row.get('context_length')}")
    print(f"\n{len(rows)} matching, {len(catalogue)} total.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
