#!/usr/bin/env python
"""Ask an independent model (Codex, on the dispatcher VM) to review a claim.

Why this exists: every grade in `docs/CLAIM_LEDGER.md` is currently produced
and checked by the same agent lineage. That is a weak independence argument.
A second model with different training and no stake in the repo's prior
conclusions is a cheap adversarial check -- not authority, but a source of
disagreement worth explaining.

Transport is deliberately boring: HTTPS POST to a small dispatcher on a VM
that runs `codex exec` locally. It is NOT ssh. This container's egress is a
default-deny allowlist (an unlisted host answers 403 to CONNECT, and raw
TCP/22 is not on it), so the dispatcher host must be added to the environment's
network policy before any of this works. See `docs/CODEX_COUNCIL.md`.

Configuration, both required, both from the environment (never a flag -- a
token in argv is a token in `ps` output):

    CODEX_COUNCIL_URL     https://<host>/ask
    CODEX_COUNCIL_TOKEN   the bearer token the dispatcher was installed with

Usage:
    echo "<claim>" | python tools/council.py redteam
    python tools/council.py review --file docs/experiment-logs/foo.md
    python tools/council.py math --json < argument.txt
    python tools/council.py --check          # health probe, no token needed

A role is a *prompt framing*, not a different model. The sketch this replaced
mapped all three roles to the same model id, which made the dict decorative.
What actually changes the answer is the preamble, so that is what a role is.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Client-side cap. The dispatcher enforces its own, lower or equal; this one
# exists so an accidental `cat data/center_col_10M.bin` fails here rather than
# after a 40 MB upload through the proxy.
MAX_PROMPT_BYTES = 100_000

DEFAULT_TIMEOUT_S = 900

# Appended to every role. These are the failure modes that have actually cost
# this repo months (CLAUDE.md, AGENTS.md "Implementation Guardrails"). A
# reviewer who does not know them re-derives generic advice; one who does can
# check the specific things that went wrong before.
REPO_GUARDRAILS = """\
Context on the repository whose work you are reviewing (rule30-foundry, \
empirical work on Wolfram's three Rule 30 prize problems). These are its \
known, expensive failure modes. Check for them explicitly, and say so when \
one applies:

1. Negative results from an underpowered search class. Before any "we searched
   class M and found no fit" conclusion, log2|M| >= n must hold; strictly below
   that the negative is guaranteed by counting alone and carries no
   information. Equality is informative, not vacuous -- at log2|M| = n the
   class has 2^n members and a no-fit outcome is not forced by cardinality, so
   do not reject boundary-case evidence. A certificate was retracted in 2026-08
   for running an experiment strictly below the threshold.
2. Bit order. Packed center-column bitstreams are written LSB-first; NumPy's
   default unpack is MSB-first. A bare np.unpackbits reverses every 8-bit block
   -- ~49.95% of positions differ while the bit mean is identical, so no
   aggregate check catches it.
3. Single seed. All three prizes concern the one deterministic single-black-cell
   initial condition. An ensemble average or random-initial-condition quantity
   is not progress on them, however well measured.
4. A ~50% bit difference between two streams is never a kernel bug -- it means
   the streams are uncorrelated, i.e. a packing or seed mismatch. A real kernel
   bug diverges late.
5. Right-censoring. "Never reached within N steps" is not "never". Reject the
   unqualified form.
6. Noise floors. A near-zero metric is not evidence of asymmetry or structure
   until a baseline has been defined and measured.
7. In a radius-1 cellular automaton, first_divergence < distance is impossible.
   Treat any such number as a hard failure, not a curiosity.

Do not defer to the framing you were given. If the claim is fine, say it is
fine and say what would falsify it. If it is not, name the specific step that
fails rather than listing generic caveats."""

ROLE_PREAMBLES = {
    "review": """\
You are an independent reviewer. You did not write this work and you have no \
stake in its conclusions. Assess whether the claim is supported by the \
evidence offered, at the strength claimed. Distinguish clearly between: proved,
empirically supported on the stated range, and asserted. Where the grade is
overstated, say what the honest grade would be.""",
    "math": """\
You are checking a mathematical argument for correctness. Work through it \
step by step. If it is valid, say so and identify which hypothesis is doing \
the real work. If it is invalid, name the exact step that fails and why -- not \
a general concern, the specific inference. If it is valid but proves something \
weaker than advertised, state precisely what it does prove.""",
    "redteam": """\
Your job is to break this claim. Assume it is wrong and find the reason. \
Prioritise, in order: a methodological error that would produce this result \
from nothing; an artifact of data handling; an overstated conclusion that the \
evidence does not reach; a gap between the quantity measured and the quantity \
the prize problem actually asks about. If after genuine effort you cannot \
break it, say so explicitly and describe the single experiment most likely to.""",
}


def render_prompt(role: str, body: str) -> str:
    """Assemble the full prompt sent to the dispatcher.

    Pure, so the wording can be tested without a network or a VM.
    """
    if role not in ROLE_PREAMBLES:
        raise KeyError(role)
    return (f"{ROLE_PREAMBLES[role]}\n\n{REPO_GUARDRAILS}\n\n"
            f"--- material under review ---\n\n{body.strip()}\n")


def build_request(url: str, token: str, role: str, body: str,
                  model: str | None = None) -> urllib.request.Request:
    """Build the POST. Separated from sending so tests can inspect it.

    The token goes in a header and never into argv, a log line, or the JSON
    body -- see redact() for the error path.
    """
    payload: dict[str, object] = {"role": role, "prompt": render_prompt(role, body)}
    if model:
        payload["model"] = model
    raw = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )


def redact(text: str, token: str | None) -> str:
    """Never let the bearer token reach stderr, a log, or a PR comment."""
    if token and token in text:
        text = text.replace(token, "<redacted>")
    return text


def _ssl_context() -> ssl.SSLContext:
    """Default context, which honours SSL_CERT_FILE.

    The agent proxy re-terminates TLS, so the certificate this client sees is
    the proxy's, signed by the CA at /root/.ccr/ca-bundle.crt -- which
    SSL_CERT_FILE already points at. Building the context explicitly documents
    that this is load-bearing rather than incidental, and keeps anyone from
    "fixing" a verification error by disabling verification.
    """
    return ssl.create_default_context()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect rather than forwarding the bearer token.

    urllib's default redirect handler rebuilds the request for the new URL and
    copies every header across except content-length and content-type -- see
    CONTENT_HEADERS in urllib.request.HTTPRedirectHandler.redirect_request.
    Authorization is not in that list, so a 302 pointing anywhere at all hands
    CODEX_COUNCIL_TOKEN to whatever answers, no warning and no error.

    Checking that the origin is unchanged would be enough, but nothing in this
    protocol has any business redirecting: the client talks to one endpoint it
    was configured with, and nginx's only redirect is :80 -> :443, which a
    correctly configured CODEX_COUNCIL_URL never hits. So refuse the lot; a
    redirect here means something is wrong and should be read, not followed.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code,
            f"refusing to follow a {code} redirect to {newurl}: urllib would "
            "forward the bearer token to that origin",
            headers, fp)


def _opener() -> urllib.request.OpenerDirector:
    """Opener with redirects refused and the proxy handling left intact.

    build_opener keeps its default handlers -- ProxyHandler among them, which
    is what routes this through HTTPS_PROXY -- and swaps in the handlers passed
    here for those of the same class.
    """
    return urllib.request.build_opener(
        NoRedirect(), urllib.request.HTTPSHandler(context=_ssl_context()))


def explain_http_error(err: urllib.error.HTTPError) -> str:
    if err.code in (401, 403) and err.headers.get("X-Dispatcher") == "codex":
        return ("dispatcher rejected the token (HTTP %d). CODEX_COUNCIL_TOKEN "
                "does not match the value the service was installed with."
                % err.code)
    if err.code == 403:
        return ("HTTP 403. If this came from the egress proxy rather than the "
                "dispatcher, the host is not on this environment's network "
                "allowlist -- that is a policy denial, not a transient error. "
                "Add the host to the environment's network policy and start a "
                "new session; do not route around it. "
                "Confirm with: curl -sS \"$HTTPS_PROXY/__agentproxy/status\"")
    if err.code == 413:
        return "prompt too large for the dispatcher's limit (HTTP 413)."
    if err.code == 504:
        return "codex exec exceeded the dispatcher's timeout (HTTP 504)."
    return f"HTTP {err.code} {err.reason}"


def ask(url: str, token: str, role: str, body: str, model: str | None,
        timeout: int) -> dict:
    req = build_request(url, token, role, body, model)
    with _opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health_url(url: str) -> str:
    """Derive the /health endpoint from the configured /ask endpoint.

    Pure, so the mapping is tested rather than assumed. Both spellings of the
    configured value are accepted, because whether a URL keeps its trailing
    slash is exactly the sort of thing that differs between the runbook and
    what someone actually pastes into an environment variable.
    """
    base = url[:-len("/ask")] if url.rstrip("/").endswith("/ask") else url
    return base.rstrip("/") + "/health"


def health(url: str, timeout: int = 30) -> dict:
    """GET /health on the dispatcher. Unauthenticated by design.

    This is the probe to run first after an allowlist change: it separates
    "the network policy now permits this host" from "the token is right",
    which otherwise fail in ways that look alike.
    """
    req = urllib.request.Request(health_url(url), method="GET")
    with _opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", nargs="?", choices=sorted(ROLE_PREAMBLES),
                    help="prompt framing to apply")
    ap.add_argument("--file", type=Path,
                    help="read the material from a file instead of stdin")
    ap.add_argument("--model", help="override the dispatcher's default model")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                    metavar="S", help=f"seconds (default {DEFAULT_TIMEOUT_S})")
    ap.add_argument("--json", action="store_true",
                    help="print the full response envelope, not just the answer")
    ap.add_argument("--check", action="store_true",
                    help="probe /health and exit; needs URL but not a token")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the assembled prompt and exit, sending nothing")
    args = ap.parse_args()

    url = os.environ.get("CODEX_COUNCIL_URL", "").strip()
    token = os.environ.get("CODEX_COUNCIL_TOKEN", "").strip()

    if args.check:
        if not url:
            print("CODEX_COUNCIL_URL is not set.", file=sys.stderr)
            return 2
        try:
            print(json.dumps(health(url), indent=2))
            return 0
        except urllib.error.HTTPError as err:
            print(f"health probe failed: {explain_http_error(err)}",
                  file=sys.stderr)
            return 1
        except urllib.error.URLError as err:
            print(f"health probe failed: {redact(str(err.reason), token)}",
                  file=sys.stderr)
            return 1

    if not args.role:
        ap.error("a role is required unless --check is given")

    body = (args.file.read_text(encoding="utf-8") if args.file
            else sys.stdin.read())
    if not body.strip():
        print("nothing to review: stdin was empty and --file was not given.",
              file=sys.stderr)
        return 2

    if args.dry_run:
        print(render_prompt(args.role, body))
        return 0

    encoded = len(render_prompt(args.role, body).encode("utf-8"))
    if encoded > MAX_PROMPT_BYTES:
        print(f"prompt is {encoded} bytes, over the {MAX_PROMPT_BYTES} limit. "
              "Send the argument, not the dataset.", file=sys.stderr)
        return 2

    missing = [n for n, v in (("CODEX_COUNCIL_URL", url),
                              ("CODEX_COUNCIL_TOKEN", token)) if not v]
    if missing:
        print(f"not configured: {', '.join(missing)} unset. "
              "See docs/CODEX_COUNCIL.md.", file=sys.stderr)
        return 2

    try:
        result = ask(url, token, args.role, body, args.model, args.timeout)
    except urllib.error.HTTPError as err:
        print(f"council request failed: {explain_http_error(err)}",
              file=sys.stderr)
        return 1
    except urllib.error.URLError as err:
        print(f"council request failed: {redact(str(err.reason), token)}",
              file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result.get("answer", "").rstrip())
        meta = (f"[{result.get('role')} · {result.get('model')} · "
                f"{result.get('duration_s', 0):.1f}s]")
        print(meta, file=sys.stderr)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
