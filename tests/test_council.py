"""Offline tests for tools/council.py.

Everything here runs without a network, a token, or a dispatcher: the point is
that the parts which are easy to get quietly wrong -- what the prompt actually
says, whether the bearer token can leak into an error path, and whether a
policy denial is diagnosed correctly -- are checked on every CI run, long
before anyone stands the VM up.

The request is never sent. `build_request` returns a urllib Request that these
tests inspect, which is why it is separate from `ask`.
"""

from __future__ import annotations

import json
import sys
import unittest
import urllib.error
from email.message import Message
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import council  # noqa: E402


class TestRenderPrompt(unittest.TestCase):
    def test_every_role_carries_the_repo_guardrails(self):
        """A reviewer that does not know the repo's failure modes re-derives
        generic advice. The guardrails are the whole value, so every role gets
        them -- not just redteam."""
        for role in council.ROLE_PREAMBLES:
            with self.subTest(role=role):
                out = council.render_prompt(role, "some claim")
                self.assertIn("log2|M| >= n", out)
                self.assertIn("LSB-first", out)
                self.assertIn("single-black-cell", out)

    def test_counting_bound_threshold_matches_the_repo_tool(self):
        """experiments/counting_bound.py marks `margin >= 0` informative, so
        log2|M| == n is informative, not vacuous. Stating the threshold as
        strict `>` would have every role reject boundary-case evidence for a
        mathematically false reason -- and Experiment S sits at that boundary.
        """
        out = council.render_prompt("redteam", "x")
        self.assertIn("log2|M| >= n", out)
        self.assertIn("Equality is informative", out)
        self.assertNotIn("must exceed n", out)

    def test_role_preamble_and_body_both_present(self):
        out = council.render_prompt("math", "the claim under test")
        self.assertIn("checking a mathematical argument", out)
        self.assertIn("the claim under test", out)

    def test_roles_are_distinguishable(self):
        """If two roles rendered the same prompt the dict would be decorative,
        which is the flaw this replaced."""
        rendered = {r: council.render_prompt(r, "x")
                    for r in council.ROLE_PREAMBLES}
        self.assertEqual(len(set(rendered.values())), len(rendered))

    def test_unknown_role_raises(self):
        with self.assertRaises(KeyError):
            council.render_prompt("cheerleader", "x")

    def test_body_is_stripped_but_not_reformatted(self):
        out = council.render_prompt("review", "  line one\n  line two  \n")
        self.assertIn("line one\n  line two", out)


class TestBuildRequest(unittest.TestCase):
    URL = "https://example.invalid/ask"

    def test_token_is_a_header_never_the_body(self):
        """A token in the JSON body ends up in dispatcher logs; a token in
        argv ends up in `ps`. It belongs in exactly one place."""
        req = council.build_request(self.URL, "s3cret", "review", "claim")
        self.assertEqual(req.get_header("Authorization"), "Bearer s3cret")
        self.assertNotIn("s3cret", req.data.decode("utf-8"))

    def test_method_and_content_type(self):
        req = council.build_request(self.URL, "t", "review", "claim")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Content-type"), "application/json")

    def test_body_is_json_with_rendered_prompt(self):
        req = council.build_request(self.URL, "t", "redteam", "claim body")
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["role"], "redteam")
        self.assertIn("claim body", payload["prompt"])
        self.assertNotIn("model", payload)

    def test_model_included_only_when_given(self):
        req = council.build_request(self.URL, "t", "math", "b", model="gpt-5.6-sol")
        self.assertEqual(json.loads(req.data.decode("utf-8"))["model"],
                         "gpt-5.6-sol")


class TestRedact(unittest.TestCase):
    def test_token_removed_from_error_text(self):
        msg = council.redact("failed to auth with abc123", "abc123")
        self.assertNotIn("abc123", msg)
        self.assertIn("<redacted>", msg)

    def test_no_token_configured_is_a_passthrough(self):
        self.assertEqual(council.redact("plain error", None), "plain error")

    def test_empty_token_does_not_redact_everything(self):
        """`"" in text` is always True -- guard against turning every error
        message into a wall of <redacted>."""
        self.assertEqual(council.redact("plain error", ""), "plain error")


class TestHealthUrl(unittest.TestCase):
    def test_ask_endpoint_maps_to_health(self):
        self.assertEqual(council.health_url("https://h.invalid/ask"),
                         "https://h.invalid/health")

    def test_trailing_slash_tolerated(self):
        self.assertEqual(council.health_url("https://h.invalid/ask/"),
                         "https://h.invalid/health")

    def test_bare_host_gets_health_appended(self):
        self.assertEqual(council.health_url("https://h.invalid"),
                         "https://h.invalid/health")


def _http_error(code: str | int, headers: dict[str, str] | None = None):
    msg = Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError("https://h.invalid/ask", code, "reason",
                                  msg, None)


class TestExplainHttpError(unittest.TestCase):
    def test_bare_403_is_diagnosed_as_an_egress_policy_denial(self):
        """The failure everyone will actually hit first. A 403 from the egress
        proxy means the host is not on the environment's allowlist, and the
        message must say so rather than implying a retry would help."""
        out = council.explain_http_error(_http_error(403))
        self.assertIn("allowlist", out)
        self.assertIn("not a transient error", out)

    def test_403_from_the_dispatcher_is_a_token_problem_instead(self):
        out = council.explain_http_error(
            _http_error(403, {"X-Dispatcher": "codex"}))
        self.assertIn("CODEX_COUNCIL_TOKEN", out)
        self.assertNotIn("allowlist", out)

    def test_401_from_the_dispatcher_is_a_token_problem(self):
        out = council.explain_http_error(
            _http_error(401, {"X-Dispatcher": "codex"}))
        self.assertIn("CODEX_COUNCIL_TOKEN", out)

    def test_413_and_504_are_named(self):
        self.assertIn("too large", council.explain_http_error(_http_error(413)))
        self.assertIn("timeout", council.explain_http_error(_http_error(504)))


class TestNoRedirect(unittest.TestCase):
    """urllib's default redirect handler copies every header except
    content-length and content-type onto the redirected request. Authorization
    is not excluded, so following a redirect hands the bearer token to whatever
    answers -- silently, with no error and no warning."""

    def _handler_call(self, code: int, newurl: str):
        req = council.build_request("https://h.invalid/ask", "s3cret",
                                    "review", "claim")
        msg = Message()
        return council.NoRedirect().redirect_request(
            req, None, code, "Found", msg, newurl)

    def test_cross_origin_redirect_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError):
            self._handler_call(302, "https://attacker.invalid/collect")

    def test_same_origin_redirect_is_also_refused(self):
        """Nothing in this protocol should redirect at all, so there is no
        same-origin exception to reason about later."""
        with self.assertRaises(urllib.error.HTTPError):
            self._handler_call(301, "https://h.invalid/ask/")

    def test_refusal_names_the_destination_and_the_reason(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._handler_call(307, "https://attacker.invalid/collect")
        text = str(ctx.exception)
        self.assertIn("attacker.invalid", text)
        self.assertIn("bearer token", text)

    def test_the_opener_installs_it(self):
        opener = council._opener()
        self.assertTrue(
            any(isinstance(h, council.NoRedirect) for h in opener.handlers))

    def test_the_opener_keeps_proxy_support(self):
        """build_opener's defaults include ProxyHandler, which is what routes
        this through HTTPS_PROXY. Replacing the redirect handler must not cost
        us that."""
        import urllib.request as ur
        opener = council._opener()
        self.assertTrue(
            any(isinstance(h, ur.ProxyHandler) for h in opener.handlers))


class TestLimits(unittest.TestCase):
    def test_prompt_cap_is_well_under_arg_max(self):
        """The dispatcher passes the prompt to codex as an argv element, so the
        cap has to leave room under a typical ~2 MB ARG_MAX."""
        self.assertLessEqual(council.MAX_PROMPT_BYTES, 1_000_000)


if __name__ == "__main__":
    unittest.main()
