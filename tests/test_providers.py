"""Tests for tools/providers.py, against a stand-in HTTP server.

The OpenRouter provider is exercised for real -- a socket, a request with
headers, a JSON response, retries, the error paths -- by pointing
`base_url` at a local server that speaks the same shapes. That tests the
client; mocking urllib would test the mock.

Nothing here reaches the network, and nothing here needs a key.
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pv = _load("providers", "tools/providers.py")


class Stub(BaseHTTPRequestHandler):
    """A stand-in OpenRouter. `script` is consumed one response per request,
    so a retry sequence can be written down in the test that needs it."""

    script: list = []
    seen: list = []

    def log_message(self, *a):  # silence
        pass

    def _respond(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        Stub.seen.append({"path": self.path,
                          "headers": dict(self.headers),
                          "body": json.loads(self.rfile.read(n) or b"{}")})
        code, payload = Stub.script.pop(0) if Stub.script else (200, {})
        self._respond(code, payload)

    def do_GET(self):
        Stub.seen.append({"path": self.path, "headers": dict(self.headers)})
        code, payload = Stub.script.pop(0) if Stub.script else (200, {})
        self._respond(code, payload)


def completion(text: str, model: str = "stub/model") -> dict:
    return {"model": model, "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}}


class ServerCase(unittest.TestCase):
    def setUp(self):
        # urllib picks up HTTPS_PROXY/HTTP_PROXY from the environment, and
        # this container has one. Without this the request to 127.0.0.1 would
        # be sent through the agent proxy, which denies it -- the test would
        # fail for a reason that has nothing to do with the code under test.
        self._no_proxy = os.environ.get("no_proxy")
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        Stub.script, Stub.seen = [], []
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}/api/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        # Backoff is 2**attempt seconds in production; a retry test would
        # otherwise take six seconds.
        self._backoff = pv.BACKOFF_BASE_S
        pv.BACKOFF_BASE_S = 1.0

    def tearDown(self):
        pv.BACKOFF_BASE_S = self._backoff
        self.srv.shutdown()
        self.srv.server_close()
        if self._no_proxy is None:
            os.environ.pop("no_proxy", None)
        else:
            os.environ["no_proxy"] = self._no_proxy

    def provider(self, **kw):
        return pv.OpenRouter(base_url=self.base, api_key="test-key", **kw)


class TestOpenRouterHappyPath(ServerCase):
    def test_returns_the_message_content(self):
        Stub.script = [(200, completion("hello from the worker"))]
        c = self.provider(model="stub/model").complete("do a thing", 30)
        self.assertTrue(c.ok)
        self.assertEqual(c.text, "hello from the worker")
        self.assertEqual(c.model, "stub/model")
        self.assertEqual(c.usage["completion_tokens"], 20)

    def test_sends_the_key_as_a_bearer_header_and_never_in_the_body(self):
        Stub.script = [(200, completion("x"))]
        self.provider().complete("prompt text", 30)
        seen = Stub.seen[0]
        self.assertEqual(seen["headers"]["Authorization"], "Bearer test-key")
        self.assertNotIn("test-key", json.dumps(seen["body"]))

    def test_sends_the_prompt_as_a_single_user_message(self):
        Stub.script = [(200, completion("x"))]
        self.provider(model="m/1").complete("prompt text", 30)
        body = Stub.seen[0]["body"]
        self.assertEqual(body["model"], "m/1")
        self.assertEqual(body["messages"],
                         [{"role": "user", "content": "prompt text"}])

    def test_hits_the_chat_completions_path(self):
        Stub.script = [(200, completion("x"))]
        self.provider().complete("p", 30)
        self.assertEqual(Stub.seen[0]["path"], "/api/v1/chat/completions")


class TestOpenRouterFailures(ServerCase):
    def test_retries_a_429_then_succeeds(self):
        Stub.script = [(429, {"error": {"message": "slow down"}}),
                       (200, completion("second time lucky"))]
        c = self.provider().complete("p", 30)
        self.assertEqual(c.text, "second time lucky")
        self.assertEqual(c.attempts, 2)

    def test_does_not_retry_a_400(self):
        """Retrying 'no' three times is three times the wrong request."""
        Stub.script = [(400, {"error": {"message": "bad model"}}),
                       (200, completion("never reached"))]
        with self.assertRaises(pv.ProviderError):
            self.provider().complete("p", 30)
        self.assertEqual(len(Stub.seen), 1)

    def test_an_error_object_in_a_200_is_not_read_as_an_empty_answer(self):
        """OpenRouter reports some upstream failures as HTTP 200 with an
        error object. Reading only the status would hand the worker an empty
        completion, which it would report as a malformed worker rather than a
        provider outage."""
        Stub.script = [(200, {"error": {"message": "upstream is down"}})]
        with self.assertRaises(pv.ProviderError) as caught:
            self.provider().complete("p", 30)
        self.assertIn("upstream is down", str(caught.exception))

    def test_no_choices_is_an_error_not_an_empty_string(self):
        Stub.script = [(200, {"model": "m", "choices": []})]
        with self.assertRaises(pv.ProviderError):
            self.provider().complete("p", 30)

    def test_401_names_the_key_as_the_problem(self):
        Stub.script = [(401, {"error": {"message": "no auth credentials"}})]
        with self.assertRaises(pv.ProviderError) as caught:
            self.provider().complete("p", 30)
        self.assertIn("OPENROUTER_API_KEY", str(caught.exception))

    def test_404_points_at_the_model_catalogue(self):
        Stub.script = [(404, {"error": {"message": "not found"}})]
        with self.assertRaises(pv.ProviderError) as caught:
            self.provider().complete("p", 30)
        self.assertIn("providers.py models", str(caught.exception))

    def test_402_names_credit(self):
        Stub.script = [(402, {"error": {"message": "insufficient credits"}})]
        with self.assertRaises(pv.ProviderError) as caught:
            self.provider().complete("p", 30)
        self.assertIn("out of credit", str(caught.exception))


class TestModels(ServerCase):
    def test_lists_the_catalogue(self):
        Stub.script = [(200, {"data": [{"id": "deepseek/x"}, {"id": "other/y"}]})]
        ids = [m["id"] for m in self.provider().models()]
        self.assertEqual(ids, ["deepseek/x", "other/y"])


class TestAvailability(unittest.TestCase):
    def test_openrouter_needs_a_key_and_says_so(self):
        ok, why = pv.OpenRouter(api_key="").available()
        self.assertFalse(ok)
        self.assertIn("OPENROUTER_API_KEY", why)

    def test_dispatcher_names_every_missing_variable(self):
        ok, why = pv.CodexDispatcher(url="", token="").available()
        self.assertFalse(ok)
        self.assertIn("CODEX_COUNCIL_URL", why)
        self.assertIn("CODEX_COUNCIL_TOKEN", why)

    def test_model_comes_from_the_environment_before_the_default(self):
        old = os.environ.get("OPENROUTER_MODEL")
        os.environ["OPENROUTER_MODEL"] = "vendor/from-env"
        try:
            self.assertEqual(pv.OpenRouter(api_key="k").model, "vendor/from-env")
        finally:
            if old is None:
                os.environ.pop("OPENROUTER_MODEL")
            else:
                os.environ["OPENROUTER_MODEL"] = old

    def test_an_explicit_model_beats_the_environment(self):
        self.assertEqual(pv.OpenRouter(model="a/b", api_key="k").model, "a/b")


class TestSelection(unittest.TestCase):
    def test_unknown_provider_lists_the_real_ones(self):
        with self.assertRaises(pv.ProviderError) as caught:
            pv.get("nope")
        self.assertIn("openrouter", str(caught.exception))

    def test_auto_prefers_openrouter_when_it_is_configured(self):
        old = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "k"
        try:
            self.assertEqual(pv.get("auto").name, "openrouter")
        finally:
            if old is None:
                os.environ.pop("OPENROUTER_API_KEY")
            else:
                os.environ["OPENROUTER_API_KEY"] = old

    def test_kwargs_are_filtered_per_provider(self):
        """`base_url` means something to openrouter and nothing to the
        dispatcher; passing it through would be a TypeError at dispatch time
        rather than a clear message."""
        self.assertNotIn("base_url", pv._kw_for("codex-dispatcher",
                                                {"base_url": "x", "model": "m"}))
        self.assertIn("model", pv._kw_for("codex-dispatcher", {"model": "m"}))


class TestErrorExplanations(unittest.TestCase):
    def test_a_proxy_denial_is_named_as_a_policy_denial(self):
        """From inside a default-deny container these look identical to a
        provider outage, and the remedies are opposite."""
        import urllib.error
        msg = pv.explain_url_error(
            urllib.error.URLError("CONNECT tunnel failed, response 403"))
        self.assertIn("network allowlist", msg)
        self.assertIn("policy denial", msg)

    def test_an_ordinary_failure_is_not_blamed_on_the_proxy(self):
        import urllib.error
        msg = pv.explain_url_error(urllib.error.URLError("timed out"))
        self.assertNotIn("allowlist", msg)


class TestTransport(unittest.TestCase):
    def test_reuses_the_opener_that_refuses_redirects(self):
        """A 302 would make urllib rebuild the request for the new URL and
        carry the Authorization header along -- handing the OpenRouter key to
        whatever answered. council.py already solved this; providers must not
        quietly open a laxer door."""
        opener = pv._council()._opener()
        self.assertTrue(any(h.__class__.__name__ == "NoRedirect"
                            for h in opener.handlers))


if __name__ == "__main__":
    unittest.main()
