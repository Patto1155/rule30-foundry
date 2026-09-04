"""Offline tests for tools/codex_dispatcher/server.py.

The dispatcher runs on the VM, so CI can never exercise it end to end. What CI
*can* pin down is the part that decides what the caller actually reads back --
the console-output parser -- and the part that decides what gets executed --
argv assembly. Both are pure. Neither needs codex installed.

Importing the module is itself a test: it must not start a server, and with no
CODEX_COUNCIL_TOKEN in the environment it must reject every credential rather
than defaulting to open.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "tools" / "codex_dispatcher"))

import server  # noqa: E402


# Real shape of `codex exec` output, from a 0.153.1 run.
SAMPLE = """\
OpenAI Codex v0.153.1
--------
workdir: /home/user
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: read-only
--------
user
Reply with exactly: Codex works

codex
Codex works

tokens used 2,980
"""


class TestExtractAnswer(unittest.TestCase):
    def test_pulls_the_reply_out_of_a_real_transcript(self):
        self.assertEqual(server.extract_answer(SAMPLE), "Codex works")

    def test_drops_the_banner_and_the_echoed_prompt(self):
        out = server.extract_answer(SAMPLE)
        self.assertNotIn("workdir:", out)
        self.assertNotIn("Reply with exactly", out)

    def test_drops_the_token_trailer(self):
        self.assertNotIn("tokens used", server.extract_answer(SAMPLE))

    def test_multiline_reply_is_kept_whole(self):
        text = SAMPLE.replace("Codex works\n\ntokens used 2,980",
                              "line one\nline two\n\ntokens used 12")
        self.assertEqual(server.extract_answer(text), "line one\nline two")

    def test_unrecognised_shape_returns_everything(self):
        """Conservative on purpose: an over-long answer is noticed, a silently
        truncated one is not."""
        odd = "no marker line anywhere in this output"
        self.assertEqual(server.extract_answer(odd), odd)

    def test_last_marker_wins(self):
        """The word `codex` can appear in the reviewed material itself; the
        reply follows the final marker, not the first."""
        text = ("codex\nthis is the echoed prompt mentioning codex\n"
                "codex\nthe real answer\n\ntokens used 5\n")
        self.assertEqual(server.extract_answer(text), "the real answer")

    def test_empty_reply_falls_back_rather_than_returning_nothing(self):
        text = "codex\n\ntokens used 5\n"
        self.assertTrue(server.extract_answer(text))


class TestBuildCodexArgv(unittest.TestCase):
    ALL = {"sandbox": True, "output_last_message": True}
    NONE: dict[str, bool] = {"sandbox": False, "output_last_message": False}

    def test_is_a_list_with_the_prompt_as_one_element(self):
        """The whole no-shell-injection argument rests on this. The prompt is
        attacker-influenced text; as a single argv element it can never become
        a command, however it is punctuated."""
        argv = server.build_codex_argv("; rm -rf / #", "gpt-5.6-sol", None,
                                       self.NONE)
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[-1], "; rm -rf / #")
        self.assertEqual(sum(1 for a in argv if a == "; rm -rf / #"), 1)

    def test_sandbox_flag_only_when_the_installed_codex_has_it(self):
        with_flag = server.build_codex_argv("p", "m", None, self.ALL)
        without = server.build_codex_argv("p", "m", None, self.NONE)
        self.assertIn("--sandbox", with_flag)
        self.assertEqual(with_flag[with_flag.index("--sandbox") + 1],
                         "read-only")
        self.assertNotIn("--sandbox", without)

    def test_output_last_message_requires_both_path_and_support(self):
        self.assertIn("--output-last-message",
                      server.build_codex_argv("p", "m", "/tmp/x", self.ALL))
        self.assertNotIn("--output-last-message",
                         server.build_codex_argv("p", "m", None, self.ALL))
        self.assertNotIn("--output-last-message",
                         server.build_codex_argv("p", "m", "/tmp/x", self.NONE))

    def test_model_and_git_check_flag_always_present(self):
        argv = server.build_codex_argv("p", "gpt-5.6-sol", None, self.NONE)
        self.assertIn("--skip-git-repo-check", argv)
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-5.6-sol")


class TestTokenOk(unittest.TestCase):
    def test_fails_closed_when_the_server_has_no_token(self):
        """No credential configured must mean "reject everything", never
        "accept anything".

        The empty token is set explicitly rather than read from the ambient
        environment. server.TOKEN is populated from CODEX_COUNCIL_TOKEN at
        import, so an assertion that it is empty holds only while nobody has
        configured a council -- and step 5 of docs/CODEX_COUNCIL.md tells the
        operator to set exactly that variable. This test would then start
        failing on the machine that had just been set up correctly.
        """
        original = server.TOKEN
        server.TOKEN = ""
        try:
            self.assertFalse(server.token_ok("Bearer anything"))
            self.assertFalse(server.token_ok(None))
            self.assertFalse(server.token_ok("Bearer "))
        finally:
            server.TOKEN = original

    def test_matches_only_the_exact_bearer_value(self):
        original = server.TOKEN
        server.TOKEN = "s" * 40
        try:
            self.assertTrue(server.token_ok("Bearer " + "s" * 40))
            self.assertFalse(server.token_ok("Bearer " + "s" * 39))
            self.assertFalse(server.token_ok("s" * 40))       # no scheme
            self.assertFalse(server.token_ok("Basic " + "s" * 40))
        finally:
            server.TOKEN = original


class TestPromptFingerprint(unittest.TestCase):
    def test_is_not_the_prompt(self):
        """Logs on this VM would otherwise become a plaintext archive of every
        claim ever reviewed."""
        secret = "a claim that should not appear in journalctl"
        self.assertNotIn(secret, server.prompt_fingerprint(secret))

    def test_is_stable_and_distinguishing(self):
        self.assertEqual(server.prompt_fingerprint("a"),
                         server.prompt_fingerprint("a"))
        self.assertNotEqual(server.prompt_fingerprint("a"),
                            server.prompt_fingerprint("b"))


class TestRoleValidation(unittest.TestCase):
    def test_accepts_the_client_roles_and_rejects_junk(self):
        for good in ("review", "math", "redteam"):
            self.assertTrue(server.VALID_ROLE.match(good))
        for bad in ("", "../etc", "Review", "a" * 40, "role;rm"):
            self.assertFalse(server.VALID_ROLE.match(bad), bad)


if __name__ == "__main__":
    unittest.main()
