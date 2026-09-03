#!/usr/bin/env python3
"""Guard the council's config block and its CI-visible constraints.

`tools/council.py` is the one tool here whose failure mode is *at dispatch
time*, minutes into a session, after a brief has been written. These tests
move the cheap half of that failure to `python tools/verify_all.py`:

  - the module must import with numpy-only dependencies (requirements-ci.txt),
    so `mcp` and `httpx` must stay out of the import path;
  - every role must name a backend that exists, so a typo in ROLES is a test
    failure rather than a KeyError after the brief is written;
  - an empty or unknown-role dispatch must be refused before it spends a token.

Nothing here makes a network call: CI has no ChatGPT subscription and must not
pretend to.
"""

import io
import pathlib
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import council  # noqa: E402


class TestRoleTable(unittest.TestCase):
    def test_every_role_names_a_real_backend(self):
        for role, cfg in council.ROLES.items():
            with self.subTest(role=role):
                self.assertIn(cfg.get("backend"), council.BACKENDS)
                self.assertTrue(cfg.get("model"), f"{role} has no model")

    def test_codex_roles_declare_a_known_reasoning_effort(self):
        allowed = {"minimal", "low", "medium", "high", "xhigh"}
        for role, cfg in council.ROLES.items():
            if cfg["backend"] == "codex":
                with self.subTest(role=role):
                    self.assertIn(cfg.get("reasoning", "medium"), allowed)

    def test_timeouts_are_sane(self):
        for role, cfg in council.ROLES.items():
            with self.subTest(role=role):
                self.assertGreater(cfg.get("timeout", council.DEFAULT_TIMEOUT), 60)


class TestDispatchGuards(unittest.TestCase):
    def test_unknown_role_is_refused(self):
        with self.assertRaises(council.CouncilError):
            council.ask("no-such-role", "hello")

    def test_empty_brief_is_refused(self):
        role = next(iter(council.ROLES))
        with self.assertRaises(council.CouncilError):
            council.ask(role, "   \n\t ")

    def test_openrouter_rejects_repo_access(self):
        """A brief that assumed repo access must fail, not silently lose it."""
        openrouter = [r for r, c in council.ROLES.items() if c["backend"] == "openrouter"]
        if not openrouter:
            self.skipTest("no openrouter-backed role configured")
        with self.assertRaises(council.CouncilError):
            council.run_openrouter(council.ROLES[openrouter[0]], "brief", repo=True)

    def test_bad_base64_seed_is_reported_clearly(self):
        """A mangled CODEX_AUTH_B64 paste must say so, not write a broken file."""
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old_home, old_blob = council.CODEX_HOME, os.environ.get("CODEX_AUTH_B64")
            council.CODEX_HOME = pathlib.Path(tmp) / ".codex"
            os.environ["CODEX_AUTH_B64"] = "not base64 !!"
            try:
                with self.assertRaises(council.CouncilError) as ctx:
                    council.ensure_codex_auth()
                self.assertIn("base64", str(ctx.exception))
            finally:
                council.CODEX_HOME = old_home
                if old_blob is None:
                    os.environ.pop("CODEX_AUTH_B64", None)
                else:
                    os.environ["CODEX_AUTH_B64"] = old_blob


class TestCli(unittest.TestCase):
    def test_roles_subcommand_is_json(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(council.main(["roles"]), 0)
        import json
        self.assertEqual(set(json.loads(buf.getvalue())), set(council.ROLES))

    def test_module_imports_without_optional_deps(self):
        """The CLI path must be stdlib-only: CI installs numpy and nothing else."""
        proc = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.modules['mcp'] = None; "
             "sys.path.insert(0, %r); import council; council.main(['roles'])" % str(REPO_ROOT / "tools")],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_setup_script_is_valid_bash(self):
        script = REPO_ROOT / "tools" / "council_env_setup.sh"
        self.assertTrue(script.exists())
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
