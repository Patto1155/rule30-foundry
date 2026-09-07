"""Unit tests for tools/codex_worker.py: the parts that must be pure.

Prompt assembly, reply parsing, the result contract, and the verdict rule are
all pure functions precisely so they can be pinned down without a VM, a
token, or a model. The pipeline itself is exercised end to end in
tests/test_codex_worker_e2e.py with a real subprocess.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cw = _load("codex_worker", "tools/codex_worker.py")


class TestModes(unittest.TestCase):
    def test_review_is_one_mode_not_the_architecture(self):
        """The correction this file exists for: review is a selectable task
        mode alongside five others, not the shape of the integration."""
        self.assertIn("review", cw.MODES)
        self.assertGreaterEqual(len(cw.MODES), 6)
        for name in ("implement", "debug", "test", "refactor", "investigate"):
            self.assertIn(name, cw.MODES)

    def test_work_modes_expect_a_patch_and_report_modes_do_not(self):
        for name in ("implement", "debug", "test", "refactor"):
            self.assertTrue(cw.MODES[name].expects_patch, name)
        for name in ("investigate", "review"):
            self.assertFalse(cw.MODES[name].expects_patch, name)


class TestPrompt(unittest.TestCase):
    spec = {"mode": "implement", "task": "Add a --json flag to lint_ledger."}

    def test_carries_the_repo_guardrails(self):
        """Same text as the council's, imported rather than copied: two lists
        of known failure modes drift, and the drifted one gets sent."""
        p = cw.render_prompt(self.spec, has_repo=True)
        self.assertIn("log2|M| >= n", p)          # the counting bound
        self.assertIn("np.unpackbits", p)          # bit order
        self.assertIn("single-black-cell", p)      # single seed
        self.assertIn("first_divergence", p)       # the radius-1 invariant

    def test_a_worker_is_not_told_it_is_a_reviewer(self):
        """Same failure-mode list as the council sends, under a lead-in that
        names what this agent is actually doing. Told it is reviewing, a model
        returns a review instead of a patch."""
        p = cw.render_prompt(self.spec, has_repo=True)
        self.assertIn("work you are doing", p)
        self.assertNotIn("work you are reviewing", p)

    def test_checkout_backend_is_told_not_to_commit(self):
        p = cw.render_prompt(self.spec, has_repo=True)
        self.assertIn("git commit", p)
        self.assertIn("worktree", p)

    def test_patchless_backend_is_told_to_return_a_diff(self):
        p = cw.render_prompt(self.spec, has_repo=False)
        self.assertIn("unified diff", p)
        self.assertIn("```diff", p)
        self.assertIn("must be empty lists", p)

    def test_acceptance_commands_are_declared_up_front(self):
        p = cw.render_prompt({**self.spec, "acceptance": ["python -m unittest"]},
                             has_repo=True)
        self.assertIn("regardless of what you report", p)
        self.assertIn("$ python -m unittest", p)

    def test_every_mode_renders(self):
        for name in cw.MODES:
            p = cw.render_prompt({"mode": name, "task": "x"}, has_repo=True)
            self.assertIn(cw.MODES[name].preamble.splitlines()[0], p)


class TestExtractJSON(unittest.TestCase):
    def test_fenced_block(self):
        obj = cw.extract_json('chatter\n```json\n{"summary": "s"}\n```\nmore')
        self.assertEqual(obj, {"summary": "s"})

    def test_last_fenced_block_wins(self):
        """A model that shows a draft and then a final report must be read as
        having meant the final one."""
        text = '```json\n{"summary": "draft"}\n```\n```json\n{"summary": "final"}\n```'
        self.assertEqual(cw.extract_json(text)["summary"], "final")

    def test_bare_object_with_a_summary(self):
        self.assertEqual(cw.extract_json('here:\n{"summary": "s", "changes": []}')
                         ["summary"], "s")

    def test_returns_none_rather_than_guessing(self):
        self.assertIsNone(cw.extract_json("no report here at all"))
        self.assertIsNone(cw.extract_json("{not json}"))

    def test_ignores_incidental_objects(self):
        """A JSON snippet quoted in prose is not a report; without the
        `summary` key the scanner leaves it alone."""
        self.assertIsNone(cw.extract_json('I edited {"a": 1} in the config.'))


class TestExtractPatch(unittest.TestCase):
    def test_fenced_diff(self):
        text = "prose\n```diff\ndiff --git a/x b/x\n+line\n```\n"
        self.assertIn("diff --git a/x b/x", cw.extract_patch(text))

    def test_unfenced_diff_is_still_found(self):
        text = "here you go\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n"
        self.assertTrue(cw.extract_patch(text).startswith("diff --git"))

    def test_none_when_there_is_no_diff(self):
        self.assertIsNone(cw.extract_patch("I changed nothing."))


class TestApplyPatch(unittest.TestCase):
    """The remote backend's edits arrive as a diff, so applying one is the
    step where a good answer is most easily thrown away."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cw-apply-"))
        import subprocess
        self.run = subprocess.run
        for argv in (["git", "init", "-q"],
                     ["git", "config", "user.email", "t@t"],
                     ["git", "config", "user.name", "t"]):
            self.run(argv, cwd=self.tmp, capture_output=True)
        (self.tmp / "f.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        self.run(["git", "add", "-A"], cwd=self.tmp, capture_output=True)
        self.run(["git", "commit", "-qm", "init"], cwd=self.tmp, capture_output=True)

    def write(self, body: str) -> Path:
        p = self.tmp / "p.diff"
        p.write_text(body, encoding="utf-8")
        return p

    def test_a_correct_patch_applies_strictly(self):
        p = self.write("diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n"
                       "@@ -1,3 +1,4 @@\n one\n two\n+two and a half\n three\n")
        applied, strategy, _ = cw.apply_patch(p, self.tmp)
        self.assertTrue(applied)
        self.assertEqual(strategy, "strict")

    def test_wrong_hunk_counts_are_recounted_rather_than_rejected(self):
        """Observed live on 2026-09-07: a model with no line numbers to count
        from wrote correct hunk *content* and wrong `@@` counts, and the
        strict apply threw the whole patch away as corrupt."""
        p = self.write("diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n"
                       "@@ -1,3 +1,3 @@\n one\n two\n+two and a half\n three\n")
        applied, strategy, _ = cw.apply_patch(p, self.tmp)
        self.assertTrue(applied)
        self.assertEqual(strategy, "recount")
        self.assertIn("two and a half",
                      (self.tmp / "f.txt").read_text(encoding="utf-8"))

    def test_a_patch_against_content_that_is_not_there_still_fails(self):
        """The ladder loosens counting, not matching. A hunk whose context
        does not exist must fail rather than land somewhere plausible."""
        p = self.write("diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n"
                       "@@ -1,3 +1,4 @@\n alpha\n beta\n+gamma\n delta\n")
        applied, strategy, errors = cw.apply_patch(p, self.tmp)
        self.assertFalse(applied)
        self.assertIsNone(strategy)
        self.assertEqual(len(errors), len(cw.APPLY_STRATEGIES))

    def test_strategies_are_ordered_strictest_first(self):
        self.assertEqual(cw.APPLY_STRATEGIES[0][0], "strict")


class TestResultContract(unittest.TestCase):
    good = {"summary": "did a thing", "changes": [{"path": "a.py"}],
            "commands_run": [], "tests": [], "artifacts": [],
            "uncertainties": [], "blockers": []}

    def test_accepts_a_complete_report(self):
        self.assertEqual(cw.validate_result(self.good), [])

    def test_uncertainties_and_blockers_are_required_keys(self):
        """An empty list is an assertion that there were none. A missing key
        is a worker that was never asked, and the two must not look alike."""
        for key in ("uncertainties", "blockers"):
            bad = {k: v for k, v in self.good.items() if k != key}
            self.assertIn(f"missing required key {key!r}", cw.validate_result(bad))

    def test_rejects_an_empty_summary(self):
        self.assertIn("summary is empty",
                      cw.validate_result({**self.good, "summary": "  "}))

    def test_rejects_a_change_without_a_path(self):
        problems = cw.validate_result({**self.good, "changes": [{"why": "x"}]})
        self.assertTrue(any("changes[0]" in p for p in problems))

    def test_rejects_a_scalar_where_a_list_belongs(self):
        problems = cw.validate_result({**self.good, "blockers": "the disk"})
        self.assertTrue(any("blockers" in p for p in problems))

    def test_rejects_a_non_object(self):
        self.assertTrue(cw.validate_result(["a list"]))


class TestVerdict(unittest.TestCase):
    ok = {"ok": True}

    def test_clean_run_is_ready_for_review(self):
        self.assertEqual(cw.decide([], self.ok, [], None, 0), cw.READY)

    def test_a_blocker_blocks_even_when_everything_else_is_green(self):
        self.assertEqual(cw.decide([], self.ok, ["no network"], None, 0),
                         cw.BLOCKED)

    def test_a_patch_that_did_not_apply_blocks(self):
        self.assertEqual(cw.decide([], self.ok, [], False, 0), cw.BLOCKED)

    def test_failed_verification_is_never_ready(self):
        self.assertEqual(cw.decide([], {"ok": False}, [], None, 0), cw.ATTENTION)

    def test_a_broken_contract_is_never_ready(self):
        self.assertEqual(cw.decide(["missing key"], self.ok, [], None, 0),
                         cw.ATTENTION)

    def test_a_timed_out_run_is_never_ready(self):
        """A run killed by the budget never produced an exit code. None is
        not zero: reading it as success is how a timeout reports READY."""
        self.assertEqual(cw.decide([], self.ok, [], None, None, True),
                         cw.ATTENTION)
        self.assertEqual(cw.decide([], self.ok, [], None, None), cw.ATTENTION)

    def test_verification_disabled_does_not_pass_as_verified(self):
        """`--verify none` reports ok=None. It must not be readable as a
        pass; the verdict may be READY only because nothing contradicted it,
        and the result field says so in words."""
        self.assertIsNone(cw.verify({}, Path("."), "none", Path("."))["ok"])
        self.assertIn("no evidence",
                      cw.verify({}, Path("."), "none", Path("."))["note"])


class TestVerificationCommands(unittest.TestCase):
    def test_full_runs_verify_all_in_strict_mode(self):
        cmds = cw.verification_commands({}, "full")
        argv = cmds[0]
        self.assertIn("tools/verify_all.py", " ".join(argv))
        # Strict mode: --allow-skip is passed, so any stage that skips
        # *without* being named fails the run. "SKIP is not PASS" has to
        # survive being run by a robot.
        self.assertTrue(any(a.startswith("--allow-skip") for a in argv))
        self.assertTrue(any("bitstream" in a for a in argv))

    def test_acceptance_commands_are_appended(self):
        cmds = cw.verification_commands({"acceptance": ["echo hi"]}, "fast")
        self.assertEqual(cmds[-1][:2], ["/bin/sh", "-c"])
        self.assertEqual(cmds[-1][2], "echo hi")


class TestContext(unittest.TestCase):
    def test_reads_named_files(self):
        text, problems = cw.gather_context(["tools/codex_worker.py"])
        self.assertIn("----- tools/codex_worker.py -----", text)
        self.assertEqual(problems, [])

    def test_missing_file_is_a_problem_not_an_exception(self):
        text, problems = cw.gather_context(["nope/nothing.py"])
        self.assertEqual(text, "")
        self.assertTrue(any("not a file" in p for p in problems))

    def test_refuses_to_send_files_outside_the_repository(self):
        _, problems = cw.gather_context(["../../etc/passwd"])
        self.assertTrue(any("outside the repository" in p for p in problems))

    def test_cap_stops_a_dataset_being_uploaded(self):
        """The failure this prevents is `--context data/center_col_10M.bin`,
        which would otherwise fail after a 40 MB upload through the proxy."""
        _, problems = cw.gather_context(["tools/codex_worker.py"], cap=10)
        self.assertTrue(any("context cap" in p for p in problems))


class TestBackendChoice(unittest.TestCase):
    def test_explicit_local_without_codex_fails_before_a_worktree_exists(self):
        """A missing binary must be a sentence, not a FileNotFoundError from
        inside the run after a worktree has already been created."""
        import os
        old = os.environ.get("CODEX_BIN")
        os.environ["CODEX_BIN"] = "definitely-not-a-real-binary-x9"
        try:
            with self.assertRaises(RuntimeError) as caught:
                cw.choose_backend("local")
            self.assertIn("not on PATH", str(caught.exception))
            self.assertEqual(cw.choose_backend("auto"), "remote")
        finally:
            if old is None:
                del os.environ["CODEX_BIN"]
            else:
                os.environ["CODEX_BIN"] = old


class TestBranchName(unittest.TestCase):
    def test_is_namespaced_slugged_and_unique(self):
        b = cw.branch_name({"mode": "debug", "task": "Fix the GPU kernel!!"}, "ab12cd34")
        self.assertTrue(b.startswith("codex/debug-"))
        self.assertTrue(b.endswith("ab12cd34"))
        self.assertNotIn("!", b)
        self.assertNotIn(" ", b)

    def test_survives_a_task_with_no_usable_characters(self):
        b = cw.branch_name({"mode": "test", "task": "!!! ???"}, "00000000")
        self.assertEqual(b, "codex/test-task-00000000")


class TestSpecValidation(unittest.TestCase):
    def test_unknown_mode_is_refused_before_a_worktree_exists(self):
        with self.assertRaises(ValueError):
            cw.submit({"mode": "vibes", "task": "do something"})

    def test_empty_task_is_refused(self):
        with self.assertRaises(ValueError):
            cw.submit({"mode": "implement", "task": "   "})


class TestPreflightRefusal(unittest.TestCase):
    def test_a_vacuous_manifest_is_refused_before_dispatch(self):
        """The 2026-08 retraction was an experiment that ran when it should
        have been refused. Delegation must not be the way around that gate:
        a manifest below the counting bound never reaches Codex at all."""
        tmp = Path(tempfile.mkdtemp())
        manifest = json.loads(
            (REPO / "queue" / "trap-vacuous-dfao.json").read_text(encoding="utf-8"))
        result = cw.submit({"mode": "implement", "task": "run the trap",
                            "manifest": manifest}, out_dir=tmp)
        self.assertEqual(result["verdict"], cw.REFUSED)
        self.assertIsNone(result["branch"])
        self.assertNotIn("backend", result)
        self.assertEqual(result["preflight"]["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
