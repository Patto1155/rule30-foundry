"""End-to-end: a task goes in, a real subprocess edits the repo, a branch
comes back.

`docs/WORKHORSE.md` had to say of the previous integration that "`--agent
codex` is unexercised -- the flag is written and its failure path is tested,
but no experiment has been implemented by Codex through it". A code path that
has never run is not an integration. This file is the fix: every stage of
tools/codex_worker.py runs for real here -- `git worktree add`, a subprocess
that edits files on disk, `git diff`, the commit, the verification
subprocesses, the verdict, `git worktree remove`.

The one substitution is the model. CODEX_BIN points at
tests/fixtures/fake_codex.py, which takes the production argv, edits the
production worktree, and prints a report in the production contract. What a
test cannot pin down is what a language model will say; what it can pin down
-- and what these tests do -- is that a real edit reaches a reviewable branch,
and that a *lying* report is caught by verification rather than believed.

For the model end, docs/CODEX_WORKER.md records the live transcript of a
`review`-mode task against the real dispatcher.

Skipped when CODEX_WORKER_NESTED is set: the worker's own verification step
runs verify_all inside the worktree, whose unittest stage discovers this file,
which invokes the worker. workhorse.py documents the same hazard for its
--dry-run path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAKE = REPO / "tests" / "fixtures" / "fake_codex.py"

NESTED = os.environ.get("CODEX_WORKER_NESTED") == "1"


def git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def worker(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CODEX_BIN"] = str(FAKE)
    env.pop("CODEX_WORKER_NESTED", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "tools/codex_worker.py", *args],
                          cwd=REPO, capture_output=True, text=True, env=env)


@unittest.skipIf(NESTED, "would re-enter verify_all through its unittest stage")
class TestEndToEnd(unittest.TestCase):
    """One coding task, submitted the way the lead agent submits one."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cw-e2e-"))
        self.branches: list[str] = []

    def tearDown(self):
        for b in self.branches:
            git("branch", "-D", b)
        git("worktree", "prune")

    def submit(self, task: str, *extra: str, mode: str = "implement",
               verify: str = "fast") -> dict:
        out = self.tmp / f"run{len(self.branches)}"
        r = worker("submit", "--mode", mode, "--task", task,
                   # HEAD, not origin/main: the test must not need a network,
                   # and what it is checking is the pipeline, not the base.
                   "--base", "HEAD", "--verify", verify, "--out", str(out),
                   *extra)
        self.assertTrue(r.stdout.strip(), f"no JSON on stdout.\n{r.stderr}")
        result = json.loads(r.stdout)
        if result.get("branch"):
            self.branches.append(result["branch"])
        result["_returncode"] = r.returncode
        return result

    # ---- the happy path, which is the thing that had never been run --------

    def test_a_coding_task_comes_back_as_a_reviewable_branch(self):
        target = "tools/_e2e_scratch.py"
        result = self.submit(
            "Add a scratch constant.\n"
            f"FAKE-WRITE {target} :: E2E_SCRATCH = 30\n")

        self.assertEqual(result["verdict"], "READY-FOR-REVIEW",
                         json.dumps(result, indent=2)[:2000])
        self.assertEqual(result["backend"], "local")
        self.assertEqual(result["_returncode"], 0)

        # A branch, and it exists in the repository the lead can read.
        self.assertTrue(result["branch"].startswith("codex/implement-"))
        self.assertEqual(git("rev-parse", "--verify", result["branch"]).returncode, 0)

        # A patch, and it is the real diff rather than the agent's account.
        patch = Path(result["patch"]).read_text(encoding="utf-8")
        self.assertIn(f"+++ b/{target}", patch)
        self.assertIn("E2E_SCRATCH = 30", patch)
        self.assertIn(f"A\t{target}", result["files_changed"])

        # The commit is on the branch, and NOT on the lead's checkout.
        show = git("show", f"{result['branch']}:{target}")
        self.assertEqual(show.returncode, 0)
        self.assertIn("E2E_SCRATCH = 30", show.stdout)
        self.assertFalse((REPO / target).exists(),
                         "the worker wrote into the lead's working tree")

        # The structured report came through intact.
        self.assertEqual(result["contract_problems"], [])
        for key in ("summary", "changes", "commands_run", "tests",
                    "artifacts", "uncertainties", "blockers"):
            self.assertIn(key, result["report"])

        # And the worker's own verification ran, and is recorded separately
        # from anything the agent claimed.
        self.assertTrue(result["verification"]["ok"])
        self.assertTrue(result["verification"]["checks"])

    def test_the_working_tree_is_untouched_and_no_worktree_leaks(self):
        before = git("status", "--porcelain").stdout
        worktrees_before = git("worktree", "list").stdout.count("\n")
        self.submit("Touch something.\nFAKE-WRITE tools/_e2e_x.py :: X = 1\n")
        self.assertEqual(git("status", "--porcelain").stdout, before)
        self.assertEqual(git("worktree", "list").stdout.count("\n"),
                         worktrees_before)

    # ---- the dishonest worker, which is the reason for the shape ----------

    def test_a_claimed_pass_does_not_survive_a_failing_acceptance_command(self):
        """The agent reports a green suite it never ran, and the acceptance
        command fails. The verdict must follow the command, not the claim --
        this is the whole reason `tests` and `verification` are separate
        fields."""
        result = self.submit(
            "Do the thing.\nFAKE-WRITE tools/_e2e_y.py :: Y = 1\nFAKE-CLAIM-PASS\n",
            "--acceptance", "exit 3")
        self.assertTrue(result["report"]["tests"][0]["passed"],
                        "fixture should have claimed a pass")
        self.assertFalse(result["verification"]["ok"])
        self.assertEqual(result["verdict"], "NEEDS-ATTENTION")
        self.assertEqual(result["_returncode"], 1)
        failed = [c for c in result["verification"]["checks"] if not c["passed"]]
        self.assertEqual(failed[0]["exit_code"], 3)

    def test_a_declared_blocker_blocks_however_green_everything_else_is(self):
        result = self.submit(
            "Try it.\nFAKE-WRITE tools/_e2e_z.py :: Z = 1\n"
            "FAKE-BLOCKER the GPU is not attached\n")
        self.assertEqual(result["verdict"], "BLOCKED")
        self.assertIn("the GPU is not attached", result["report"]["blockers"])

    def test_a_report_missing_a_required_key_is_not_ready(self):
        result = self.submit(
            "Do it.\nFAKE-WRITE tools/_e2e_w.py :: W = 1\nFAKE-OMIT uncertainties\n")
        self.assertEqual(result["verdict"], "NEEDS-ATTENTION")
        self.assertIn("missing required key 'uncertainties'",
                      result["contract_problems"])

    def test_no_report_at_all_is_recorded_rather_than_inferred(self):
        result = self.submit(
            "Do it quietly.\nFAKE-WRITE tools/_e2e_v.py :: V = 1\nFAKE-NO-JSON\n")
        self.assertIsNone(result["report"])
        self.assertIn("no JSON report found in the reply",
                      result["contract_problems"])
        self.assertEqual(result["verdict"], "NEEDS-ATTENTION")
        # The work still reached a branch: a worker that did something useful
        # and reported it badly should not have its diff thrown away.
        self.assertIsNotNone(result["patch"])

    def test_a_nonzero_exit_is_not_ready(self):
        result = self.submit("Fail.\nFAKE-WRITE tools/_e2e_u.py :: U = 1\n"
                             "FAKE-EXIT 1\n")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["verdict"], "NEEDS-ATTENTION")

    # ---- report-only modes -------------------------------------------------

    def test_investigate_returns_findings_and_an_empty_branch(self):
        result = self.submit("What does gates.py enforce?", mode="investigate")
        self.assertEqual(result["verdict"], "READY-FOR-REVIEW")
        self.assertIsNone(result["patch"])
        self.assertEqual(result["files_changed"], [])
        self.assertTrue(result["report"]["summary"])

    # ---- verification is not optional by accident -------------------------

    def test_verify_none_is_recorded_as_evidence_free(self):
        result = self.submit("Do it.\nFAKE-WRITE tools/_e2e_t.py :: T = 1\n",
                             verify="none")
        self.assertIsNone(result["verification"]["ok"])
        self.assertIn("no evidence", result["verification"]["note"])

    def test_full_verification_runs_verify_all_in_the_worktree(self):
        result = self.submit("Do it.\nFAKE-WRITE tools/_e2e_s.py :: S = 1\n",
                             verify="full")
        cmds = " ".join(c["cmd"] for c in result["verification"]["checks"])
        self.assertIn("tools/verify_all.py", cmds)
        self.assertTrue(result["verification"]["ok"],
                        (REPO / result["out"] / "verification.txt").read_text()
                        if (Path(result["out"]) / "verification.txt").exists()
                        else "verify_all failed in the worktree")


@unittest.skipIf(NESTED, "would re-enter verify_all through its unittest stage")
class TestDryRun(unittest.TestCase):
    def test_dry_run_dispatches_nothing_and_prints_the_prompt(self):
        r = worker("submit", "--mode", "debug", "--task", "why is it red",
                   "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--- task ---", r.stdout)
        self.assertIn("why is it red", r.stdout)
        self.assertEqual(git("worktree", "list").stdout.count("codex-wt-"), 0)


if __name__ == "__main__":
    unittest.main()
