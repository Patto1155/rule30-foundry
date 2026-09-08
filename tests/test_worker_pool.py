"""tools/worker_pool.py, run for real across concurrent workers.

Every task here goes through the whole pipeline -- worktree, subprocess,
`git diff`, commit, verification, verdict -- with CODEX_BIN pointing at
tests/fixtures/fake_codex.py. The substitution is the model; the concurrency,
the git locking, and the aggregation are the production code.

What this is for: N tasks at once against ONE `.git` is where a fan-out
breaks. Two `git worktree add` calls racing on the worktree registry is not a
failure a single-task test can produce.

Skipped when CODEX_WORKER_NESTED is set: verification runs verify_all inside
each worktree, whose unittest stage discovers this file.
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


def pool(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CODEX_BIN"] = str(FAKE)
    env.pop("CODEX_WORKER_NESTED", None)
    return subprocess.run([sys.executable, "tools/worker_pool.py", *args],
                          cwd=REPO, capture_output=True, text=True, env=env)


@unittest.skipIf(NESTED, "would re-enter verify_all through its unittest stage")
class TestPool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pool-"))
        self.queue = self.tmp / "queue"
        self.queue.mkdir()
        self.branches: list[str] = []

    def tearDown(self):
        for b in self.branches:
            git("branch", "-D", b)
        git("worktree", "prune")

    def spec(self, name: str, task: str, mode: str = "implement", **extra) -> Path:
        p = self.queue / f"{name}.json"
        p.write_text(json.dumps({"mode": mode, "task": task, **extra}),
                     encoding="utf-8")
        return p

    def run_pool(self, *paths: Path, concurrency: int = 4,
                 verify: str = "fast") -> dict:
        r = pool("run", *[str(p) for p in paths], "--backend", "local",
                 "--base", "HEAD", "--concurrency", str(concurrency),
                 "--verify", verify, "--out", str(self.tmp / "out"))
        self.assertTrue(r.stdout.strip(), f"no JSON on stdout:\n{r.stderr}")
        summary = json.loads(r.stdout)
        for row in summary["results"]:
            if row.get("branch"):
                self.branches.append(row["branch"])
        summary["_returncode"] = r.returncode
        summary["_stderr"] = r.stderr
        return summary

    # ---- the thing a single-task test cannot show -------------------------

    def test_four_concurrent_tasks_each_get_their_own_branch(self):
        specs = [self.spec(f"t{i}", f"Task {i}.\nFAKE-WRITE tools/_pool_{i}.py :: "
                                    f"POOL_{i} = {i}\n")
                 for i in range(4)]
        s = self.run_pool(*specs, concurrency=4)

        self.assertEqual(len(s["results"]), 4)
        for row in s["results"]:
            self.assertEqual(row["verdict"], "READY-FOR-REVIEW", row)
        branches = {r["branch"] for r in s["results"]}
        self.assertEqual(len(branches), 4, "branches collided")
        for b in branches:
            self.assertEqual(git("rev-parse", "--verify", b).returncode, 0)

    def test_concurrent_worktrees_do_not_corrupt_the_main_checkout(self):
        """Four `git worktree add` calls against one .git. Without the lock
        this is where the registry races."""
        before = git("status", "--porcelain").stdout
        worktrees_before = git("worktree", "list").stdout.count("\n")
        specs = [self.spec(f"w{i}", f"W.\nFAKE-WRITE tools/_pw_{i}.py :: A = {i}\n")
                 for i in range(4)]
        self.run_pool(*specs, concurrency=4)
        self.assertEqual(git("status", "--porcelain").stdout, before)
        self.assertEqual(git("worktree", "list").stdout.count("\n"),
                         worktrees_before)

    def test_each_task_sees_only_its_own_edit(self):
        """Separate worktrees, so two workers touching the tree at once
        cannot leak into each other's diff."""
        specs = [self.spec(f"i{i}", f"I.\nFAKE-WRITE tools/_pi_{i}.py :: B = {i}\n")
                 for i in range(3)]
        s = self.run_pool(*specs, concurrency=3)
        for i, row in enumerate(s["results"]):
            # Its own edit and its own report, filed under its own task id --
            # nothing from the two tasks running beside it.
            self.assertEqual(row["files_changed"],
                             [f"A\tqueue/results/{row['task_id']}.json",
                              f"A\ttools/_pi_{i}.py"], row)

    def test_tasks_actually_overlap_rather_than_running_one_at_a_time(self):
        """Concurrency the pool claims but nothing else here proves. Four
        tasks that each take 1.5s finish in well under 6s only if they ran at
        the same time; serialised, the assertion fails."""
        import time
        specs = [self.spec(f"s{i}", f"S.\nFAKE-SLEEP 1.5\n"
                                    f"FAKE-WRITE tools/_pz_{i}.py :: F = {i}\n")
                 for i in range(4)]
        started = time.time()
        s = self.run_pool(*specs, concurrency=4, verify="none")
        elapsed = time.time() - started
        for row in s["results"]:
            self.assertEqual(row["verdict"], "READY-FOR-REVIEW", row)
        self.assertLess(elapsed, 4.5,
                        f"4 x 1.5s tasks took {elapsed:.1f}s -- they were "
                        "serialised, not pooled")

    def test_concurrency_one_serialises(self):
        """The negative control for the test above: with a pool of one the
        same work must take at least as long as the sum of its parts, so the
        timing assertion is measuring concurrency rather than a fast machine."""
        import time
        specs = [self.spec(f"q{i}", f"Q.\nFAKE-SLEEP 1.0\n"
                                    f"FAKE-WRITE tools/_pq_{i}.py :: G = {i}\n")
                 for i in range(3)]
        started = time.time()
        self.run_pool(*specs, concurrency=1, verify="none")
        self.assertGreaterEqual(time.time() - started, 3.0)

    # ---- one bad task must not take the pool down -------------------------

    def test_a_failing_task_does_not_stop_the_others(self):
        good = self.spec("good", "G.\nFAKE-WRITE tools/_pg.py :: G = 1\n")
        blocked = self.spec("blocked", "B.\nFAKE-WRITE tools/_pb.py :: B = 1\n"
                                       "FAKE-BLOCKER no GPU\n")
        s = self.run_pool(good, blocked, concurrency=2)
        by_task = {r["task"]: r for r in s["results"]}
        self.assertEqual(by_task["good.json"]["verdict"], "READY-FOR-REVIEW")
        self.assertEqual(by_task["blocked.json"]["verdict"], "BLOCKED")
        self.assertEqual(s["_returncode"], 1, "a blocked task must fail the run")

    def test_an_invalid_spec_is_refused_without_a_worktree(self):
        bad = self.spec("bad", "", mode="implement")
        good = self.spec("ok", "O.\nFAKE-WRITE tools/_po.py :: O = 1\n")
        s = self.run_pool(bad, good, concurrency=2)
        by_task = {r["task"]: r for r in s["results"]}
        self.assertEqual(by_task["bad.json"]["verdict"], "REFUSED")
        self.assertIn("task is empty", by_task["bad.json"]["reason"])
        self.assertIsNone(by_task["bad.json"].get("branch"))
        self.assertEqual(by_task["ok.json"]["verdict"], "READY-FOR-REVIEW")

    def test_an_unknown_mode_is_refused_by_name(self):
        s = self.run_pool(self.spec("m", "x", mode="vibes"), concurrency=1)
        self.assertEqual(s["results"][0]["verdict"], "REFUSED")
        self.assertIn("mode must be one of", s["results"][0]["reason"])

    # ---- the summary is what the lead reads -------------------------------

    def test_results_are_sorted_and_the_summary_is_written(self):
        specs = [self.spec(n, f"T.\nFAKE-WRITE tools/_ps_{n}.py :: C = 1\n")
                 for n in ("zebra", "alpha", "mango")]
        s = self.run_pool(*specs, concurrency=3)
        self.assertEqual([r["task"] for r in s["results"]],
                         ["alpha.json", "mango.json", "zebra.json"])
        written = json.loads((self.tmp / "out" / "summary.json").read_text())
        self.assertEqual(written["results"], s["results"])

    def test_verification_result_is_carried_per_task(self):
        s = self.run_pool(
            self.spec("v", "V.\nFAKE-WRITE tools/_pv.py :: D = 1\n"
                           "FAKE-CLAIM-PASS\n",
                      acceptance=["exit 4"]), concurrency=1)
        row = s["results"][0]
        self.assertFalse(row["verification_ok"])
        self.assertEqual(row["verdict"], "NEEDS-ATTENTION")

    def test_uncertainties_and_blockers_reach_the_summary(self):
        s = self.run_pool(
            self.spec("u", "U.\nFAKE-WRITE tools/_pu.py :: E = 1\n"
                           "FAKE-BLOCKER the disk is full\n"), concurrency=1)
        self.assertIn("the disk is full", s["results"][0]["blockers"])


@unittest.skipIf(NESTED, "would re-enter verify_all through its unittest stage")
class TestPoolCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="poolcli-"))
        self.spec_path = self.tmp / "a.json"
        self.spec_path.write_text(json.dumps({"mode": "implement", "task": "x"}))

    def test_dry_run_creates_no_worktree_and_sends_nothing(self):
        before = git("worktree", "list").stdout
        r = pool("run", str(self.spec_path), "--dry-run", "--concurrency", "10")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["concurrency"], 10)
        self.assertEqual(out["would_run"], [str(self.spec_path)])
        self.assertEqual(git("worktree", "list").stdout, before)

    def test_concurrency_is_capped_rather_than_obeyed_blindly(self):
        """A typo in --concurrency should not fork a hundred test suites."""
        out = json.loads(pool("run", str(self.spec_path), "--dry-run",
                              "--concurrency", "500").stdout)
        self.assertEqual(out["concurrency"], 16)

    def test_max_tasks_caps_the_run(self):
        b = self.tmp / "b.json"
        b.write_text(json.dumps({"mode": "implement", "task": "y"}))
        out = json.loads(pool("run", str(self.spec_path), str(b), "--dry-run",
                              "--max-tasks", "1").stdout)
        self.assertEqual(len(out["would_run"]), 1)

    def test_list_reports_problems_without_running_anything(self):
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"mode": "implement", "task": ""}))
        rows = json.loads(pool("list", str(bad)).stdout)
        self.assertIn("task is empty", rows[0]["problems"][0])

    def test_an_empty_queue_is_an_error_not_a_silent_success(self):
        r = pool("run", str(self.tmp / "nothing-here"), "--dry-run")
        self.assertNotEqual(r.returncode, 0)

    def test_the_agent_backend_is_selectable(self):
        """The pool must be able to ask for the tool-using loop.

        codex_worker has had `--backend agent` since the loop landed, but the
        pool's own choices did not list it, so the one backend that can read a
        file it was not handed was unreachable through the fan-out.
        """
        out = json.loads(pool("run", str(self.spec_path), "--dry-run",
                              "--backend", "agent").stdout)
        self.assertEqual(out["backend"], "agent")

    def test_the_default_backend_is_the_tool_using_loop(self):
        out = json.loads(pool("run", str(self.spec_path), "--dry-run").stdout)
        self.assertEqual(out["backend"], "agent")


class TestSpecValidation(unittest.TestCase):
    """A typo in a spec should cost nothing, not fifteen worktrees.

    Both `tools` and `limits` fail silently if unchecked: an unknown budget
    key is dropped by Budget, so the author believes a cap is in force that is
    not, and an unknown tool name used to select nothing at all.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="poolval-"))

    def problems(self, spec):
        p = self.tmp / "s.json"
        p.write_text(json.dumps(spec))
        return json.loads(pool("list", str(p)).stdout)[0]["problems"]

    def test_a_good_spec_has_no_problems(self):
        self.assertEqual(self.problems({
            "mode": "test", "task": "add a test",
            "tools": ["read_file", "write_file", "run"],
            "limits": {"max_turns": 10, "max_tool_calls": 20,
                       "max_cost_usd": 0.5, "wall_clock_s": 600}}), [])

    def test_an_unknown_tool_is_caught_before_dispatch(self):
        got = " ".join(self.problems(
            {"mode": "test", "task": "x", "tools": ["read_file", "nope"]}))
        self.assertIn("nope", got)

    def test_an_unknown_limit_key_is_caught_rather_than_dropped(self):
        got = " ".join(self.problems(
            {"mode": "test", "task": "x", "limits": {"max_calls": 5}}))
        self.assertIn("max_calls", got)
        self.assertIn("max_tool_calls", got)

    def test_a_nonpositive_budget_is_refused(self):
        got = " ".join(self.problems(
            {"mode": "test", "task": "x", "limits": {"max_cost_usd": 0}}))
        self.assertIn("max_cost_usd", got)

    def test_limits_must_be_an_object(self):
        got = " ".join(self.problems(
            {"mode": "test", "task": "x", "limits": [1, 2]}))
        self.assertIn("limits must be an object", got)


class TestShippedQueue(unittest.TestCase):
    """The specs committed to queue/tasks/ must be dispatchable.

    They are reviewed like code because they instruct an outside model to edit
    this repository; a spec that cannot be validated is one nobody can review.
    """

    def test_every_committed_spec_validates(self):
        queue = REPO / "queue" / "tasks"
        specs = sorted(queue.glob("*.json"))
        self.assertTrue(specs, "no task specs are committed")
        rows = json.loads(pool("list", *[str(p) for p in specs]).stdout)
        bad = {r["path"]: r["problems"] for r in rows if r["problems"]}
        self.assertEqual(bad, {})


if __name__ == "__main__":
    unittest.main()
