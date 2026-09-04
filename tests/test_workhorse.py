"""Tests for tools/workhorse.py, the pull-based experiment runner.

Everything runs with --dry-run: preflight, execute, postflight, into a
temporary directory, with no branch, commit, or push. The script under test
is experiments/counting_bound.py with a --verdict flag, which completes in
milliseconds, so the whole pipeline is exercised for real rather than mocked.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def write_manifest(tmp: Path, **over) -> Path:
    m = {
        "name": "t-fast", "kind": "measurement", "seed": "single-black-cell",
        "theory_gate": "OPEN", "script": "experiments/counting_bound.py",
        "argv": ["--verdict", "5:10"], "claims": [],
        "reads_packed_bitstream": False, "budget": {"minutes": 1},
    }
    m.update(over)
    p = tmp / f"{m['name']}.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return p


def workhorse(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "tools/workhorse.py", *args],
                          cwd=REPO, capture_output=True, text=True)


class TestDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_full_pipeline_on_a_fast_script(self):
        p = write_manifest(self.tmp)
        r = workhorse("run", str(p), "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        s = json.loads(r.stdout)
        self.assertEqual(s["status"], "ok")
        self.assertEqual(s["exit_code"], 0)
        self.assertEqual(s["postflight"], "PASS")
        out = Path(s["out"])
        for f in ("result.json", "postflight.json", "stdout.txt", "stderr.txt",
                  "hashes.json"):
            self.assertTrue((out / f).exists(), f)

    def test_result_carries_provenance_and_parsed_stdout(self):
        p = write_manifest(self.tmp)
        s = json.loads(workhorse("run", str(p), "--dry-run").stdout)
        res = json.loads((Path(s["out"]) / "result.json").read_text())
        self.assertEqual(res["agent"], "script")
        self.assertEqual(res["manifest"]["name"], "t-fast")
        self.assertIn("head", res["provenance"])
        self.assertEqual(len(res["provenance"]["head"]), 40)
        # counting_bound prints JSON, so the runner parses it.
        self.assertEqual(res["stdout_json"]["artifact_type"], "rule30.counting_bound")

    def test_hashes_cover_every_output_file(self):
        p = write_manifest(self.tmp)
        s = json.loads(workhorse("run", str(p), "--dry-run").stdout)
        out = Path(s["out"])
        hashes = json.loads((out / "hashes.json").read_text())
        expected = {f.name for f in out.iterdir() if f.is_file()} - {"hashes.json"}
        self.assertEqual(set(hashes), expected)
        for v in hashes.values():
            self.assertRegex(v, r"^[0-9a-f]{64}$")

    def test_a_refused_manifest_never_executes(self):
        """The trap: preflight refuses it, nothing runs, and the refusal is
        recorded under queue/refused/ rather than swallowed."""
        p = write_manifest(self.tmp, name="t-trap", kind="search",
                           claims=["negative"],
                           search={"class": "dfao", "states": 24, "base": 2,
                                   "prefix_bits": 10_000})
        before = set((REPO / "queue" / "refused").glob("t-trap-*.json"))
        r = workhorse("run", str(p), "--dry-run")
        after = set((REPO / "queue" / "refused").glob("t-trap-*.json"))
        try:
            self.assertEqual(r.returncode, 1)
            s = json.loads(r.stdout)
            self.assertEqual(s["status"], "refused")
            self.assertEqual(s["preflight"]["verdict"], "FAIL")
            self.assertEqual(len(after - before), 1)
            rec = json.loads(next(iter(after - before)).read_text())
            self.assertEqual(rec["report"]["verdict"], "FAIL")
        finally:
            for f in after - before:
                f.unlink()

    def test_nonzero_exit_is_needs_attention(self):
        p = write_manifest(self.tmp, name="t-bad", argv=["--verdict", "not-a-spec"])
        r = workhorse("run", str(p), "--dry-run")
        self.assertEqual(r.returncode, 1)
        s = json.loads(r.stdout)
        self.assertEqual(s["status"], "needs-attention")
        self.assertNotEqual(s["exit_code"], 0)

    def test_budget_timeout_is_enforced(self):
        """A run that exceeds its budget is stopped and marked, not left
        running. The council asked for resource limits; this is the one that
        matters on a rented box."""
        p = write_manifest(self.tmp, name="t-slow",
                           script="tests/fixtures/sleep_forever.py", argv=[])
        r = workhorse("run", str(p), "--dry-run", "--timeout", "2")
        self.assertEqual(r.returncode, 1)
        s = json.loads(r.stdout)
        self.assertTrue(s["timed_out"])
        self.assertEqual(s["status"], "needs-attention")

    def test_dry_run_does_not_shell_out_to_verify_all(self):
        """Regression. preflight's external gates run verify_all, whose
        unittest stage runs this very file, which runs the runner -- so a dry
        run on the external path recurses without bound. It hung the suite
        once. --dry-run must imply --no-external, and the proof is that the
        three subprocess gates report SKIP."""
        p = write_manifest(self.tmp, name="t-recurse", reads_packed_bitstream=True)
        r = workhorse("run", str(p), "--dry-run", "--pretty")
        self.assertEqual(r.returncode, 0, r.stderr)
        for gate in ("verify-all", "bitorder-lint", "golden-self-test"):
            self.assertRegex(r.stderr, rf"SKIP\s+{gate}", gate)

    def test_codex_agent_without_codex_on_path_fails_clearly(self):
        p = write_manifest(self.tmp, name="t-codex")
        r = workhorse("run", str(p), "--dry-run", "--agent", "codex")
        self.assertEqual(r.returncode, 1)
        self.assertIn("codex", r.stderr.lower())


class TestList(unittest.TestCase):
    def test_lists_committed_queue_with_verdicts(self):
        r = workhorse("list")
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = {row["name"]: row for row in json.loads(r.stdout)}
        self.assertEqual(rows["b1-pattern-map-walk-32"]["verdict"], "PASS")
        trap = rows["TRAP-vacuous-dfao-24"]
        self.assertEqual(trap["verdict"], "FAIL")
        self.assertEqual(trap["failing"], ["counting-bound"])

    def test_pretty_goes_to_stderr(self):
        r = workhorse("list", "--pretty")
        json.loads(r.stdout)
        self.assertIn("refused by: counting-bound", r.stderr)


if __name__ == "__main__":
    unittest.main()
