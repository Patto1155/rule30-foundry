#!/usr/bin/env python3
"""Run tools/lint_ledger.py, and prove it catches the defect it was built for.

The ledger is the honesty mechanism. In August 2026 it was found to have been
wrong for two weeks: a retracted row promised a replacement row that did not
exist, and the experiment log under a "Claim Level: Certificate" header was
still a template full of `RESULT_SECTION` placeholders. A lint that passes on
a clean repo but would not have caught that is worth nothing, so the detection
cases below are the point of this file.
"""

import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import lint_ledger

HEADER = ("| Claim | Level | Evidence | What would promote it |\n"
          "|---|---|---|---|\n")


class RepoIsCleanTest(unittest.TestCase):
    def test_ledger_cites_nothing_missing(self):
        findings = lint_ledger.check_ledger(REPO_ROOT)
        self.assertEqual(
            findings, [], "\n".join(str(f) for f in findings))

    def test_experiment_logs_have_no_placeholders(self):
        findings = lint_ledger.check_logs(REPO_ROOT)
        self.assertEqual(
            findings, [], "\n".join(str(f) for f in findings))

    def test_it_actually_reads_rows(self):
        """A lint that parses zero rows passes vacuously."""
        rows = [line for line in
                (REPO_ROOT / "docs" / "CLAIM_LEDGER.md").read_text(
                    encoding="utf-8").splitlines()
                if lint_ledger.is_table_row(line)]
        self.assertGreater(len(rows), 20)


class LedgerDetectionTest(unittest.TestCase):
    def check(self, ledger_body, manifest=None):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "docs").mkdir()
            (root / "data").mkdir()
            (root / "docs" / "CLAIM_LEDGER.md").write_text(
                HEADER + ledger_body, encoding="utf-8", newline="")
            if manifest is not None:
                (root / "data" / "MANIFEST.sha256").write_text(
                    manifest, encoding="utf-8", newline="")
            return lint_ledger.check_ledger(root)

    def test_flags_a_cited_path_that_does_not_exist(self):
        findings = self.check(
            "| A claim. | Observation | `docs/experiment-logs/ghost.md` | x |\n")
        self.assertEqual([f.kind for f in findings], ["MISSING"])
        self.assertIn("ghost.md", findings[0].detail)

    def test_flags_a_placeholder_in_a_row(self):
        findings = self.check(
            "| A claim. | Certificate | `tools/x.py` RESULT_SECTION | x |\n")
        self.assertIn("PLACEHOLDER", [f.kind for f in findings])

    def test_flags_a_certificate_with_no_verifier_script(self):
        findings = self.check(
            "| A claim. | **Certificate** | Trust me. | x |\n")
        self.assertEqual([f.kind for f in findings], ["NO-VERIFIER"])

    def test_does_not_demand_a_verifier_from_a_lower_grade(self):
        findings = self.check(
            "| A claim. | Observation | Trust me. | x |\n")
        self.assertEqual(findings, [])

    def test_ignores_maths_in_backticks(self):
        """`log2|M| >= n` and `2^28.2` are not file paths."""
        findings = self.check(
            "| A claim. | Theorem | `log2\\|M\\| >= n` and `D_d(t+1)` | x |\n")
        self.assertEqual(findings, [])

    def test_extracts_a_path_out_of_a_command_line(self):
        findings = self.check(
            "| A claim. | Observation | "
            "`python tools/nope.py --out data/nope.json` | x |\n")
        self.assertEqual({f.kind for f in findings}, {"MISSING"})
        self.assertEqual(len(findings), 2)

    def test_strips_a_line_number_reference(self):
        findings = self.check(
            "| A claim. | Observation | `experiments/ghost.py:36` | x |\n")
        self.assertEqual(len(findings), 1)
        self.assertIn("experiments/ghost.py", findings[0].detail)
        self.assertNotIn(":36", findings[0].detail)

    def test_gitignored_bitstream_is_ok_when_anchored(self):
        findings = self.check(
            "| A claim. | **Certificate** | `python tools/lint_ledger.py "
            "--bitstream data/center_col_10M.bin` | x |\n",
            manifest="deadbeef  data/center_col_10M.bin\n")
        # tools/lint_ledger.py does not exist in the temp root, so that one
        # is reported; the bitstream is not, because it is anchored.
        self.assertNotIn("center_col_10M.bin",
                         " ".join(f.detail for f in findings))

    def test_gitignored_bitstream_is_flagged_when_not_anchored(self):
        findings = self.check(
            "| A claim. | Observation | `data/center_col_10M.bin` | x |\n",
            manifest="# nothing anchored\n")
        self.assertEqual([f.kind for f in findings], ["UNANCHORED"])


class LogDetectionTest(unittest.TestCase):
    def check_logs(self, name, body, *, encoding="utf-8"):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            logs = root / "docs" / "experiment-logs"
            logs.mkdir(parents=True)
            (logs / name).write_bytes(body.encode(encoding))
            return lint_ledger.check_logs(root)

    def test_flags_the_template_placeholder_that_shipped(self):
        findings = self.check_logs(
            "x.md", "# Log\n\nClaim Level: Certificate\n\nRESULT_SECTION\n")
        self.assertEqual([f.kind for f in findings], ["PLACEHOLDER"])
        self.assertIn("RESULT_SECTION", findings[0].detail)

    def test_accepts_a_filled_in_log(self):
        findings = self.check_logs(
            "x.md", "# Log\n\n## Result\n\ns*(32) = 10.\n")
        self.assertEqual(findings, [])

    def test_reports_a_non_utf8_log_instead_of_crashing(self):
        findings = self.check_logs("x.md", "em — dash\n", encoding="cp1252")
        self.assertEqual([f.kind for f in findings], ["ENCODING"])


class MarkdownEncodingTest(unittest.TestCase):
    """Every tracked markdown file must be UTF-8.

    Six logs were cp1252, which renders as mojibake on GitHub and raises
    UnicodeDecodeError in any tool that reads them.
    """

    def test_all_tracked_markdown_is_utf8(self):
        import subprocess
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout
        paths = [REPO_ROOT / rel for rel in out.split()]
        self.assertGreater(len(paths), 10)
        for path in paths:
            with self.subTest(path=path.name):
                try:
                    path.read_text(encoding="utf-8")
                except UnicodeDecodeError as exc:
                    self.fail(f"{path.relative_to(REPO_ROOT)} is not UTF-8: "
                              f"{exc.reason} at byte {exc.start}")


if __name__ == "__main__":
    unittest.main()
