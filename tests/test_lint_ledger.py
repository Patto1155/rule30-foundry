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


class StatusIsCurrentTest(unittest.TestCase):
    """docs/STATUS.md exists, is placeholder-free, and is not behind the logs."""

    def test_repo_status_is_clean(self):
        findings = lint_ledger.check_status(REPO_ROOT)
        self.assertEqual(findings, [], "\n".join(str(f) for f in findings))

    def test_newest_log_is_picked_by_filename_date_not_mtime(self):
        """A fresh clone gives every file the same mtime; dates must come from
        the filename or this check passes vacuously."""
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            logs = root / "docs" / "experiment-logs"
            logs.mkdir(parents=True)
            # Written newest-first, so mtime order is the reverse of date order.
            for name in ("2026-08-30-late.md", "2026-01-02-early.md",
                         "S_linear_complexity.md"):
                (logs / name).write_text("x", encoding="utf-8", newline="")
            self.assertEqual(
                lint_ledger.newest_dated_log(root), "2026-08-30-late.md")

    def check(self, status_body, logs=("2026-08-30-newest.md",)):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "docs" / "experiment-logs").mkdir(parents=True)
            for name in logs:
                (root / "docs" / "experiment-logs" / name).write_text(
                    "x", encoding="utf-8", newline="")
            if status_body is not None:
                (root / "docs" / "STATUS.md").write_text(
                    status_body, encoding="utf-8", newline="")
            return lint_ledger.check_status(root)

    def test_flags_a_missing_status_file(self):
        findings = self.check(None)
        self.assertTrue(any(f.kind == "FAIL" for f in findings), findings)

    def test_flags_a_status_that_is_behind_the_newest_log(self):
        findings = self.check("Updated: whenever. Nothing cited here.\n")
        self.assertTrue(
            any(f.kind == "STALE-STATUS" for f in findings), findings)

    def test_accepts_a_status_that_cites_the_newest_log(self):
        findings = self.check(
            "See `docs/experiment-logs/2026-08-30-newest.md`.\n")
        self.assertEqual(findings, [], "\n".join(str(f) for f in findings))

    def test_flags_a_placeholder_in_status(self):
        findings = self.check(
            "2026-08-30-newest.md\n\nRESULT_SECTION\n")
        self.assertTrue(
            any(f.kind == "PLACEHOLDER" for f in findings), findings)


class SingleStatusHomeTest(unittest.TestCase):
    """Only docs/STATUS.md may declare current state."""

    def test_repo_has_exactly_one_status_home(self):
        findings = lint_ledger.check_single_status_home(REPO_ROOT)
        self.assertEqual(findings, [], "\n".join(str(f) for f in findings))

    def check(self, rel, body):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8", newline="")
            return lint_ledger.check_single_status_home(root)

    def test_flags_the_snapshot_that_actually_went_stale(self):
        """The real defect: AGENTS.md, frozen at 2026-04-01 for five months."""
        findings = self.check(
            "AGENTS.md", "## Current Frontier\n\nCurrent state as of "
                         "`2026-04-01`:\n\n- `O` and `P` are still open.\n")
        self.assertEqual(len(findings), 2, findings)
        self.assertTrue(all(f.kind == "DUPLICATE-STATUS" for f in findings))

    def test_does_not_flag_status_itself(self):
        self.assertEqual(
            self.check("docs/STATUS.md", "Current state as of 2026-09-02\n"), [])

    def test_does_not_flag_an_archived_handover(self):
        self.assertEqual(
            self.check("docs/handover/archive/2026-06-13-old.md",
                       "Current state as of 2026-06-13\n"), [])

    def test_does_not_flag_a_dated_fact(self):
        """'Refuted 2026-08-19' is a fact with a date, not a frozen snapshot.
        Flagging it would make the lint unusable in the ledger."""
        self.assertEqual(
            self.check("docs/CLAIM_LEDGER.md",
                       "| Claim | **Refuted 2026-08-19** (was: Proof "
                       "candidate) | Admission Rule (added 2026-08-15) |\n"), [])

    def test_an_annotated_exemption_is_honoured(self):
        self.assertEqual(
            self.check("AGENTS.md",
                       "<!-- status-exempt: documents the rule -->\n"
                       "Do not write \"current state as of\" anywhere else.\n"),
            [])

    def test_a_bare_marker_with_no_reason_does_not_silence(self):
        """The mandatory reason is what stops the marker becoming a silencer."""
        findings = self.check(
            "AGENTS.md", "<!-- status-exempt: -->\nCurrent state as of X\n")
        self.assertTrue(findings, "bare marker should not exempt")
