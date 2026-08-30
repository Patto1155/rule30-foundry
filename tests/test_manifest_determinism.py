#!/usr/bin/env python3
"""Pin byte-reproducibility of the data tree and its manifest.

`.gitattributes` marks `data/** -text`, which stops git from normalising line
endings on checkout. That was the right fix - it makes the manifest mean the
same thing on every platform - but it also removed the layer that had been
silently hiding two CRLF defects:

  - `csv.writer` defaults to CRLF, so re-running an experiment produced
    byte-different CSVs and broke manifest hashes while the *content* was
    identical. All ten writers now pass `lineterminator="\\n"`.
  - `Path.write_text` translates "\\n" to `os.linesep`, so
    `tools/make_manifest.py` emitted a CRLF manifest on Windows and an LF one
    everywhere else. Fixed by writing with `newline=""`.

Nothing structural stopped writer number eleven, so these tests do.
"""

import ast
import pathlib
import subprocess
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import make_manifest

MANIFEST = REPO_ROOT / "data" / "MANIFEST.sha256"
CSV_WRITERS = frozenset({"writer", "DictWriter"})


def tracked(pattern: str) -> list[pathlib.Path]:
    out = subprocess.run(["git", "ls-files", pattern], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [REPO_ROOT / rel for rel in out.split()]


class NoCarriageReturnsTest(unittest.TestCase):
    def test_manifest_has_no_carriage_returns(self):
        self.assertTrue(MANIFEST.exists(), f"missing {MANIFEST}")
        raw = MANIFEST.read_bytes()
        self.assertNotIn(
            b"\r", raw,
            "data/MANIFEST.sha256 contains CR: it was written with newline "
            "translation on, so it is byte-different across platforms")

    def test_tracked_data_csvs_have_no_carriage_returns(self):
        paths = tracked("data/*.csv") + tracked("data/**/*.csv")
        self.assertGreater(len(paths), 0, "no tracked data CSVs found")
        for path in paths:
            with self.subTest(path=path.name):
                self.assertNotIn(b"\r", path.read_bytes())


class CsvWriterLineTerminatorTest(unittest.TestCase):
    """Static guard against writer number eleven."""

    def _bare_writers(self, source: str, relpath: str) -> list[str]:
        bare = []
        for node in ast.walk(ast.parse(source, filename=relpath)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name not in CSV_WRITERS:
                continue
            if any(kw.arg in ("lineterminator", None) for kw in node.keywords):
                continue
            bare.append(f"{relpath}:{node.lineno}: csv.{name}() without "
                        "lineterminator (defaults to CRLF)")
        return bare

    def test_every_csv_writer_pins_the_line_terminator(self):
        bare = []
        for root in ("experiments", "gpu", "tools"):
            for path in sorted((REPO_ROOT / root).rglob("*.py")):
                bare += self._bare_writers(
                    path.read_text(encoding="utf-8"),
                    path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(bare, [], "\n".join(bare))

    def test_the_guard_detects_a_bare_writer(self):
        bare = self._bare_writers(
            "import csv\nw = csv.writer(f)\n", "<test>")
        self.assertEqual(len(bare), 1)

    def test_the_guard_accepts_an_explicit_terminator(self):
        bare = self._bare_writers(
            "import csv\nw = csv.writer(f, lineterminator='\\n')\n", "<test>")
        self.assertEqual(bare, [])


class ManifestDeterminismTest(unittest.TestCase):
    def test_building_twice_gives_identical_bytes(self):
        first, _ = make_manifest.build_manifest_text(REPO_ROOT)
        second, _ = make_manifest.build_manifest_text(REPO_ROOT)
        self.assertEqual(first, second)

    def test_built_text_uses_lf_only(self):
        text, _ = make_manifest.build_manifest_text(REPO_ROOT)
        self.assertNotIn("\r", text)

    def test_manifest_on_disk_is_up_to_date(self):
        """The committed manifest is what the tool regenerates.

        This holds on a machine with the canonical bitstreams and on one
        without: an absent bitstream keeps its anchored hash line verbatim, so
        the file is machine-independent.
        """
        text, _ = make_manifest.build_manifest_text(REPO_ROOT)
        self.assertEqual(
            MANIFEST.read_bytes(), text.encode("utf-8"),
            "data/MANIFEST.sha256 is stale; run python tools/make_manifest.py")


class AnchorPreservationTest(unittest.TestCase):
    """Regenerating on a machine without the bitstreams must not drop them.

    The bitstreams are gitignored, so this is the common case, not the exotic
    one. An earlier version wrote `# UNANCHORED` here, which would have erased
    the only hashes tying the A-H results to specific bytes.
    """

    ANCHOR = "6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e"

    def setUp(self):
        self._real = make_manifest.tracked_data_files
        make_manifest.tracked_data_files = lambda root=None: []
        self.addCleanup(setattr, make_manifest, "tracked_data_files",
                        self._real)

    def _build(self, tmp: pathlib.Path) -> str:
        text, _ = make_manifest.build_manifest_text(tmp)
        return text

    def test_absent_bitstream_keeps_its_anchored_hash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            (tmp / "data").mkdir()
            (tmp / "data" / "MANIFEST.sha256").write_text(
                f"# header\n{self.ANCHOR}  data/center_col_10M.bin\n",
                encoding="utf-8", newline="")
            text = self._build(tmp)
            self.assertIn(f"{self.ANCHOR}  data/center_col_10M.bin", text)
            self.assertNotIn("UNANCHORED  data/center_col_10M.bin", text)

    def test_a_never_seen_bitstream_is_marked_unanchored(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            (tmp / "data").mkdir()
            text = self._build(tmp)
            self.assertIn("# UNANCHORED  data/center_col_10M.bin", text)

    def test_a_changed_bitstream_hash_raises_a_warning(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            (tmp / "data").mkdir()
            (tmp / "data" / "MANIFEST.sha256").write_text(
                f"# header\n{self.ANCHOR}  data/center_col_10M.bin\n",
                encoding="utf-8", newline="")
            (tmp / "data" / "center_col_10M.bin").write_bytes(b"not the same")
            _, warnings = make_manifest.build_manifest_text(tmp)
            self.assertEqual(len(warnings), 1)
            self.assertIn("CHANGED", warnings[0])


if __name__ == "__main__":
    unittest.main()
