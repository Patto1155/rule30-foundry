#!/usr/bin/env python3
"""Run tools/lint_bitorder.py as part of the test suite.

The lint is what stops a *sixth* bare `np.unpackbits` from being written. It
is only useful if it runs without anyone remembering to run it.
"""

import pathlib
import sys
import textwrap
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import lint_bitorder


class RepoIsCleanTest(unittest.TestCase):
    def test_no_bare_packbits_calls_in_repo(self):
        findings = lint_bitorder.scan_repo(REPO_ROOT)
        self.assertEqual(
            findings, [],
            "bare packbits/unpackbits:\n" + "\n".join(str(f) for f in findings))

    def test_it_actually_scans_something(self):
        """A lint that scans nothing passes vacuously."""
        self.assertGreater(len(lint_bitorder.python_files(REPO_ROOT)), 20)


class DetectionTest(unittest.TestCase):
    """The lint must fail on the code that caused the incident."""

    def scan(self, source):
        return lint_bitorder.scan_source(textwrap.dedent(source), "<test>")

    def test_flags_the_original_defect(self):
        findings = self.scan("""
            import numpy as np
            bits = np.unpackbits(data)[:10_000_000]
        """)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].func, "unpackbits")

    def test_flags_a_bare_call_without_the_np_prefix(self):
        findings = self.scan("""
            from numpy import packbits
            raw = packbits(bits)
        """)
        self.assertEqual(len(findings), 1)

    def test_accepts_an_explicit_bitorder(self):
        findings = self.scan("""
            import numpy as np
            bits = np.unpackbits(data, bitorder='little')
        """)
        self.assertEqual(findings, [])

    def test_accepts_an_exemption_with_a_reason(self):
        findings = self.scan("""
            import numpy as np
            # bitorder-exempt: MSB-first here is deliberate.
            bits = np.unpackbits(data)
        """)
        self.assertEqual(findings, [])

    def test_accepts_a_multi_line_exemption(self):
        findings = self.scan("""
            import numpy as np
            # bitorder-exempt: MSB-first here is deliberate, because this
            # module must not share a convention with the kernel it checks.
            bits = np.unpackbits(data)
        """)
        self.assertEqual(findings, [])

    def test_rejects_an_exemption_with_no_reason(self):
        findings = self.scan("""
            import numpy as np
            # bitorder-exempt:
            bits = np.unpackbits(data)
        """)
        self.assertEqual(len(findings), 1)

    def test_an_exemption_does_not_leak_past_a_blank_line(self):
        findings = self.scan("""
            import numpy as np
            # bitorder-exempt: applies to the call directly below only.
            a = np.unpackbits(x)

            b = np.unpackbits(y)
        """)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].lineno, 6)

    def test_an_exemption_does_not_leak_past_intervening_code(self):
        findings = self.scan("""
            import numpy as np
            # bitorder-exempt: applies to the call directly below only.
            a = np.unpackbits(x)
            b = np.unpackbits(y)
        """)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].lineno, 5)

    def test_ignores_unrelated_calls(self):
        findings = self.scan("""
            import numpy as np
            x = np.frombuffer(data)
            y = obj.pack(bits)
        """)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
