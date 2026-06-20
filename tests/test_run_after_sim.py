#!/usr/bin/env python3
"""Guards against analysing a truncated packed artifact as a full 46M-bit run."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_after_sim


class ArtifactSizeStatusTest(unittest.TestCase):
    def setUp(self):
        self.expected = run_after_sim.EXPECTED_BYTES

    def test_exact_size_is_complete(self):
        self.assertEqual(
            run_after_sim.artifact_size_status(self.expected, self.expected),
            "complete",
        )

    def test_one_byte_short_is_incomplete(self):
        self.assertEqual(
            run_after_sim.artifact_size_status(self.expected - 1, self.expected),
            "incomplete",
        )

    def test_ninety_nine_percent_is_rejected(self):
        # The old `size >= expected * 0.99` gate accepted this truncated file
        # and let up to 1% of missing data be analysed as a complete run.
        size = int(self.expected * 0.99)
        self.assertLess(size, self.expected)
        self.assertEqual(
            run_after_sim.artifact_size_status(size, self.expected),
            "incomplete",
        )

    def test_oversized_is_flagged(self):
        self.assertEqual(
            run_after_sim.artifact_size_status(self.expected + 1, self.expected),
            "oversized",
        )


if __name__ == "__main__":
    unittest.main()
