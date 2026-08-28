#!/usr/bin/env python3
"""CLI parsing regressions for ca_lab."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ca_lab


class PrettyFlagParsingTest(unittest.TestCase):
    def test_pretty_before_subcommand_is_preserved(self):
        args = ca_lab.build_parser().parse_args(
            ["--pretty", "sim", "--rule", "30"]
        )
        self.assertTrue(args.pretty)

    def test_pretty_after_subcommand_is_preserved(self):
        args = ca_lab.build_parser().parse_args(
            ["sim", "--rule", "30", "--pretty"]
        )
        self.assertTrue(args.pretty)

    def test_pretty_defaults_to_false(self):
        args = ca_lab.build_parser().parse_args(["sim", "--rule", "30"])
        self.assertFalse(args.pretty)


if __name__ == "__main__":
    unittest.main()
