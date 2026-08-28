#!/usr/bin/env python3
"""Regression tests for the batch runner's process exit status."""

import contextlib
import io
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_all


class BatchExitCodeTest(unittest.TestCase):
    def test_all_success_returns_zero(self):
        self.assertEqual(run_all.batch_exit_code({"first": 0, "second": 0}), 0)

    def test_child_failure_returns_nonzero(self):
        self.assertEqual(run_all.batch_exit_code({"first": 0, "second": 2}), 1)

    def test_missing_script_returns_nonzero(self):
        self.assertEqual(run_all.batch_exit_code({"first": 0, "missing": -1}), 1)

    def test_empty_batch_returns_nonzero(self):
        self.assertEqual(run_all.batch_exit_code({}), 1)

    def test_main_propagates_child_failure(self):
        experiments = [("example", "Example", "data/example.log")]
        with (
            mock.patch.object(run_all, "EXPERIMENTS", experiments),
            mock.patch.object(run_all, "run_experiment", return_value=3),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run_all.main(), 1)


if __name__ == "__main__":
    unittest.main()
