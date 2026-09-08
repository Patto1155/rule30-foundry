"""Independent truth-table and expanding-row checks for finite controls."""

import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments import periodic_trace_constraints as experiment


def reference(initial, steps):
    # Expanding finite support, with a Wolfram rule-number lookup instead of
    # the experiment's Boolean formula and shrinking light cone.
    row = {i for i, bit in initial.items() if bit}
    result = [int(0 in row)]
    for _ in range(steps):
        low, high = min(row, default=0) - 1, max(row, default=0) + 1
        next_row = set()
        for i in range(low, high + 1):
            code = 4 * (i - 1 in row) + 2 * (i in row) + (i + 1 in row)
            if (30 >> code) & 1:
                next_row.add(i)
        row = next_row
        result.append(int(0 in row))
    return result


class PeriodicTraceConstraintsTests(unittest.TestCase):
    def test_truth_table_and_defects(self):
        for background in itertools.product((0, 1), repeat=3):
            code = sum(b << (2 - i) for i, b in enumerate(background))
            self.assertEqual(experiment.rule30(*background), (30 >> code) & 1)
            for defect in itertools.product((0, 1), repeat=3):
                other_code = sum((b ^ d) << (2 - i)
                                 for i, (b, d) in enumerate(zip(background, defect)))
                self.assertEqual(experiment.defect_update(background, defect),
                                 ((30 >> code) ^ (30 >> other_code)) & 1)

    def test_all_short_prescribed_words(self):
        for length in range(1, 9):
            for target in itertools.product((0, 1), repeat=length):
                initial = experiment.realize_trace(target)
                self.assertTrue(all(-length < i <= 0 for i in initial))
                self.assertEqual(reference(initial, length - 1), list(target))

    def test_single_seed_and_general_forward(self):
        target = reference({0: 1}, 80)
        self.assertEqual(experiment.center_trace({0: 1}, 80), target)
        self.assertEqual({i: b for i, b in experiment.realize_trace(target).items() if b},
                         {0: 1})
        for initial in ({}, {-5: 1, 0: 1, 7: 1}, {64: 1, -65: 1}):
            self.assertEqual(experiment.center_trace(initial, 80), reference(initial, 80))

    def test_scan_witness_and_candidate_mismatches(self):
        result = experiment.tail_scan(96, 8, 16)
        seed = reference({0: 1}, 96)
        self.assertEqual(result["candidate_count"], 17 * 8)
        for row in result["candidates"]:
            onset, period = row["onset"], row["period"]
            expected = next((t for t in range(onset + period, 97)
                             if seed[t] != seed[t - period]), None)
            self.assertEqual(row["first_mismatch_time"], expected)
        witness = result["changed_seed_witness"]
        initial = {i: 1 for i in witness["initial_nonzero_coordinates"]}
        self.assertEqual(reference(initial, 64), witness["target_trace"])
        self.assertEqual(reference({0: 1}, 64), witness["seed_trace"])
        self.assertEqual(-witness["nearest_nonzero_negative_coordinate"],
                         witness["first_output_mismatch_time"])

    def test_unresolved_is_explicit(self):
        result = experiment.tail_scan(0, 1, 0)
        self.assertEqual(result["unresolved_count"], 1)
        self.assertIsNone(result["candidates"][0]["first_mismatch_time"])

    def test_invalid_input(self):
        for call in (lambda: experiment.rule30(0, 2, 0),
                     lambda: experiment.realize_trace([]),
                     lambda: experiment.realize_trace([1, "0"]),
                     lambda: experiment.center_trace({0: 3}, 0),
                     lambda: experiment.center_trace({0.5: 1}, 1),
                     lambda: experiment.center_trace({}, -1),
                     lambda: experiment.defect_update([0, 1], [0, 0, 0]),
                     lambda: experiment.defect_update([0, 1, 0], [0, 2, 0]),
                     lambda: experiment.tail_scan(10, 0, 0),
                     lambda: experiment.tail_scan(10, 5, 10)):
            with self.assertRaises(ValueError):
                call()

    def test_cli_json_and_determinism(self):
        script = Path(experiment.__file__).resolve()
        result = subprocess.run([sys.executable, str(script), "--self-test"],
                                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["prescribed_words"], 510)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            command = [sys.executable, str(script), "--tail-scan", "--steps", "16",
                       "--max-period", "3", "--max-onset", "4", "--out", str(output)]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            second = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(output.read_text(encoding="utf-8"), first.stdout)


if __name__ == "__main__":
    unittest.main()
