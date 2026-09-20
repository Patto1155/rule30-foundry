"""Exact tests for shared-predecessor slanted triple transitions."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

from experiments import coherent_predecessor as cp


class TruthTableTests(unittest.TestCase):
    def test_rule30_formula_matches_all_eight_truth_table_entries(self):
        for left in (0, 1):
            for centre in (0, 1):
                for right in (0, 1):
                    self.assertEqual(
                        cp.truth_table_cell(30, left, centre, right),
                        left ^ (centre | right),
                    )

    def test_controls_use_their_actual_truth_tables(self):
        for left in (0, 1):
            for centre in (0, 1):
                for right in (0, 1):
                    self.assertEqual(
                        cp.truth_table_cell(15, left, centre, right), 1 ^ left
                    )
                    self.assertEqual(
                        cp.truth_table_cell(90, left, centre, right), left ^ right
                    )


class CoherentRelationTests(unittest.TestCase):
    def test_shared_predecessor_removes_exactly_four_walks(self):
        summary = cp.relation_summary(30)
        self.assertEqual(summary["one_update_relation_count"], 20)
        self.assertEqual(summary["independently_composed_two_update_count"], 48)
        self.assertEqual(summary["shared_predecessor_two_update_count"], 44)
        self.assertEqual(
            summary["spurious_independent_walks"],
            [
                "000->100->111",
                "001->101->110",
                "010->111->100",
                "011->110->101",
            ],
        )

    def test_rule15_and_rule90_controls_are_not_rule30_relations(self):
        rule15 = cp.relation_summary(15)
        rule90 = cp.relation_summary(90)
        self.assertEqual(
            (rule15["one_update_relation_count"],
             rule15["independently_composed_two_update_count"],
             rule15["shared_predecessor_two_update_count"]),
            (8, 8, 8),
        )
        self.assertEqual(
            (rule90["one_update_relation_count"],
             rule90["independently_composed_two_update_count"],
             rule90["shared_predecessor_two_update_count"]),
            (32, 128, 128),
        )


class TemporalSufficiencyTests(unittest.TestCase):
    def test_two_update_windows_first_fail_at_four_updates(self):
        two = cp.realized_paths(30, 2)
        three_local = cp.compose_two_update_windows(two, 3)
        four_local = cp.compose_two_update_windows(two, 4)
        self.assertEqual(three_local, cp.realized_paths(30, 3))
        self.assertEqual(len(three_local), 100)
        self.assertEqual(len(four_local), 224)
        self.assertEqual(len(cp.realized_paths(30, 4)), 208)
        self.assertEqual(len(four_local - cp.realized_paths(30, 4)), 16)

    def test_first_four_update_obstruction_has_exact_certificate(self):
        report = cp.build_report()
        certificate = report["evidence"]["first_obstruction_certificate"]
        self.assertEqual(
            report["result"]["lexicographically_first_four_update_spurious_path"],
            "000->000->100->011->010",
        )
        self.assertTrue(all(certificate["two_update_window_witnesses"]))
        self.assertEqual(certificate["first_two_updates_force_seven_bit_original"],
                         "0000001")
        self.assertEqual(
            certificate["compatible_nine_bit_originals_through_three_updates"],
            ["000000100", "000000101", "000000110", "000000111"],
        )
        self.assertEqual(certificate["four_bit_continuations_checked"], 16)
        self.assertEqual(certificate["forced_fourth_state_counts"],
                         {"100": 10, "110": 6})
        self.assertNotIn(certificate["requested_fourth_state"],
                         certificate["forced_fourth_state_counts"])
        self.assertIn("requires A=0",
                      certificate["truth_table_contradiction"][-1])

    def test_report_lists_all_sixteen_four_update_obstructions(self):
        report = cp.build_report()
        paths = report["result"]["all_four_update_spurious_paths"]
        self.assertEqual(len(paths), 16)
        self.assertEqual(paths, sorted(paths))


class CliTests(unittest.TestCase):
    def test_cli_output_is_deterministic_and_matches_api(self):
        script = pathlib.Path(cp.__file__)
        with tempfile.TemporaryDirectory() as directory:
            first = pathlib.Path(directory) / "first.json"
            second = pathlib.Path(directory) / "second.json"
            for output in (first, second):
                subprocess.run(
                    [sys.executable, str(script), "--output", str(output)],
                    check=True,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(first.read_text()), cp.build_report())


if __name__ == "__main__":
    unittest.main()
