import unittest
import copy
import json
from pathlib import Path
import tempfile

from experiments.periodic_boundary_density import (
    controls,
    direct_period_wrap_mismatches,
    finite_cone_mismatches,
    orbit_report,
    period_four_ether_witness,
    rotate_time,
    rule30_ring_step,
    run,
    verify_report,
)


class PeriodicBoundaryDensityTest(unittest.TestCase):
    def test_time_rotation_uses_next_phase(self):
        self.assertEqual(rotate_time(0b0101, 4), 0b1010)
        self.assertEqual(rotate_time(0b0001, 4), 0b1000)

    def test_exact_one_seventh_cycle(self):
        report = orbit_report(boundary_word=0b10, period=2, max_states=100_000)
        self.assertEqual(report["status"], "cycle")
        self.assertEqual(report["transient_length"], 1)
        self.assertEqual(report["cycle_length"], 7)
        self.assertEqual(report["cycle_ones"], 1)
        self.assertEqual(report["cycle_output_lsb_spatial_order"], "0000001")
        self.assertGreater(report["charge_slope_numerator"], 0)

    def test_one_seventh_cycle_is_physical_space_time_ether(self):
        rows = ["1000000", "1100001", "0010011", "1111110", "1000000"]
        self.assertEqual([rule30_ring_step(row) for row in rows[:-1]], rows[1:])
        witness = period_four_ether_witness()
        self.assertEqual(witness["truth_table_step_mismatches"], 0)
        self.assertEqual(witness["alternating_centre_column"], "0101")
        self.assertEqual(witness["uniform_alpha_lower_bound"], 7)

    def test_independent_recurrence_checks(self):
        self.assertEqual(direct_period_wrap_mismatches(3), 0)
        self.assertEqual(finite_cone_mismatches(3), 0)

    def test_left_permutive_controls_are_distinct(self):
        report = controls()
        self.assertTrue(report["rule15"]["all_boundaries_have_zero_left_prefix"])
        self.assertEqual(report["rule15"]["trace_mismatches"], 0)
        self.assertEqual(report["rule90"]["distinct_left_prefixes"], 4)
        self.assertEqual(report["rule90"]["trace_mismatches"], 0)

    def test_declared_scan_finds_no_falsifier(self):
        report = run(max_period=6, max_states=100_000)
        self.assertEqual(report["total_right_censored"], 0)
        self.assertEqual(report["global_minimum_cycle_density"],
                         {"numerator": 1, "denominator": 7})
        self.assertEqual(report["cycles_below_one_eighth"], [])
        self.assertFalse(report["falsifies_uniform_charge_bound"])
        self.assertEqual(report["periods"][-1]["boundary_words"], 64)

    def test_saved_report_replays_exactly_and_rejects_tampering(self):
        report = run(max_period=6, max_states=100_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            verified = verify_report(path)
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["boundary_words"], 126)

            tamperings = []
            count = copy.deepcopy(report)
            count["periods"][5]["boundary_words"] = 63
            tamperings.append(count)
            cycle = copy.deepcopy(report)
            cycle["periods"][1]["minimum_examples"][0][
                "cycle_output_lsb_spatial_order"] = "1000000"
            tamperings.append(cycle)
            density = copy.deepcopy(report)
            density["global_minimum_cycle_density"] = {
                "numerator": 1, "denominator": 8}
            tamperings.append(density)
            for tampered in tamperings:
                path.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "differs from exact replay"):
                    verify_report(path)

            scope = copy.deepcopy(report)
            scope["declared_limits"]["max_period"] = 5
            path.write_text(json.dumps(scope), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declared_limits mismatch"):
                verify_report(path)


if __name__ == "__main__":
    unittest.main()
