"""Hostile mutations for the standalone coherent-density verifier."""

import copy
import json
from pathlib import Path
import unittest

from tools.verify_coherent_density import (
    COHERENT_STATES,
    COHERENT_TRIPLES,
    TEMPORAL_EDGES,
    TEMPORAL_STATES,
    forward_left_cone,
    verify_candidate,
    verify_report,
    verify_stronger_obstruction,
    verify_temporal_candidate,
    verify_temporal_report,
)


class VerifyCoherentDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((Path(__file__).resolve().parents[1] /
            "runs/theory/coherent-density-obstructions.json").read_text())
        cls.temporal_report = json.loads((Path(__file__).resolve().parents[1] /
            "runs/theory/temporal-pair-density.json").read_text())

    def test_replays_all_four_exact_integer_contradictions(self):
        result = verify_report(self.report)
        self.assertEqual(result["integer_contradictions"], [2, 2, 6, 6])
        self.assertEqual((result["coherent_states"], result["coherent_edges"]),
                         (20, 44))

    def test_shared_predecessor_control_excludes_resampled_walk(self):
        self.assertEqual(len(COHERENT_STATES), 20)
        self.assertEqual(len(COHERENT_TRIPLES), 44)
        self.assertNotIn(((0, 0, 0), (1, 0, 0), (1, 1, 1)),
                         COHERENT_TRIPLES)

    def test_rejects_tampered_positive_rhs_summary(self):
        damaged = copy.deepcopy(self.report["candidates"][0])
        damaged["verification"]["contradiction"] = 3
        damaged["verification"]["identity"] = "0 >= 3"
        with self.assertRaises(ValueError):
            verify_candidate(damaged)

    def test_rejects_tampered_top_label(self):
        damaged = copy.deepcopy(self.report["candidates"][0])
        row = next(entry for entry in damaged["contributing_constraints"]
                   if entry["label"][0] == "top")
        row["label"][4][0] ^= 1
        with self.assertRaisesRegex(ValueError, "top-boundary"):
            verify_candidate(damaged)

    def test_rejects_phase_outside_scope(self):
        damaged = copy.deepcopy(self.report["candidates"][0])
        row = next(entry for entry in damaged["contributing_constraints"]
                   if entry["label"][0] == "interior")
        row["label"][2] = 1
        with self.assertRaisesRegex(ValueError, "phase"):
            verify_candidate(damaged)

    def test_rejects_noncoherent_transition(self):
        damaged = copy.deepcopy(self.report["candidates"][0])
        row = next(entry for entry in damaged["contributing_constraints"]
                   if entry["label"][0] == "interior")
        row["label"][3:] = [[0, 0, 0], [1, 0, 0], [1, 1, 1]]
        with self.assertRaisesRegex(ValueError, "shared-predecessor"):
            verify_candidate(damaged)

    def test_rejects_illegal_terminal_transition(self):
        damaged = copy.deepcopy(self.report["candidates"][0])
        row = next(entry for entry in damaged["contributing_constraints"]
                   if entry["label"][0] == "terminal_flow" and
                   entry["label"][3] == [0, 1])
        row["label"][4] = [0, 0]
        with self.assertRaisesRegex(ValueError, "terminal pair transition"):
            verify_candidate(damaged)

    def test_rejects_changed_labeled_multiplier(self):
        damaged = copy.deepcopy(self.report["candidates"][-1])
        damaged["contributing_constraints"][0]["multiplier"] += 1
        with self.assertRaises(ValueError):
            verify_candidate(damaged)

    def test_rejects_changed_dense_multiplier(self):
        damaged = copy.deepcopy(self.report["candidates"][-1])
        index = next(i for i, value in enumerate(damaged["farkas_multipliers"])
                     if value)
        damaged["farkas_multipliers"][index] += 1
        with self.assertRaisesRegex(ValueError, "multipliers disagree"):
            verify_candidate(damaged)

    def test_rejects_unknown_scope(self):
        damaged = copy.deepcopy(self.report["candidates"][0])
        damaged["alpha"] = 7
        with self.assertRaisesRegex(ValueError, "unknown candidate scope"):
            verify_candidate(damaged)

    def test_witnesses_omit_endpoints_and_use_zero_emission_bottoms(self):
        result = verify_report(self.report)
        self.assertEqual(set(result["omitted_constraint_kinds"]),
                         {"initial", "terminal_bound", "slope_floor"})
        self.assertTrue(result["all_bottom_emissions_zero"])
        self.assertTrue(result["finite_alpha_independent"])
        self.assertTrue(result["endpoint_deficit_independent"])

    def test_n129_forward_cone_and_required_latches(self):
        result = verify_stronger_obstruction()
        self.assertEqual(result["cuts"], {
            "3": "0101", "4": "00110", "5": "001011",
            "6": "0001010", "7": "00010101",
        })
        self.assertEqual(result["required_latches"], {"1": 1, "3": 1, "5": 1})
        self.assertEqual(result["odd_diagonal_endpoint_bits"],
                         {"1": 1, "3": 0, "5": 1, "7": 0})
        self.assertEqual(result["triple_paths"], {
            "4": ["000", "110", "001"],
            "5": ["000", "100", "011", "110"],
            "6": ["000", "000", "110", "101", "010"],
            "7": ["000", "000", "100", "011", "100", "111"],
        })
        # The cone is stepped from n=129 directly, rather than reconstructed
        # from the expected diagonal strings in this test.
        self.assertEqual(forward_left_cone(129, 7)[3][1], 1)

    def test_stronger_obstruction_has_exact_multiplicities_and_density(self):
        result = verify_stronger_obstruction()
        self.assertEqual(result["spatial_multiplicities"],
                         {"4": 2, "5": 2, "6": 1, "7": 1})
        self.assertEqual(result["phi_coefficients"], {"3": -2, "5": 1, "7": 1})
        self.assertEqual(result["alpha_coefficient"], 0)
        self.assertEqual(result["spatial_lower_bound"], 6)
        self.assertEqual(result["bottom_correction_residual"], {})
        self.assertEqual(result["density_coefficients"], [
            {"variable": ["F", 1, 0, 0, 0], "coefficient": 1},
            {"variable": ["F", 1, 1, 0, 0], "coefficient": 2},
            {"variable": ["F", 1, 0, 1, 1], "coefficient": 1},
            {"variable": ["F", 1, 1, 1, 0], "coefficient": 1},
            {"variable": ["F", 1, 0, 0, 1], "coefficient": 1},
        ])

    def test_same_density_is_sum_of_three_terminal_cycles(self):
        result = verify_stronger_obstruction()
        self.assertEqual(result["terminal_cycles"], [
            ["00->00", "00->00"],
            ["11->00", "00->11"],
            ["01->10", "10->01"],
        ])
        self.assertEqual(result["terminal_upper_bound"], 0)
        self.assertIn(">= 6 and <= 0", result["contradiction"])

    def test_temporal_pair_replays_four_exact_obstructions(self):
        result = verify_temporal_report(self.temporal_report)
        self.assertEqual(result["integer_contradictions"], [4, 4, 4, 4])
        self.assertTrue(result["all_alpha_coefficients_cancel"])
        self.assertEqual(result["initialization_constraints"], 0)

    def test_temporal_projection_and_endpoint_controls(self):
        result = verify_temporal_report(self.temporal_report)["controls"]
        self.assertEqual((len(TEMPORAL_STATES), len(TEMPORAL_EDGES)), (10, 22))
        self.assertEqual(result["spatial_telescope_depths"], [3, 4, 5, 6, 7])
        self.assertEqual(result["q3_interior_edges"], 0)
        self.assertEqual(result["terminal_extrema_domain_sizes"],
                         {"bottom": 10, "top_even": 2, "top_odd": 4})

    def test_temporal_rejects_illegal_projected_transition(self):
        damaged = copy.deepcopy(self.temporal_report["candidates"][0])
        row = next(entry for entry in damaged["contributing_constraints"]
                   if entry["label"][0] == "terminal_flow")
        source = tuple(tuple(pair) for pair in row["label"][3])
        illegal = next(state for state in TEMPORAL_STATES
                       if (source, state) not in TEMPORAL_EDGES)
        row["label"][4] = [list(pair) for pair in illegal]
        with self.assertRaisesRegex(ValueError, "projected transition"):
            verify_temporal_candidate(damaged)

    def test_temporal_rejects_top_multiplier_rhs_and_scope_tampering(self):
        base = self.temporal_report["candidates"][-1]
        damaged = copy.deepcopy(base)
        top = next(entry for entry in damaged["contributing_constraints"]
                   if entry["label"][0] == "top")
        top["label"][4][0][0] ^= 1
        with self.assertRaises(ValueError):
            verify_temporal_candidate(damaged)
        damaged = copy.deepcopy(base)
        damaged["contributing_constraints"][0]["multiplier"] += 1
        with self.assertRaises(ValueError):
            verify_temporal_candidate(damaged)
        damaged = copy.deepcopy(base)
        damaged["verification"]["contradiction"] = 5
        damaged["verification"]["identity"] = "0 >= 5"
        with self.assertRaises(ValueError):
            verify_temporal_candidate(damaged)
        damaged = copy.deepcopy(base)
        damaged["alpha"] = 7
        with self.assertRaisesRegex(ValueError, "scope"):
            verify_temporal_candidate(damaged)


if __name__ == "__main__":
    unittest.main()
