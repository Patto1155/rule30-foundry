"""Independent checks for the interface-obstruction proofs."""

import unittest

from tools.verify_interface_obstructions import (
    ETHER,
    advance_interface,
    cell,
    interface_step,
    full_row_first_centre_failure,
    macro_residues,
    small_return_map_order,
    triangular_exponent_bound,
    verify,
    verify_small_edit_family,
    verify_physical_full_row_corollary,
)


class InterfaceObstructionVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = verify()

    def test_symbolic_unitriangular_exponent(self):
        self.assertEqual(triangular_exponent_bound(1), 1)
        self.assertEqual(triangular_exponent_bound(28), 1 << 27)
        self.assertEqual(self.result["prefix_L_exponent_formula"], "2^(L-1)")
        # "Divides" is essential: equality already fails at width four.
        self.assertEqual(small_return_map_order(4), 4)
        self.assertNotEqual(small_return_map_order(4), 1 << (4 - 1))
        self.assertIn("not a formal proof", self.result["induction_check_scope"])

    def test_cut_phase_is_required_for_inverse_survivor(self):
        self.assertEqual(interface_step(3, -2, -2, 16), 1)
        self.assertEqual(interface_step(1, -1, -2, 16), 1)
        # The formerly used cut-zero phase convention would be wrong here.
        self.assertNotEqual(interface_step(3, -2, 0, 16), 1)

    def test_formal_arrival_and_boundary_merge(self):
        formal = self.result["formal_survivor"]
        self.assertEqual(formal["formal_defects_at_times_minus2_minus1_zero"],
                         [3, 1, 1])
        self.assertEqual(formal["arrival_ether_phase"], 2)
        self.assertEqual(formal["first_boundary_after_arrival"], 0)
        self.assertEqual(formal["merge_phase"], 3)
        self.assertEqual(formal["possibly_affected_merge_outputs"], [0, 1, 2])

    def test_finite_approximant_phase_and_prefix(self):
        for width in range(2, 9):
            k = width - 1
            cut = 28 * (1 << k) - 2
            self.assertEqual(advance_interface(1, 0, cut, cut, width), 3)

    def test_ether_boundary_is_compatible_before_arrival(self):
        for time in range(28):
            self.assertEqual(ETHER[time % 4][6], time & 1)

    def test_fixed_hole_q_and_locality(self):
        fixed = self.result["fixed_hole"]
        self.assertEqual(fixed["q_n6"], 21)
        self.assertEqual(fixed["residues_through_q"],
                         "1212121212121212121213")
        self.assertEqual(fixed["dependency_indices_through_failure"], [0, 43])
        self.assertEqual(fixed["next_possible_outer_one"], 49)

    def test_small_bit_18_arrival_and_delay(self):
        family = verify_small_edit_family()
        self.assertEqual(family["arrival_macrotime"], 9)
        self.assertEqual(family["arrival_prefix_width"], 22)
        self.assertEqual(family["arrival_bits_low_index_first"],
                         "1010100001000001011010")
        self.assertEqual(family["post_arrival_residues_delays_0_through_10"],
                         [1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 0])
        self.assertEqual(family["post_arrival_failure_delay"], 10)
        self.assertEqual(family["q_for_all_r_at_least_6"], 19)
        self.assertEqual(family["initial_dependency_indices_through_failure"],
                         [0, 39])
        self.assertEqual(family["first_omitted_ether_one_lower_bound"], 42)

    def test_physical_full_row_fixed_bit_gain(self):
        corollary = verify_physical_full_row_corollary()
        self.assertEqual(corollary["ether_centre_trace_period"], [0, 1, 0, 1])
        self.assertEqual(corollary["left_half_bit_42_physical_site"], -43)
        self.assertEqual(corollary["restored_exact_first_failure"], "7R+1")
        self.assertEqual(
            corollary["deleted_exact_first_failure_for_R_at_least_7"], 43)
        self.assertEqual(corollary["restoration_prefix_gain"], "7R-42")
        self.assertIn("no eventual-trace claim", corollary["claim_scope"])
        for radius in (7, 8, 9):
            self.assertEqual(
                full_row_first_centre_failure(radius, False, 7 * radius + 1),
                7 * radius + 1)
            self.assertEqual(
                full_row_first_centre_failure(radius, True, 43), 43)

    def test_truth_table_replay_is_not_a_packed_kernel(self):
        n6 = sum(1 << (7 * j) for j in range(6))
        self.assertEqual(macro_residues(n6, 21, rule=30)[-1], 3)
        self.assertEqual(cell(30, 1, 0, 0), 1)
        self.assertEqual(cell(30, 0, 0, 0), 0)


if __name__ == "__main__":
    unittest.main()
