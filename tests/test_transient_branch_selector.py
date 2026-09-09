import unittest

from experiments import diagonal_recursion
from experiments import transient_branch_selector as selector


class TransientBranchSelectorTests(unittest.TestCase):
    def test_selected_simulation_matches_full_reference(self):
        got = selector.simulate_selected([0, 2, 7, 19], 50)
        expected = diagonal_recursion.left_diagonals(50, 20)
        for d, column in got.items():
            self.assertEqual(list(column), expected[:, d].tolist())

    def test_last_reset_selects_each_even_parity_branch(self):
        for period in (2, 4, 8, 16):
            for predecessor in range(1 << period):
                if predecessor.bit_count() & 1:
                    continue
                candidates = selector.branch_words(predecessor, period)
                for last_one in range(period):
                    predicted = selector.predict_from_last_reset(
                        predecessor, period, last_one)
                    self.assertIn(predicted, candidates)

    def test_odd_predecessor_has_no_same_period_branch(self):
        with self.assertRaises(ValueError):
            selector.branch_words(0b1011, 4)

    def test_origin_boundary_selects_when_zero_from_start(self):
        predecessor = 0b1111
        self.assertIn(selector.predict_from_last_reset(predecessor, 4, None),
                      selector.branch_words(predecessor, 4))

    def test_one_tail_bit_always_selects_a_candidate(self):
        for predecessor in range(256):
            if predecessor.bit_count() & 1:
                continue
            candidates = selector.branch_words(predecessor, 8)
            for phase in range(8):
                for value in (0, 1):
                    predicted = selector.predict_from_tail_state(
                        predecessor, 8, phase, value)
                    self.assertIn(predicted, candidates)

    def test_tail_period_and_word_are_phase_locked(self):
        column = bytearray((0, 1, 1, 0)[t % 4] for t in range(40))
        self.assertEqual(selector.minimal_tail_period(column), 4)
        self.assertEqual(selector.periodic_word(column, 4), 0b0110)


if __name__ == "__main__":
    unittest.main()
