import unittest

from experiments.alternating_right_coherence import (
    COUNTEREXAMPLE_BOUNDARY,
    LOW_TRIPLE_GRAPH,
    allowed_boundary_words,
    admits_coherent_latches,
    boundary_word,
    check_coherent_bound,
    required_left_bits,
    right_macrostep,
    verify_five_bit_transitions,
)


class AlternatingRightCoherenceTest(unittest.TestCase):
    def test_nor_and_forced_low_triple_chain(self):
        check = verify_five_bit_transitions()
        self.assertEqual(check["five_bit_rows"], 32)
        self.assertEqual(check["truth_table_map_mismatches"], 0)
        self.assertEqual(check["nor_mismatches"], 0)
        self.assertEqual(check["forced_chain_mismatches"], 0)
        self.assertFalse(check["factor_graph_mismatches"])
        self.assertEqual(check["zero_rank_mismatches"], 0)
        self.assertEqual(LOW_TRIPLE_GRAPH[6], {0, 4})
        self.assertEqual(
            [right_macrostep(value) & 1 for value in range(8)],
            [1, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_exact_small_boundary_language(self):
        self.assertEqual(allowed_boundary_words(2), {"00", "01", "10"})
        self.assertNotIn("00000", allowed_boundary_words(5))
        words6 = allowed_boundary_words(6)
        self.assertNotIn("101001", words6)
        self.assertEqual(len(words6), 17)

    def test_cone_width_is_sufficient(self):
        # Bits beyond index 2m-2 cannot affect an m-bit trace.
        for length in range(1, 5):
            width = 2 * length - 1
            for value in range(1 << width):
                self.assertEqual(
                    boundary_word(value, length),
                    boundary_word(value | (0b101 << width), length),
                )

    def test_counterexample_boundary_is_incoherent(self):
        self.assertEqual(
            [i for i in range(len(COUNTEREXAMPLE_BOUNDARY) - 1)
             if COUNTEREXAMPLE_BOUNDARY[i:i + 2] == "11"],
            [8, 13, 16, 24],
        )

    def test_rule_specific_centre_constraints(self):
        for right in (0, 1):
            self.assertEqual(required_left_bits(30, 0, 1, right), (1 - right,))
            self.assertEqual(required_left_bits(30, 1, 0, right), (1,))
            self.assertEqual(required_left_bits(15, 0, 1, right), (0,))
            self.assertEqual(required_left_bits(15, 1, 0, right), (1,))
            self.assertEqual(required_left_bits(90, 0, 1, right), (1 - right,))
            self.assertEqual(required_left_bits(90, 1, 0, right), (right,))

    def test_coherent_filter_matches_direct_small_replay(self):
        # Independent centre-truth-table checking is inside admits_latches;
        # the packed transfer supplies only the inferred even-time s word.
        self.assertTrue(admits_coherent_latches(1, 3))
        self.assertFalse(admits_coherent_latches(1, 4))
        report = check_coherent_bound(weight=2, bound=8)
        self.assertTrue(report["verified_bound"])
        self.assertTrue(report["sharp"])
        self.assertIsNotNone(report["witness_surviving_bound_latches"])


if __name__ == "__main__":
    unittest.main()
