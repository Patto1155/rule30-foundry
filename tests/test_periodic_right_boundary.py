import random
import unittest
from experiments.periodic_right_boundary import trace, naive_trace, seed_snapshots, suffix_report, loop_completion_report
from experiments.periodic_strip_graph import output_loops


class RightBoundaryTests(unittest.TestCase):
    def test_independent_reference_across_word_boundary(self):
        rng = random.Random(30)
        for boundary in ("0", "1", "01", "10"):
            for initial in (0, 1, 1 << 65, rng.getrandbits(80), rng.getrandbits(8)):
                self.assertEqual(trace(initial, boundary, 90), naive_trace(initial, boundary, 90))

    def test_seed_snapshots_orientation(self):
        self.assertEqual(list(seed_snapshots(3)), [(0, 0, 1), (1, 1, 1), (2, 2, 0), (3, 7, 1)])

    def test_suffix_and_defect(self):
        self.assertEqual(suffix_report([1, 0] + [0, 1]*10, 4),
                         {"smallest_suffix_period": 2, "suffix_start": 10, "last_defect_time": 3})
        self.assertEqual(suffix_report([0]*20, 4)["smallest_suffix_period"], 1)
        with self.assertRaises(ValueError):
            suffix_report([0]*5, 4)

    def test_lightcone_and_controls(self):
        self.assertEqual(trace(0, "0", 90), [0]*91)
        a, b = trace(0, "01", 90), trace(1 << 65, "01", 90)
        self.assertEqual(a[:65], b[:65])

    def test_complete_right_state_removes_free_choices(self):
        for width in (3, 5, 9, 15):
            certificate = output_loops(width, "01")
            result = loop_completion_report(certificate)
            self.assertLessEqual(sum(c["right_completion_compatible"]
                                     for c in result["choices"]), 1)
            # Validate required-left comparisons using the independent row
            # simulator's cell-1 output, not the packed continuation.
            right = naive_trace(result["finite_right_initial"], "01", result["block_length"])
            for choice, block in enumerate(certificate["output_blocks"]):
                required = [((t+1) % 2) ^ ((t % 2) | right[t]) for t in range(len(block))]
                expected = next((t for t in range(len(block)) if required[t] != block[t]), None)
                self.assertEqual(result["choices"][choice]["first_required_left_mismatch"], expected)


if __name__ == "__main__":
    unittest.main()
