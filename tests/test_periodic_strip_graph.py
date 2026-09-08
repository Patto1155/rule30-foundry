import unittest
from experiments.periodic_strip_graph import analyze, build_graph, components


class StripGraphTests(unittest.TestCase):
    def test_edges_against_truth_table_and_free_boundaries(self):
        for word in ("0", "1", "01"):
            width = 5
            graph = build_graph(width, word)
            for (phase, row), edges in graph.items():
                expected = []
                for other in range(1 << width):
                    if (other >> 2) & 1 != int(word[(phase + 1) % len(word)]):
                        continue
                    valid = True
                    for i in range(1, width - 1):
                        code = sum(((row >> (i + j - 1)) & 1) << (2-j)
                                   for j in range(3))
                        valid &= ((other >> i) & 1) == ((30 >> code) & 1)
                    if valid:
                        expected.append(((phase+1) % len(word), other))
                self.assertEqual(set(edges), set(expected))

    def test_scc_against_known_graph(self):
        graph = {0: [1], 1: [0, 2], 2: [3], 3: [2], 4: []}
        self.assertEqual({frozenset(s) for s in components(graph)},
                         {frozenset((0, 1)), frozenset((2, 3)), frozenset((4,))})

    def test_constant_one_forces_left_zero(self):
        for width in (3, 5, 7, 9):
            result = analyze(width, "1")
            self.assertGreater(result["recurrent_states"], 0)
            self.assertTrue(result["left_eventually_has_word_period"])

    def test_no_sync_control_and_closed_walk_witness(self):
        result = analyze(3, "0")
        self.assertFalse(result["left_eventually_has_word_period"])
        for width in (3, 5, 7, 9):
            for word in ("0", "01"):
                result = analyze(width, word)
                self.assertGreater(result["recurrent_states"], 0)
                witness = result["witness"]
                if witness is None:
                    continue
                graph = build_graph(width, word)
                closed = witness["closed_walk"]
                self.assertEqual(closed[0], closed[-1])
                for a, b in zip(closed, closed[1:]):
                    self.assertIn(b, graph[a])
                path = witness["mismatch_path"]
                self.assertEqual(len(path), len(word) + 1)
                self.assertEqual(((path[0][1] ^ path[-1][1]) >> (width//2-1)) & 1, 1)


if __name__ == "__main__":
    unittest.main()
