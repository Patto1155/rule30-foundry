import unittest
import copy
from experiments.periodic_strip_graph import analyze, build_graph, components, output_loops, verify_output_loops


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

    def test_nonperiodic_output_construction(self):
        # Distinct equal-length output blocks may be spliced at a common full
        # state. A non-eventually-periodic selector then gives such an output.
        for width in (3, 5, 9, 15):
            result = output_loops(width, "01")
            self.assertTrue(result["found"])
            self.assertTrue(verify_output_loops(result)["verified"])
            paths = result["closed_walks"]
            self.assertEqual(paths[0][0], paths[1][0])
            graph = build_graph(width, "01")
            for path, block in zip(paths, result["output_blocks"]):
                self.assertEqual(path[0], path[-1])
                self.assertEqual(len(path)-1, result["block_length"])
                for a, b in zip(path, path[1:]):
                    self.assertIn(b, graph[a])
                self.assertEqual(block, [(v[1] >> (width//2-1)) & 1 for v in path[:-1]])
            self.assertNotEqual(*result["output_blocks"])
            encoded = set()
            for choices in range(16):
                bits = []
                for j in range(4):
                    bits.extend(result["output_blocks"][(choices >> j) & 1])
                encoded.add(tuple(bits))
            self.assertEqual(len(encoded), 16)

    def test_reject_tampered_output_certificate(self):
        original = output_loops(5, "01")
        changed = copy.deepcopy(original)
        changed["output_blocks"][0][0] ^= 1
        with self.assertRaises(ValueError):
            verify_output_loops(changed)
        changed = copy.deepcopy(original)
        phase, row = changed["closed_walks"][0][1]
        changed["closed_walks"][0][1] = (phase, row ^ 2)
        with self.assertRaises(ValueError):
            verify_output_loops(changed)

    def test_no_branching_output_for_constant_centers(self):
        for word in ("0", "1"):
            for width in (5, 9):
                self.assertFalse(output_loops(width, word)["found"])


if __name__ == "__main__":
    unittest.main()
