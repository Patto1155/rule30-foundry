"""Independent geometry, edge-domain and hostile certificate mutation checks."""
import copy
import unittest

from experiments.alternating_left_weight import induced_left_prefix
from experiments.slanted_cone_potential import (
    FeatureSpec, diagonal_cone, feature, forward_cut, full_cut,
    genuine_edges, negative_cycle, pack, quotient_graph, verify_cycle, verify_report,
)


class SlantedConeTests(unittest.TestCase):
    def test_full_diagonals_against_independent_forward_truth_table(self):
        for macrosteps in range(1, 7):
            for s in range(1 << macrosteps):
                diagonals = diagonal_cone(s, macrosteps)
                initial = pack([word[0] for word in diagonals[1:]])
                self.assertEqual(list(induced_left_prefix(s, macrosteps)),
                                 [word[0] for word in diagonals[1:]])
                for q in range(2 * macrosteps + 1):
                    self.assertEqual(full_cut(diagonals, q), forward_cut(initial, q, 30))

    def test_rule15_has_zero_initial_prefix_and_negative_cycle(self):
        diagonals = diagonal_cone(0, 8, rule=15)
        self.assertEqual(pack([word[0] for word in diagonals[1:]]), 0)
        self.assertEqual(full_cut(diagonals, 16), forward_cut(0, 16, 15))
        nodes, edges = quotient_graph(genuine_edges(8, rule=15), FeatureSpec())
        certificate = {"features": {"bottom": 0, "top": 0, "count": "none"},
                       "rule": 15, "block_space": 8,
                       "cycle": negative_cycle(len(nodes), edges)}
        self.assertEqual(verify_cycle(certificate)["cycle_charge"], -8)

    def make_certificate(self):
        nodes, edges = quotient_graph(genuine_edges(8), FeatureSpec())
        return {"features": {"bottom": 0, "top": 0, "count": "none"},
                "rule": 30, "block_space": 8,
                "cycle": negative_cycle(len(nodes), edges)}

    def test_exact_spurious_cycle_and_missing_full_cut(self):
        certificate = self.make_certificate()
        checked = verify_cycle(certificate)
        self.assertEqual(checked["cycle_edges"], 1)
        self.assertEqual(checked["cycle_charge"], -8)
        self.assertFalse(checked["full_cut_seams"][0]["full_cuts_equal"])

    def test_verifier_rejects_changed_charge_domain_rule_and_invalid_latch(self):
        certificate = self.make_certificate()
        for field, replacement in [("charge", -9), ("start_depth", 7),
                                   ("end_depth", 17), ("rule", 15), ("initial_left", 0)]:
            damaged = copy.deepcopy(certificate)
            damaged["cycle"][0][field] = replacement
            with self.assertRaises(ValueError):
                verify_cycle(damaged)

    def test_bellman_ford_nontrivial_cycle_and_acyclic_negative_edge(self):
        edges = [{"source": 0, "target": 1, "charge": -2},
                 {"source": 1, "target": 0, "charge": 1}]
        cycle = negative_cycle(2, edges)
        self.assertEqual(len(cycle), 2)
        self.assertEqual(sum(edge["charge"] for edge in cycle), -1)
        self.assertIsNone(negative_cycle(2, edges[:1]))

    def test_targeted_four_bottom_bit_cycle(self):
        certificate = {"features": {"bottom": 4, "top": 0, "count": "none"},
                       "rule": 30, "block_space": 8, "cycle": [
                           {"start_depth": 24, "end_depth": 32, "initial_left": 2126145,
                            "charge": -8, "rule": 30},
                           {"start_depth": 24, "end_depth": 32, "initial_left": 536874262,
                            "charge": 0, "rule": 30}]}
        self.assertEqual(verify_cycle(certificate)["cycle_charge"], -8)
        self.assertEqual(verify_report({"candidates": [certificate]})["verified_negative_cycles"], 1)

    def test_nonpositive_population_potential_is_falsified(self):
        # Genuine 8->16 zero-input tile. Both cut populations INCREASE,
        # so -alpha*wt(Dq-1)-beta*wt(Dq), alpha,beta>=0, cannot pay -8.
        source, target = forward_cut(129, 8, 30), forward_cut(129, 16, 30)
        self.assertEqual([sum(word) for word in source], [3, 6])
        self.assertEqual([sum(word) for word in target], [8, 7])
        self.assertEqual(8 * (129 >> 8).bit_count() - 8, -8)


if __name__ == "__main__":
    unittest.main()
