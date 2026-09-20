import argparse
import tempfile
import unittest
from pathlib import Path

from tools.jev_conjecture_search import evaluate, exact_rows, make_cards, run


class GrammarTest(unittest.TestCase):
    def test_cards_are_unique_and_nontrivial(self):
        cards = make_cards(2)
        self.assertGreater(len(cards), 4000)
        self.assertEqual(len(cards), len({card["id"] for card in cards}))
        for card in cards:
            if card["lag"] == 0:
                self.assertNotIn(card["target"], [offset for offset, _ in card["premises"]])

    def test_more_premises_strictly_expands_the_grammar(self):
        self.assertLess(len(make_cards(0)), len(make_cards(1)))
        self.assertLess(len(make_cards(1)), len(make_cards(2)))


class ExactFalsifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cols = exact_rows(2, 8)

    def test_known_alternating_identity_survives(self):
        card = {"phase": 1, "premises": (), "target": -1, "lag": 0, "value": 1}
        result = evaluate(card, self.cols, 2, 3)
        self.assertGreater(result["support"], 0)
        self.assertEqual(result["verdict"], "PROVED_LOCAL_EXHAUSTIVE")

    def test_negated_identity_has_a_replayable_witness(self):
        card = {"phase": 1, "premises": (), "target": -1, "lag": 0, "value": 0}
        result = evaluate(card, self.cols, 2, 3)
        self.assertEqual(result["verdict"], "FALSIFIED")
        self.assertIsNotNone(result["witness"])
        self.assertEqual(result["witness"]["radius"], 2)


class CampaignTest(unittest.TestCase):
    def test_fixed_campaign_writes_a_reproducible_pool_and_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                out=Path(directory) / "run", policy="fixed", jev_provider="auto",
                preflight_radius=2, radius=3, window=3, steps=8,
                max_cards=12, max_tests=4, batch=4, max_premises=1,
                concurrency=2, seed=30, timeout=5.0)
            summary = run(args)
            self.assertEqual(summary["tests"], 4)
            self.assertEqual(summary["jev_calls"], 0)
            self.assertTrue((args.out / "pool.json").exists())
            self.assertTrue((args.out / "trace.json").exists())


if __name__ == "__main__":
    unittest.main()
