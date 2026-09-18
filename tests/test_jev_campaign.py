"""Scheduling checks for tools/jev_campaign.py.

The driver certifies nothing; these guard the plan shapes it hands to
jev_search.load_plan, which is where the real validation lives.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.jev_campaign import CONTROLS, shard_plan
from tools.jev_search import load_plan


def write(tmp, plan):
    path = tmp / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


class ShardPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "_campaign_tmp"
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self):
        for f in self.tmp.iterdir():
            f.unlink()
        self.tmp.rmdir()

    def test_shard_plan_is_accepted_by_the_real_validator(self):
        for n in (20, 32, 64):
            plan = shard_plan("goal", n, [4, 5, 6, 7], seed=30)
            self.assertEqual(load_plan(write(self.tmp, plan))["scope"], "frontier")

    def test_shard_overlapping_a_control_drops_the_duplicate_instance(self):
        # control-center16-s4 already pins (center, n=16, states=4); emitting a
        # frontier card there makes load_plan reject the whole plan.
        plan = shard_plan("goal", 16, [3, 4, 5], seed=30)
        ids = [c["id"] for c in plan["cards"]]
        self.assertNotIn("center16-s4", ids)
        self.assertNotIn("random16-s4", ids)
        self.assertIn("center16-s3", ids)
        load_plan(write(self.tmp, plan))

    def test_every_center_frontier_card_has_a_matched_random_null(self):
        plan = shard_plan("goal", 40, [8, 9, 10], seed=7)
        centers = [c for c in plan["cards"]
                   if c["role"] == "frontier" and c["sequence"] == "center"]
        self.assertTrue(centers)
        for c in centers:
            self.assertTrue(any(
                x["sequence"] == "random" and x["role"] == "frontier"
                and all(x[k] == c[k] for k in ("n", "states", "base", "direction"))
                for x in plan["cards"]))

    def test_center_cards_are_scheduled_ahead_of_their_nulls(self):
        # The fixed policy walks the plan in order and a null runs 3-4x longer
        # than the center card it controls, so an interleaved plan lets a
        # control strand the frontier card behind it.
        plan = shard_plan("goal", 64, [12, 13, 14], seed=30)
        frontier = [c for c in plan["cards"] if c["role"] == "frontier"]
        last_center = max(i for i, c in enumerate(frontier) if c["sequence"] == "center")
        first_null = min(i for i, c in enumerate(frontier) if c["sequence"] == "random")
        self.assertLess(last_center, first_null)

    def test_controls_are_present_and_carry_expected_answers(self):
        plan = shard_plan("goal", 44, [9, 10], seed=30)
        controls = [c for c in plan["cards"] if c["role"] == "calibration"]
        self.assertEqual(len(controls), len(CONTROLS))
        self.assertEqual({c["expected"] for c in controls}, {"SAT", "UNSAT"})

    def test_frontier_cards_never_embed_an_answer_label(self):
        plan = shard_plan("goal", 48, [11, 12], seed=30)
        for c in plan["cards"]:
            if c["role"] == "frontier":
                self.assertNotIn("expected", c)


if __name__ == "__main__":
    unittest.main()
