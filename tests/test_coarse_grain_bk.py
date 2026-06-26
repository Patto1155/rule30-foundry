#!/usr/bin/env python3
"""Regression tests for b>=3 projection re-scoring semantics."""

import math
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))

import coarse_grain_bk
import coarse_grain_bk_verdict


class ProjectionScoreTest(unittest.TestCase):
    def tiny_prep(self):
        center = np.array([0, 1, 0, 1], dtype=np.int64)
        return {
            "nbrs": [
                np.array([0, 0, 1, 1], dtype=np.int64),
                center,
                np.array([1, 0, 1, 0], dtype=np.int64),
            ],
            "tgt": center,
            "M": 4,
            "n_pat": 2,
            "r": 1,
            "xp": np,
        }

    def test_score_projection_preserves_entropy_gate(self):
        score = coarse_grain_bk.score_projection(self.tiny_prep(), [0, 0], h_min=0.85)

        self.assertFalse(score["valid"])
        self.assertLess(score["entropy"], 0.85)
        self.assertTrue(math.isinf(score["excess"]))
        self.assertLess(score["excess"], 0.0)

    def test_score_projection_accepts_balanced_projection(self):
        score = coarse_grain_bk.score_projection(self.tiny_prep(), [0, 1], h_min=0.85)

        self.assertTrue(score["valid"])
        self.assertAlmostEqual(score["entropy"], 1.0)
        self.assertTrue(math.isfinite(score["excess"]))


class VerdictRescoreTest(unittest.TestCase):
    def test_invalid_full_rescore_cannot_win_best_shear(self):
        field = np.zeros((4, 4), dtype=np.uint8)

        def fake_prep_blocks(_field, _b, shear, _r, **_kwargs):
            return {"shear": shear, "M": 4}

        def fake_search_projection(_prep, **_kwargs):
            return {"best_pi": [0, 1], "best_closure": 0.9, "evals": 1}

        def fake_score_projection(prep, _pi, h_min):
            self.assertEqual(h_min, 0.85)
            if prep["shear"] == 0.0:
                return {
                    "closure": 0.99,
                    "excess": float("-inf"),
                    "entropy": 0.0,
                    "valid": False,
                }
            return {
                "closure": 0.5,
                "excess": 0.1,
                "entropy": 1.0,
                "valid": True,
            }

        with mock.patch.object(coarse_grain_bk_verdict, "prep_blocks", fake_prep_blocks):
            with mock.patch.object(coarse_grain_bk_verdict, "search_projection", fake_search_projection):
                with mock.patch.object(coarse_grain_bk_verdict, "score_projection", fake_score_projection):
                    best, per_shear = coarse_grain_bk_verdict.search_best_over_shears(
                        field, b=3, shears=[0.0, 1.0], r=1, budget=1,
                        msearch=4, hmin=0.85, sseed=1,
                    )

        self.assertEqual(best["shear"], 1.0)
        self.assertIsNone(per_shear["0.0"]["full_excess"])
        self.assertFalse(per_shear["0.0"]["full_valid"])


if __name__ == "__main__":
    unittest.main()
