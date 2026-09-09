import copy
import json
import unittest
from pathlib import Path

from tools import verify_dfao_n56 as verifier


ROOT = Path(__file__).resolve().parents[1]


class DfaoN56VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        cls.artifact = load("runs/dfao-n56-msd-s12-2026-09-09.json")
        cls.curve = load("data/prize/2026-08-15-dfao-min-state-curve.json")
        cls.anchor = load("data/prize/2026-08-30-dfao-drat-proofs.json")

    def test_static_dependencies_and_upper_witness(self):
        result = verifier.verify(self.artifact, self.curve, self.anchor)
        self.assertTrue(result["static_checks_verified"])
        self.assertFalse(result["certificate_reproved"])
        self.assertEqual(result["s_star_56_msd"], 13)

    def test_cnf_hash_tamper_is_rejected(self):
        changed = copy.deepcopy(self.artifact)
        changed["result"]["cnf_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verifier.verify(changed, self.curve, self.anchor)

    def test_uncertified_anchor_is_rejected(self):
        changed = copy.deepcopy(self.anchor)
        row = verifier.one(changed["results"], sequence="center",
                           direction="msd", base=2, n=48)
        row["lower_bound_certified"] = False
        with self.assertRaises(ValueError):
            verifier.verify(self.artifact, self.curve, changed)


if __name__ == "__main__":
    unittest.main()
