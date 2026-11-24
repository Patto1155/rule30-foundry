"""tools/verify_lean.py must refuse a weakened or unfinished proof.

These run without a Lean toolchain: they exercise the static checks and the
output parser, which are what stop a `sorry`, a smuggled axiom or a quietly
weakened statement from being reported as a proof.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_lean  # noqa: E402

REL = "formal/Rule30/PeriodOne.lean"
SPEC = verify_lean.FILES[REL]
GOOD = "\n".join(
    f"'{t}' depends on axioms: [propext, Classical.choice, Quot.sound]"
    for t in SPEC["theorems"])


class StaticChecks(unittest.TestCase):
    def test_committed_source_is_clean(self):
        self.assertEqual(verify_lean.check_source(REL, SPEC), [])

    def _mutated(self, old, new):
        path = ROOT / REL
        original = path.read_text(encoding="utf-8")
        self.assertIn(old, original)
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            return verify_lean.check_source(REL, SPEC)
        finally:
            path.write_text(original, encoding="utf-8")

    def test_sorry_is_rejected(self):
        self.assertTrue(self._mutated("exact rule_latch _", "sorry"))

    def test_weakened_statement_is_rejected(self):
        # dropping the nonzero hypothesis' partner: allow only b = true
        self.assertTrue(self._mutated(
            "T ≤ t → evolve x t c = b := by", "T ≤ t → evolve x t c = true := by"))

    def test_changed_rule_is_rejected(self):
        self.assertTrue(self._mutated("l ^^ (c || r)", "l ^^ (c ^^ r)"))

    def test_added_axiom_is_rejected(self):
        self.assertTrue(self._mutated(
            "namespace Rule30\n", "namespace Rule30\naxiom cheat : False\n"))


class OutputChecks(unittest.TestCase):
    def test_good_output_passes(self):
        self.assertEqual(verify_lean.check_output(SPEC, 0, GOOD), [])

    def test_sorry_axiom_is_rejected(self):
        bad = GOOD.replace("propext", "propext, sorryAx", 1)
        self.assertTrue(verify_lean.check_output(SPEC, 0, bad))

    def test_nonzero_exit_is_rejected(self):
        self.assertTrue(verify_lean.check_output(SPEC, 1, GOOD))

    def test_missing_report_is_rejected(self):
        self.assertTrue(verify_lean.check_output(SPEC, 0, GOOD.split("\n")[0]))


if __name__ == "__main__":
    unittest.main()
