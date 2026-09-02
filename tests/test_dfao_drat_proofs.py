#!/usr/bin/env python3
"""Logic tests for the s*(n) DRAT certification harness.

The parts that need a SAT solver and a DRAT checker on disk live in
``python experiments/dfao_drat_proofs.py --self-test`` (wired into
tools/verify_all.py, and skipped when the toolchain is absent). What is tested
here is the part that decides *which* instances have to be proved - get that
wrong and the certificate covers less than it claims while still reporting
100% verified.
"""

import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import dfao_drat_proofs
except (Exception, SystemExit) as exc:  # pragma: no cover - dependency-driven
    # SystemExit inherits from BaseException, not Exception. A bare
    # `except Exception` here silently deleted 23 tests when an optional
    # dependency was absent and an imported module raised SystemExit at
    # module scope -- see tests/test_import_safety.py, which now forbids that
    # at source. This guard is the second line of defence, not the first.
    raise unittest.SkipTest(f"cannot import dfao_drat_proofs: {exc}")


def row(sequence="center", n=16, direction="msd", s_star=5, exact=True,
        seed=None):
    return {
        "sequence": sequence, "n": n, "direction": direction, "base": 2,
        "s_star": s_star, "s_star_exact": exact, "seed": seed,
        "sha256_bits_ascii": "0" * 64, "per_state": [],
    }


class InstanceEnumerationTest(unittest.TestCase):
    def test_covers_every_state_below_s_star(self):
        todo = dfao_drat_proofs.lower_bound_instances(
            {"results": [row(s_star=5)]}, None)
        self.assertEqual([t["states"] for t in todo], [1, 2, 3, 4])

    def test_does_not_try_to_refute_s_star_itself(self):
        """s* is the SAT verdict. Asking for its refutation would be a bug."""
        todo = dfao_drat_proofs.lower_bound_instances(
            {"results": [row(s_star=5)]}, None)
        self.assertNotIn(5, [t["states"] for t in todo])

    def test_skips_rows_without_an_exact_s_star(self):
        """A row that timed out has no lower bound to certify."""
        todo = dfao_drat_proofs.lower_bound_instances(
            {"results": [row(exact=False, s_star=None)]}, None)
        self.assertEqual(todo, [])

    def test_sequence_filter(self):
        curve = {"results": [row(sequence="center", s_star=3),
                             row(sequence="random", s_star=4, seed=30)]}
        todo = dfao_drat_proofs.lower_bound_instances(curve, ["center"])
        self.assertEqual({t["row"]["sequence"] for t in todo}, {"center"})
        self.assertEqual(len(todo), 2)

    def test_no_filter_takes_every_sequence(self):
        curve = {"results": [row(sequence="center", s_star=3),
                             row(sequence="random", s_star=4, seed=30)]}
        todo = dfao_drat_proofs.lower_bound_instances(curve, None)
        self.assertEqual(len(todo), 2 + 3)


class CurveCoverageTest(unittest.TestCase):
    """The real curve artifact, if present, must enumerate what we expect."""

    CURVE = (REPO_ROOT / "data" / "prize" /
             "2026-08-15-dfao-min-state-curve.json")

    def setUp(self):
        if not self.CURVE.exists():
            raise unittest.SkipTest(f"{self.CURVE.name} absent")
        import json
        self.curve = json.loads(self.CURVE.read_text(encoding="utf-8"))

    def test_every_exact_row_is_fully_covered(self):
        todo = dfao_drat_proofs.lower_bound_instances(self.curve, None)
        by_row = {}
        for item in todo:
            r = item["row"]
            by_row.setdefault((r["sequence"], r["direction"], r["n"]),
                              []).append(item["states"])
        for r in self.curve["results"]:
            if not r["s_star_exact"]:
                continue
            key = (r["sequence"], r["direction"], r["n"])
            self.assertEqual(sorted(by_row[key]), list(range(1, r["s_star"])),
                             f"incomplete lower-bound coverage for {key}")

    def test_it_covers_the_implied_verdicts_too(self):
        """The point of the re-run: UNSAT_IMPLIED must not be taken on faith."""
        todo = dfao_drat_proofs.lower_bound_instances(self.curve, None)
        implied = 0
        for item in todo:
            prev = next((p for p in item["row"]["per_state"]
                         if p["states"] == item["states"]), None)
            if prev and prev["status"] == "UNSAT_IMPLIED":
                implied += 1
        self.assertGreater(implied, 100,
                           "most instances in the original run were discharged "
                           "by monotonicity; they must be in the re-proof set")


class StandaloneEvaluatorTest(unittest.TestCase):
    """The witness evaluator must not agree with prize_lab by construction.

    It is written from the DFAO definition precisely so that a mis-encoding in
    prize_lab cannot validate its own witnesses. That only means anything if
    the two implementations actually agree on the recorded witnesses - and if
    the standalone one is right about a sequence whose automaton is known.
    """

    def test_thue_morse_two_state_dfao(self):
        """Thue-Morse: output = parity of the popcount of the index."""
        candidate = {
            "base": 2, "direction": "msd", "initial_state": 0,
            "transitions": [[0, 1], [1, 0]], "outputs": [0, 1], "states": 2,
        }
        got = [dfao_drat_proofs.eval_dfao(candidate, i) for i in range(64)]
        want = [bin(i).count("1") % 2 for i in range(64)]
        self.assertEqual(got, want)

    def test_index_zero_is_a_single_zero_digit(self):
        """The empty digit string would leave the DFAO in its initial state."""
        candidate = {
            "base": 2, "direction": "msd", "initial_state": 0,
            "transitions": [[1, 0], [1, 1]], "outputs": [0, 1], "states": 2,
        }
        self.assertEqual(dfao_drat_proofs.eval_dfao(candidate, 0), 1)

    def test_direction_changes_the_output(self):
        """msd and lsd are genuinely different readings, not a no-op flag."""
        base = {"base": 2, "initial_state": 0,
                "transitions": [[0, 1], [2, 2], [0, 0]],
                "outputs": [0, 1, 0], "states": 3}
        msd = [dfao_drat_proofs.eval_dfao({**base, "direction": "msd"}, i)
               for i in range(32)]
        lsd = [dfao_drat_proofs.eval_dfao({**base, "direction": "lsd"}, i)
               for i in range(32)]
        self.assertNotEqual(msd, lsd)

    def test_agrees_with_prize_lab_on_every_recorded_witness(self):
        import json
        curve = (REPO_ROOT / "data" / "prize" /
                 "2026-08-15-dfao-min-state-curve.json")
        if not curve.exists():
            raise unittest.SkipTest(f"{curve.name} absent")
        from prize_lab import run_dfao
        checked = 0
        for row in json.loads(curve.read_text(encoding="utf-8"))["results"]:
            sat = next((p for p in row["per_state"]
                        if p["status"] == "SAT" and "candidate" in p), None)
            if sat is None:
                continue
            cand = sat["candidate"]
            mine = [dfao_drat_proofs.eval_dfao(cand, i) for i in range(row["n"])]
            theirs = [run_dfao(cand, i) for i in range(row["n"])]
            self.assertEqual(mine, theirs,
                             f"{row['sequence']}/{row['direction']}/n={row['n']}")
            checked += 1
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
