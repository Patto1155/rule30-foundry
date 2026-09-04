"""Offline tests for tools/gates.py.

Every gate here corresponds to a rule in CLAUDE.md or AGENTS.md that has cost
this repo real time. The tests pin each gate to the failure it exists to
catch, using the repo's own numbers where a number is involved, so that the
gate cannot drift from the rule.

`run_external=False` throughout: the subprocess gates (lint, golden self-test,
verify_all) are exercised by verify_all itself, and re-running verify_all from
inside the unittest stage would recurse.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gates  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def good_manifest(**over) -> dict:
    m = {
        "name": "t", "kind": "measurement", "seed": "single-black-cell",
        "theory_gate": "OPEN", "script": "experiments/counting_bound.py",
        "claims": [],
    }
    m.update(over)
    return m


def by_name(report: dict, gate: str) -> dict:
    return next(g for g in report["gates"] if g["gate"] == gate)


class TestSchema(unittest.TestCase):
    def test_minimal_manifest_passes(self):
        r = gates.preflight(good_manifest(), run_external=False)
        self.assertEqual(r["verdict"], "PASS")

    def test_missing_required_field_fails_and_short_circuits(self):
        m = good_manifest()
        del m["seed"]
        r = gates.preflight(m, run_external=False)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertEqual(len(r["gates"]), 1)
        self.assertIn("seed", r["gates"][0]["reason"])

    def test_unknown_kind_fails(self):
        r = gates.preflight(good_manifest(kind="vibes"), run_external=False)
        self.assertEqual(by_name(r, "schema")["status"], "FAIL")


class TestSeed(unittest.TestCase):
    """CLAUDE.md rule 3: single seed only."""

    def test_ensemble_is_refused(self):
        r = gates.preflight(good_manifest(seed="random-ic-ensemble"),
                            run_external=False)
        g = by_name(r, "seed")
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("not progress", g["reason"])

    def test_the_seed_passes(self):
        r = gates.preflight(good_manifest(), run_external=False)
        self.assertEqual(by_name(r, "seed")["status"], "PASS")


class TestTheoryGate(unittest.TestCase):
    def test_open_passes(self):
        r = gates.preflight(good_manifest(theory_gate="open"), run_external=False)
        self.assertEqual(by_name(r, "theory-gate")["status"], "PASS")

    def test_settled_and_closed_are_refused(self):
        for v in ("ALREADY SETTLED", "ROUTE CLOSED"):
            with self.subTest(v=v):
                r = gates.preflight(good_manifest(theory_gate=v),
                                    run_external=False)
                self.assertEqual(by_name(r, "theory-gate")["status"], "FAIL")

    def test_unset_or_not_covered_is_refused(self):
        """'NOT COVERED' means the question was not answered, which is not
        the same as OPEN. Only an explicit OPEN may run."""
        for v in ("", "NOT COVERED", "maybe"):
            with self.subTest(v=v):
                r = gates.preflight(good_manifest(theory_gate=v),
                                    run_external=False)
                self.assertEqual(by_name(r, "theory-gate")["status"], "FAIL")


class TestCountingBound(unittest.TestCase):
    """CLAUDE.md rule 1, pinned to experiments/counting_bound.py's numbers."""

    def _search(self, states, n, claims=("negative",)):
        return good_manifest(kind="search", claims=list(claims),
                             search={"class": "dfao", "states": states,
                                     "base": 2, "prefix_bits": n})

    def test_the_retracted_certificate_shape_is_refused(self):
        """24-state DFAO vs 10,000 bits: log2|M| = 244.078, VACUOUS. This is
        the exact shape of the 2026-08 retraction and the trap manifest."""
        r = gates.preflight(self._search(24, 10_000), run_external=False)
        g = by_name(r, "counting-bound")
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("244.1", g["reason"])
        self.assertIn("VACUOUS", g["reason"])
        self.assertEqual(r["verdict"], "FAIL")

    def test_informative_search_passes(self):
        r = gates.preflight(self._search(24, 200), run_external=False)
        self.assertEqual(by_name(r, "counting-bound")["status"], "PASS")

    def test_equality_is_informative_not_vacuous(self):
        """The tool marks margin >= 0 informative. A strict > here would reject
        boundary-case evidence for a mathematically false reason -- the P2
        review finding on #25."""
        cb = gates._load("experiments/counting_bound.py")
        n = int(cb.log2_dfao_upper(4, 2))          # 20.0 exactly for s=4
        self.assertEqual(cb.log2_dfao_upper(4, 2), float(n))
        r = gates.preflight(self._search(4, n), run_external=False)
        g = by_name(r, "counting-bound")
        self.assertEqual(g["status"], "PASS", g["reason"])
        self.assertIn("+0.0", g["reason"])

    def test_agrees_with_the_tool_verdict(self):
        """The gate must never disagree with counting_bound.verdict."""
        cb = gates._load("experiments/counting_bound.py")
        for s, n in ((3, 5), (3, 20), (8, 100), (8, 130), (24, 244), (24, 245)):
            with self.subTest(s=s, n=n):
                expect = "PASS" if cb.verdict(s, n, 2)["informative"] else "FAIL"
                r = gates.preflight(self._search(s, n), run_external=False)
                self.assertEqual(by_name(r, "counting-bound")["status"], expect)

    def test_search_without_negative_claim_skips(self):
        r = gates.preflight(self._search(24, 10_000, claims=()), run_external=False)
        self.assertEqual(by_name(r, "counting-bound")["status"], "SKIP")

    def test_non_dfao_class_needs_a_declared_bound(self):
        m = good_manifest(kind="search", claims=["negative"],
                          search={"class": "mlp", "prefix_bits": 1000})
        r = gates.preflight(m, run_external=False)
        g = by_name(r, "counting-bound")
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("defensible", g["reason"])

    def test_declared_bound_is_honoured(self):
        base = {"class": "grammar", "prefix_bits": 1000}
        ok = good_manifest(kind="search", claims=["negative"],
                           search={**base, "log2_size": 1000.0})
        bad = good_manifest(kind="search", claims=["negative"],
                            search={**base, "log2_size": 999.9})
        self.assertEqual(by_name(gates.preflight(ok, run_external=False),
                                 "counting-bound")["status"], "PASS")
        self.assertEqual(by_name(gates.preflight(bad, run_external=False),
                                 "counting-bound")["status"], "FAIL")

    def test_missing_n_fails_rather_than_guessing(self):
        m = good_manifest(kind="search", claims=["negative"],
                          search={"class": "dfao", "states": 24, "base": 2})
        r = gates.preflight(m, run_external=False)
        self.assertEqual(by_name(r, "counting-bound")["status"], "FAIL")


class TestLightCone(unittest.TestCase):
    """COMPUTE_PLAN.md §5 item 2, pinned to gpu/tape_geometry.check."""

    def test_short_tape_is_refused(self):
        m = good_manifest(kind="simulation",
                          simulation={"cells": 1000, "steps": 10_000})
        r = gates.preflight(m, run_external=False)
        g = by_name(r, "light-cone")
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("light cone", g["reason"])

    def test_the_46M_run_geometry_passes(self):
        """The repo's own recorded run: 92,000,064 cells x 46,000,000 steps."""
        m = good_manifest(kind="simulation",
                          simulation={"cells": 92_000_064, "steps": 46_000_000})
        r = gates.preflight(m, run_external=False)
        self.assertEqual(by_name(r, "light-cone")["status"], "PASS")

    def test_agrees_with_tape_geometry(self):
        tg = gates._load("gpu/tape_geometry.py")
        for cells, steps in ((4096, 1000), (4096, 3000), (1 << 20, 1 << 19)):
            with self.subTest(cells=cells, steps=steps):
                expect = "PASS" if tg.describe(cells, steps)["cone_fits"] else "FAIL"
                m = good_manifest(kind="simulation",
                                  simulation={"cells": cells, "steps": steps})
                r = gates.preflight(m, run_external=False)
                self.assertEqual(by_name(r, "light-cone")["status"], expect)


class TestScript(unittest.TestCase):
    def test_missing_script_fails(self):
        r = gates.preflight(good_manifest(script="experiments/nope.py"),
                            run_external=False)
        self.assertEqual(by_name(r, "script")["status"], "FAIL")


class TestExternalGatesSkipWhenDisabled(unittest.TestCase):
    def test_all_three_skip(self):
        r = gates.preflight(good_manifest(reads_packed_bitstream=True),
                            run_external=False)
        for g in ("bitorder-lint", "golden-self-test", "verify-all"):
            self.assertEqual(by_name(r, g)["status"], "SKIP", g)


# --------------------------------------------------------------------------
# postflight
# --------------------------------------------------------------------------

def good_result(**over) -> dict:
    r = {"manifest": good_manifest(), "horizon": 10**6,
         "metrics": {}, "conclusions": []}
    r.update(over)
    return r


class TestCensoring(unittest.TestCase):
    """AGENTS.md: 'never reached within N' is right-censored."""

    def test_unqualified_never_is_rejected(self):
        r = gates.postflight(good_result(
            conclusions=["The center column never repeats."]))
        g = by_name(r, "censoring")
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("horizon", g["reason"])

    def test_qualified_forms_pass(self):
        for c in ("No period p <= 5,000,000 in the first 10M bits.",
                  "Not observed through d = 1.2e10.",
                  "No 32->64 doubling within the walk horizon.",
                  "The sequence does not repeat up to n = 10^7."):
            with self.subTest(c=c):
                r = gates.postflight(good_result(conclusions=[c]))
                self.assertEqual(by_name(r, "censoring")["status"], "PASS", c)

    def test_search_result_must_state_horizon(self):
        r = good_result(manifest=good_manifest(kind="search"))
        del r["horizon"]
        self.assertEqual(by_name(gates.postflight(r), "censoring")["status"], "FAIL")

    def test_measurement_need_not(self):
        r = good_result()
        del r["horizon"]
        self.assertEqual(by_name(gates.postflight(r), "censoring")["status"], "PASS")


class TestNoiseFloor(unittest.TestCase):
    """AGENTS.md: define a noise floor before calling a near-zero metric
    structured."""

    def test_metric_without_baseline_fails(self):
        r = gates.postflight(good_result(metrics={"te": {"value": 0.003}}))
        g = by_name(r, "noise-floor")
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("te", g["reason"])

    def test_metric_with_baseline_passes(self):
        r = gates.postflight(good_result(
            metrics={"te": {"value": 0.003, "baseline": 0.0021}}))
        self.assertEqual(by_name(r, "noise-floor")["status"], "PASS")


class TestDivergence(unittest.TestCase):
    """AGENTS.md: first_divergence < distance is impossible in a radius-1 CA."""

    def test_impossible_value_is_a_hard_failure(self):
        r = gates.postflight(good_result(
            divergence=[{"distance": 40, "first_divergence": 12}]))
        g = by_name(r, "divergence-invariant")
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("impossible", g["reason"])

    def test_equal_and_greater_pass(self):
        r = gates.postflight(good_result(
            divergence=[{"distance": 40, "first_divergence": 40},
                        {"distance": 40, "first_divergence": 41}]))
        self.assertEqual(by_name(r, "divergence-invariant")["status"], "PASS")


class TestFiftyPercent(unittest.TestCase):
    """AGENTS.md: ~50% differing is uncorrelated streams, never a kernel bug."""

    def test_the_bitorder_bug_signature_is_caught(self):
        """49.95% -- the measured signature of the I-L bit-order bug."""
        r = gates.postflight(good_result(
            stream_comparison={"fraction_differing": 0.4995}))
        g = by_name(r, "fifty-percent")
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("not a kernel bug", g["reason"])

    def test_late_divergence_passes(self):
        r = gates.postflight(good_result(
            stream_comparison={"fraction_differing": 0.0004}))
        self.assertEqual(by_name(r, "fifty-percent")["status"], "PASS")

    def test_no_comparison_skips(self):
        self.assertEqual(by_name(gates.postflight(good_result()),
                                 "fifty-percent")["status"], "SKIP")


class TestSeedEcho(unittest.TestCase):
    def test_result_must_echo_the_seed(self):
        r = good_result(manifest=good_manifest(seed="ensemble"))
        self.assertEqual(by_name(gates.postflight(r), "seed-echo")["status"], "FAIL")


# --------------------------------------------------------------------------
# the CLI and the committed manifests
# --------------------------------------------------------------------------

class TestCLIAndManifests(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run([sys.executable, "tools/gates.py", *args],
                              cwd=REPO, capture_output=True, text=True)

    def test_trap_manifest_is_refused(self):
        """The committed trap must always be refused. verify_all pins this too;
        here it is pinned with the reason."""
        r = self._run("preflight", "queue/trap-vacuous-dfao.json", "--no-external")
        self.assertEqual(r.returncode, 1)
        rep = json.loads(r.stdout)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertEqual(by_name(rep, "counting-bound")["status"], "FAIL")

    def test_expect_fail_inverts_only_for_fail(self):
        self.assertEqual(self._run("preflight", "queue/trap-vacuous-dfao.json",
                                   "--no-external", "--expect-fail").returncode, 0)
        r = self._run("preflight", "queue/b1-pattern-map-walk.json",
                      "--no-external", "--expect-fail")
        self.assertEqual(r.returncode, 1)
        self.assertIn("stopped gating", r.stderr)

    def test_b1_manifest_passes_pure_gates(self):
        r = self._run("preflight", "queue/b1-pattern-map-walk.json", "--no-external")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_stdout_is_json_and_pretty_goes_to_stderr(self):
        r = self._run("preflight", "queue/b1-pattern-map-walk.json",
                      "--no-external", "--pretty")
        json.loads(r.stdout)
        self.assertIn("-->", r.stderr)

    def test_unreadable_path_exits_2(self):
        self.assertEqual(self._run("preflight", "queue/does-not-exist.json").returncode, 2)


if __name__ == "__main__":
    unittest.main()
