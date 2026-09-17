"""Failure-focused checks for model selection and finite evidence handling."""
import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import gates, jev_search as j


def plan():
    return j.load_plan(j.ROOT / "queue/jev/smoke.json")


def response(choice="a", probs=None):
    return {"answers": {"next": {"type": "choice", "choice": choice,
            "probabilities": probs or {"a": .8, "return-to-lead": .2}}},
            "usage": {"input_tokens": 50}}


class JevSearchTests(unittest.TestCase):
    def test_plans_require_controls_matching_null_and_no_duplicate_instances(self):
        for mutation in (lambda p: p["cards"].pop(0),
                         lambda p: p["cards"][2].update(id="return-to-lead"),
                         lambda p: p["cards"].append(copy.deepcopy(p["cards"][-1])),
                         lambda p: p["cards"].__setitem__(2, {**p["cards"][2], "n": True}),
                         lambda p: p["cards"].__setitem__(2, {**p["cards"][2], "n": 21})):
            p = plan()
            mutation(p)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "plan.json"
                j.save(path, p)
                with self.assertRaises(ValueError):
                    j.load_plan(path)

    def test_monotonicity_only_with_verified_matching_family(self):
        c = plan()["cards"][2]
        lower = {**c, "n": 16, "states": 5}
        h = [{"card": lower, "status": "UNSAT", "verified": True}]
        self.assertEqual(j.implication(c, h)["status"], "UNSAT")
        for status, verified in (("UNKNOWN", True), ("UNSAT", False), ("ERROR", True)):
            self.assertIsNone(j.implication(c, [{**h[0], "status": status, "verified": verified}]))
        for change in ({"direction": "lsd"}, {"base": 3}, {"sequence": "random"}, {"n": 32}, {"states": 3}):
            self.assertIsNone(j.implication(c, [{**h[0], "card": {**lower, **change}}]))
        sat = {"card": {**c, "n": 32, "states": 3}, "status": "SAT", "verified": True}
        self.assertEqual(j.implication(c, [sat])["status"], "SAT")
        rc = {**c, "sequence": "random", "seed": 31}
        self.assertIsNone(j.implication(rc, [{**sat, "card": {**sat["card"], "sequence": "random", "seed": 30}}]))

    def test_choice_rejects_invented_options_nan_and_nonmaximal_choices(self):
        options = {"a", "return-to-lead"}
        self.assertEqual(j.validate_choice(response(), options), ("a", .8))
        for r in (response("injected-shell-command"),
                  response(probs={"a": float("nan"), "return-to-lead": .2}),
                  response(probs={"a": .8, "return-to-lead": .8}),
                  response(probs={"a": .1, "return-to-lead": .9}),
                  response(probs={"a": True, "return-to-lead": 0})):
            with self.assertRaises((ValueError, KeyError)):
                j.validate_choice(r, options)

    def test_live_adapter_writes_evidence_and_escalates_uncertainty(self):
        c = {**plan()["cards"][2], "id": "a"}
        with tempfile.TemporaryDirectory() as td:
            with patch.object(j.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(response()).encode())):
                result = j.jev_select(plan(), [c], [], Path(td), "secret-for-test", 1, .9)
            self.assertTrue(result["escalate"])
            self.assertEqual(result["input_tokens"], 50)
            self.assertNotIn("secret-for-test", (Path(td) / "request.json").read_text())

    def test_independent_generator_disagreement_aborts(self):
        c = plan()["cards"][1]
        with patch.object(j, "sequence_bits", return_value=[0] * c["n"]):
            with self.assertRaisesRegex(ValueError, "independent naive"):
                j.checked_bits(c)

    def test_postflight_scopes_controls_and_replications_like_preflight(self):
        for c in (plan()["cards"][0], plan()["cards"][3]):
            r = gates.postflight({"manifest": j.manifest(c), "horizon": c["n"]})
            self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(gates.gate_seed_echo({"manifest": {"purpose": "prize-claim", "seed": "random"}}).status, "FAIL")
        m = {"purpose": "replication", "seed": "random", "replicates": {"source": "log", "seed": "random"}}
        self.assertEqual(gates.gate_seed_echo({"manifest": m}).status, "PASS")
        m["seed"] = "different"
        self.assertEqual(gates.gate_seed_echo({"manifest": m}).status, "FAIL")

    def test_timeout_is_unknown_and_rejected_proof_is_not_verified(self):
        c = plan()["cards"][1]
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            with patch.object(j, "command", return_value=(None, "", .1)):
                r = j.solve_card(c, d, "solver", "checker", 1, 1, time.monotonic() + 5)
            self.assertEqual(r["status"], "UNKNOWN")
            self.assertFalse(r["verified"])
            (d / "instance.drat").write_text("bad proof")
            with patch.object(j, "command", side_effect=[(20, "", .1), (1, "s NOT VERIFIED", .1)]):
                r = j.solve_card(c, d, "solver", "checker", 1, 1, time.monotonic() + 5)
            self.assertFalse(r["verified"])

    def test_replay_rejects_witness_and_cnf_tampering(self):
        c = plan()["cards"][0]
        bits = j.checked_bits(c)
        cnf, _ = j.dfao_sat_cnf(bits, states=2, base=2, direction="msd")
        candidate = {"states": 2, "base": 2, "direction": "msd", "initial_state": 0,
                     "transitions": [[0, 1], [1, 0]], "outputs": [0, 1]}
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "instance.cnf").write_text(cnf)
            j.save(d / "witness.json", candidate)
            j.save(d / "result.json", {"card": c, "status": "SAT", "verified": False,
                   "cnf_sha256": j.sha256_file(d / "instance.cnf"), "bits_sha256": j.digest(bits)})
            self.assertTrue(j.verify_result(d, "unused", 1)["verified"])
            candidate["outputs"] = [1, 0]
            j.save(d / "witness.json", candidate)
            with self.assertRaisesRegex(ValueError, "witness mismatch"):
                j.verify_result(d, "unused", 1)
            (d / "instance.cnf").write_text("p cnf 0 0\n")
            with self.assertRaisesRegex(ValueError, "CNF"):
                j.verify_result(d, "unused", 1)

    def test_model_budget_stops_before_api_and_control_failure_stops_search(self):
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(policy="jev", out=Path(td) / "run", seed=30,
                cadical=None, drat_trim=None, seconds=5, solve_seconds=1, check_seconds=1,
                max_steps=20, max_calls=10, max_input_tokens=1, min_probability=.5)
            def solved(c, *unused):
                return {"card": c, "status": c.get("expected", "UNKNOWN"), "verified": True}
            with patch.dict(j.os.environ, {"TYPESAFE_API_KEY": "unused"}), \
                 patch.object(j, "find_tool", return_value=Path(__file__)), \
                 patch.object(j, "solve_card", side_effect=solved), \
                 patch.object(j, "jev_select") as api:
                result = j.run(plan(), args)
                self.assertEqual(result["stop_reason"], "model-budget")
                self.assertEqual(result["jev_calls"], 0)
                api.assert_not_called()
            args.out = Path(td) / "failed-control"
            args.policy = "fixed"
            with patch.object(j, "find_tool", return_value=Path(__file__)), \
                 patch.object(j, "solve_card", side_effect=lambda c, *a: {"card": c, "status": "UNKNOWN", "verified": False}) as solve:
                result = j.run(plan(), args)
                self.assertEqual(result["stop_reason"], "calibration-failed")
                self.assertEqual(solve.call_count, 1)


if __name__ == "__main__":
    unittest.main()
