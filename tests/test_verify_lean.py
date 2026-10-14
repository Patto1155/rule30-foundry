"""Axiom audit must not accept an empty or admitted proof report."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('verify_lean', Path(__file__).resolve().parents[1] / 'tools/verify_lean.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AxiomAudit(unittest.TestCase):
    def test_empty_output_is_not_an_audit(self):
        self.assertTrue(module.audit_axioms('', {'a'}))

    def test_partial_output_is_not_an_audit(self):
        self.assertTrue(module.audit_axioms("'Rule30.a' does not depend on any axioms", {'a', 'b'}))

    def test_admission_and_custom_axiom_rejected(self):
        for axiom in ['sorryAx', 'Rule30.assumedGrowth']:
            with self.subTest(axiom=axiom):
                self.assertTrue(module.audit_axioms(f"'Rule30.a' depends on axioms: [{axiom}]", {'a'}))

    def test_standard_and_axiom_free_accepted(self):
        output = "'Rule30.a' does not depend on any axioms\n'Rule30.b' depends on axioms: [propext, Classical.choice, Quot.sound]"
        self.assertEqual(module.audit_axioms(output, {'a', 'b'}), [])
