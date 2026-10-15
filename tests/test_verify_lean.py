"""Axiom audit must not accept an empty or admitted proof report."""
import importlib.util
from pathlib import Path
import unittest
import tempfile

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


class SourceAudit(unittest.TestCase):
    def test_nested_namespaces_modifiers_and_multiline_declarations(self):
        source = """namespace Rule30
namespace Nested
@[simp] protected theorem
  unicode_α : True := by trivial
end Nested
section
private lemma helper : True := by trivial
end
end Rule30
"""
        self.assertEqual(module.source_theorems(source), {'Nested.unicode_α', 'helper'})

    def test_nested_comments_and_strings_are_not_declarations(self):
        source = 'namespace Rule30\n/- theorem fake : False := by /- nested -/ contradiction -/\n-- lemma fake2 : False := by contradiction\ndef text := "theorem fake3"\ntheorem real : True := by trivial\nend Rule30\n'
        self.assertEqual(module.source_theorems(source), {'real'})

    def test_nested_axiom_reports(self):
        self.assertEqual(module.audit_axioms(
            "'Rule30.Nested.a' does not depend on any axioms", {'Nested.a'}), [])

    def test_unsupported_names_fail_closed(self):
        with self.assertRaises(ValueError):
            module.source_theorems('namespace Rule30\ntheorem «odd name» : True := by trivial')

    def test_actual_source_files_all_counted(self):
        formal = Path(__file__).resolve().parents[1] / 'formal'
        declarations = set()
        for path in module.lean_sources(formal):
            declarations.update(module.source_theorems(path.read_text()))
        self.assertIn('triangular_uniqueness', declarations)
        self.assertIn('onset_reduction_evolve', declarations)

    def test_secondary_source_discovered_without_handwritten_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            formal = Path(temporary)
            (formal / 'Rule30').mkdir()
            (formal / '.lake').mkdir()
            (formal / 'Rule30/Added.lean').write_text(
                'namespace Rule30\ntheorem added : True := by trivial\nend Rule30\n')
            (formal / '.lake/Generated.lean').write_text('theorem generated : True := by trivial')
            sources = module.lean_sources(formal)
            self.assertEqual([path.name for path in sources], ['Added.lean'])
            self.assertEqual(module.source_theorems(sources[0].read_text()), {'added'})
