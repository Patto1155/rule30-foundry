#!/usr/bin/env python3
"""Run tools/jwst_u13151_consistency.py as part of the test suite.

The checker validates a frozen reproduction specification transcribed from
arXiv:2610.26560v1. Its whole value is that it catches a mistyped decimal point
before a multi-gigabyte download, so it is only useful if it runs without
anyone remembering to run it -- hence a test, like tools/lint_bitorder.py.

It is stdlib-only on purpose: the machine that attempts the JWST reproduction
will have astropy and numpy, this one does not, and the spec must be checkable
on either.

The mutation tests matter more than the pass tests. A consistency checker that
cannot fail is decoration; these perturb the frozen inputs and assert the
corresponding check goes red.
"""

import json
import pathlib
import subprocess
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import jwst_u13151_consistency as spec

TOOL = REPO_ROOT / "tools" / "jwst_u13151_consistency.py"


class FrozenSpecIsConsistentTest(unittest.TestCase):
    def test_all_checks_pass_or_are_recorded_discrepancies(self):
        checks, _ = spec.run_all()
        fails, _ = spec.summarise(checks)
        self.assertEqual(
            fails, [],
            "transcription failures in the frozen spec:\n"
            + "\n".join(f"{c.name}: {c.detail}" for c in fails))

    def test_the_registry_actually_contains_every_check(self):
        """A checker whose list is empty passes vacuously."""
        checks, diag = spec.run_all()
        self.assertGreaterEqual(len(checks), 7)
        for name in ("check_wavelength_calibration_sigma",
                     "check_redshift_consistency", "check_doublet_ratio",
                     "check_specific_sfr", "check_mstar_over_sfr",
                     "check_metallicity_percent",
                     "check_o32_definition_offset"):
            self.assertTrue(hasattr(spec, name), name)
        self.assertGreaterEqual(len(diag), 8)

    def test_sigma_z_reproduces_the_paper_footnote(self):
        """750 km/s at z=9.7995 must give the quoted 0.0270."""
        got = spec.sigma_z_from_velocity(9.7995, 750.0)
        self.assertAlmostEqual(got, 0.0270, places=4)

    def test_o32_offset_is_verified_in_flux_not_wavelength(self):
        """The 1.3 offset is a rounded 1.33 in flux.

        Re-deriving it from wavelengths gives 1.99 and a spurious 0.185 dex
        discrepancy. That error is easy to make and was made once while writing
        this tool, so it is pinned here.
        """
        check = spec.check_o32_definition_offset()
        self.assertTrue(check.ok, check.detail)
        self.assertAlmostEqual(check.values["from_theory"], 0.1257, places=3)
        self.assertAlmostEqual(check.values["from_wavelengths"], 0.2989, places=3)
        self.assertGreater(check.values["from_wavelengths"]
                           - check.values["from_theory"], 0.17)


class MutationsBreakTheRightCheckTest(unittest.TestCase):
    """Perturb the frozen inputs; exactly the dependent check must go red."""

    def setUp(self):
        self._fluxes = dict(spec.FLUXES)
        self._saved = {k: getattr(spec, k) for k in
                       ("Z_NIRSPEC_ERR", "LOG_MSTAR", "MOH_TE")}

    def tearDown(self):
        spec.FLUXES.clear()
        spec.FLUXES.update(self._fluxes)
        for key, value in self._saved.items():
            setattr(spec, key, value)

    def test_a_transposed_doublet_flux_fails_the_doublet_check(self):
        spec.FLUXES["[O III]5007"] = (100.0, 14.7, False)
        self.assertFalse(spec.check_doublet_ratio().ok)

    def test_a_five_percent_flux_shift_still_passes_the_doublet_check(self):
        """The check must have loose-enough errors to be about physics."""
        f5007, e5007, _ = spec.FLUXES["[O III]5007"]
        spec.FLUXES["[O III]5007"] = (f5007 * 1.05, e5007, False)
        self.assertTrue(spec.check_doublet_ratio().ok)

    def test_a_mistyped_redshift_error_fails_the_calibration_check(self):
        spec.Z_NIRSPEC_ERR = 0.27
        self.assertFalse(spec.check_wavelength_calibration_sigma().ok)

    def test_a_mistyped_stellar_mass_fails_the_ssfr_checks(self):
        spec.LOG_MSTAR = 8.68  # a decimal point
        self.assertFalse(spec.check_specific_sfr().ok)
        self.assertFalse(spec.check_mstar_over_sfr().ok)

    def test_a_metallicity_outside_the_abstract_range_fails(self):
        spec.MOH_TE = 8.10
        self.assertFalse(spec.check_metallicity_percent().ok)


class DiscrepancyIsNotAFailureTest(unittest.TestCase):
    def test_a_recorded_discrepancy_does_not_fail_the_build(self):
        """The build is about our transcription, not the paper's prose."""
        flagged = spec.Check("paper contradicts itself", ok=False,
                             discrepancy=True, detail="x")
        broken = spec.Check("we mistyped a flux", ok=False, detail="y")
        fails, discrepancies = spec.summarise([flagged, broken])
        self.assertEqual(fails, [broken])
        self.assertEqual(discrepancies, [flagged])


class CommandLineTest(unittest.TestCase):
    def test_exit_code_zero_and_json_parses(self):
        proc = subprocess.run([sys.executable, str(TOOL), "--json"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["source"], "arXiv:2610.26560v1")
        self.assertEqual(payload["n_fail"], 0)
        self.assertIn("derived_diagnostics", payload)
        self.assertGreaterEqual(len(payload["documented_refusals"]), 3)

    def test_quiet_prints_nothing(self):
        proc = subprocess.run([sys.executable, str(TOOL), "--quiet"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
