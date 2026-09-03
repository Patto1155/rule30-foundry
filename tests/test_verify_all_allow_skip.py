#!/usr/bin/env python3
"""Pin the rule that CI cannot go green on a stage that checked nothing.

`tools/verify_all.py` reports a stage whose input is absent as SKIP rather
than FAIL, which is the honest thing to do on a fresh clone: the canonical
bitstreams are gitignored. But SKIP is not evidence, and in CI nobody reads
the log -- so a stage that starts skipping because its input quietly vanished
would go green forever, which is precisely the failure mode the tool's own
docstring says it exists to prevent.

`--allow-skip` is the fix: it names the stages known to be unrunnable, and
turns every other SKIP into a failure. This test pins that behaviour so a
later refactor cannot silently restore "any SKIP is fine" under CI.

The test targets `skip_permitted` directly rather than running verify_all as
a subprocess. verify_all's last stage is `unittest discover -s tests`, so a
test that invoked it would re-enter the suite that contains it.
"""

import fnmatch
import importlib.util
import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "verify_all", REPO_ROOT / "tools" / "verify_all.py")
verify_all = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_all)

skip_permitted = verify_all.skip_permitted


class TestSkipPermitted(unittest.TestCase):

    def test_none_is_permissive(self):
        """No --allow-skip at all keeps the interactive default: SKIP is fine."""
        self.assertTrue(skip_permitted("drat-toolchain", None))
        self.assertTrue(skip_permitted("anything-at-all", None))

    def test_empty_allowlist_is_strict(self):
        """`allowed = ()` is strict, not permissive.

        argparse's append action yields None when the flag is absent and a
        list when it is present, so an empty tuple can only mean the caller
        asked for strictness and named nothing. Collapsing it to permissive
        would make `--allow-skip` silently optional.
        """
        self.assertFalse(skip_permitted("drat-toolchain", ()))

    def test_exact_name_matches(self):
        self.assertTrue(skip_permitted("drat-toolchain", ("drat-toolchain",)))

    def test_glob_matches_the_bitstream_family(self):
        """The workflow relies on one glob covering both bitstream stages."""
        for name in ("bitstream:center_col_10M.bin",
                     "bitstream:center_col_46M.bin"):
            with self.subTest(stage=name):
                self.assertTrue(skip_permitted(name, ("bitstream:*",)))

    def test_unlisted_stage_is_refused(self):
        """A stage outside the allowlist must not be permitted to skip."""
        allowed = ("bitstream:*", "drat-toolchain")
        self.assertFalse(skip_permitted("lint-bitorder", allowed))
        self.assertFalse(skip_permitted("unittest", allowed))
        self.assertFalse(skip_permitted("manifest-current", allowed))

    def test_glob_does_not_match_beyond_its_prefix(self):
        """`bitstream:*` must not license a skip of an unrelated stage."""
        self.assertFalse(skip_permitted("golden-self-test", ("bitstream:*",)))


class TestWorkflowAllowlist(unittest.TestCase):
    """The globs CI actually passes must license exactly the skippable stages.

    This reads `--allow-skip` out of the workflow file rather than restating
    it, so the two cannot drift apart. The failure it is really guarding
    against is an over-broad glob: `--allow-skip '*'` would make CI green on a
    run that executed nothing, which is the same "everything passed on a
    machine that checked nothing" that verify_all exists to prevent -- only
    now spelled in YAML where the tool's own safeguards cannot see it.
    """

    WORKFLOW = REPO_ROOT / ".github" / "workflows" / "verify.yml"

    # Parsed rather than hardcoded: --allow-skip 'glob' or --allow-skip glob.
    PATTERN = re.compile(r"--allow-skip\s+'([^']+)'|--allow-skip\s+(\S+)")

    @classmethod
    def skippable_stages(cls) -> set:
        """Stage names whose input can legitimately be absent.

        Derived from verify_all's own constants, so adding a bitstream keeps
        this test correct without an edit here.
        """
        names = {f"bitstream:{pathlib.Path(rel).name}"
                 for rel in verify_all.BITSTREAMS}
        names.add("drat-toolchain")
        return names

    def workflow_globs(self) -> list:
        text = self.WORKFLOW.read_text(encoding="utf-8")
        globs = [q or bare for q, bare in self.PATTERN.findall(text)]
        self.assertTrue(globs, f"no --allow-skip found in {self.WORKFLOW}")
        return globs

    def test_workflow_exists(self):
        """A2 is only done if the workflow is actually in the repo."""
        self.assertTrue(self.WORKFLOW.is_file(),
                        f"{self.WORKFLOW} is missing: CI does not run "
                        "verify_all")

    def test_every_glob_licenses_something_real(self):
        """A glob matching no stage is a typo that silently does nothing."""
        skippable = self.skippable_stages()
        for glob in self.workflow_globs():
            with self.subTest(glob=glob):
                self.assertTrue(
                    any(fnmatch.fnmatch(name, glob) for name in skippable),
                    f"--allow-skip {glob!r} matches no skippable stage")

    def test_no_glob_licenses_a_mandatory_stage(self):
        """The real hazard: a glob broad enough to excuse a stage that ran."""
        skippable = self.skippable_stages()
        mandatory = [s.name for s in verify_all.build_stages()
                     if s.name not in skippable]
        self.assertTrue(mandatory, "verify_all has no mandatory stages")
        for glob in self.workflow_globs():
            for name in mandatory:
                with self.subTest(glob=glob, stage=name):
                    self.assertFalse(
                        fnmatch.fnmatch(name, glob),
                        f"--allow-skip {glob!r} would let the mandatory stage "
                        f"{name!r} skip without failing CI")


if __name__ == "__main__":
    unittest.main()
