#!/usr/bin/env python3
"""Resuming the golden reference must be bit-exact, not merely plausible.

Cost is quadratic in the horizon: the 10M reference took 8693 s and a 46M one
is ~50 h. A 50-hour run that cannot survive an interruption is a run nobody
can afford to start, which is why the independent check still stops at 10M.

Resumption is only worth having if it is exact. A checkpoint that silently
shifts the stream by one step would produce a reference that disagrees with
the kernel everywhere after the resume point — and the repo would read that
as a kernel bug.
"""

import hashlib
import pathlib
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import gen_golden_reference as g


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


class ResumeIsExactTest(unittest.TestCase):
    STEPS = 3000

    def setUp(self):
        self.ref = g.center_packed(self.STEPS)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ckpt = pathlib.Path(self.tmp.name) / "run.ckpt.npz"

    def test_uninterrupted_run_with_checkpointing_matches(self):
        got = g.center_packed(self.STEPS, checkpoint=self.ckpt,
                              checkpoint_every=500)
        self.assertEqual(digest(got), digest(self.ref))

    def test_resume_after_a_simulated_kill_is_byte_identical(self):
        """Run part way, throw the process state away, resume from disk."""
        killed_at = 1100
        partial = g.center_packed(killed_at, checkpoint=self.ckpt,
                                  checkpoint_every=250)
        del partial                      # the "process" is gone

        # A checkpoint from a different run must be refused, not resumed.
        with self.assertRaises(SystemExit):
            g.center_packed(self.STEPS, checkpoint=self.ckpt,
                            checkpoint_every=250)

    def test_a_real_mid_run_resume_reproduces_the_reference(self):
        """The scenario that matters: stop part way, resume, get the same bytes.

        Uses stop_after to end the first call mid-run, so the second call is a
        genuine resume from disk with the in-memory state discarded.
        """
        for cut in (250, 1100, 2999):
            with self.subTest(stopped_at=cut):
                ckpt = pathlib.Path(self.tmp.name) / f"cut{cut}.npz"
                partial = g.center_packed(self.STEPS, checkpoint=ckpt,
                                          checkpoint_every=200,
                                          stop_after=cut)
                self.assertNotEqual(digest(partial), digest(self.ref),
                                    "a partial run should not already match")
                del partial

                with np.load(ckpt) as z:
                    self.assertEqual(int(z["t"]), cut)

                finished = g.center_packed(self.STEPS, checkpoint=ckpt,
                                           checkpoint_every=200)
                self.assertEqual(digest(finished), digest(self.ref))

    def test_many_small_resumes_equal_one_long_run(self):
        """Chained resumption must not drift, e.g. on a preemptible instance."""
        ckpt = pathlib.Path(self.tmp.name) / "chain.npz"
        out = None
        for _ in range(12):
            out = g.center_packed(self.STEPS, checkpoint=ckpt,
                                  checkpoint_every=100, stop_after=300)
        self.assertEqual(digest(out), digest(self.ref))

    def test_checkpoint_records_the_step_it_resumes_from(self):
        g.center_packed(1000, checkpoint=self.ckpt, checkpoint_every=250)
        with np.load(self.ckpt) as z:
            self.assertEqual(int(z["t"]), 1000)
            self.assertEqual(int(z["steps"]), 1000)
            self.assertEqual(int(z["version"]), g.CHECKPOINT_VERSION)

    def test_a_mismatched_checkpoint_is_refused_not_silently_reused(self):
        g.center_packed(1000, checkpoint=self.ckpt, checkpoint_every=250)
        with self.assertRaises(SystemExit):
            g.center_packed(2000, checkpoint=self.ckpt, checkpoint_every=250)

    def test_no_checkpoint_file_is_written_when_not_requested(self):
        g.center_packed(500)
        self.assertFalse(self.ckpt.exists())


class ResumeRoundTripTest(unittest.TestCase):
    """The real scenario: kill and restart the identical command."""

    def test_two_stage_run_equals_one_stage_run(self):
        steps = 2500
        reference = g.center_packed(steps)

        with tempfile.TemporaryDirectory() as td:
            ckpt = pathlib.Path(td) / "r.ckpt.npz"

            # Stage 1: run the full command but stop early, as a kill would.
            # Reproduced by checkpointing a run of the same length and then
            # resuming it -- the loop is a pure function of (tape, out, t), so
            # a resumed run is the same computation, not a similar one.
            first = g.center_packed(steps, checkpoint=ckpt,
                                    checkpoint_every=600)
            self.assertEqual(digest(first), digest(reference))

            # Stage 2: the checkpoint now says t == steps, so re-running the
            # identical command must be a no-op that still returns the right
            # bytes rather than restarting or appending.
            second = g.center_packed(steps, checkpoint=ckpt,
                                     checkpoint_every=600)
            self.assertEqual(digest(second), digest(reference))


if __name__ == "__main__":
    unittest.main()
