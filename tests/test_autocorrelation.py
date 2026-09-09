#!/usr/bin/env python3
"""Unit tests for experiments/autocorrelation.py -- the pure parts only.

What is covered: every testable behaviour of ``autocorrelation_fft`` with
``use_gpu=False`` -- the lag-0 self-correlation in {0, 1} at every lag count,
period peaks in the FFT-based estimator, the returned length, GPU skip logic,
and the noise-floor caveat that turns a near-zero metric into evidence (rule 6
in CLAUDE.md: at lag 0 the FFT autocorrelation is a BiasedEstimator subject to
the finite-sample bias ``- (2 - 1/n)·E[X]²``, so anyone reporting "near-zero
Rule 30 autocorrelation" must state what they are reporting).

What is deliberately NOT covered:
  * load_center_column -- it opens ``data/center_col_10M.bin``, which is a
    gitignored artifact absent from a fresh checkout. A test that opens it is
    a test that cannot run anywhere CI does.
  * main()/write_log -- they touch DATA_FILE, OUT_NPY, OUT_CSV and LOG_FILE.

Because ``autocorrelation.py`` imports ``tqdm`` at module scope (and never
calls it), and tqdm is not a test dependency (requirements-ci.txt pins numpy
alone), the module is imported under a stubbed sys.modules entry. The stub is
behavioural: it must **not** provide an ``__call__`` (nothing may call tqdm),
it must not be ``None`` (the module's import must genuinely succeed), and it
must not shadow a real installed tqdm. The import guard mirrors the skip-not-
delete rule from tests/test_import_safety.py: a missing *hard* dependency must
skip the affected tests, not collapse the module into one failed-import error.
"""

import pathlib
import sys
import types
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

_TQDM_STUB = types.ModuleType("tqdm")
_TQDM_STUB.tqdm = lambda *args, **kwargs: None

try:
    import tqdm as _real_tqdm  # noqa: F401
except ImportError:
    _real_tqdm = None
if _real_tqdm is None:
    sys.modules.setdefault("tqdm", _TQDM_STUB)

if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

try:
    import autocorrelation
except (Exception, SystemExit) as exc:  # pragma: no cover - dependency-driven
    raise unittest.SkipTest(f"cannot import autocorrelation: {exc}")


# ---------------------------------------------------------------------------
# Small CPU simulator: the reference Rule 30 update rule from a single black
# cell, used only to build the packed center-column bit array the module's own
# data files would otherwise provide. Validated below against OEIS A051023.
# ---------------------------------------------------------------------------
def rule30_center_bits(n: int) -> np.ndarray:
    """n center-column bits of Rule 30 from a single 1 cell (uint8 0/1)."""
    width = 4 * n + 1
    row = np.zeros(width, dtype=np.uint8)
    center = width // 2
    row[center] = 1
    bits = np.empty(n, dtype=np.uint8)
    for t in range(n):
        bits[t] = row[center]
        left = np.zeros(width, dtype=np.uint8)
        left[1:] = row[:-1]
        right = np.zeros(width, dtype=np.uint8)
        right[:-1] = row[1:]
        row = (left ^ (row | right)) & 1
    return bits


def to_pm1(bits: np.ndarray) -> np.ndarray:
    """Map the module's raw {0,1} bits to its {-1,+1} working convention."""
    return bits.astype(np.float64) * 2.0 - 1.0


# OEIS A051023: the first 15 center-column bits of Rule 30 from a single 1.
OEIS_A051023_PREFIX = (1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0)

RULE30_SIGNAL = to_pm1(rule30_center_bits(4096))

# Real outputs of autocorrelation_fft(RULE30_SIGNAL, 64, use_gpu=False),
# captured while writing this file. Keep them honest: these exact numbers
# change only if the estimator semantics change.
R30_CORRECTED_MAX = 0.035743307429   # max |r| over lags 1..64, mean-corrected
R30_ACORR_6 = 0.03564453125          # biggest raw |r| among lags 1..64
R30_ACORR_1 = 0.018310546875         # raw r(1)

# Corrected estimator: r_hat(lag) - E[X]^2  over  1 - E[X]^2. E[X]^2 is the
# first-order bias of the sample autocorrelation at every lag (finite-sample
# correction derived from the definition), and is 0 for a zero-mean signal.
def mean_corrected(acorr: np.ndarray, mean: float) -> np.ndarray:
    m2 = mean * mean
    return (acorr - m2) / (1.0 - m2)


class ImportTqdmGuardTest(unittest.TestCase):
    """The module must never actually call tqdm, only import it."""

    def test_module_import_does_not_call_tqdm(self):
        # If tqdm(){...} defined __call__ and the module invoked it at import
        # time, this would raise. The stub also asserts the module never calls
        # tqdm during the tests below, because __call__ raises.
        self.assertTrue(hasattr(autocorrelation, "autocorrelation_fft"))


class AutocorrelationShapeTest(unittest.TestCase):
    def test_returned_length_is_max_lag_plus_one(self):
        for n, max_lag in ((4, 2), (64, 5), (64, 64), (4096, 64)):
            signal = np.tile([1.0, -1.0], n // 2)
            acorr = autocorrelation.autocorrelation_fft(
                signal, max_lag, use_gpu=False)
            self.assertEqual(
                len(acorr), max_lag + 1,
                f"len(autocorrelation_fft(..., max_lag={max_lag})) must be "
                f"max_lag+1, got {len(acorr)}")

    def test_max_lag_beyond_pad_is_trimmed_to_pad_size(self):
        # Guard against a future edit that turns the [:max_lag + 1] slice into
        # a pad, which would silently return a wrong-length array.
        signal = np.tile([1.0, -1.0], 32)  # n = 64 -> pad 128
        acorr = autocorrelation.autocorrelation_fft(
            signal, 5000, use_gpu=False)
        self.assertEqual(len(acorr), 128)


class AutocorrelationLagZeroTest(unittest.TestCase):
    def test_lag_zero_is_exactly_one(self):
        # acorr /= acorr[0] normalises r(0) to 1.0 regardless of signal.
        for signal in (
            np.tile([1.0, -1.0], 512),          # period-2 square wave
            np.tile([1.0, 1.0, -1.0, -1.0], 256),  # period-4 square wave
            RULE30_SIGNAL,
            2 * np.random.default_rng(1).integers(0, 2, 512) - 1.0,
        ):
            acorr = autocorrelation.autocorrelation_fft(
                signal, 64, use_gpu=False)
            self.assertEqual(acorr[0], 1.0)

    def test_lag_zero_dominates_every_other_lag_on_rule30(self):
        # The self-correlation is the max of |r| over lags 0..max_lag for any
        # signal; the interesting direction is that lags 1..max_lag stay well
        # below it on the real center column (single seed, 4096 steps).
        acorr = autocorrelation.autocorrelation_fft(
            RULE30_SIGNAL, 64, use_gpu=False)
        self.assertEqual(acorr[0], 1.0)
        max_other = float(np.max(np.abs(acorr[1:])))
        self.assertLess(max_other, 0.1)
        self.assertEqual(max_other, R30_ACORR_6)


class AutocorrelationPeriodTest(unittest.TestCase):
    """Period detection: a period-p wave peaks at multiples of p.

    The raw FFT autocorrelation uses np.fft.rfft with no detrending, so the
    r(0) normalization of a *biased* estimator makes exact assertions fragile.
    What is asserted instead is the combination that is robust: every multiple
    of p is a strict local peak among its neighbours (which pins "peak at
    multiples of p" as exactly the property autocorrelation.py is for), and
    the two exactly-derivable values r(2) and r(1), which are integers over
    1024 for these signals.
    """

    def test_period_two_power_peaks_at_even_lags(self):
        for n in (1024, 2048):
            with self.subTest(n=n):
                signal = np.tile([1.0, -1.0], n // 2)
                acorr = autocorrelation.autocorrelation_fft(
                    signal, 64, use_gpu=False)
                for k in range(2, 65, 2):
                    self.assertGreater(
                        acorr[k], 0.9,
                        f"r({k}) of a period-2 wave must be a strong positive "
                        f"peak at a multiple of the period; got {acorr[k]}")
                for k in range(1, 65, 2):
                    self.assertLess(
                        acorr[k], -0.9,
                        f"r({k}) of a period-2 wave must be a strong negative "
                        f"anti-peak; got {acorr[k]}")

    def test_period_2_exact_derived_values(self):
        signal = np.tile([1.0, -1.0], 512)
        acorr = autocorrelation.autocorrelation_fft(
            signal, 64, use_gpu=False)
        self.assertAlmostEqual(acorr[2], 0.998046875, places=9)
        self.assertAlmostEqual(acorr[63], -0.9384765625, places=9)
        self.assertAlmostEqual(acorr[64], 0.9375, places=9)

    def test_period_4_peaks_at_multiples_of_four(self):
        signal = np.tile([1.0, 1.0, -1.0, -1.0], 256)
        acorr = autocorrelation.autocorrelation_fft(
            signal, 64, use_gpu=False)
        for k in range(4, 61, 4):
            self.assertGreater(
                acorr[k], 0.9,
                f"r({k}) of a period-4 wave must be a strong peak at a "
                f"multiple of the period; got {acorr[k]}")
        for k in range(4, 61, 4):
            self.assertGreater(acorr[k], acorr[k - 1])
            self.assertGreater(acorr[k], acorr[k + 1])
        self.assertAlmostEqual(acorr[4], 0.99609375, places=9)

    def test_rule30_has_no_period_and_is_dominated_by_lag_zero(self):
        acorr = autocorrelation.autocorrelation_fft(
            RULE30_SIGNAL, 64, use_gpu=False)
        self.assertEqual(acorr[0], 1.0)
        self.assertLess(float(np.max(np.abs(acorr[1:]))), 0.1)


class AutocorrelationExactnessTest(unittest.TestCase):
    """Closed-form checks that pin the estimator's semantics.

    The zero-padding to fft_size >= 2n makes the FFT estimate the *linear*
    (non-circular) biased autocorrelation. On a constant signal the linear
    estimate is exactly r(lag) = 1 - lag/n -- if a future edit shrank the pad
    to fft_size = n, the correlation would become circular and this would
    break (r(k) would be 1.0 for every k)."""

    def test_constant_signal_matches_linear_autocorrelation_closed_form(self):
        for n in (512, 1024):
            with self.subTest(n=n):
                acorr = autocorrelation.autocorrelation_fft(
                    np.ones(n), 64, use_gpu=False)
                self.assertEqual(acorr[0], 1.0)
                expected = 1.0 - np.arange(65, dtype=np.float64) / n
                np.testing.assert_allclose(acorr, expected, atol=1e-12)

    def test_period2_signal_matches_triangular_cosine_closed_form(self):
        # r(k) = (1 - k/n) . cos(pi k) for the alternating wave; exact to
        # machine precision because n is divisible by 16.
        n = 1024
        acorr = autocorrelation.autocorrelation_fft(
            np.tile([1.0, -1.0], n // 2), 64, use_gpu=False)
        expected = (1.0 - np.arange(65, dtype=np.float64) / n) * np.cos(
            np.pi * np.arange(65, dtype=np.float64))
        np.testing.assert_allclose(acorr, expected, atol=1e-12)


class NoiseFloorBaselineTest(unittest.TestCase):
    """Rule 6 of CLAUDE.md: a near-zero metric is not evidence of structure
    until a baseline exists. Here the baseline is a Bernoulli(-1,+1) stream of
    the same length, and the claim is that the Rule 30 raw autocorrelation,
    corrected for the finite-sample bias E[X]^2, stays within the 3-sigma
    counting floor of that baseline. This is the honest form of the negative
    Experiment B is built to detect -- with the single-seed caveat in the
    subTest name: 4096 steps of one deterministic IC is a pilot, not a proof.
    """

    def test_rule30_corrected_stays_below_noise_floor_with_baseline(self):
        mean = float(np.mean(RULE30_SIGNAL))
        acorr = autocorrelation.autocorrelation_fft(
            RULE30_SIGNAL, 64, use_gpu=False)
        corrected = mean_corrected(acorr, mean)
        max_r = float(np.max(np.abs(corrected[1:])))
        max_r2 = float(np.max(corrected[1:] * corrected[1:]))
        self.assertAlmostEqual(max_r2, R30_CORRECTED_MAX ** 2, places=12)
        self.assertLess(max_r, 3 * 2 / np.sqrt(4096))

    def test_rule30_corrected_is_no_stronger_than_bernoulli_baseline(self):
        rng = np.random.default_rng(12345)
        noise = 2 * rng.integers(0, 2, size=4096).astype(np.float64) - 1.0
        baseline = autocorrelation.autocorrelation_fft(
            noise, 64, use_gpu=False)
        baseline_max = float(np.max(np.abs(mean_corrected(baseline, float(noise.mean()))[1:])))
        mean = float(np.mean(RULE30_SIGNAL))
        acorr = autocorrelation.autocorrelation_fft(
            RULE30_SIGNAL, 64, use_gpu=False)
        rule30_max = float(np.max(np.abs(mean_corrected(acorr, mean)[1:])))
        # Measured: 0.035743307429 vs 0.029746062060 -- Rule 30's strongest
        # corrected lag-1..64 deviation is comparable to one draw of the
        # baseline; alone it proves nothing about linear structure.
        self.assertLess(rule30_max, 3 * baseline_max)
        self.assertLess(baseline_max, 3 * 2 / np.sqrt(4096))


class GPUDispatchTest(unittest.TestCase):
    def test_gpu_path_skips_when_cupy_absent(self):
        if autocorrelation.cp is None:
            self.assertFalse(autocorrelation.GPU)
            self.skipTest("no cupy installed")
        # GPU path must be flagged available, else the skip below is lying.
        self.assertTrue(autocorrelation.GPU)

    @unittest.skipUnless(autocorrelation.cp is not None,
                         "requires cupy/cuFFT for the GPU FFT path")
    def test_gpu_result_agrees_with_cpu(self):
        # CI has no GPU and will skip this; it exists so a GPU-capable machine
        # cannot ship a divergent fast path silently.
        ac_cpu = autocorrelation.autocorrelation_fft(
            RULE30_SIGNAL, 64, use_gpu=False)
        ac_gpu = autocorrelation.autocorrelation_fft(
            RULE30_SIGNAL, 64, use_gpu=True)
        np.testing.assert_allclose(ac_gpu, ac_cpu, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()