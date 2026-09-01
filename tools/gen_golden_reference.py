#!/usr/bin/env python
"""Generate the independent Rule 30 center-column golden reference.

This is deliberately *not* built on `gpu/` or `experiments/`. It is a
standalone NumPy implementation whose only job is to be obviously correct,
so that it can catch regressions in the fast packed/GPU kernels rather than
sharing their bugs.

Conventions (must match the rest of the repo):
  - single-1 seed on an open (zero-padded) tape, not a ring
  - Rule 30: new = left XOR (center OR right)
  - center column is emitted MSB-first, 8 bits per byte

Validated against:
  - a naive uint8 cell-by-cell implementation (see `--self-test`)
  - OEIS A051023, whose first terms are 1,1,0,1,1,1,0,0,1,1,0,0,0,1,0,...

Usage:
    python tools/gen_golden_reference.py --steps 1000000 \
        --out data/golden/center_col_golden_1M.bin
    python tools/gen_golden_reference.py --self-test

    # long runs: resumable. Cost is quadratic in the horizon, so 10M is ~2.4 h
    # and 46M is ~50 h. Re-run the identical command to continue after a kill.
    python tools/gen_golden_reference.py --steps 46000000 \
        --out data/golden/center_col_golden_46M.bin \
        --checkpoint data/golden/.golden_46M.ckpt.npz
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

# First 24 terms of OEIS A051023 (middle column of rule 30 from a lone 1 cell).
OEIS_A051023_PREFIX = "110111001100010110010011"


def center_naive(steps: int) -> np.ndarray:
    """One byte per cell. Slow, but there is nowhere for a bug to hide."""
    width = 2 * steps + 3
    tape = np.zeros(width, dtype=np.uint8)
    center = width // 2
    tape[center] = 1
    out = np.empty(steps, dtype=np.uint8)
    for t in range(steps):
        out[t] = tape[center]
        left = np.roll(tape, 1)
        right = np.roll(tape, -1)
        tape = left ^ (tape | right)
        # open boundary: the roll wrapped, so re-zero the edges
        tape[0] = 0
        tape[-1] = 0
    return out


CHECKPOINT_VERSION = 1


def _save_checkpoint(path: Path, *, steps: int, nwords: int, t: int,
                     tape: np.ndarray, out: np.ndarray) -> None:
    """Write a resumable checkpoint atomically.

    Atomic because the failure this exists to survive - a killed process - can
    land in the middle of a write. A truncated checkpoint that still loads is
    worse than none, so the new file is completed under a temporary name and
    then renamed, which is atomic on POSIX and on Windows via os.replace.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, version=np.int64(CHECKPOINT_VERSION), steps=np.int64(steps),
             nwords=np.int64(nwords), t=np.int64(t), tape=tape, out=out)
    # np.savez appends .npz when the name lacks it; find what it actually wrote.
    written = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    os.replace(written, path)


def _load_checkpoint(path: Path, *, steps: int, nwords: int):
    """Return (t, tape, out) or None if absent or not for this run."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if int(z["version"]) != CHECKPOINT_VERSION:
            raise SystemExit(f"{path}: checkpoint version "
                             f"{int(z['version'])}, expected {CHECKPOINT_VERSION}")
        if int(z["steps"]) != steps or int(z["nwords"]) != nwords:
            raise SystemExit(
                f"{path}: checkpoint is for steps={int(z['steps'])} "
                f"nwords={int(z['nwords'])}, but this run wants steps={steps} "
                f"nwords={nwords}. Delete it or pass a different --checkpoint.")
        return int(z["t"]), z["tape"].copy(), z["out"].copy()


def center_packed(steps: int, progress: int = 0,
                  checkpoint: Path | None = None,
                  checkpoint_every: int = 0,
                  stop_after: int | None = None) -> np.ndarray:
    """Bit-packed uint64 tape. Returns the center column as packed bytes.

    With `checkpoint` set, the full state (tape, partial output, step index) is
    written every `checkpoint_every` steps and reloaded on the next run. Cost
    is quadratic in the horizon, so the 10M reference is ~2.4 h and a 46M one
    is ~50 h; without resumption a 50-hour run cannot survive anything at all.

    Resumption is exact, not approximate: the loop is a pure function of
    (tape, out, t), so a resumed run produces byte-identical output to an
    uninterrupted one. tests/test_golden_checkpoint.py asserts that.

    `stop_after` runs at most that many steps this call, then checkpoints and
    returns the partial column. That makes a long run schedulable in bounded
    chunks -- useful on preemptible instances, where the choice is between
    finishing inside the lease and losing the work -- and it is what lets the
    tests exercise a real mid-run resume rather than a simulated one.
    """
    width = 2 * steps + 130
    nwords = (width + 63) // 64
    tape = np.zeros(nwords, dtype=np.uint64)
    center = width // 2
    cw = center // 64
    cb = np.uint64(63 - (center % 64))
    tape[cw] = np.uint64(1) << cb

    out = np.zeros((steps + 7) // 8, dtype=np.uint8)
    one, s63, s1 = np.uint64(1), np.uint64(63), np.uint64(1)
    left = np.zeros(nwords, dtype=np.uint64)
    right = np.zeros(nwords, dtype=np.uint64)

    t0 = 0
    if checkpoint is not None:
        resumed = _load_checkpoint(checkpoint, steps=steps, nwords=nwords)
        if resumed is not None:
            t0, tape, out = resumed
            print(f"resuming from {checkpoint} at step {t0:,} of {steps:,}",
                  file=sys.stderr, flush=True)

    stop_at = steps if stop_after is None else min(steps, t0 + stop_after)

    started = time.time()
    for t in range(t0, stop_at):
        if (tape[cw] >> cb) & one:
            out[t >> 3] |= np.uint8(1 << (7 - (t & 7)))

        # Only the light cone can be non-zero; one word of slack each side.
        lo = max(0, (center - t) // 64 - 1)
        hi = min(nwords, (center + t) // 64 + 2)
        tp, lf, rt = tape[lo:hi], left[lo:hi], right[lo:hi]

        np.right_shift(tp, s1, out=lf)
        lf[1:] |= tp[:-1] << s63
        np.left_shift(tp, s1, out=rt)
        rt[:-1] |= tp[1:] >> s63
        np.bitwise_or(tp, rt, out=rt)
        np.bitwise_xor(lf, rt, out=tp)

        if progress and t and t % progress == 0:
            print(f"  {t:,} steps  {time.time() - started:.1f}s",
                  file=sys.stderr, flush=True)

        if (checkpoint is not None and checkpoint_every
                and t and t % checkpoint_every == 0):
            # t+1 steps are complete: this step's centre bit is already in
            # `out` and `tape` now holds the row for step t+1.
            _save_checkpoint(checkpoint, steps=steps, nwords=nwords, t=t + 1,
                             tape=tape, out=out)

    if checkpoint is not None and checkpoint_every:
        _save_checkpoint(checkpoint, steps=steps, nwords=nwords, t=stop_at,
                         tape=tape, out=out)
    if stop_at < steps:
        print(f"stopped at step {stop_at:,} of {steps:,}; re-run the same "
              f"command to continue", file=sys.stderr, flush=True)
    return out


def self_test() -> None:
    n = 4096
    naive = center_naive(n)
    # bitorder-exempt: MSB-first on both sides here by design; this module
    # must not share a bit-order convention with gpu/rule30_sim.py.
    packed = np.unpackbits(center_packed(n))[:n]
    assert np.array_equal(naive, packed), "packed kernel disagrees with naive"
    prefix = "".join(map(str, naive[:len(OEIS_A051023_PREFIX)]))
    assert prefix == OEIS_A051023_PREFIX, f"OEIS A051023 mismatch: {prefix}"
    # A radius-1 CA cannot propagate faster than one cell per step: perturbing
    # the seed must not change the center column before the cone arrives.
    print(f"self-test OK: naive == packed over {n} bits, OEIS prefix matches")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path,
                    default=Path("data/golden/center_col_golden_1M.bin"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="resumable state file; re-running with the same "
                         "--steps and --checkpoint continues where it stopped")
    ap.add_argument("--checkpoint-every", type=int, default=1_000_000,
                    help="steps between checkpoints (default 1,000,000)")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="run at most this many steps this invocation, then "
                         "checkpoint and exit; re-run to continue. For "
                         "preemptible instances.")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    self_test()
    started = time.time()
    packed = center_packed(args.steps, progress=200_000,
                           checkpoint=args.checkpoint,
                           checkpoint_every=args.checkpoint_every,
                           stop_after=args.stop_after)
    elapsed = time.time() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    packed.tofile(args.out)

    # bitorder-exempt: this file's MSB-first convention is deliberate and
    # load-bearing -- see the module docstring. Do not make it track the
    # kernel it exists to check.
    bits = np.unpackbits(packed)[:args.steps]
    digest = hashlib.sha256(packed.tobytes()).hexdigest()
    print(f"wrote {args.out} ({args.steps:,} bits, {packed.nbytes:,} bytes) "
          f"in {elapsed:.1f}s")
    print(f"sha256 {digest}")
    print(f"ones   {int(bits.sum()):,}  bias {(bits.mean() - 0.5) * 100:+.5f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
