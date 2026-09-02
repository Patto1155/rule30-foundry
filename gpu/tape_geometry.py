#!/usr/bin/env python
"""Tape sizing rules for the Rule 30 kernels. Pure Python, no CUDA, no numpy.

Why this is a separate module
-----------------------------
`gpu/rule30_sim.py` imports cupy at module scope, so nothing in it can be
imported or unit-tested on a machine without a GPU. The sizing arithmetic is
the part most likely to be got wrong and the most expensive to get wrong, so
it lives here where it can be tested anywhere.

The rule
--------
The kernels zero-pad outside the array: a thread at word 0 reads
`prev_word = 0`, and at word `n_words-1` reads `next_word = 0`. On an infinite
zero background that is *correct*, because Rule 30 maps all-zero to all-zero -
the padding only becomes wrong once the pattern actually reaches the array
edge.

Rule 30 spreads at most one cell per step in each direction, so from a single
seed at cell `c` the pattern after `t` steps is contained in `[c-t, c+t]`. The
simulation is therefore exact for every cell while

    t <= min(c, n_cells - 1 - c)

and beyond that the edge cells are wrong, with the error propagating inward at
one cell per step until it reaches the center column.

**This matters far more than it looks.** A tape that is too short does not
crash and does not look wrong: the first bits are fine, the bit mean stays
0.5, and the `--center` sanity check on the first 20 bits still passes. It
fails *late*, which is exactly the signature `docs/handover/2026-08-29-*.md`
warns is a real kernel bug rather than a packing mismatch. Sizing a run to fit
VRAM without checking this produces a plausible, hash-anchored, wrong
bitstream after hours of compute.
"""

from __future__ import annotations

WORD_BITS = 64
# gpu/rule30_sim.py seeds bit 32 of the middle word.
SEED_BIT_IN_WORD = 32


def word_count(n_cells: int) -> int:
    """Words needed for `n_cells`, matching the kernel's rounding."""
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}")
    return (n_cells + WORD_BITS - 1) // WORD_BITS


def rounded_cells(n_cells: int) -> int:
    """The kernel rounds the tape up to a whole number of words."""
    return word_count(n_cells) * WORD_BITS


def seed_cell(n_cells: int) -> int:
    """Linear index of the seeded cell, as the kernel places it."""
    return (word_count(n_cells) // 2) * WORD_BITS + SEED_BIT_IN_WORD


def max_safe_steps(n_cells: int) -> int:
    """Largest `n_steps` for which every cell stays exact on this tape."""
    cells = rounded_cells(n_cells)
    c = seed_cell(n_cells)
    return min(c, cells - 1 - c)


def min_cells_for_steps(n_steps: int) -> int:
    """Smallest tape width that keeps `n_steps` exact.

    Solved by construction rather than search: the binding side is the right
    edge, `cells - 1 - c >= n_steps`, and `c` is just under half the tape.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    cells = rounded_cells(max(1, 2 * n_steps))
    while max_safe_steps(cells) < n_steps:
        cells += WORD_BITS
    return cells


def describe(n_cells: int, n_steps: int) -> dict:
    """Everything a caller needs to decide whether a run is sound."""
    cells = rounded_cells(n_cells)
    safe = max_safe_steps(n_cells)
    return {
        "requested_cells": n_cells,
        "rounded_cells": cells,
        "n_words": word_count(n_cells),
        "seed_cell": seed_cell(n_cells),
        "n_steps": n_steps,
        "max_safe_steps": safe,
        "cone_fits": n_steps <= safe,
        "shortfall_steps": max(0, n_steps - safe),
        "min_cells_for_steps": min_cells_for_steps(n_steps),
        "tape_bytes": word_count(n_cells) * 8 * 2,      # double-buffered
        "center_buffer_bytes": n_steps,                  # one byte per step
    }


class ConeTooLarge(ValueError):
    """The light cone reaches the tape edge before the run ends."""


def check(n_cells: int, n_steps: int) -> dict:
    """Raise unless the run is exact for every step. Returns the description."""
    info = describe(n_cells, n_steps)
    if not info["cone_fits"]:
        raise ConeTooLarge(
            f"light cone escapes the tape: {n_steps:,} steps on "
            f"{info['rounded_cells']:,} cells, but only the first "
            f"{info['max_safe_steps']:,} steps are exact "
            f"({info['shortfall_steps']:,} short).\n"
            f"  Rule 30 spreads 1 cell/step, so N valid center bits need "
            f"~2N cells.\n"
            f"  Use --cells {info['min_cells_for_steps']:,} for "
            f"{n_steps:,} steps "
            f"({info['min_cells_for_steps'] // 8 // 1024 // 1024} MiB packed, "
            f"x2 double-buffered), or --steps {info['max_safe_steps']:,} on "
            f"this tape.\n"
            f"  A short tape does not crash and does not look wrong: the bit "
            f"mean stays 0.5 and the first-20-bit check still passes. It goes "
            f"wrong LATE. Pass --allow-truncated-cone only if you have a "
            f"specific reason and will not anchor the output."
            .replace(",", ","))
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=int)
    ap.add_argument("--steps", type=int, required=True)
    args = ap.parse_args()
    if args.cells:
        info = describe(args.cells, args.steps)
        for k, v in info.items():
            print(f"{k:24s} {v:,}" if isinstance(v, int) else f"{k:24s} {v}")
    else:
        cells = min_cells_for_steps(args.steps)
        print(f"{args.steps:,} steps needs at least {cells:,} cells "
              f"({cells // 8 / 1e6:.1f} MB packed, "
              f"{cells // 8 * 2 / 1e6:.1f} MB double-buffered, "
              f"+ {args.steps / 1e6:.1f} MB center buffer)")
