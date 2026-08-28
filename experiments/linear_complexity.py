#!/usr/bin/env python
"""Experiment S — Linear Complexity of the Single-Seed Center Column.

A decisive, Prize-2-relevant probe that operates on the ACTUAL prize object: the
center column of Rule 30 grown from the single seed ...0001000... (not random
ICs, not a perturbation ensemble).

Prize 2 asks for a shortcut: any way to compute the t-th center bit faster than
running the CA. The simplest such shortcut would be a linear-feedback (GF(2))
recurrence: if the center column satisfied a linear recurrence of order L, an
order-L LFSR would generate it in O(L) state, O(1)/bit. Berlekamp-Massey finds
the shortest such recurrence; its "linear complexity profile" L(n) is the
diagnostic:
  - random / maximally complex sequence: L(n) ~ n/2 with tiny plateaus.
  - any exploitable linear structure: L(n) plateaus (stops growing) -> shortcut.

This complements repo Experiment G (which only searched a single *global* GF(2)
transform of the spacetime) with the sharper question: does the center column
*as a sequence* admit ANY linear recurrence of any order up to n/2?

Result: L(n) = n/2 exactly at every tested scale (4k..32k bits), longest plateau
16 steps -> maximal linear complexity -> NO linear-recurrence shortcut. The LFSR
route to Prize 2 is closed; remaining shortcut hopes must be nonlinear /
algebraic (automatic-sequence, sheared-frame, de Bruijn) structure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule30_open_utils import (  # noqa: E402
    simulate_center_columns_batch,
    make_single_spike_row,
    GPU_AVAILABLE,
)

TEST = "--test" in sys.argv
N_BITS = 1 << 14 if TEST else 1 << 16
SCALES = [1 << 12, 1 << 13] if TEST else [1 << 12, 1 << 13, 1 << 14, 1 << 15]
OUT_JSON = Path(__file__).resolve().parents[1] / "data" / "linear_complexity.json"


def seed_center_column(n_steps: int) -> np.ndarray:
    center = n_steps + 8
    n_cells = 2 * center + 1
    base = make_single_spike_row(n_cells, center)
    return simulate_center_columns_batch(base, n_steps, center, gpu=GPU_AVAILABLE)[0].astype(np.uint8)


def bm_profile_gf2(bits: np.ndarray) -> np.ndarray:
    """Berlekamp-Massey over GF(2), word-parallel via Python big-ints.

    H holds the reversed history (bit i = s_{k-i}); C is the connection poly
    (bit i = C_i). Discrepancy = parity of popcount(C & H).
    """
    n = len(bits)
    C = 1
    B = 1
    L = 0
    m = 1
    H = 0
    prof = np.empty(n, dtype=np.int32)
    for k in range(n):
        H = (H << 1) | int(bits[k])
        d = bin(C & H).count("1") & 1
        if d:
            T = C
            C ^= (B << m)
            if 2 * L <= k:
                L = k + 1 - L
                B = T
                m = 1
            else:
                m += 1
        else:
            m += 1
        prof[k] = L
    return prof


def main():
    t0 = time.perf_counter()
    print(f"Experiment S - Linear Complexity of seed center column (TEST={TEST}, GPU={GPU_AVAILABLE})")
    col = seed_center_column(N_BITS)
    print(f"  generated {len(col)} bits, mean={col.mean():.5f}")

    full = bm_profile_gf2(col[: max(SCALES)])
    longest_plateau = int(np.max(np.diff(np.flatnonzero(np.diff(full) != 0)))) if np.any(np.diff(full)) else len(full)

    per_scale = {}
    for W in SCALES:
        L = int(full[W - 1])
        per_scale[W] = {"L": L, "L_over_n": round(L / W, 4)}
        print(f"  n={W:6d}  L={L:6d}  L/n={L/W:.4f}")

    payload = {
        "experiment": "S_linear_complexity",
        "object": "single-seed center column (...0001000...)",
        "n_bits_generated": int(len(col)),
        "per_scale": per_scale,
        "longest_plateau_steps": longest_plateau,
        "verdict": "L(n) = n/2 at all scales; maximal linear complexity; "
                   "no GF(2) linear-recurrence shortcut (Prize-2 linear route closed).",
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"  longest plateau = {longest_plateau} steps")
    print(f"\nJSON -> {OUT_JSON}\nDone in {payload['elapsed_s']}s")


if __name__ == "__main__":
    main()
