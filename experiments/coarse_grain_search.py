#!/usr/bin/env python
"""Experiment T — Coarse-Graining / Renormalization Search (with sheared supercells).

Pitch & prior art
-----------------
Israeli & Goldenfeld (PRL 2004; PRE 2006) and Song & Grochow (2020) searched for
*exact, total, deterministic* coarse-grainings of elementary CA — a projection π
from b-cell supercells to a smaller-radius rule that reproduces the dynamics with
ZERO error for all configurations. They swept all elementary rules up to supercell
size N=7. Rule 30 is in the set with **no exact coarse-graining** (it is the
canonical irreducible rule). So re-searching for *exact* coarse-graining at small
b just reproduces a known negative — we don't do that blindly; we use it as a
pipeline check.

What is genuinely open (and what this probe targets)
----------------------------------------------------
1. APPROXIMATE closure: not 100%-deterministic, but "coarse field predictable to
   within ε". A CNN/predictor measures this; I-G never reported it. A high
   approximate closure would be a real compressibility/predictability result.
2. SHEARED supercells. Everyone used axis-aligned blocks. Rule 30's causal
   geometry is anisotropic — our damage-velocity work (Exp R) measured rightward
   front speed v_R = 1 and leftward bulk speed v_L ≈ 0.244. Supercells sheared to
   follow those fronts are a coordinate frame nobody has tried.

Method
------
- Generate a bulk Rule-30 spacetime patch from a RANDOM IC (coarse-graining is a
  property of the *rule*, so we study generic bulk dynamics, not the spike).
- For a shear slope σ, shift fine row t left by round(σ·t), then block into b×b
  supercells and project to a binary coarse field via π.
- Closure(σ, π) = accuracy of the OPTIMAL deterministic predictor of coarse[T+1,X]
  from the (2r+1)-neighborhood of coarse[T,·] (max-likelihood per neighborhood
  pattern — the exact best a local coarse rule could do). Constrained to
  non-trivial π (coarse-field entropy ≥ H_MIN, so π≠const).
- b=2: ENUMERATE all 2^(b·b) binary projections (exact, decisive).
- Null: identical pipeline on an i.i.d. fair-coin field → the closure a
  structureless field achieves by finite-sample luck (the bar to beat).
- (Optional) b=3 learned projection via PyTorch straight-through search — the CNN
  the search question asked about, where enumeration (2^512) is impossible.

Read-out: does Rule 30's best coarse field close *above the i.i.d. null* at any
shear? Axis-aligned should not (consistent with I-G); the live question is whether
a cone-aligned shear does.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule30_open_utils import simulate_spacetime, GPU_AVAILABLE  # noqa: E402


# ---------------------------------------------------------------------------
# Field generation
# ---------------------------------------------------------------------------
def bulk_spacetime(n_steps: int, width: int, seed: int) -> np.ndarray:
    """Random-IC Rule-30 spacetime, interior window (boundary trimmed)."""
    rng = np.random.default_rng(seed)
    margin = n_steps + 32
    n_cells = width + 2 * margin
    row = rng.integers(0, 2, size=n_cells, dtype=np.uint8)
    st = simulate_spacetime(row, n_steps, gpu=GPU_AVAILABLE)  # [n_steps, n_cells]
    return st[:, margin:margin + width].copy()


# ---------------------------------------------------------------------------
# Shear + blocking + projection
# ---------------------------------------------------------------------------
def sheared_blocks(field: np.ndarray, b: int, shear: float) -> np.ndarray:
    """Return coarse grid of b*b block-patterns as integer codes [Tc, Xc].

    Row t is rolled left by round(shear*t) before axis-aligned b×b blocking, so a
    shear σ aligns supercells to a front moving at σ cells/step.
    """
    H, W = field.shape
    if shear != 0.0:
        shifted = np.empty_like(field)
        for t in range(H):
            shifted[t] = np.roll(field[t], -int(round(shear * t)))
        field = shifted
    Tc, Xc = H // b, W // b
    field = field[: Tc * b, : Xc * b]
    # pack each b×b block into an integer code 0..2^(b*b)-1
    blocks = field.reshape(Tc, b, Xc, b).transpose(0, 2, 1, 3).reshape(Tc, Xc, b * b)
    weights = (1 << np.arange(b * b)).astype(np.int64)
    return (blocks.astype(np.int64) * weights).sum(axis=2)  # [Tc, Xc] codes


def _bits_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def prep_transitions(field: np.ndarray, b: int, shear: float, r: int,
                     m_max: int, seed: int = 0):
    """Precompute per-shear transition arrays: neighbor block-codes + target code,
    each value in 0..2^(b*b)-1. Subsampled to <= m_max transitions for speed."""
    codes = sheared_blocks(field, b, shear)          # [Tc, Xc] block codes
    Tc, Xc = codes.shape
    if Tc < 2 or Xc < 2 * r + 1:
        return None
    src, tgt = codes[:-1], codes[1:]
    nbrs = [src[:, r + dx: Xc - r + dx].ravel() for dx in range(-r, r + 1)]
    t = tgt[:, r: Xc - r].ravel()
    M = t.size
    if M > m_max:
        sel = np.random.default_rng(seed).choice(M, m_max, replace=False)
        nbrs = [a[sel] for a in nbrs]
        t = t[sel]
        M = m_max
    return nbrs, t, M


def enumerate_b2(field: np.ndarray, shear: float, r: int, h_min: float,
                 m_max: int = 20000):
    """Best (closure, entropy, π) over ALL 2^4 non-trivial binary projections.

    closure = accuracy of the optimal deterministic local predictor (per-
    neighborhood-pattern majority), computed with a single bincount per π.
    """
    b = 2
    prep = prep_transitions(field, b, shear, r, m_max)
    if prep is None:
        return {"closure": float("nan"), "entropy": 0.0, "pi": None}
    nbrs, tgt, M = prep
    n_pat = 1 << (b * b)                               # 16 block patterns
    npat_nbr = 1 << (2 * r + 1)
    # Select by EXCESS closure = closure - marginal baseline max(p,1-p), so the
    # score measures predictability ABOVE the trivial "predict majority class"
    # rather than rewarding lopsided projections.
    best = {"excess": -1.0, "closure": 0.0, "baseline": 0.0, "entropy": 0.0, "pi": None}
    for bits in range(1, (1 << n_pat) - 1):
        lut = np.array([(bits >> p) & 1 for p in range(n_pat)], dtype=np.int64)
        p1 = float(lut[nbrs[r]].mean())
        h = _bits_entropy(p1)
        if h < h_min:
            continue
        baseline = max(p1, 1.0 - p1)
        pat = np.zeros(M, dtype=np.int64)
        for a in nbrs:
            pat = pat * 2 + lut[a]
        key = pat * 2 + lut[tgt]
        counts = np.bincount(key, minlength=npat_nbr * 2).reshape(-1, 2)
        closure = counts.max(1).sum() / M
        excess = closure - baseline
        if excess > best["excess"]:
            best = {"excess": float(excess), "closure": float(closure),
                    "baseline": float(baseline), "entropy": h, "pi": lut.tolist()}
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    n_steps = 600 if args.test else 1600
    width = 600 if args.test else 1600
    r = 1                                   # coarse neighborhood radius
    h_min = 0.85                            # near-balanced coarse field (bits)
    shears = [0.0, 0.25, 0.5, 1.0, -0.25, -1.0]
    seed = 7

    print(f"Experiment T - Coarse-Graining Search (GPU={GPU_AVAILABLE}, test={args.test})")
    t0 = time.perf_counter()
    field = bulk_spacetime(n_steps, width, seed)
    print(f"  bulk Rule-30 field {field.shape}, density={field.mean():.4f}  "
          f"({time.perf_counter()-t0:.1f}s)")
    null = np.random.default_rng(99).integers(0, 2, size=field.shape, dtype=np.uint8)

    print(f"\n  b=2 exact enumeration, neighborhood r={r}, H_min={h_min}")
    print("  EXCESS = closure - max(p,1-p)  (predictability above marginal; "
          "i.i.d. null ~ 0)")
    print(f"  {'shear':>6} | {'R30 excess':>10} {'clos':>6} | "
          f"{'null excess':>11} {'clos':>6} | {'R30-null':>9}")
    rows = []
    for sigma in shears:
        rg = enumerate_b2(field, sigma, r, h_min)
        ng = enumerate_b2(null, sigma, r, h_min)
        gap = rg["excess"] - ng["excess"]
        rows.append((sigma, rg, ng, gap))
        print(f"  {sigma:6.2f} | {rg['excess']:+10.4f} {rg['closure']:6.3f} | "
              f"{ng['excess']:+11.4f} {ng['closure']:6.3f} | {gap:+9.4f}")

    best = max(rows, key=lambda x: x[3])
    print(f"\n  Best shear by (R30 - null) excess gap: shear={best[0]:.2f}, "
          f"gap={best[3]:+.4f}, R30 excess={best[1]['excess']:+.4f}")
    interp = ("NO coarse-graining signal above i.i.d. null - consistent with "
              "Israeli-Goldenfeld irreducibility of Rule 30."
              if best[3] < 0.03 else
              "Cone-aligned shear is predictable ABOVE the null - investigate "
              "(possible approximate reducibility in a sheared frame).")
    print(f"  Interpretation: {interp}")
    print(f"\n  total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
