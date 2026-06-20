#!/usr/bin/env python
"""GPU-vectorized b=2 coarse-grain enumeration (drop-in for enumerate_b2, r=1).

The reference `coarse_grain_search.enumerate_b2` loops over all 2^16 binary
projections in Python, doing a bincount over the M sampled transitions for each —
~15 s/field. That is the bottleneck that makes cross-rule sweeps slow and b>=3
intractable.

This version computes the same quantity with NO per-projection Python loop:

  1. Build the joint block-code transition histogram ONCE: a 16^4 tensor
     H[c_{-1}, c_0, c_1, c_tgt] counting (3-neighbour, target) block-code tuples.
     This is independent of the projection and replaces the M-dependent bincount.
  2. Enumerate all projections as a (N, 16) bit matrix. p1 (center-cell density)
     is a single matmul against the center marginal -> entropy filter, all at once.
  3. For the survivors, reduce H -> T2[a,b,c,d] (a 2x2x2x2 table per projection)
     by four batched group-contractions on the GPU, then
     closure = sum_{a,b,c} max_d T2 / M.

Verified bit-for-bit against enumerate_b2 on the same subsample (see verify()).
Falls back to enumerate_b2 for r != 1.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_grain_search import prep_transitions, enumerate_b2  # noqa: E402

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

_GPU_LINALG = None


def _gpu_linalg_ok() -> bool:
    """cuPy einsum needs cuBLAS; probe once and cache (some installs lack it)."""
    global _GPU_LINALG
    if _GPU_LINALG is None:
        try:
            float(cp.matmul(cp.ones((2, 2)), cp.ones((2, 2))).sum())
            _GPU_LINALG = True
        except Exception:
            _GPU_LINALG = False
    return bool(_GPU_LINALG)


def _entropy_vec(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.where((p > 0) & (p < 1), h, 0.0)


def enumerate_b2_fast(
    field: np.ndarray,
    shear: float,
    r: int,
    h_min: float,
    m_max: int = 20000,
    seed: int = 0,
    gpu: bool = True,
    chunk: int = 4096,
):
    """Best (excess, closure, pi) over all non-trivial binary b=2 projections.

    Matches coarse_grain_search.enumerate_b2 exactly for r=1; delegates otherwise.
    """
    if r != 1:
        return enumerate_b2(field, shear, r, h_min, m_max)

    prep = prep_transitions(field, 2, shear, r, m_max, seed)
    if prep is None:
        return {"closure": float("nan"), "entropy": 0.0, "pi": None}
    nbrs, tgt, M = prep
    n0, nc, n1 = (a.astype(np.int64) for a in nbrs)
    tgt = tgt.astype(np.int64)

    # 1) joint 16^4 transition histogram (neighbour_-1, center, neighbour_+1, target)
    key = ((n0 * 16 + nc) * 16 + n1) * 16 + tgt
    H = np.bincount(key, minlength=16 ** 4).astype(np.float64).reshape(16, 16, 16, 16)
    m_center = H.sum(axis=(0, 2, 3))  # marginal over the center block code

    # 2) all non-trivial projections; entropy filter via a single matmul
    P = np.arange(1, 16 ** 4 - 1, dtype=np.int64)            # exclude all-0 / all-1
    bits = ((P[:, None] >> np.arange(16)) & 1).astype(np.float64)  # (N,16)
    p1 = bits @ m_center / M
    h = _entropy_vec(p1)
    mask = h >= h_min
    if not mask.any():
        return {"excess": -1.0, "closure": 0.0, "baseline": 0.0, "entropy": 0.0, "pi": None}
    Gs = bits[mask]
    Ps = P[mask]
    p1s = p1[mask]
    hs = h[mask]
    baseline = np.maximum(p1s, 1.0 - p1s)

    # 3) closure for each survivor via batched group-contraction of H
    xp = cp if (gpu and cp is not None and _gpu_linalg_ok()) else np
    Hx = xp.asarray(H)
    closures = np.empty(Gs.shape[0], dtype=np.float64)
    for s0 in range(0, Gs.shape[0], chunk):
        g = xp.asarray(Gs[s0:s0 + chunk])                 # (c,16) in {0,1}
        gm = xp.stack([1.0 - g, g], axis=2)               # (c,16,2): code -> group one-hot
        # reduce each 16-axis of H into its 2 projection groups, append group index
        Y = xp.einsum("ijkl,sia->sjkla", Hx, gm)          # (c,16,16,16,2)  [j,k,l,a]
        Z = xp.einsum("sjkla,sjb->sklab", Y, gm)          # (c,16,16,2,2)   [k,l,a,b]
        W = xp.einsum("sklab,skc->slabc", Z, gm)          # (c,16,2,2,2)    [l,a,b,c]
        T = xp.einsum("slabc,sld->sabcd", W, gm)          # (c,2,2,2,2)     [a,b,c,d]
        clo = T.max(axis=4).sum(axis=(1, 2, 3)) / M       # (c,)
        closures[s0:s0 + chunk] = cp.asnumpy(clo) if xp is cp else np.asarray(clo)

    excess = closures - baseline
    i = int(np.argmax(excess))
    lut = ((Ps[i] >> np.arange(16)) & 1).astype(np.int64)
    return {"excess": float(excess[i]), "closure": float(closures[i]),
            "baseline": float(baseline[i]), "entropy": float(hs[i]), "pi": lut.tolist()}


def verify(seed: int = 0) -> None:
    """Match enumerate_b2 bit-for-bit on identical subsamples, several fields."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eca_sim import simulate_spacetime_rule, GPU_AVAILABLE

    def bulk(rule, n, w, sd):
        rng = np.random.default_rng(sd)
        margin = n + 32
        nc = w + 2 * margin
        row = rng.integers(0, 2, size=nc, dtype=np.uint8)
        st = simulate_spacetime_rule(row, n, rule, gpu=GPU_AVAILABLE)
        return st[:, margin:margin + w].copy()

    cases = [(30, 0.0), (30, 0.25), (45, 0.0), (90, 0.0), (110, 1.0)]
    for rule, shear in cases:
        field = bulk(rule, 300, 300, 7)
        ref = enumerate_b2(field, shear, 1, 0.85, m_max=20000)
        fast = enumerate_b2_fast(field, shear, 1, 0.85, m_max=20000, seed=0,
                                 gpu=GPU_AVAILABLE)
        # prep_transitions uses seed=0 by default in enumerate_b2's call path.
        ok_clo = abs(ref["closure"] - fast["closure"]) < 1e-12
        ok_exc = abs(ref["excess"] - fast["excess"]) < 1e-12
        same_pi = ref["pi"] == fast["pi"]
        tag = "OK" if (ok_clo and ok_exc) else "MISMATCH"
        print(f"  rule {rule:>3} shear {shear:>4}: ref excess {ref['excess']:+.6f} "
              f"clos {ref['closure']:.6f} | fast {fast['excess']:+.6f} {fast['closure']:.6f} "
              f"| pi {'==' if same_pi else '!='} {tag}")
        if not (ok_clo and ok_exc):
            raise RuntimeError(f"enumerate_b2_fast != enumerate_b2 for rule {rule} shear {shear}")
    print("  all fast == reference (closure & excess exact)  OK")


def benchmark(seed: int = 0) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eca_sim import simulate_spacetime_rule, GPU_AVAILABLE
    rng = np.random.default_rng(7)
    n = w = 1200
    margin = n + 32
    nc = w + 2 * margin
    row = rng.integers(0, 2, size=nc, dtype=np.uint8)
    field = simulate_spacetime_rule(row, n, 30, gpu=GPU_AVAILABLE)[:, margin:margin + w].copy()

    t0 = time.perf_counter()
    enumerate_b2(field, 0.0, 1, 0.85, m_max=20000)
    t_ref = time.perf_counter() - t0

    enumerate_b2_fast(field, 0.0, 1, 0.85, gpu=GPU_AVAILABLE)  # warm up
    t0 = time.perf_counter()
    enumerate_b2_fast(field, 0.0, 1, 0.85, gpu=GPU_AVAILABLE)
    t_fast = time.perf_counter() - t0
    print(f"\n  enumerate_b2 (loop) : {t_ref*1000:8.1f} ms")
    print(f"  enumerate_b2_fast   : {t_fast*1000:8.1f} ms   speedup {t_ref/t_fast:5.1f}x")


if __name__ == "__main__":
    print("Vectorized b=2 coarse-grain enumeration")
    verify()
    benchmark()
