#!/usr/bin/env python
"""General-b coarse-grain closure: GPU-resident, uncapped, two evaluation paths.

This is the foundation for the b>=3 reducibility search. It generalizes the b=2
closure machinery (`coarse_grain_search` / `coarse_grain_fast`) to arbitrary block
size and removes the m_max subsample cap, which was a leftover from the old
O(M)-per-projection loop and threw away ~98% of the transitions.

Two evaluation paths (both verified to agree at b=2):

  closure_enumerate_b2(prep)
      The code-histogram path: build the 16^4 block-code transition histogram once
      (cost ~ one bincount, independent of how many projections we score), then
      score ALL 2^16 projections by contracting it. Only feasible for b=2 (the
      16^4 tensor fits; 512^4 does not). Used for the exhaustive b=2 baseline.

  closure_batch(prep, luts)
      The apply-projection path: for a batch of projections (LUTs), map block codes
      -> coarse bits by gather, then bincount over the 2^(2r+1) x 2 coarse patterns.
      O(M * P), works for ANY b. This is the workhorse for the b=3 population
      search, where enumeration is impossible (2^512 projections).

Closure = (1/M) * sum_pattern max_target counts[pattern, target]  (accuracy of the
optimal local predictor of the next coarse cell). excess = closure - max(p1,1-p1).

Everything stays on the GPU when available (cuPy); falls back to NumPy otherwise.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_grain_search import sheared_blocks  # noqa: E402  (works for any b)

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

_GPU_LINALG = None


def _gpu_linalg_ok() -> bool:
    global _GPU_LINALG
    if _GPU_LINALG is None:
        try:
            float(cp.matmul(cp.ones((2, 2)), cp.ones((2, 2))).sum())
            _GPU_LINALG = True
        except Exception:
            _GPU_LINALG = False
    return bool(_GPU_LINALG)


def _xp(gpu: bool):
    return cp if (gpu and cp is not None) else np


def prep_blocks(field, b, shear, r, gpu=True, m_max=None, seed=0):
    """Transition arrays of block CODES (0..2^(b*b)-1), uncapped by default.

    Returns dict: nbrs (list of 2r+1 code arrays, len M), tgt (len M), M, n_codes,
    n_pat (=2^(b*b)), xp. Arrays live on the GPU when gpu and cuPy are available.
    """
    codes = sheared_blocks(np.asarray(field), b, shear)   # [Tc, Xc] int codes (CPU)
    Tc, Xc = codes.shape
    if Tc < 2 or Xc < 2 * r + 1:
        return None
    src, tgt = codes[:-1], codes[1:]
    nbrs = [src[:, r + dx: Xc - r + dx].ravel() for dx in range(-r, r + 1)]
    t = tgt[:, r: Xc - r].ravel()
    M = t.size
    if m_max is not None and M > m_max:
        sel = np.random.default_rng(seed).choice(M, m_max, replace=False)
        nbrs = [a[sel] for a in nbrs]
        t = t[sel]
        M = m_max
    xp = _xp(gpu)
    nbrs = [xp.asarray(a.astype(np.int64)) for a in nbrs]
    t = xp.asarray(t.astype(np.int64))
    return {"nbrs": nbrs, "tgt": t, "M": int(M), "n_pat": 1 << (b * b),
            "r": r, "xp": xp}


def _entropy(p, xp):
    p = xp.clip(p, 1e-12, 1 - 1e-12)
    return -p * xp.log2(p) - (1 - p) * xp.log2(1 - p)


def closure_batch(prep, luts, h_min=0.0, chunk=None):
    """Closure/excess for a batch of projections via apply-projection + bincount.

    luts: (P, n_pat) array of {0,1}. Returns dict of arrays (len P): closure,
    excess, p1, entropy, valid (entropy >= h_min). General b; the workhorse for
    the b=3 search. Projections failing the entropy filter get excess = -inf.
    """
    xp = prep["xp"]
    nbrs, tgt, M, r = prep["nbrs"], prep["tgt"], prep["M"], prep["r"]
    k = 2 * r + 1
    luts = xp.asarray(luts).astype(xp.int64)
    P = luts.shape[0]
    n_key = (1 << k) * 2

    closure = xp.empty(P, dtype=xp.float64)
    p1 = xp.empty(P, dtype=xp.float64)
    chunk = P if chunk is None else chunk
    for s0 in range(0, P, chunk):
        L = luts[s0:s0 + chunk]                       # (c, n_pat)
        c = L.shape[0]
        # coarse neighbour bits: gather lut value at each block code -> (c, M)
        pat = xp.zeros((c, M), dtype=xp.int64)
        for j in range(k):
            pat = pat * 2 + L[:, nbrs[j]]             # (c, M) running pattern 0..2^k-1
        tb = L[:, tgt]                                # (c, M) target bit
        key = pat * 2 + tb                            # (c, M) 0..n_key-1
        # per-projection histogram over n_key bins, then closure
        counts = xp.zeros((c, n_key), dtype=xp.float64)
        for v in range(n_key):
            counts[:, v] = (key == v).sum(axis=1)
        counts = counts.reshape(c, 1 << k, 2)
        closure[s0:s0 + c] = counts.max(axis=2).sum(axis=1) / M
        # p1 from the center neighbour's coarse density
        p1[s0:s0 + c] = L[:, nbrs[r]].mean(axis=1)
    ent = _entropy(p1, xp)
    baseline = xp.maximum(p1, 1 - p1)
    valid = ent >= h_min
    excess = xp.where(valid, closure - baseline, -xp.inf)
    return {"closure": closure, "excess": excess, "p1": p1,
            "entropy": ent, "baseline": baseline, "valid": valid}


def closure_enumerate_b2(prep, h_min):
    """Exhaustive best over all 2^16 b=2 projections via the code histogram.

    Uncapped in M (histogram build is one bincount; the contraction is
    M-independent). Returns best dict {excess, closure, baseline, entropy, pi}.
    """
    xp = prep["xp"]
    if prep["n_pat"] != 16:
        raise ValueError("closure_enumerate_b2 requires b=2 (n_pat=16)")
    nbrs, tgt, M = prep["nbrs"], prep["tgt"], prep["M"]
    n0, nc, n1 = nbrs
    key = ((n0 * 16 + nc) * 16 + n1) * 16 + tgt
    # histogram on the same device as the data
    if xp is cp:
        H = cp.bincount(key, minlength=16 ** 4).astype(cp.float64).reshape(16, 16, 16, 16)
    else:
        H = np.bincount(cp.asnumpy(key) if cp is not None and isinstance(key, cp.ndarray) else key,
                        minlength=16 ** 4).astype(np.float64).reshape(16, 16, 16, 16)
    m_center = H.sum(axis=(0, 2, 3))

    P = np.arange(1, 16 ** 4 - 1, dtype=np.int64)
    bits = ((P[:, None] >> np.arange(16)) & 1).astype(np.float64)   # (N,16) CPU
    p1 = bits @ (cp.asnumpy(m_center) if xp is cp else m_center) / M
    h = np.where((p1 > 0) & (p1 < 1),
                 -p1 * np.log2(np.clip(p1, 1e-12, 1)) - (1 - p1) * np.log2(np.clip(1 - p1, 1e-12, 1)),
                 0.0)
    mask = h >= h_min
    if not mask.any():
        return {"excess": -1.0, "closure": 0.0, "baseline": 0.0, "entropy": 0.0, "pi": None}
    Gs, Ps, p1s, hs = bits[mask], P[mask], p1[mask], h[mask]
    baseline = np.maximum(p1s, 1 - p1s)

    use_gpu = (xp is cp) and _gpu_linalg_ok()
    yp = cp if use_gpu else np
    Hx = yp.asarray(cp.asnumpy(H) if (xp is cp and not use_gpu) else H)
    closures = np.empty(Gs.shape[0], dtype=np.float64)
    step = 4096
    for s0 in range(0, Gs.shape[0], step):
        g = yp.asarray(Gs[s0:s0 + step])
        gm = yp.stack([1.0 - g, g], axis=2)
        Y = yp.einsum("ijkl,sia->sjkla", Hx, gm)
        Z = yp.einsum("sjkla,sjb->sklab", Y, gm)
        W = yp.einsum("sklab,skc->slabc", Z, gm)
        T = yp.einsum("slabc,sld->sabcd", W, gm)
        clo = T.max(axis=4).sum(axis=(1, 2, 3)) / M
        closures[s0:s0 + step] = cp.asnumpy(clo) if yp is cp else np.asarray(clo)

    excess = closures - baseline
    i = int(np.argmax(excess))
    lut = ((Ps[i] >> np.arange(16)) & 1).astype(np.int64)
    return {"excess": float(excess[i]), "closure": float(closures[i]),
            "baseline": float(baseline[i]), "entropy": float(hs[i]), "pi": lut.tolist()}


# --------------------------------------------------------------------------- #
# b>=3 population search (GPU-resident)
# --------------------------------------------------------------------------- #

def seed_projections(b, xp):
    """Structured starting projections: each single block cell, and block parity.

    These are strong priors for additive/linear rules (so the validity gate fires
    fast) and harmless for chaotic ones.
    """
    n_pat = 1 << (b * b)
    codes = np.arange(n_pat, dtype=np.int64)
    seeds = [((codes >> c) & 1) for c in range(b * b)]          # single-cell reads
    pc = np.zeros(n_pat, dtype=np.int64)                        # parity (popcount&1)
    for c in range(b * b):
        pc ^= (codes >> c) & 1
    seeds.append(pc)
    return xp.asarray(np.stack(seeds).astype(np.int64))


def search_projection(prep, budget, pop=256, elite=24, p_flip=None,
                      restart_frac=0.15, seed=0, h_min=0.85, b=None,
                      chunk=64, log_every=0):
    """(mu+lambda) evolutionary search for the closure-maximizing projection.

    Population resident on the GPU; only NEW candidates are scored each generation
    (elite scores carried), so `budget` (total scored projections) is spent well.
    Returns {best_excess, best_closure, best_pi, evals, generations, history}.
    """
    xp = prep["xp"]
    n_pat = prep["n_pat"]
    if b is None:
        b = int(round(np.log2(n_pat) ** 0.5))
    if p_flip is None:
        p_flip = max(1, n_pat // 64) / n_pat        # ~ flip a handful of bits
    rng = np.random.default_rng(seed)

    def rand_pop(n):
        return xp.asarray(rng.integers(0, 2, size=(n, n_pat), dtype=np.int64))

    # initial population: structured seeds + random fill
    seeds = seed_projections(b, xp)
    init = xp.concatenate([seeds, rand_pop(max(0, pop - seeds.shape[0]))], axis=0)[:pop]
    res = closure_batch(prep, init, h_min, chunk=chunk)
    P, exc, clo = init, res["excess"], res["closure"]
    evals = int(P.shape[0])

    def _best(exc, clo, P):
        i = int(xp.argmax(exc))
        return float(exc[i]), float(clo[i]), xp.asnumpy(P[i]) if xp is cp else np.asarray(P[i])
    bx, bc, bpi = _best(exc, clo, P)
    history = [(evals, bx, bc)]

    gen = 0
    while evals < budget:
        gen += 1
        order = xp.argsort(exc)[::-1]
        elites = P[order[:elite]]
        ex_el = exc[order[:elite]]
        cl_el = clo[order[:elite]]
        n_new = pop - elite
        n_restart = int(n_new * restart_frac)
        n_child = n_new - n_restart
        # children: pick random elite parents, flip bits
        pidx = rng.integers(0, elite, size=n_child)
        parents = elites[xp.asarray(pidx)]
        mut = (xp.asarray(rng.random((n_child, n_pat))) < p_flip).astype(xp.int64)
        children = parents ^ mut
        newcomers = xp.concatenate([children, rand_pop(n_restart)], axis=0)
        r2 = closure_batch(prep, newcomers, h_min, chunk=chunk)
        evals += int(newcomers.shape[0])
        # combine elites (carried) + newcomers, keep top `pop`
        P = xp.concatenate([elites, newcomers], axis=0)
        exc = xp.concatenate([ex_el, r2["excess"]], axis=0)
        clo = xp.concatenate([cl_el, r2["closure"]], axis=0)
        keep = xp.argsort(exc)[::-1][:pop]
        P, exc, clo = P[keep], exc[keep], clo[keep]
        cbx, cbc, cbpi = _best(exc, clo, P)
        if cbx > bx:
            bx, bc, bpi = cbx, cbc, cbpi
        if log_every and gen % log_every == 0:
            print(f"    gen {gen:4d} evals {evals:7d} best_excess {bx:+.5f} best_clo {bc:.5f}")
        history.append((evals, bx, bc))

    return {"best_excess": bx, "best_closure": bc, "best_pi": bpi.tolist(),
            "evals": evals, "generations": gen, "history": history}


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify(seed: int = 0) -> None:
    from eca_sim import simulate_spacetime_rule, GPU_AVAILABLE
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from coarse_grain_fast import enumerate_b2_fast

    def bulk(rule, n, w, sd):
        rng = np.random.default_rng(sd)
        margin = n + 32
        nc = w + 2 * margin
        row = rng.integers(0, 2, size=nc, dtype=np.uint8)
        st = simulate_spacetime_rule(row, n, rule, gpu=GPU_AVAILABLE)
        return st[:, margin:margin + w].copy()

    gpu = GPU_AVAILABLE
    for rule, shear in [(30, 0.0), (45, 0.0), (90, 0.0), (110, 0.0)]:
        field = bulk(rule, 400, 400, 7)

        # (a) uncapped enumerate matches the m_max=20000 reference path on a
        #     MATCHED subsample (same m_max, same seed) -> bit-for-bit.
        prep_cap = prep_blocks(field, 2, shear, 1, gpu=gpu, m_max=20000, seed=0)
        best_cap = closure_enumerate_b2(prep_cap, 0.85)
        ref = enumerate_b2_fast(field, shear, 1, 0.85, m_max=20000, seed=0, gpu=gpu)
        ok_a = abs(best_cap["closure"] - ref["closure"]) < 1e-12 and best_cap["pi"] == ref["pi"]

        # (b) closure_batch on the chosen pi reproduces the enumerate closure
        #     (same prep) -> the two evaluation paths agree exactly.
        lut = np.array(best_cap["pi"], dtype=np.int64)[None, :]
        cb = closure_batch(prep_cap, lut, h_min=0.0)
        cb_clo = float(cb["closure"][0]) if cp is None or not isinstance(cb["closure"], cp.ndarray) else float(cp.asnumpy(cb["closure"])[0])
        ok_b = abs(cb_clo - best_cap["closure"]) < 1e-12

        # (c) uncapped run (all transitions) — just report, not a match (different M)
        prep_full = prep_blocks(field, 2, shear, 1, gpu=gpu)
        best_full = closure_enumerate_b2(prep_full, 0.85)

        tag = "OK" if (ok_a and ok_b) else "MISMATCH"
        print(f"  rule {rule:>3} sh {shear:>3}: cap_clo {best_cap['closure']:.6f} "
              f"(==ref {ok_a}) | batch==enum {ok_b} | "
              f"M cap {prep_cap['M']}->full {prep_full['M']}  full_clo {best_full['closure']:.6f}  {tag}")
        if not (ok_a and ok_b):
            raise RuntimeError(f"closure foundation mismatch rule {rule} shear {shear}")
    print("  foundation verified: enumerate(uncapped)==ref(capped) & batch==enumerate  OK")


if __name__ == "__main__":
    print("General-b coarse-grain closure foundation")
    t0 = time.perf_counter()
    verify()
    print(f"done in {time.perf_counter()-t0:.1f}s")
