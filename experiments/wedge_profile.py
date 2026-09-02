"""Structured/chaotic decomposition of the single-seed Rule 30 light cone.

Prize object: the orbit of the single black cell. No ensemble, no perturbation.

Left diagonal d is  D_d(t) = a(t, -t+d), the cell at fixed offset d inward from
the LEFT edge of the light cone. Each diagonal begins in the chaotic region near
the centre and later *settles* into eventual periodicity with a small period.

    settle(d) = last t at which D_d(t) != D_d(t-p), minimised over candidate p

Measurements emitted:

  1. settling law        settle(d) ~ s*d              (linear fit + subrange stability)
  2. window prediction   first-failure depth on [aT,T] should be ~ a*T/s
  3. period-cap check    answer must be invariant to the candidate-period cap
  4. entropy split       pre-settle vs post-settle compressibility, vs a random control

Derived: structured wedge area fraction 1/(2s); wedge inner-boundary speed 1-1/s.

Rule 30, bit index increasing to the right:
    a(t+1,i) = a(t,i-1) XOR ( a(t,i) OR a(t,i+1) )

Usage:
    python experiments/wedge_profile.py --steps 65536 --diagonals 49152 --pretty
    python experiments/wedge_profile.py --steps 8192 --diagonals 6144   # quick

Memory is about steps * diagonals / 8 bytes (65536 x 49152 ~ 400 MB).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zlib
from collections import Counter

import numpy as np

ARTIFACT_VERSION = 1
DEFAULT_PERIODS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
OEIS_A051023_PREFIX = "1101110011000101"


def simulate_left_window(steps: int, diagonals: int, progress: bool = False) -> np.ndarray:
    """Bit-packed (steps+1, diagonals/8) view of the left edge of the cone.

    Row t holds diagonals d = 0..diagonals-1, i.e. cells at positions -t+d,
    little-endian so that bit d of row t is D_d(t).
    """
    if diagonals % 8:
        raise ValueError("diagonals must be a multiple of 8")
    wb = diagonals // 8
    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    wmask = (1 << diagonals) - 1

    buf = np.empty((steps + 1, wb), dtype=np.uint8)
    state = 1 << centre
    for t in range(steps + 1):
        buf[t] = np.frombuffer(
            ((state >> (centre - t)) & wmask).to_bytes(wb, "little"), dtype=np.uint8
        )
        state = ((state << 1) ^ (state | (state >> 1))) & mask
        if progress and t % 8192 == 0:
            sys.stderr.write(f"\r  sim {t}/{steps}")
            sys.stderr.flush()
    if progress:
        sys.stderr.write("\r" + " " * 40 + "\r")
    return buf


def centre_column(steps: int) -> list[int]:
    """Exact centre column, for the ground-truth gate."""
    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    state = 1 << centre
    out = []
    for _ in range(steps + 1):
        out.append((state >> centre) & 1)
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return out


def sanity_gate(buf: np.ndarray, steps: int) -> dict:
    """Hard failure checks before any measurement is trusted.

    - centre column must match OEIS A051023
    - diagonals 0 and 1 are provably identically 1 for t >= 1 (leftmost two
      live cells of the cone); if not, the extraction indexing is wrong
    """
    cc = "".join(str(b) for b in centre_column(min(steps, 64))[:16])
    if cc != OEIS_A051023_PREFIX:
        raise RuntimeError(f"centre column {cc!r} != OEIS A051023 {OEIS_A051023_PREFIX!r}")
    d0 = np.unpackbits(buf[1:, :1], axis=1, bitorder="little")[:, 0]
    d1 = np.unpackbits(buf[1:, :1], axis=1, bitorder="little")[:, 1]
    if not (d0.all() and d1.all()):
        raise RuntimeError("diagonals 0/1 are not identically 1 - indexing is wrong")
    return {"centre_prefix": cc, "oeis": "A051023", "diag0_diag1_all_ones": True, "ok": True}


def settle_times(buf: np.ndarray, diagonals: int, periods=DEFAULT_PERIODS,
                 block: int = 2048, progress: bool = False) -> np.ndarray:
    """settle(d) for every diagonal, vectorised in blocks of columns."""
    out = np.zeros(diagonals, dtype=np.int64)
    n = buf.shape[0]
    for b0 in range(0, diagonals, block):
        b1 = min(b0 + block, diagonals)
        sub = np.unpackbits(buf[:, b0 // 8:b1 // 8], axis=1, bitorder="little")
        best = np.full(sub.shape[1], n, dtype=np.int64)
        for p in periods:
            if n <= 2 * p:
                continue
            v = sub[p:] != sub[:-p]
            rev = v[::-1]
            has = rev.any(axis=0)
            last = np.where(has, (len(v) - 1 - rev.argmax(axis=0)) + p, 0)
            best = np.minimum(best, last)
        out[b0:b1] = best
        if progress:
            sys.stderr.write(f"\r  settle {b1}/{diagonals}")
            sys.stderr.flush()
    if progress:
        sys.stderr.write("\r" + " " * 40 + "\r")
    return out


def fit_slope(settle: np.ndarray, steps: int) -> dict:
    d = np.arange(len(settle))
    m = (settle > 200) & (settle < steps * 0.92) & (d > 100)
    if m.sum() < 100:
        raise RuntimeError("too few diagonals settle inside the run; raise --steps")
    slope, intercept = np.polyfit(d[m], settle[m], 1)
    sub = []
    w = len(settle)
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        mm = m & (d >= lo * w) & (d < hi * w)
        if mm.sum() > 100:
            sub.append({
                "range": [round(lo, 2), round(hi, 2)],
                "n": int(mm.sum()),
                "slope": round(float(np.polyfit(d[mm], settle[mm], 1)[0]), 5),
            })
    slopes = [s["slope"] for s in sub]
    return {
        "n_fit_points": int(m.sum()),
        "slope": round(float(slope), 5),
        "intercept": round(float(intercept), 2),
        "subrange_fits": sub,
        "subrange_spread": round(max(slopes) - min(slopes), 5) if slopes else None,
        "wedge_area_fraction": round(1.0 / (2.0 * slope), 5),
        "wedge_boundary_speed": round(1.0 - 1.0 / slope, 5),
        "max_speedup_factor": round(1.0 / (1.0 - 1.0 / (2.0 * slope)), 4),
    }


def first_failure_depth(buf, diagonals, lo, hi, periods, block=2048) -> int:
    """Smallest d that is not periodic on the window [lo, hi]. -1 if none found."""
    b1 = 0
    for b0 in range(0, diagonals, block):
        b1 = min(b0 + block, diagonals)
        sub = np.unpackbits(buf[lo:hi + 1, b0 // 8:b1 // 8], axis=1, bitorder="little")
        good = np.zeros(sub.shape[1], dtype=bool)
        for p in periods:
            if sub.shape[0] > 2 * p:
                good |= (sub[:-p] == sub[p:]).all(axis=0)
        if not good.all():
            return b0 + int(np.flatnonzero(~good)[0])
    return -1


def window_predictions(buf, diagonals, steps, slope, periods=DEFAULT_PERIODS) -> list[dict]:
    out = []
    for alpha in (0.25, 0.5, 0.75, 0.90, 0.95):
        lo = int(alpha * steps)
        d = first_failure_depth(buf, diagonals, lo, steps, periods)
        pred = alpha * steps / slope
        out.append({
            "alpha": alpha,
            "measured_depth": d,
            "predicted_depth": round(pred, 1),
            "ratio": round(d / pred, 4) if d > 0 else None,
        })
    return out


def period_cap_check(buf, diagonals, steps, caps=(16, 64, 256, 1024)) -> list[dict]:
    lo = steps // 2
    return [
        {"cap": c,
         "depth": first_failure_depth(buf, diagonals, lo, steps,
                                      tuple(p for p in DEFAULT_PERIODS + (96, 128, 192, 256, 384, 512, 768, 1024)
                                            if p <= c))}
        for c in caps
    ]


def block_entropy_rate(bits: np.ndarray, k: int = 6) -> tuple[float, int]:
    n = len(bits)
    if n < 40 * (2 ** k):
        k = max(2, int(math.log2(max(n / 40, 4))))
    cnt: Counter = Counter()
    packed, m = 0, (1 << k) - 1
    for i, b in enumerate(bits.tolist()):
        packed = ((packed << 1) | b) & m
        if i >= k - 1:
            cnt[packed] += 1
    tot = sum(cnt.values())
    h = -sum((c / tot) * math.log2(c / tot) for c in cnt.values())
    return h / k, k


def zlib_ratio(bits: np.ndarray) -> float:
    # bitorder-exempt: re-packs an in-memory bit array into bytes for zlib;
    # any consistent convention gives the same ratio.
    raw = np.packbits(bits.astype(np.uint8)).tobytes()
    return len(zlib.compress(raw, 9)) / max(1, len(raw))


def _column(buf: np.ndarray, d: int) -> np.ndarray:
    """Bits of diagonal d straight out of the packed buffer.

    Unpacking the whole buffer would be steps*diagonals bytes (3.2 GB at
    T=65536, W=49152); extract the single byte-column and mask instead.
    """
    return (buf[:, d >> 3] >> (d & 7)) & np.uint8(1)


def entropy_split(buf, settle, steps, diagonals, n_samples=16, seed=30) -> dict:
    lo, hi = int(0.05 * diagonals), int(0.90 * diagonals)
    rows = []
    for d in np.linspace(lo, hi, n_samples, dtype=int):
        sd = int(settle[d])
        col = _column(buf, int(d))
        pre, post = col[d:sd], col[sd:steps + 1]
        if len(pre) < 512 or len(post) < 512:
            continue
        hpre, kpre = block_entropy_rate(pre)
        hpost, kpost = block_entropy_rate(post)
        rows.append({
            "d": int(d), "settle": sd, "settle_over_d": round(sd / d, 4),
            "pre_len": int(len(pre)), "pre_block_entropy_rate": round(hpre, 5),
            "pre_block_k": kpre, "pre_zlib_ratio": round(zlib_ratio(pre), 5),
            "post_len": int(len(post)), "post_block_entropy_rate": round(hpost, 5),
            "post_zlib_ratio": round(zlib_ratio(post), 5),
        })
    rng = np.random.default_rng(seed)
    ctl = rng.integers(0, 2, 20000).astype(np.uint8)
    hctl, kctl = block_entropy_rate(ctl)
    return {
        "samples": rows,
        "random_control": {
            "block_entropy_rate": round(hctl, 5), "block_k": kctl,
            "zlib_ratio": round(zlib_ratio(ctl), 5), "seed": seed,
        },
        "summary": {
            "pre_entropy_min": round(min(r["pre_block_entropy_rate"] for r in rows), 5),
            "pre_zlib_min": round(min(r["pre_zlib_ratio"] for r in rows), 5),
            "post_zlib_max": round(max(r["post_zlib_ratio"] for r in rows), 5),
        } if rows else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=65536)
    ap.add_argument("--diagonals", type=int, default=49152,
                    help="left diagonals tracked; ~0.75*steps is a good default")
    ap.add_argument("--entropy-samples", type=int, default=16)
    ap.add_argument("--out", default=None, help="write JSON artifact here")
    ap.add_argument("--pretty", action="store_true", help="human tables on stderr")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    prog = (not args.no_progress) and sys.stderr.isatty()
    t0 = time.perf_counter()
    buf = simulate_left_window(args.steps, args.diagonals, progress=prog)
    gate = sanity_gate(buf, args.steps)
    settle = settle_times(buf, args.diagonals, progress=prog)
    fit = fit_slope(settle, args.steps)
    wins = window_predictions(buf, args.diagonals, args.steps, fit["slope"])
    caps = period_cap_check(buf, args.diagonals, args.steps)
    ent = entropy_split(buf, settle, args.steps, args.diagonals, args.entropy_samples)

    out = {
        "artifact_type": "rule30.wedge_profile",
        "artifact_version": ARTIFACT_VERSION,
        "params": {"steps": args.steps, "diagonals": args.diagonals,
                   "periods": list(DEFAULT_PERIODS)},
        "sanity_gate": gate,
        "settling_fit": fit,
        "window_predictions": wins,
        "period_cap_check": caps,
        "entropy_split": ent,
        "settle_sha256": hashlib.sha256(settle.tobytes()).hexdigest(),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }

    if args.pretty:
        w = sys.stderr
        w.write(f"\nsanity gate: centre prefix {gate['centre_prefix']} == OEIS A051023 OK\n")
        w.write(f"\nsettling law  settle(d) = {fit['slope']:.5f} * d + {fit['intercept']:.1f}"
                f"   (n={fit['n_fit_points']}, subrange spread {fit['subrange_spread']})\n")
        w.write(f"  wedge area fraction   {fit['wedge_area_fraction']:.4f}\n")
        w.write(f"  boundary speed        {fit['wedge_boundary_speed']:.4f}\n")
        w.write(f"  max speedup factor    {fit['max_speedup_factor']:.3f}x\n")
        w.write("\nwindow prediction check\n")
        for r in wins:
            w.write(f"  alpha={r['alpha']:.2f}  measured {r['measured_depth']:>7}"
                    f"  predicted {r['predicted_depth']:>9.0f}  ratio {r['ratio']}\n")
        w.write("\nperiod-cap invariance: "
                + ", ".join(f"cap{c['cap']}->{c['depth']}" for c in caps) + "\n")
        if ent["summary"]:
            s = ent["summary"]
            c = ent["random_control"]
            w.write(f"\nentropy split (pre-settle vs post-settle)\n"
                    f"  pre  : min block-entropy rate {s['pre_entropy_min']:.4f}, "
                    f"min zlib ratio {s['pre_zlib_min']:.4f}\n"
                    f"  post : max zlib ratio {s['post_zlib_max']:.4f}\n"
                    f"  random control: entropy {c['block_entropy_rate']:.4f}, "
                    f"zlib {c['zlib_ratio']:.4f}\n")
        w.write(f"\nelapsed {out['elapsed_s']}s\n\n")
        w.flush()

    if args.out:
        from pathlib import Path
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
