#!/usr/bin/env python
"""Experiment O - 2D spacetime geometry / box counting for Rule 30."""

import io
import json
import time
import zlib
import datetime
import sys
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    GPU = True
except ImportError:
    cp = None
    GPU = False

TEST = "--test" in sys.argv
N_STEPS = 512 if TEST else 8192
BOX_SIZES = [1, 2, 4, 8, 16, 32, 64] if TEST else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DIAG_LAGS = [1, 2, 4, 8, 16, 32] if TEST else [1, 2, 4, 8, 16, 32, 64, 128, 256]

OUT_JSON = Path(r"D:\APATPROJECTS\rule30-research\data\fractal_dimension.json")
LOG_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\O_2d_complexity_fractal.md")
PLOT_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\fractal_dimension.png")
PROG_LOG = Path(r"D:\APATPROJECTS\rule30-research\data\fractal_dimension.progress.log")

KERNEL_SRC = r"""
extern "C" __global__
void rule30_step(
    const unsigned long long* tape,
    unsigned long long* out,
    int n_words
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;
    unsigned long long c = tape[idx];
    unsigned long long prev = (idx > 0) ? tape[idx - 1] : 0ULL;
    unsigned long long next = (idx < n_words - 1) ? tape[idx + 1] : 0ULL;
    unsigned long long left = (c << 1) | (prev >> 63);
    unsigned long long right = (c >> 1) | (next << 63);
    out[idx] = left ^ (c | right);
}
"""


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def naive_spacetime(n_steps: int) -> np.ndarray:
    n_cells = 2 * n_steps + 1
    center = n_cells // 2
    row = np.zeros(n_cells, dtype=np.uint8)
    row[center] = 1
    out = np.empty((n_steps, n_cells), dtype=np.uint8)
    for t in range(n_steps):
        out[t] = row
        nxt = np.zeros_like(row)
        for i in range(n_cells):
            left = row[i - 1] if i > 0 else 0
            center_bit = row[i]
            right = row[i + 1] if i + 1 < n_cells else 0
            nxt[i] = left ^ (center_bit | right)
        row = nxt
    return out


def verify_kernel() -> None:
    check_steps = 96
    n_cells = 2 * check_steps + 1
    packed = simulate_spacetime(check_steps, gpu=False)
    naive = naive_spacetime(check_steps)
    if not np.array_equal(packed, naive):
        raise RuntimeError("Packed Rule 30 spacetime failed naive verification.")


def simulate_spacetime(n_steps: int, gpu: bool) -> np.ndarray:
    n_cells = 2 * n_steps + 1
    n_words = (n_cells + 63) // 64
    center = n_cells // 2
    center_word = center // 64
    center_bit = center % 64

    if gpu:
        kernel = cp.RawKernel(KERNEL_SRC, "rule30_step")
        cur = cp.zeros(n_words, dtype=cp.uint64)
        nxt = cp.zeros_like(cur)
        cur[center_word] = cp.uint64(1) << cp.uint64(center_bit)
        packed_rows = cp.empty((n_steps, n_words), dtype=cp.uint64)
        block = 256
        grid = (n_words + block - 1) // block
        for step in range(n_steps):
            packed_rows[step] = cur
            kernel((grid,), (block,), (cur, nxt, np.int32(n_words)))
            cur, nxt = nxt, cur
        packed_rows = cp.asnumpy(packed_rows)
    else:
        cur = np.zeros(n_words, dtype=np.uint64)
        nxt = np.zeros_like(cur)
        cur[center_word] = np.uint64(1) << np.uint64(center_bit)
        packed_rows = np.empty((n_steps, n_words), dtype=np.uint64)
        for step in range(n_steps):
            packed_rows[step] = cur
            left = (cur << np.uint64(1))
            left[1:] |= (cur[:-1] >> np.uint64(63))
            right = (cur >> np.uint64(1))
            right[:-1] |= (cur[1:] << np.uint64(63))
            nxt[:] = left ^ (cur | right)
            cur, nxt = nxt, cur

    bytes_view = packed_rows.view(np.uint8).reshape(n_steps, n_words * 8)
    bits = np.unpackbits(bytes_view, bitorder="little", axis=1)[:, :n_cells]
    return bits.astype(np.uint8)


def causal_mask(n_steps: int, n_cells: int) -> np.ndarray:
    center = n_cells // 2
    rows = np.arange(n_steps, dtype=np.int32)[:, None]
    cols = np.arange(n_cells, dtype=np.int32)[None, :]
    return (np.abs(cols - center) <= rows)


def box_count(arr: np.ndarray, box_size: int) -> int:
    pad_t = (-arr.shape[0]) % box_size
    pad_x = (-arr.shape[1]) % box_size
    padded = np.pad(arr.astype(bool), ((0, pad_t), (0, pad_x)), constant_values=False)
    n_t = padded.shape[0] // box_size
    n_x = padded.shape[1] // box_size
    blocks = padded.reshape(n_t, box_size, n_x, box_size)
    occupied = blocks.any(axis=(1, 3))
    return int(occupied.sum())


def centered_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt(np.mean(x * x) * np.mean(y * y))
    if denom == 0:
        return 0.0
    return float(np.mean(x * y) / denom)


def diagonal_corr(arr: np.ndarray, mask: np.ndarray, lag: int, slope: int) -> float:
    if slope > 0:
        a = arr[:-lag, :-lag]
        b = arr[lag:, lag:]
        valid = mask[:-lag, :-lag] & mask[lag:, lag:]
    else:
        a = arr[:-lag, lag:]
        b = arr[lag:, :-lag]
        valid = mask[:-lag, lag:] & mask[lag:, :-lag]
    if not np.any(valid):
        return 0.0
    return centered_corr(a[valid], b[valid])


def neighbor_anisotropy(arr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid_h = mask[:, :-1] & mask[:, 1:]
    valid_v = mask[:-1, :] & mask[1:, :]
    valid_d1 = mask[:-1, :-1] & mask[1:, 1:]
    valid_d2 = mask[:-1, 1:] & mask[1:, :-1]
    return {
        "horizontal": centered_corr(arr[:, :-1][valid_h], arr[:, 1:][valid_h]),
        "vertical": centered_corr(arr[:-1, :][valid_v], arr[1:, :][valid_v]),
        "diag_pos": centered_corr(arr[:-1, :-1][valid_d1], arr[1:, 1:][valid_d1]),
        "diag_neg": centered_corr(arr[:-1, 1:][valid_d2], arr[1:, :-1][valid_d2]),
    }


def png_size(arr: np.ndarray) -> int | None:
    try:
        from PIL import Image
    except Exception:
        return None
    image = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return len(buf.getvalue())


def raw_zlib_ratio(arr: np.ndarray) -> float:
    raw = arr.astype(np.uint8).tobytes()
    return len(zlib.compress(raw, level=9)) / len(raw)


def main() -> None:
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()
    log("=== Experiment O - 2D spacetime geometry ===")
    log(f"TEST={TEST} GPU={GPU} N_STEPS={N_STEPS:,}")
    t0 = time.perf_counter()

    verify_kernel()
    log("Naive verification passed.")

    log("Simulating Rule 30 spacetime ...")
    bits = simulate_spacetime(N_STEPS, gpu=GPU)
    n_steps, n_cells = bits.shape
    mask = causal_mask(n_steps, n_cells)
    active = bits[mask]
    p = float(active.mean())
    log(f"Spacetime ready: {n_steps:,} x {n_cells:,}; active density={p:.6f}")

    rng = np.random.default_rng(42)
    rand = np.zeros_like(bits)
    rand[mask] = rng.binomial(1, p, size=int(mask.sum())).astype(np.uint8)

    log("Computing box counts ...")
    box_rows = []
    for r in BOX_SIZES:
        n_r30 = box_count(bits & mask, r)
        n_rand = box_count(rand, r)
        box_rows.append({"box_size": r, "rule30": n_r30, "random": n_rand})
        log(f"  box={r:>3}: rule30={n_r30:,} random={n_rand:,}")

    xs = np.log([1.0 / row["box_size"] for row in box_rows])
    ys_r30 = np.log([row["rule30"] for row in box_rows])
    ys_rand = np.log([row["random"] for row in box_rows])
    dim_r30 = float(np.polyfit(xs, ys_r30, 1)[0])
    dim_rand = float(np.polyfit(xs, ys_rand, 1)[0])

    log("Computing diagonal correlations ...")
    diag_pos = []
    diag_neg = []
    for lag in DIAG_LAGS:
        c_pos = diagonal_corr(bits, mask, lag, +1)
        c_neg = diagonal_corr(bits, mask, lag, -1)
        diag_pos.append(c_pos)
        diag_neg.append(c_neg)
        log(f"  lag={lag:>3}: diag+={c_pos:+.6f} diag-={c_neg:+.6f}")

    anis = neighbor_anisotropy(bits, mask)
    log(
        "Neighbor anisotropy: "
        + ", ".join(f"{k}={v:+.6f}" for k, v in anis.items())
    )

    png_rule30 = png_size(bits & mask)
    png_rand = png_size(rand)
    z_rule30 = raw_zlib_ratio(bits & mask)
    z_rand = raw_zlib_ratio(rand)
    log(
        f"Compression: zlib rule30={z_rule30:.6f} random={z_rand:.6f}"
        + (
            f"; png rule30={png_rule30} random={png_rand}"
            if png_rule30 is not None and png_rand is not None
            else "; png unavailable"
        )
    )

    elapsed = time.perf_counter() - t0
    log(f"Done in {elapsed:.1f}s")

    result = {
        "n_steps": n_steps,
        "n_cells": n_cells,
        "active_density": round(p, 8),
        "box_counting": box_rows,
        "fractal_dimension_rule30": round(dim_r30, 6),
        "fractal_dimension_random": round(dim_rand, 6),
        "diag_lags": DIAG_LAGS,
        "diag_corr_pos": [round(v, 8) for v in diag_pos],
        "diag_corr_neg": [round(v, 8) for v in diag_neg],
        "neighbor_anisotropy": {k: round(v, 8) for k, v in anis.items()},
        "zlib_ratio_rule30": round(z_rule30, 8),
        "zlib_ratio_random": round(z_rand, 8),
        "png_bytes_rule30": png_rule30,
        "png_bytes_random": png_rand,
        "elapsed_s": round(elapsed, 2),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(f"JSON -> {OUT_JSON}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Experiment O - 2D spacetime geometry\n"
            f"Rule 30, {n_steps:,} steps, dimension={dim_r30:.4f} vs random {dim_rand:.4f}",
            fontsize=12,
            fontweight="bold",
        )

        ax = axes[0, 0]
        sample_rows = min(1024, n_steps)
        step_stride = max(1, n_steps // sample_rows)
        ax.imshow(bits[::step_stride], cmap="binary", aspect="auto", interpolation="nearest")
        ax.set_title("Rule 30 spacetime")
        ax.set_xlabel("Cell")
        ax.set_ylabel("Sampled step")

        ax = axes[0, 1]
        rs = [row["box_size"] for row in box_rows]
        ax.loglog(rs, [row["rule30"] for row in box_rows], "o-", label="Rule 30")
        ax.loglog(rs, [row["random"] for row in box_rows], "s--", label="Random baseline")
        ax.set_title("Box counting")
        ax.set_xlabel("Box size")
        ax.set_ylabel("Occupied boxes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogx(DIAG_LAGS, diag_pos, "o-", label="diag +1")
        ax.semilogx(DIAG_LAGS, diag_neg, "s--", label="diag -1")
        ax.axhline(0.0, color="black", lw=1)
        ax.set_title("Diagonal correlation")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        labels = list(anis)
        vals = [anis[k] for k in labels]
        ax.bar(labels, vals, color=["steelblue", "tomato", "orange", "green"])
        ax.axhline(0.0, color="black", lw=1)
        ax.set_title("Nearest-neighbor anisotropy")
        ax.set_ylabel("Correlation")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Plot -> {PLOT_FILE}")
    except Exception as e:
        log(f"Plot skipped: {e}")

    interpretation = []
    if abs(dim_r30 - dim_rand) < 0.05:
        interpretation.append("Box-counting dimension is close to the matched random baseline; no strong low-dimensional fractal signature.")
    else:
        interpretation.append("Box-counting dimension differs materially from the matched random baseline; this suggests nontrivial 2D geometry.")
    if max(abs(v) for v in diag_pos + diag_neg) > 0.02:
        interpretation.append("Diagonal correlations persist above a small-noise level, indicating directional 2D structure beyond the causal-cone mask.")
    else:
        interpretation.append("Diagonal correlations are weak at the tested lags once the trivial causal-cone geometry is controlled for.")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"""# Experiment Log - 2D Complexity and Fractal Dimension

- Date: {datetime.date.today().isoformat()}
- Goal: Characterise Rule 30 as a 2D spacetime object rather than a 1D center-column projection.
- Setup: N_STEPS={n_steps}, width={n_cells}, single-seed initial condition, matched random baseline inside the same causal cone, GPU={GPU}
- Method:
  - Simulate the full spacetime diagram with corrected packed-bit Rule 30 kernel
  - Verify packed simulation against a naive implementation on a small case
  - Measure box-counting occupied boxes at scales {BOX_SIZES}
  - Measure diagonal correlations at lags {DIAG_LAGS}
  - Compare zlib/PNG compression against a matched random baseline inside the same causal cone
- Result:
  - Fractal dimension (Rule 30): {dim_r30:.6f}
  - Fractal dimension (random baseline): {dim_rand:.6f}
  - zlib ratio: rule30={z_rule30:.6f}, random={z_rand:.6f}
  - Max |diag correlation|: {max(abs(v) for v in diag_pos + diag_neg):.6f}
  - Neighbor anisotropy: {", ".join(f"{k}={v:+.4f}" for k, v in anis.items())}
- Interpretation:
  - {interpretation[0]}
  - {interpretation[1]}
- Next Step: If this still looks close to the matched random baseline, move to Experiment P with random initial conditions rather than recycling the center column.
"""
        )
    log(f"Log -> {LOG_FILE}")


if __name__ == "__main__":
    main()
