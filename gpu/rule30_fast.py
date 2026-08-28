#!/usr/bin/env python
"""Fused multi-step Rule 30 kernels (shared-memory halo tiling), three output modes.

Bottleneck this fixes
---------------------
The reference `simulate_center_columns_batch` / `simulate_spacetime` (and the
causal experiments built on them) launch ONE CUDA kernel per simulated step. For
long runs that is *launch-latency bound*: the GPU sits mostly idle while Python
dispatches millions of ~microsecond kernels. (Earlier causal sweeps launched
~1.6M kernels and ran for minutes with the 1060 barely warm.)

Idea
----
Advance K steps per launch. Each thread block owns a tile of T packed uint64
words and loads a halo of `m = ceil(K/64)` words on each side into shared memory.
It iterates K Rule-30 steps entirely in shared memory (no global sync, no Python
dispatch), then writes back the T inner words — which stay correct because
corruption from the phantom-zero tile edges spreads at exactly 1 cell/step and so
cannot reach the inner region within K < 64*m steps.

Result: ~K-fold fewer kernel launches (K=48 ⇒ ~48× fewer dispatches), with a tiny
(T+2m)/T compute-redundancy overhead (<1% for T=256, m=1).

Three modes, one kernel
-----------------------
The single kernel `rule30_multistep` covers all three by two flags and a variant
dimension on `blockIdx.y`:

  * write_center=1, batched  -> simulate_center_columns_batch_fast
                                (many tapes, center column of each)
  * write_center=1, 1 tape   -> simulate_center_column_fast
                                (the original single-tape path; a batch of one)
  * write_rows=1,   1 tape   -> simulate_spacetime_fast
                                (full space-time field; VRAM-capped, time-chunked)

Whichever output is disabled is passed a 1-element dummy array so the kernel
launch signature is uniform.

Correctness is verified against the trusted single-step implementations in
`experiments/rule30_open_utils` (themselves checked against a naive Rule 30).
Center modes are big wins; the spacetime mode is a modest win — it must write
O(area) and its output is VRAM-bound, so it is time-chunked over `base_step`.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from rule30_open_utils import (  # noqa: E402
    simulate_center_columns_batch,
    simulate_spacetime,
    make_single_spike_row,
    pack_rows,
    unpack_rows,
    GPU_AVAILABLE,
)

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


# K steps per launch. m = ceil(K/64) halo words; corruption spreads 1 cell/step,
# so the inner tile stays clean as long as K < 64*m. K=48, m=1 leaves 16 cells of
# slack on a 64-cell halo — safe (the safety check is asserted at call time).
DEFAULT_K = 48
DEFAULT_T = 256        # output words per block
DEFAULT_THREADS = 128

# Soft budget (bytes) for the GPU-resident space-time output chunk. The 1060 has
# 6 GB; 256 MB per chunk leaves ample headroom for the two ping-pong tapes.
SPACETIME_CHUNK_BYTES = 256 * 1024 * 1024

MULTISTEP_SRC = r"""
extern "C" __global__
void rule30_multistep(
    const unsigned long long* cur_in,
    unsigned long long*       nxt_out,
    unsigned char*            center_out,   // [n_variants * center_stride] or dummy
    unsigned long long*       rows_out,     // [chunk_steps * n_words]      or dummy
    int n_words, unsigned long long last_word_mask, int T, int m, int n_sub,
    int base_step, int center_word, int center_bit,
    int center_stride, int write_center, int write_rows)
{
    extern __shared__ unsigned long long sh[];
    int L = T + 2 * m;
    unsigned long long* A = sh;
    unsigned long long* B = sh + L;

    int var = blockIdx.y;                       // which tape (batch dimension)
    long long tape_base = (long long)var * n_words;

    int tile0 = blockIdx.x * T;                 // first output global word
    int load0 = tile0 - m;                      // global word at shared index 0

    for (int j = threadIdx.x; j < L; j += blockDim.x) {
        int g = load0 + j;
        A[j] = (g >= 0 && g < n_words) ? cur_in[tape_base + g] : 0ULL;
    }
    __syncthreads();

    int center_idx = center_word - load0;
    bool owns_center = (write_center && center_word >= tile0 && center_word < tile0 + T);
    long long center_base = (long long)var * center_stride;

    for (int s = 0; s < n_sub; ++s) {
        // Emit current-row outputs (time = base_step + s) BEFORE stepping.
        if (owns_center && threadIdx.x == 0) {
            center_out[center_base + base_step + s] =
                (unsigned char)((A[center_idx] >> center_bit) & 1ULL);
        }
        if (write_rows) {
            long long row_base = (long long)(base_step + s) * n_words;
            for (int j = threadIdx.x; j < T; j += blockDim.x) {
                int g = tile0 + j;
                if (g < n_words) rows_out[row_base + g] = A[m + j];
            }
        }
        // One Rule-30 step in shared memory.
        for (int j = threadIdx.x; j < L; j += blockDim.x) {
            unsigned long long c  = A[j];
            unsigned long long pw = (j > 0)     ? A[j - 1] : 0ULL;
            unsigned long long nw = (j < L - 1) ? A[j + 1] : 0ULL;
            unsigned long long left  = (c << 1) | (pw >> 63);
            unsigned long long right = (c >> 1) | (nw << 63);
            // Pin words beyond the true tape to 0 every step (open boundary):
            // halo words at g<0 / g>=n_words must NOT evolve, else their bits leak
            // back across the edge. Internal-tile halos (real neighbors) are kept.
            int gj = load0 + j;
            unsigned long long value =
                (gj < 0 || gj >= n_words) ? 0ULL : (left ^ (c | right));
            B[j] = (gj == n_words - 1) ? (value & last_word_mask) : value;
        }
        __syncthreads();
        unsigned long long* tmp = A; A = B; B = tmp;
        __syncthreads();
    }

    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        int g = tile0 + j;
        if (g < n_words) nxt_out[tape_base + g] = A[m + j];
    }
}
"""

_KERNEL = None


def _last_word_mask(n_cells: int) -> np.uint64:
    used_bits = n_cells % 64
    return np.uint64((1 << used_bits) - 1 if used_bits else (1 << 64) - 1)


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(MULTISTEP_SRC, "rule30_multistep")
    return _KERNEL


def _require_gpu():
    if cp is None or not GPU_AVAILABLE:
        raise RuntimeError("fused kernel requires CuPy + a CUDA device")


@contextmanager
def _single_step_reference():
    """Temporarily disable the fused delegation in rule30_open_utils."""
    old = os.environ.get("RULE30_NO_FAST")
    os.environ["RULE30_NO_FAST"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RULE30_NO_FAST", None)
        else:
            os.environ["RULE30_NO_FAST"] = old


# --------------------------------------------------------------------------- #
# Mode 1/2 — center columns (batched and single-tape)
# --------------------------------------------------------------------------- #

def center_columns_from_packed_fast(
    packed_rows: np.ndarray,
    n_steps: int,
    center_cell: int,
    K: int = DEFAULT_K,
    T: int = DEFAULT_T,
    threads: int = DEFAULT_THREADS,
    n_cells: int | None = None,
) -> np.ndarray:
    """Center column [n_variants, n_steps+1] from already-packed tapes (GPU only).

    This is the core that the public wrappers and the open-utils delegation call.
    """
    _require_gpu()
    rows = np.asarray(packed_rows, dtype=np.uint64)
    if rows.ndim == 1:
        rows = rows[None, :]
    n_variants, n_words = rows.shape
    if n_cells is None:
        n_cells = n_words * 64
    if not 0 < n_cells <= n_words * 64:
        raise ValueError(f"n_cells={n_cells} is incompatible with {n_words} packed words")
    if not 0 <= center_cell < n_cells:
        raise ValueError(f"center_cell={center_cell} is outside n_cells={n_cells}")
    last_word_mask = _last_word_mask(n_cells)
    m = (K + 63) // 64
    if K >= 64 * m:
        raise ValueError(f"unsafe K={K}: requires K < 64*m (m={m})")
    center_word = center_cell // 64
    center_bit = center_cell % 64
    out_steps = n_steps + 1

    cur = cp.asarray(np.ascontiguousarray(rows))
    cur[:, -1] &= cp.uint64(last_word_mask)
    nxt = cp.zeros_like(cur)
    center_out = cp.zeros((n_variants, out_steps), dtype=cp.uint8)
    rows_dummy = cp.zeros(1, dtype=cp.uint64)

    n_blocks = (n_words + T - 1) // T
    L = T + 2 * m
    shmem = 2 * L * 8
    kern = _kernel()

    base = 0
    while base < n_steps:
        n_sub = min(K, n_steps - base)
        kern(
            (n_blocks, n_variants), (threads,),
            (cur, nxt, center_out, rows_dummy,
             np.int32(n_words), last_word_mask,
             np.int32(T), np.int32(m), np.int32(n_sub),
             np.int32(base), np.int32(center_word), np.int32(center_bit),
             np.int32(out_steps), np.int32(1), np.int32(0)),
            shared_mem=shmem,
        )
        cur, nxt = nxt, cur
        base += n_sub

    # Final step's center bit (time = n_steps) lives in `cur` now.
    center_out[:, n_steps] = (
        (cur[:, center_word] >> cp.uint64(center_bit)) & cp.uint64(1)
    ).astype(cp.uint8)
    return cp.asnumpy(center_out)


def simulate_center_columns_batch_fast(
    initial_rows: np.ndarray,
    n_steps: int,
    center_cell: int,
    K: int = DEFAULT_K,
    T: int = DEFAULT_T,
    threads: int = DEFAULT_THREADS,
) -> np.ndarray:
    """Center columns [n_variants, n_steps+1] for a batch of tapes."""
    if cp is None or not GPU_AVAILABLE:
        return simulate_center_columns_batch(initial_rows, n_steps, center_cell, gpu=False)
    rows = np.asarray(initial_rows, dtype=np.uint8)
    packed = pack_rows(rows)
    return center_columns_from_packed_fast(
        packed, n_steps, center_cell, K, T, threads, n_cells=rows.shape[-1]
    )


def simulate_center_column_fast(
    initial_row: np.ndarray,
    n_steps: int,
    center_cell: int,
    K: int = DEFAULT_K,
    T: int = DEFAULT_T,
    threads: int = DEFAULT_THREADS,
) -> np.ndarray:
    """Center column [n_steps+1] for one tape (a batch of one)."""
    if cp is None or not GPU_AVAILABLE:
        return simulate_center_columns_batch(initial_row, n_steps, center_cell, gpu=False)[0]
    row = np.asarray(initial_row, dtype=np.uint8)
    packed = pack_rows(row)  # (1, n_words)
    return center_columns_from_packed_fast(
        packed, n_steps, center_cell, K, T, threads, n_cells=len(row)
    )[0]


# --------------------------------------------------------------------------- #
# Mode 3 — full space-time field (single tape, time-chunked)
# --------------------------------------------------------------------------- #

def simulate_spacetime_fast(
    initial_row: np.ndarray,
    n_steps: int,
    K: int = DEFAULT_K,
    T: int = DEFAULT_T,
    threads: int = DEFAULT_THREADS,
    chunk_bytes: int = SPACETIME_CHUNK_BYTES,
) -> np.ndarray:
    """Space-time field [n_steps, n_cells] for one tape (rows at times 0..n_steps-1).

    Output is VRAM-bound, so it is produced in time-chunks: a chunk of rows is
    written on the GPU, copied to a host packed buffer, then simulation continues
    from the carried state. Matches `simulate_spacetime`'s convention exactly
    (n_steps rows, the final time = n_steps row is NOT recorded).
    """
    if cp is None or not GPU_AVAILABLE:
        return simulate_spacetime(initial_row, n_steps, gpu=False)

    row = np.asarray(initial_row, dtype=np.uint8)
    n_cells = len(row)
    packed = pack_rows(row)[0]
    n_words = int(packed.shape[0])
    last_word_mask = _last_word_mask(n_cells)
    m = (K + 63) // 64
    if K >= 64 * m:
        raise ValueError(f"unsafe K={K}: requires K < 64*m (m={m})")

    n_blocks = (n_words + T - 1) // T
    L = T + 2 * m
    shmem = 2 * L * 8
    kern = _kernel()

    rows_per_chunk = max(1, chunk_bytes // (n_words * 8))
    host_rows = np.empty((n_steps, n_words), dtype=np.uint64)

    cur = cp.asarray(packed.reshape(1, -1))
    nxt = cp.zeros_like(cur)
    center_dummy = cp.zeros(1, dtype=cp.uint8)

    g0 = 0
    while g0 < n_steps:
        chunk = min(rows_per_chunk, n_steps - g0)
        rows_gpu = cp.empty((chunk, n_words), dtype=cp.uint64)
        base = 0
        while base < chunk:
            n_sub = min(K, chunk - base)
            kern(
                (n_blocks, 1), (threads,),
                (cur, nxt, center_dummy, rows_gpu,
                 np.int32(n_words), last_word_mask,
                 np.int32(T), np.int32(m), np.int32(n_sub),
                 np.int32(base), np.int32(0), np.int32(0),
                 np.int32(1), np.int32(0), np.int32(1)),
                shared_mem=shmem,
            )
            cur, nxt = nxt, cur
            base += n_sub
        host_rows[g0:g0 + chunk] = cp.asnumpy(rows_gpu)
        g0 += chunk

    return unpack_rows(host_rows, n_cells)


# --------------------------------------------------------------------------- #
# Verification + benchmark
# --------------------------------------------------------------------------- #

def verify(seed: int = 0) -> None:
    """Byte-verify all three modes against the trusted single-step references."""
    if not GPU_AVAILABLE:
        print("GPU unavailable — skipping fused-kernel verification.")
        return
    rng = np.random.default_rng(seed)

    # Mode 2: single-tape center column (spike + random, crossing word boundaries).
    for name, ns, off in [("spike", 777, 40), ("random", 1011, 40)]:
        center = ns + off
        n_cells = 2 * center + 1
        row = (make_single_spike_row(n_cells, center) if name == "spike"
               else rng.integers(0, 2, size=n_cells, dtype=np.uint8))
        with _single_step_reference():
            ref = simulate_center_columns_batch(row, ns, center, gpu=GPU_AVAILABLE)[0]
        fast = simulate_center_column_fast(row, ns, center)
        if not np.array_equal(ref, fast):
            d = int(np.flatnonzero(ref != fast)[0])
            raise RuntimeError(f"center_column mismatch ({name}) first at step {d}")
        print(f"  verify center_column {name:6s}: fast == reference over {ns+1} steps  OK")

    # Mode 1: batched center columns (mixed spike + random tapes, same center).
    ns = 900
    center = ns + 32
    n_cells = 2 * center + 1
    batch = np.stack([
        make_single_spike_row(n_cells, center),
        rng.integers(0, 2, size=n_cells, dtype=np.uint8),
        rng.integers(0, 2, size=n_cells, dtype=np.uint8),
        make_single_spike_row(n_cells, center) ^ rng.integers(0, 2, size=n_cells, dtype=np.uint8),
    ])
    with _single_step_reference():
        ref = simulate_center_columns_batch(batch, ns, center, gpu=GPU_AVAILABLE)
    fast = simulate_center_columns_batch_fast(batch, ns, center)
    if not np.array_equal(ref, fast):
        v, d = (int(x[0]) for x in np.where(ref != fast))
        raise RuntimeError(f"batch mismatch first at variant {v}, step {d}")
    print(f"  verify center_columns_batch: fast == reference, {batch.shape[0]} tapes × {ns+1} steps  OK")

    # A non-word-aligned tape must keep padded bits pinned to zero. Otherwise
    # those bits evolve as real cells and leak back across the right boundary.
    row = np.array([1, 0], dtype=np.uint8)
    ref = simulate_center_columns_batch(row, 12, 1, gpu=False)[0]
    fast = simulate_center_column_fast(row, 12, 1)
    if not np.array_equal(ref, fast):
        d = int(np.flatnonzero(ref != fast)[0])
        raise RuntimeError(f"center_column padding-boundary mismatch first at step {d}")
    print("  verify center_column padding boundary: fast == reference  OK")

    # Mode 3: full space-time field. Use a RANDOM IC (active at the tape edges)
    # and force >1 time-chunk — a centered spike leaves the edges 0 and would hide
    # boundary-leak bugs. Test both spike and random.
    for name, ns, mk in [
        ("spike", 400, lambda nc: make_single_spike_row(nc, nc // 2)),
        ("random", 400, lambda nc: rng.integers(0, 2, size=nc, dtype=np.uint8)),
    ]:
        n_cells = 2 * ns + 131
        row = mk(n_cells)
        ref = simulate_spacetime(row, ns, gpu=GPU_AVAILABLE)
        fast = simulate_spacetime_fast(row, ns, chunk_bytes=64 * (n_cells // 64 + 2) * 8)
        if not np.array_equal(ref, fast):
            t, x = (int(a[0]) for a in np.where(ref != fast))
            raise RuntimeError(f"spacetime mismatch ({name}) first at step {t}, cell {x}")
        print(f"  verify spacetime {name:6s}: fast == reference, {ns} rows × {n_cells} cells (multi-chunk)  OK")


def benchmark(n_steps: int = 30000) -> None:
    if not GPU_AVAILABLE:
        print("GPU unavailable — skipping benchmark.")
        return
    # The reference is now wired to delegate to this fast path; force the true
    # per-step kernel for an honest baseline.
    sync = cp.cuda.Stream.null.synchronize

    def timed(fn):
        sync(); t0 = time.perf_counter(); out = fn(); sync()
        return out, time.perf_counter() - t0

    # Center column (single tape).
    center = n_steps + 64
    n_cells = 2 * center + 1
    row = make_single_spike_row(n_cells, center)
    with _single_step_reference():
        ref, t_ref = timed(lambda: simulate_center_columns_batch(row, n_steps, center, gpu=True)[0])
    fast, t_fast = timed(lambda: simulate_center_column_fast(row, n_steps, center))
    assert np.array_equal(ref, fast), "center benchmark mismatch!"
    print(f"\n  center column  n_steps={n_steps:,}")
    print(f"    single-step : {t_ref*1000:8.1f} ms  ({n_steps:,} launches)")
    print(f"    fused K={DEFAULT_K}   : {t_fast*1000:8.1f} ms  ({(n_steps+DEFAULT_K-1)//DEFAULT_K:,} launches)")
    print(f"    speedup     : {t_ref/t_fast:5.2f}x")

    # Space-time field (smaller — O(area) output).
    st = min(n_steps, 4000)
    n_cells = 2 * st + 131
    row = make_single_spike_row(n_cells, st + 8)
    ref, t_ref = timed(lambda: simulate_spacetime(row, st, gpu=True))
    fast, t_fast = timed(lambda: simulate_spacetime_fast(row, st))
    assert np.array_equal(ref, fast), "spacetime benchmark mismatch!"
    print(f"\n  space-time     n_steps={st:,}  n_cells={n_cells:,}")
    print(f"    single-step : {t_ref*1000:8.1f} ms")
    print(f"    fused K={DEFAULT_K}   : {t_fast*1000:8.1f} ms")
    print(f"    speedup     : {t_ref/t_fast:5.2f}x")


if __name__ == "__main__":
    print(f"Rule30 fused multistep kernel (GPU={GPU_AVAILABLE}, K={DEFAULT_K}, T={DEFAULT_T})")
    verify()
    benchmark(30000)
