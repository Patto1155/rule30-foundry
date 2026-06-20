#!/usr/bin/env python
"""Rule-parameterized elementary CA, open boundary — verified packed CPU + GPU.

Why this exists
---------------
The coarse-grain search (Experiment T) compares Rule 30's coarse closure against
an i.i.d. fair-coin null. That null is too weak: ANY locally-correlated field
beats it, so a positive excess does not distinguish "Rule 30 is approximately
reducible" from "any chaotic local rule leaks the same way." The honest control
is a SAME-LOCAL-STATISTICS null — run the identical pipeline on other chaotic
elementary rules (45, 90) and ask whether Rule 30 stands out.

To do that we need to simulate an arbitrary Wolfram rule with the same bit-packed
speed as the Rule 30 path. This module provides that WITHOUT modifying the
trusted Rule 30 reference in `rule30_open_utils` — it is a separate, independently
verified implementation, and `verify()` confirms its rule-30 output is identical
to that reference.

Arbitrary-rule bit-packed update
--------------------------------
For neighborhood (l, c, r) the new cell is bit `4l+2c+r` of the 8-bit rule number
(Wolfram convention). On bit-planes that is a sum of disjoint minterms:

    out = OR_{i: rule_i=1}  (l-plane matches i_2) & (c matches i_1) & (r matches i_0)

The eight minterms are disjoint, so OR == XOR == +; each contributes one masked
plane. Same halo-tiling, K-step fusion, and `K < 64*m` safety as the Rule 30
kernel (see docs/GPU_KERNELS.md).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule30_open_utils import (  # noqa: E402
    make_single_spike_row,
    pack_rows,
    unpack_rows,
    simulate_spacetime,
    GPU_AVAILABLE,
)

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

DEFAULT_K = 48
DEFAULT_T = 256
DEFAULT_THREADS = 128
SPACETIME_CHUNK_BYTES = 256 * 1024 * 1024

_U64 = np.uint64
_FULL = _U64(0xFFFFFFFFFFFFFFFF)
_ONE = _U64(1)
_S63 = _U64(63)


# --------------------------------------------------------------------------- #
# CPU references
# --------------------------------------------------------------------------- #

def step_naive_rule(row: np.ndarray, rule: int) -> np.ndarray:
    """Naive cell-by-cell open-boundary step for an arbitrary rule (ground truth)."""
    out = np.zeros_like(row)
    n = len(row)
    for i in range(n):
        l = int(row[i - 1]) if i > 0 else 0
        c = int(row[i])
        r = int(row[i + 1]) if i + 1 < n else 0
        out[i] = (rule >> (4 * l + 2 * c + r)) & 1
    return out


def step_packed_cpu_rule(packed_row: np.ndarray, rule: int) -> np.ndarray:
    """Bit-packed open-boundary step for an arbitrary rule, via 8 disjoint minterms."""
    cur = np.asarray(packed_row, dtype=_U64)
    left = cur << _ONE
    left[1:] |= cur[:-1] >> _S63
    right = cur >> _ONE
    right[:-1] |= cur[1:] << _S63
    center = cur
    out = np.zeros_like(cur)
    for i in range(8):
        if (rule >> i) & 1:
            term = np.full_like(cur, _FULL)
            term &= left if (i & 4) else ~left
            term &= center if (i & 2) else ~center
            term &= right if (i & 1) else ~right
            out |= term
    return out


def simulate_spacetime_rule_cpu(initial_row: np.ndarray, n_steps: int, rule: int) -> np.ndarray:
    row = np.asarray(initial_row, dtype=np.uint8)
    n_cells = len(row)
    cur = pack_rows(row)[0]
    n_words = len(cur)
    packed_rows = np.empty((n_steps, n_words), dtype=_U64)
    for step in range(n_steps):
        packed_rows[step] = cur
        cur = step_packed_cpu_rule(cur, rule)
    return unpack_rows(packed_rows, n_cells)


# --------------------------------------------------------------------------- #
# GPU fused multistep, rule-parameterized (space-time output, time-chunked)
# --------------------------------------------------------------------------- #

MULTISTEP_RULE_SRC = r"""
extern "C" __global__
void eca_multistep(
    const unsigned long long* cur_in,
    unsigned long long*       nxt_out,
    unsigned long long*       rows_out,
    int n_words, int T, int m, int n_sub, int base_step, int rule)
{
    extern __shared__ unsigned long long sh[];
    int L = T + 2 * m;
    unsigned long long* A = sh;
    unsigned long long* B = sh + L;

    int tile0 = blockIdx.x * T;
    int load0 = tile0 - m;

    for (int j = threadIdx.x; j < L; j += blockDim.x) {
        int g = load0 + j;
        A[j] = (g >= 0 && g < n_words) ? cur_in[g] : 0ULL;
    }
    __syncthreads();

    for (int s = 0; s < n_sub; ++s) {
        long long row_base = (long long)(base_step + s) * n_words;
        for (int j = threadIdx.x; j < T; j += blockDim.x) {
            int g = tile0 + j;
            if (g < n_words) rows_out[row_base + g] = A[m + j];
        }
        for (int j = threadIdx.x; j < L; j += blockDim.x) {
            unsigned long long c  = A[j];
            unsigned long long pw = (j > 0)     ? A[j - 1] : 0ULL;
            unsigned long long nw = (j < L - 1) ? A[j + 1] : 0ULL;
            unsigned long long lft = (c << 1) | (pw >> 63);
            unsigned long long rgt = (c >> 1) | (nw << 63);
            unsigned long long out = 0ULL;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if ((rule >> i) & 1) {
                    unsigned long long term = ~0ULL;
                    term &= (i & 4) ? lft : ~lft;
                    term &= (i & 2) ? c   : ~c;
                    term &= (i & 1) ? rgt : ~rgt;
                    out |= term;
                }
            }
            // Pin words beyond the true tape to 0 every step (open boundary):
            // halo words at g<0 / g>=n_words must NOT evolve, else their bits leak
            // back across the edge. Internal-tile halos (real neighbors) are kept.
            int gj = load0 + j;
            B[j] = (gj < 0 || gj >= n_words) ? 0ULL : out;
        }
        __syncthreads();
        unsigned long long* tmp = A; A = B; B = tmp;
        __syncthreads();
    }

    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        int g = tile0 + j;
        if (g < n_words) nxt_out[tile0 + j] = A[m + j];
    }
}
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(MULTISTEP_RULE_SRC, "eca_multistep")
    return _KERNEL


def simulate_spacetime_rule(
    initial_row: np.ndarray,
    n_steps: int,
    rule: int,
    gpu: bool = True,
    K: int = DEFAULT_K,
    T: int = DEFAULT_T,
    threads: int = DEFAULT_THREADS,
    chunk_bytes: int = SPACETIME_CHUNK_BYTES,
) -> np.ndarray:
    """Space-time field [n_steps, n_cells] for an arbitrary rule (times 0..n_steps-1)."""
    if not (gpu and cp is not None and GPU_AVAILABLE):
        return simulate_spacetime_rule_cpu(initial_row, n_steps, rule)

    row = np.asarray(initial_row, dtype=np.uint8)
    n_cells = len(row)
    packed = pack_rows(row)[0]
    n_words = int(packed.shape[0])
    m = (K + 63) // 64
    if K >= 64 * m:
        raise ValueError(f"unsafe K={K}: requires K < 64*m (m={m})")

    n_blocks = (n_words + T - 1) // T
    L = T + 2 * m
    shmem = 2 * L * 8
    kern = _kernel()

    rows_per_chunk = max(1, chunk_bytes // (n_words * 8))
    host_rows = np.empty((n_steps, n_words), dtype=_U64)

    cur = cp.asarray(packed)
    nxt = cp.zeros_like(cur)

    g0 = 0
    while g0 < n_steps:
        chunk = min(rows_per_chunk, n_steps - g0)
        rows_gpu = cp.empty((chunk, n_words), dtype=cp.uint64)
        base = 0
        while base < chunk:
            n_sub = min(K, chunk - base)
            kern(
                (n_blocks,), (threads,),
                (cur, nxt, rows_gpu,
                 np.int32(n_words), np.int32(T), np.int32(m), np.int32(n_sub),
                 np.int32(base), np.int32(rule)),
                shared_mem=shmem,
            )
            cur, nxt = nxt, cur
            base += n_sub
        host_rows[g0:g0 + chunk] = cp.asnumpy(rows_gpu)
        g0 += chunk

    return unpack_rows(host_rows, n_cells)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def verify(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rules = [30, 45, 90, 110, 184, 60, 150]

    # 1) packed-CPU step matches naive across a 64-bit word boundary, every rule.
    # Width is a multiple of 64 so the packed tape has NO padding cells — then the
    # packed and naive tapes have identical open boundaries (a padded width would
    # let the packed sim evolve extra edge cells the naive tape doesn't have).
    n_cells = 192  # 3 words; dynamics must cross both internal word boundaries
    for rule in rules:
        row = rng.integers(0, 2, size=n_cells, dtype=np.uint8)
        cur_p = pack_rows(row)[0]
        cur_n = row.copy()
        for step in range(120):
            got = unpack_rows(cur_p, n_cells)[0]
            if not np.array_equal(got, cur_n):
                raise RuntimeError(f"packed-CPU != naive, rule {rule} at step {step}")
            cur_p = step_packed_cpu_rule(cur_p, rule)
            cur_n = step_naive_rule(cur_n, rule)
    print(f"  verify packed-CPU == naive for rules {rules} over 120 steps  OK")

    # 2) rule-30 path reproduces the trusted reference simulate_spacetime EXACTLY.
    n_steps = 200
    n_cells = 2 * n_steps + 131
    spike = make_single_spike_row(n_cells, n_steps + 8)
    ref30 = simulate_spacetime(spike, n_steps, gpu=GPU_AVAILABLE)
    mine30_cpu = simulate_spacetime_rule(spike, n_steps, 30, gpu=False)
    if not np.array_equal(ref30, mine30_cpu):
        raise RuntimeError("rule-30 CPU path != rule30_open_utils.simulate_spacetime")
    print(f"  verify rule-30 CPU path == rule30_open_utils reference  OK")

    if GPU_AVAILABLE:
        # 3) GPU == CPU for several rules, random IC, multi-chunk forced.
        for rule in rules:
            ns = 300
            nc = 2 * ns + 131
            row = rng.integers(0, 2, size=nc, dtype=np.uint8)
            cpu = simulate_spacetime_rule(row, ns, rule, gpu=False)
            gpu = simulate_spacetime_rule(row, ns, rule, gpu=True,
                                          chunk_bytes=64 * (nc // 64 + 2) * 8)
            if not np.array_equal(cpu, gpu):
                raise RuntimeError(f"GPU != CPU for rule {rule}")
        print(f"  verify GPU == CPU for rules {rules} (multi-chunk)  OK")
    else:
        print("  GPU unavailable — skipped GPU/CPU cross-check")


if __name__ == "__main__":
    print(f"ECA rule-parameterized sim (GPU={GPU_AVAILABLE})")
    t0 = time.perf_counter()
    verify()
    print(f"all checks passed in {time.perf_counter()-t0:.1f}s")
