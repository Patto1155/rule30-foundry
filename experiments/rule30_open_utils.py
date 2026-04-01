#!/usr/bin/env python
"""Shared open-boundary Rule 30 helpers for verified follow-up experiments."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    try:
        GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        GPU_AVAILABLE = False
except ImportError:
    cp = None
    GPU_AVAILABLE = False


BATCH_KERNEL_SRC = r"""
extern "C" __global__
void rule30_batch_step(
    const unsigned long long* tapes,
    unsigned long long* out,
    int n_variants,
    int n_words
) {
    int var = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (var >= n_variants || idx >= n_words) return;

    int base = var * n_words;
    unsigned long long c = tapes[base + idx];
    unsigned long long prev = (idx > 0) ? tapes[base + idx - 1] : 0ULL;
    unsigned long long next = (idx < n_words - 1) ? tapes[base + idx + 1] : 0ULL;
    unsigned long long left = (c << 1) | (prev >> 63);
    unsigned long long right = (c >> 1) | (next << 63);
    out[base + idx] = left ^ (c | right);
}
"""


def make_single_spike_row(n_cells: int, center_cell: int | None = None) -> np.ndarray:
    center = n_cells // 2 if center_cell is None else center_cell
    row = np.zeros(n_cells, dtype=np.uint8)
    row[center] = 1
    return row


def pack_rows(rows: np.ndarray) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_variants, n_cells = arr.shape
    pad_bits = (-n_cells) % 64
    if pad_bits:
        arr = np.pad(arr, ((0, 0), (0, pad_bits)), constant_values=0)
    arr = np.ascontiguousarray(arr)
    packed_bytes = np.packbits(arr, bitorder="little", axis=1)
    return packed_bytes.view(np.uint64).reshape(n_variants, -1)


def unpack_rows(packed_rows: np.ndarray, n_cells: int) -> np.ndarray:
    arr = np.asarray(packed_rows, dtype=np.uint64)
    if arr.ndim == 1:
        arr = arr[None, :]
    bytes_view = arr.view(np.uint8).reshape(arr.shape[0], arr.shape[1] * 8)
    return np.unpackbits(bytes_view, bitorder="little", axis=1)[:, :n_cells].astype(np.uint8)


def step_naive_open(row: np.ndarray) -> np.ndarray:
    out = np.zeros_like(row)
    for i in range(len(row)):
        left = row[i - 1] if i > 0 else 0
        center = row[i]
        right = row[i + 1] if i + 1 < len(row) else 0
        out[i] = left ^ (center | right)
    return out


def step_packed_cpu_open(packed_row: np.ndarray) -> np.ndarray:
    cur = np.asarray(packed_row, dtype=np.uint64)
    left = cur << np.uint64(1)
    left[1:] |= cur[:-1] >> np.uint64(63)
    right = cur >> np.uint64(1)
    right[:-1] |= cur[1:] << np.uint64(63)
    return left ^ (cur | right)


def simulate_naive_center_columns(initial_rows: np.ndarray, n_steps: int, center_cell: int) -> np.ndarray:
    rows = np.asarray(initial_rows, dtype=np.uint8)
    if rows.ndim == 1:
        rows = rows[None, :]
    cols = np.empty((rows.shape[0], n_steps + 1), dtype=np.uint8)
    for idx, row0 in enumerate(rows):
        row = row0.copy()
        for step in range(n_steps + 1):
            cols[idx, step] = row[center_cell]
            if step < n_steps:
                row = step_naive_open(row)
    return cols


def simulate_center_columns_batch(
    initial_rows: np.ndarray,
    n_steps: int,
    center_cell: int,
    gpu: bool = True,
) -> np.ndarray:
    packed = pack_rows(initial_rows)
    return simulate_center_columns_batch_from_packed(packed, n_steps, center_cell, gpu=gpu)


def simulate_center_columns_batch_from_packed(
    packed_rows: np.ndarray,
    n_steps: int,
    center_cell: int,
    gpu: bool = True,
) -> np.ndarray:
    rows = np.asarray(packed_rows, dtype=np.uint64)
    if rows.ndim == 1:
        rows = rows[None, :]
    n_variants, n_words = rows.shape
    center_word = center_cell // 64
    center_bit = center_cell % 64
    out_steps = n_steps + 1

    if gpu and GPU_AVAILABLE:
        kernel = cp.RawKernel(BATCH_KERNEL_SRC, "rule30_batch_step")
        cur = cp.asarray(rows)
        nxt = cp.zeros_like(cur)
        center_cols = cp.empty((n_variants, out_steps), dtype=cp.uint8)
        threads = 128
        blocks_x = (n_words + threads - 1) // threads
        blocks_y = n_variants
        for step in range(out_steps):
            center_cols[:, step] = (
                (cur[:, center_word] >> cp.uint64(center_bit)) & cp.uint64(1)
            ).astype(cp.uint8)
            if step < n_steps:
                kernel(
                    (blocks_x, blocks_y),
                    (threads,),
                    (cur, nxt, np.int32(n_variants), np.int32(n_words)),
                )
                cur, nxt = nxt, cur
        return cp.asnumpy(center_cols)

    cur = rows.copy()
    nxt = np.zeros_like(cur)
    center_cols = np.empty((n_variants, out_steps), dtype=np.uint8)
    for step in range(out_steps):
        center_cols[:, step] = ((cur[:, center_word] >> np.uint64(center_bit)) & np.uint64(1)).astype(np.uint8)
        if step < n_steps:
            for idx in range(n_variants):
                nxt[idx] = step_packed_cpu_open(cur[idx])
            cur, nxt = nxt, cur
    return center_cols


def simulate_spacetime(initial_row: np.ndarray, n_steps: int, gpu: bool = True) -> np.ndarray:
    row = np.asarray(initial_row, dtype=np.uint8)
    n_cells = len(row)
    packed = pack_rows(row)[0]
    n_words = len(packed)

    if gpu and GPU_AVAILABLE:
        kernel = cp.RawKernel(BATCH_KERNEL_SRC, "rule30_batch_step")
        cur = cp.asarray(packed.reshape(1, -1))
        nxt = cp.zeros_like(cur)
        packed_rows = cp.empty((n_steps, n_words), dtype=cp.uint64)
        threads = 128
        blocks_x = (n_words + threads - 1) // threads
        for step in range(n_steps):
            packed_rows[step] = cur[0]
            kernel((blocks_x, 1), (threads,), (cur, nxt, np.int32(1), np.int32(n_words)))
            cur, nxt = nxt, cur
        return unpack_rows(cp.asnumpy(packed_rows), n_cells)

    cur = packed.copy()
    packed_rows = np.empty((n_steps, n_words), dtype=np.uint64)
    for step in range(n_steps):
        packed_rows[step] = cur
        cur = step_packed_cpu_open(cur)
    return unpack_rows(packed_rows, n_cells)


def first_divergence_steps(reference_cols: np.ndarray, variant_cols: np.ndarray, censored_value: int) -> np.ndarray:
    ref = np.asarray(reference_cols, dtype=np.uint8)
    vars_ = np.asarray(variant_cols, dtype=np.uint8)
    if vars_.ndim == 1:
        vars_ = vars_[None, :]
    out = np.full(vars_.shape[0], censored_value, dtype=np.int32)
    for idx in range(vars_.shape[0]):
        diff = np.flatnonzero(vars_[idx] != ref)
        if diff.size:
            out[idx] = int(diff[0])
    return out


def verify_single_spike_direction_and_boundary() -> None:
    n_steps = 96
    n_cells = 2 * n_steps + 131
    center = 96
    base = make_single_spike_row(n_cells, center)
    ref_packed = simulate_center_columns_batch(base, n_steps, center, gpu=False)[0]
    ref_naive = simulate_naive_center_columns(base, n_steps, center)[0]
    if not np.array_equal(ref_packed, ref_naive):
        raise RuntimeError("Packed open-boundary kernel failed center-column verification.")

    for side_name, side_sign in [("left", -1), ("right", 1)]:
        for dist in [1, 2, 3, 31, 63, 64, 65, 79]:
            row = base.copy()
            row[center + side_sign * dist] ^= 1
            packed = simulate_center_columns_batch(row, n_steps, center, gpu=False)[0]
            naive = simulate_naive_center_columns(row, n_steps, center)[0]
            if not np.array_equal(packed, naive):
                raise RuntimeError(
                    f"Packed open-boundary kernel failed perturbation verification for {side_name} d={dist}."
                )


def verify_random_batch_against_naive(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n_cells = 193
    center = n_cells // 2
    rows = rng.integers(0, 2, size=(4, n_cells), dtype=np.uint8)
    rows[0] = make_single_spike_row(n_cells, center)
    packed = simulate_center_columns_batch(rows, 64, center, gpu=False)
    naive = simulate_naive_center_columns(rows, 64, center)
    if not np.array_equal(packed, naive):
        raise RuntimeError("Packed open-boundary kernel failed random-row verification.")


def verify_spacetime_against_naive() -> None:
    n_steps = 80
    n_cells = 223
    center = 111
    row = make_single_spike_row(n_cells, center)
    packed = simulate_spacetime(row, n_steps, gpu=False)
    naive_rows = np.empty((n_steps, n_cells), dtype=np.uint8)
    cur = row.copy()
    for step in range(n_steps):
        naive_rows[step] = cur
        cur = step_naive_open(cur)
    if not np.array_equal(packed, naive_rows):
        raise RuntimeError("Packed spacetime simulation failed naive verification.")
