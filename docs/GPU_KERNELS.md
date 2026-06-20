# GPU Kernels — Fused Multi-Step Rule 30

This documents the GPU simulation kernels used across the experiments, why they
are correct, and how to verify and benchmark them. **Read the safety argument
before changing `K`, `T`, or the halo logic.**

## Two kernels

| File | Kernel | Launches | Use |
|---|---|---|---|
| `experiments/rule30_open_utils.py` | `rule30_batch_step` | one per step | trusted reference; CPU twin verified against a naive cell-by-cell Rule 30 |
| `gpu/rule30_fast.py` | `rule30_multistep` | one per **K** steps | fused fast path; byte-verified against the reference |

The reference is the source of truth. The fast kernel only ever *delegates* for
it — and falls through to it on any error (see "Wiring", below), so behavior is
never worse than the reference.

## The fused idea (halo tiling)

The per-step kernel is launch-latency bound: long single-tape runs dispatch
millions of ~microsecond kernels while the 1060 sits idle. The fused kernel
advances **K steps per launch** entirely in shared memory.

Each thread block owns a tile of `T` packed `uint64` words and loads a halo of
`m = ceil(K/64)` words on each side into shared memory (`L = T + 2m` words total).
It iterates K Rule-30 steps in shared memory — no global sync, no Python
dispatch — then writes back the `T` inner words.

Compute redundancy is `(T + 2m)/T` (≈ 0.8% for `T=256, m=1`). The win is ~K-fold
fewer launches.

### Safety argument: why `K < 64·m`

Open-boundary Rule 30 has radius 1: information (and therefore any error)
propagates at **exactly one cell per step**. Inside a block, the only incorrect
inputs are the phantom zeros just beyond the loaded `L`-word window — at cell
offsets `< 0` and `≥ 64·L` within the shared buffer.

- The inner (written-back) region begins at cell offset `64·m` (after the left
  halo of `m` words = `64·m` cells).
- After `s` steps, left-edge corruption has reached cell offset `s`.
- So the inner region stays clean for every step `s` while `s < 64·m`.

Since each launch runs `n_sub ≤ K` steps and we require **`K < 64·m`**, the inner
`T` words — written back at the end *and* read out at every intermediate step in
the space-time mode — are always correct. The default `K=48, m=1` leaves 16 cells
of slack on the 64-cell halo. `center_columns_from_packed_fast` and
`simulate_spacetime_fast` both assert `K < 64·m` and raise `ValueError` otherwise.

At the true tape boundary the phantom zeros are *genuinely* correct (open
boundary = zeros beyond the tape), so the leftmost/rightmost tiles need no special
case.

## Three modes, one kernel

`rule30_multistep` covers all three via two flags and a variant dimension on
`blockIdx.y`. The disabled output is passed a 1-element dummy array so the launch
signature is uniform.

| Mode | Flags | `blockIdx.y` | Python entry point | Output shape |
|---|---|---|---|---|
| Batched center columns | `write_center=1` | one per tape | `simulate_center_columns_batch_fast(initial_rows, n_steps, center_cell)` | `[n_variants, n_steps+1]` |
| Single center column | `write_center=1` | 1 | `simulate_center_column_fast(initial_row, n_steps, center_cell)` | `[n_steps+1]` |
| Full space-time field | `write_rows=1` | 1 | `simulate_spacetime_fast(initial_row, n_steps)` | `[n_steps, n_cells]` |

Core (packed input, used by the wiring below):
`center_columns_from_packed_fast(packed_rows, n_steps, center_cell)`.

All entry points fall back to the reference CPU path when no GPU is present.

### Mode notes

- **Center modes** are the big wins: output is `O(steps)` (or `O(variants·steps)`),
  tiny relative to the work, so the launch saving dominates.
- **Space-time** is a modest win and is **VRAM-bound** — it must write `O(area)`.
  `simulate_spacetime_fast` produces the field in **time-chunks**: it fills a GPU
  chunk of rows (≤ `SPACETIME_CHUNK_BYTES`, default 256 MB), copies it to a host
  packed buffer, then continues from the carried tape state. It records `n_steps`
  rows (times `0 .. n_steps-1`) — matching `simulate_spacetime` exactly (the
  `time = n_steps` row is *not* recorded; the center modes, by contrast, record
  `n_steps+1` values including the final time).

## Wiring (zero API change)

`rule30_open_utils.simulate_center_columns_batch_from_packed` lazily imports the
fast kernel at the top of its GPU branch and delegates to
`center_columns_from_packed_fast`. On **any** exception it falls through to the
per-step reference kernel. Set `RULE30_NO_FAST=1` to force the reference path
(used for the equivalence check below). Every experiment built on
`simulate_center_columns_batch*` therefore gets the speedup with no code change.

## Verification recipe

```bash
# All three modes, byte-identical vs the reference, then a perf table:
PYTHONUTF8=1 python gpu/rule30_fast.py
```

`verify()` checks: single-tape center column (spike + random IC, crossing 64-bit
word boundaries), batched center columns (4 mixed tapes), and the full space-time
field (forced to use >1 time-chunk to exercise carry-over). Any mismatch raises
with the first differing `(variant, step)`.

End-to-end equivalence inside a real experiment:

```bash
# Same JSON output with and without the fast path:
PYTHONUTF8=1 python experiments/linear_complexity.py --test
RULE30_NO_FAST=1 PYTHONUTF8=1 python experiments/linear_complexity.py --test
```

## Performance (GTX 1060, SM 6.1, K=48, T=256)

| Workload | Reference (per-step) | Fused (K=48) | Speedup |
|---|---|---|---|
| Center column, 30,000 steps | ~4700 ms (30,000 launches) | ~24 ms (625 launches) | **~196×** |
| Space-time, 4,000 steps × 8,131 cells | ~220 ms | ~54 ms | **~4–6×** |
| `linear_complexity.py --test` (end-to-end) | 2.9 s | 0.4 s | ~7× |

Center-mode speedup grows with step count (it is launch-bound); space-time is
capped by the `O(area)` writeback and host copy. Numbers vary run-to-run; the
benchmark forces `RULE30_NO_FAST=1` for the reference baseline (the reference is
otherwise wired to delegate to the fast path). Reproduce with `benchmark()` in
`gpu/rule30_fast.py`.

## Boundary correctness (a bug worth recording)

The halo argument bounds *internal*-tile contamination, but the true tape edges
need extra care: the kernel loads phantom halo words beyond the tape (`g < 0` or
`g >= n_words`) as 0, and those words must be **re-pinned to 0 after every step**.
If they are allowed to evolve, Rule 30 fills them with nonzero bits that leak back
across the edge within ~2 steps — corrupting the edge cells. The kernels pin them
explicitly (`B[j] = (g<0||g>=n_words) ? 0 : ...`). This was caught only by a
**random-IC** space-time test; a centered-spike test leaves the edges 0 and hides
it. Lesson encoded in `verify()`: always exercise a random IC, not just a spike.
