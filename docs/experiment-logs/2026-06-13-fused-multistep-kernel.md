# Implementation Note - Fused Multi-Step Rule 30 Kernel (≈185× speedup)

- Date: 2026-06-13
- Title: Fused multi-step GPU kernel (`gpu/rule30_fast.py`)
- Goal: Remove the kernel-launch bottleneck in long single-tape simulations so the
  follow-on experiments (coarse-graining search, period/velocity sweeps) run cheaply.
- Bottleneck diagnosed: `simulate_center_columns_batch` launches **one CUDA kernel
  per step** (plus a per-step center-bit extraction). For long runs this is
  launch-latency bound — the GTX 1060 sits idle while Python dispatches millions of
  ~µs kernels. (Earlier causal sweeps launched ~1.6M kernels and ran for minutes.)
- Method: A shared-memory **halo-tiling** kernel that advances **K=48 steps per
  launch**. Each block owns T=256 packed uint64 words and loads `m=ceil(K/64)=1`
  halo word per side; it iterates K Rule-30 steps entirely in shared memory. The
  inner T words stay correct because corruption from the phantom-zero tile edges
  spreads at exactly 1 cell/step, so it cannot reach the inner region within
  K < 64·m = 64 steps. The block owning the center column emits the center bit for
  every intermediate step. Net: ~K-fold fewer launches, <1% compute redundancy.
- Verification: fused center column is **bit-identical** to the trusted single-step
  reference (`simulate_center_columns_batch`, itself checked vs naive Rule 30) for
  both a single-spike IC (778 steps) and a random IC (1012 steps).
- Result (n_steps = 30,000, single tape, GTX 1060):

  | implementation              | launches | wall time | speedup |
  |-----------------------------|----------|-----------|---------|
  | single-step (per-step)      | 30,000   | 4320.6 ms | 1×      |
  | fused K=48 multistep         | 625      | 23.4 ms   | **185×** |

  (The factor exceeds the 48× launch reduction because the old path also paid a
  per-step extraction kernel + host-side slice each step.)
- Use: `from gpu.rule30_fast import simulate_center_column_fast`. CPU fallback
  defers to the reference implementation.
- Caveat / tuning: K is capped by the 1-word halo (K<64). For longer fused
  stretches raise `m` (and shared mem) accordingly; K·blocks must keep shared mem
  ≤ 48 KB. Multi-variant batching is not yet fused (single-tape only).
