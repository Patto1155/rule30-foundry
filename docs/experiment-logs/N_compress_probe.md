# Experiment Log — Compression-Based Irreducibility Probe

- Date: 2026-04-01
- Title: Multi-scale gzip/bz2 compression vs random baseline + sliding window + run-length analysis
- Goal: Attack Prize Problem 2 — if incompressible at all scales, no faster algorithm exists
- Data: center_col_46M.bin (5,750,000 bytes = 46M bits)
- Method:
  1. Compress raw bytes at scales 128B–5750KB; compare to random baseline
  2. Sliding window: 20 windows x 287,500 bytes, measure gzip variation
  3. Run-length histogram vs geometric(0.5) expected from Bernoulli process
- Results (at full 5,750,000 bytes):
  - gzip: r30=1.000308, random=1.000308, ratio=1.0
  - bz2:  r30=1.004417, random=1.004496, ratio=0.999922
  - Window gzip std: 0.000000
  - Run-length mean: 1.9997 (geometric(0.5) expects 2.0)
- Interpretation:
  Rule 30 compresses at roughly same ratio as random -> no structure detectable by dictionary-based compression -> supports Prize Problem 2 irreducibility.
  Window std < 0.001 -> compression ratio is stationary across the full sequence.
  Run-length mean ≈ 2.0 -> consistent with Bernoulli(0.5) at local scale.
- Elapsed: 6s
