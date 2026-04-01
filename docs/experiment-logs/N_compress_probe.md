# Experiment Log — Compression-Based Irreducibility Probe

- Date: 2026-04-01
- Title: Multi-scale gzip/bz2 compression vs random baseline + sliding window + run-length analysis
- Goal: Test whether simple compressors and run statistics distinguish the center column from matched random baselines
- Data: center_col_46M.bin (5,750,000 bytes = 46M bits)
- Method:
  1. Compress raw bytes at scales 128B–5750KB; compare to random baseline
  2. Sliding window: 20 windows x 287,500 bytes, compare Rule 30 and fresh random windows
  3. Run-length histogram vs geometric(0.5) expected from Bernoulli process
- Results (at full 5,750,000 bytes):
  - gzip: r30=1.000308, random=1.000308, ratio=1.0
  - bz2:  r30=1.004417, random=1.004496, ratio=0.999922
  - Window gzip: rule30 mean=1.00037565, random mean=1.00037565, delta mean=+0.00000000
  - Run-length mean: 1.9997 (geometric(0.5) expects 2.0)
  - Run-length chi-square vs geometric(0.5): 1.588
- Interpretation:
  Tested dictionary compressors do not distinguish Rule 30 from matched bytewise-random baselines at the tested scales.
  Window deltas stay near zero, so there is no obvious non-stationarity by this compression metric.
  Run-length statistics are locally consistent with Bernoulli(0.5).
  This is evidence about the tested compressors and statistics only; it is not a proof about irreducibility.
- Elapsed: 9s
