# Experiment Log

- Date: 2026-03-27
- Title: GPU Baseline — Rule 30 Simulation Performance on GTX 1060
- Goal: Establish what the GTX 1060 6GB can do for raw Rule 30 simulation
- Setup: NVIDIA GeForce GTX 1060 6GB (SM 6.1, Pascal), CuPy RawKernel, uint64 bit-packing
- Method: Simulate Rule 30 on 1M-cell tape with bit-packed uint64 words (15,625 words). CUDA kernel handles step + center column extraction via atomicOr. Double-buffered tape, 256 threads/block.
- Result:
  - **Short validation (1M cells, 10K steps, no center extraction):** 5,113 steps/s = 5.1 Gcells/s, 2.0s total
  - **Short validation (1M cells, 1K steps, with center extraction v1):** 2,506 steps/s = 2.5 Gcells/s (per-step GPU→CPU sync bottleneck)
  - **Optimized (1M cells, 10M steps, GPU-side center extraction):** 83,170 steps/s = 83.2 Gcells/s, 120s total
  - **Center column verification:** PASS (first 20 bits match CPU reference)
  - VRAM used: 1.4 MB (tape is tiny — GPU is massively underutilized for compute density)
- Interpretation: The GTX 1060 can simulate 1M-cell Rule 30 at 83k steps/sec when center extraction is done on-GPU. The bottleneck is kernel launch overhead, not compute — each kernel processes only 15K uint64 words. For larger tapes, throughput would be higher per step but steps/sec would drop. 10M center column steps complete in 2 minutes.
- Next Step: Run full experiment suite (A-H) on the 10M center column data. Consider 100M-cell tape for Phase 2b full benchmark when time permits.
