# ML / Analysis Directions for Rule 30 — Evaluation (2026-03-27)

Evaluated 10 approaches from cryptanalysis, ML, and information theory perspectives.
Filtered by: GTX 1060 6GB, Python/numba/cupy, solo researcher, empirical focus.

## In scope this session (Experiments F, G, H)

### F — Cryptanalysis framing
Treat center column as stream cipher output. Run correlation attacks, distinguishing
attacks (is it truly random vs biased?), state reconstruction via partial leakage.
Rule 30 is literally used as a PRNG (Mathematica). Well-defined, testable, clean results.

### G — GF(2) representation search (symmetry breaking)
Search over XOR bases, linear transforms over GF(2), local Fourier transforms.
Objective: minimize entropy of transformed sequence. GPU-friendly (matrix ops over GF(2)).
Directly attacks Problem 2 — if any transform makes the sequence compressible, that's major.

### H — Compute-bounded prediction scaling laws
Meta-experiment on top of Markov predictor (Exp D). Fix horizon, increase model order
and context length, measure prediction gain vs compute. If accuracy plateaus early,
that's irreducibility evidence. The *shape* of the scaling curve is the real answer.

## Backlog (next session)
- Kolmogorov structure probing — genetic programming to find short programs reproducing prefixes
- Causal scrambling / sensitivity maps — flip bits, measure divergence, map influence cones
- Motif mining with compression feedback — extends block frequency analysis iteratively

## Skipped (wrong hardware or vague)
- Inverse CA via amortized inference (needs diffusion models, >6GB VRAM)
- Neural field / continuous relaxation (no clear falsifiable experiment)
- Adversarial game framing (complexity without clear win)
- Temporal ensemble / particle filters (compute-hungry, hard to pin down)

## Key insight
The real question is: *Is there any representation where prediction is easier than simulation?*
If even a tiny edge scales with context or generalizes across seeds, that's genuinely interesting.
If not, that's a clean "computational irreducibility" negative result.
