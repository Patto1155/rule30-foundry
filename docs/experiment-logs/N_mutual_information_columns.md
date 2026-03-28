# Experiment N — Mutual Information Between Columns

**date:** planned 2026-03-28
**status:** not run

## Goal

Measure mutual information (MI) between pairs of columns separated by distance d. All previous experiments (A–L) analysed the center column in isolation — a 1D projection of a 2D spacetime diagram. This experiment examines cross-column structure.

## Why This Matters

If Rule 30 is truly mixing in the ergodic sense, then:
- MI(column_i, column_j) → 0 as |i - j| → ∞
- Columns become statistically independent at large separations

If MI decays *slowly* (power law rather than exponential), that implies long-range spatial correlations — structure that is completely invisible to any single-column analysis. This would be a significant finding: the sequence *looks* random marginally, but has multi-column dependencies.

## Setup

```python
# For separation d, extract columns i and i+d from the spacetime diagram
# Estimate MI using k-NN estimator or binned histogram

for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    col_a = spacetime[:, center]
    col_b = spacetime[:, center + d]
    mi = mutual_information(col_a, col_b)  # bits
    mi_random_baseline = mutual_information(shuffle(col_a), col_b)
    record(d, mi, mi_random_baseline)
```

**GPU approach:** Store a strip of columns (e.g., center ± 1024) in VRAM during simulation. At 1 bit/cell, 1M steps × 2048 columns = 256 MB — fits in the 6GB budget.

**MI estimator:** Use binned histogram (fast, GPU-amenable) for binary sequences: MI = H(X) + H(Y) - H(X,Y) where H is Shannon entropy. For binary columns this is exact, no approximation needed.

## What to Measure

| Metric | Description |
|--------|-------------|
| `mi(d)` | Mutual information at separation d (bits) |
| `decay_rate` | Fit exponential or power law to mi(d) curve |
| `independence_threshold` | Smallest d where mi(d) ≈ mi_baseline |

## Theoretical Prediction

- Exponential decay: consistent with strong mixing, supports irreducibility
- Power-law decay: consistent with long-range correlations, potential structure
- No decay: columns are not independent → very significant finding

## Parameters

- Steps: 10,000,000 (10M sufficient for stable MI estimates)
- Column strip: center ± 1024
- Separations d: logarithmically spaced 1 → 1024

## Script

New — `experiments/mutual_information_columns.py` (to be written)

## Next Step

→ Experiment O: 2D Complexity and Fractal Dimension
