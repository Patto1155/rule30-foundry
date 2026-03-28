# Experiment O — 2D Complexity and Fractal Dimension

**date:** planned 2026-03-28
**status:** not run

## Goal

Measure the complexity of the Rule 30 *spacetime diagram* as a 2D object — not just the center column. The spacetime diagram is visually fractal and self-similar; this experiment quantifies that.

## Why This Matters

All experiments A–L treat Rule 30 as a 1D sequence generator. But Rule 30 *is* a 2D object: a binary image of shape (steps × width). The 1D center column is a single pixel-column of that image.

The 2D structure may contain:
- **Self-similarity**: the diagram looks similar at different scales (Wolfram observed this visually)
- **Fractal dimension**: a non-integer Hausdorff dimension would be a formal complexity measure
- **Anisotropy**: structure along diagonals (causal cones) that doesn't appear in rows or columns

This is a genuinely different question from "is the center column random?" — it asks about the *geometry* of the computation.

## Setup

### 2a — Box-Counting Fractal Dimension

```python
# Treat spacetime diagram as a binary image
# Count boxes of size r needed to cover the 1-cells
for r in [1, 2, 4, 8, 16, 32, 64, 128]:
    n_boxes = count_covering_boxes(spacetime, box_size=r)
    record(r, n_boxes)

# Fit: log(n_boxes) = D * log(1/r) + const → slope D is fractal dimension
# Expected: D=2 for random, D<2 for structured, D=1 for fully regular
```

### 2b — Diagonal Correlation (Causal Cone Structure)

```python
# Measure autocorrelation along diagonals (slope +1 and -1)
# These correspond to the two causal directions in Rule 30
diagonal_acf = autocorrelate_along_diagonal(spacetime, slope=+1)
```

### 2c — 2D Compression Ratio

```python
# Compress the spacetime diagram as a PNG/WebP image
# Compare to random binary image of same size
import subprocess
ratio_r30  = compressed_size(spacetime) / raw_size(spacetime)
ratio_rand = compressed_size(random_bits) / raw_size(random_bits)
```

PNG compression exploits 2D spatial structure — if Rule 30 compresses better than random, there is 2D structure even if 1D k-mer tests pass.

## What to Measure

| Metric | Description |
|--------|-------------|
| `fractal_dim` | Box-counting dimension of spacetime diagram |
| `diagonal_acf` | Autocorrelation along causal cone diagonals |
| `2d_compression_ratio` | Compressed/raw vs. random baseline |
| `isotropy` | Compare row-wise vs column-wise vs diagonal statistics |

## Parameters

- Steps: 10,000 (small enough to visualise, large enough for statistics)
- Width: 10,001
- Box sizes: powers of 2 from 1 to 128

## Script

New — `experiments/fractal_dimension.py` (to be written)

## Next Step

→ Experiment P: Invariant Measure Analysis
