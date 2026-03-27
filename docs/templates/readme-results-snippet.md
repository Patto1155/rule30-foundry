## Results Snapshot (Local GPU)

This section is a public, low-detail snapshot. Full logs live under `docs/experiment-logs/`.
Large artifacts are kept local (not committed).

### GPU Benchmark (GTX 1060 6GB)

| Date | Workload | Cells | Steps | Steps/s | Gcells/s | Notes |
|---|---:|---:|---:|---:|---:|---|
| YYYY-MM-DD | Rule 30 + center col | 21,000,000 | 10,000,000 | 27,500 | 579 | `fraction_ones ~ 0.5002`, verification PASS |

Local artifacts (not in repo):

- `D:\APATPROJECTS\rule30-research\data\center_col_10M.bin` (size: 1,250,000 bytes)
- `D:\APATPROJECTS\rule30-research\data\center_col_10M_results.json` (small summary)

### Quick Visual (ASCII)

```
Rule 30 center column (first 64 bits, little-endian bit order)
11011100110001011001...
```

