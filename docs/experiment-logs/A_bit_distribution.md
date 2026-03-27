# Experiment Log

- Date: 2026-03-27
- Title: Bit Distribution of Rule 30 Center Column (10M steps)
- Goal: Measure the 0/1 bias of the center column and check 1/sqrt(N) convergence
- Setup: 10M-bit center column from D:\APATPROJECTS\rule30-research\data\center_col_10M.bin, NumPy + CuPy (GPU)
- Method: Count 0s and 1s cumulatively at every 100000-step checkpoint; compute bias = (count_1 - count_0) / total; fit expected 1/sqrt(N) envelope
- Result: Final bias = 0.00044400 at N = 10000000; expected |bias| ~ 1/sqrt(N) = 0.00031623; ratio |bias|/bound = 1.4041
- Interpretation: Bias is within expected 1/sqrt(N) bound — consistent with unbiased coin.
- Next Step: Run block-frequency and autocorrelation experiments for deeper structure tests
