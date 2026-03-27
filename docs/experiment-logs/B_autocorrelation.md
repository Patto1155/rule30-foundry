# Experiment Log

- Date: 2026-03-27
- Title: Autocorrelation Scan of Rule 30 Center Column (lags 1..100,000)
- Goal: Detect any linear dependence between the center column and lagged copies of itself
- Setup: 10M-bit center column mapped to +/-1; FFT-based autocorrelation; NumPy CPU
- Method: Compute normalised autocorrelation r(lag) = E[x_t * x_(t+lag)] / Var(x) via FFT for lags 1..100,000; rank by |r|
- Result: Top-20 lags by |autocorrelation|:
    lag=  70013  r=-0.00138490
    lag=  66314  r=-0.00132660
    lag=  19123  r=-0.00128490
    lag=  88922  r=-0.00128340
    lag=  54754  r=+0.00127860
    lag=  72458  r=+0.00126820
    lag=  73349  r=+0.00125390
    lag=  39384  r=-0.00125160
    lag=  39815  r=-0.00125030
    lag=  18250  r=-0.00123580
    lag=   7952  r=-0.00122620
    lag=  50902  r=-0.00122560
    lag=   9024  r=-0.00122200
    lag=  53464  r=+0.00121880
    lag=  81323  r=-0.00121090
    lag=  51397  r=-0.00120890
    lag=  24844  r=-0.00120300
    lag=  68241  r=-0.00120030
    lag=   3161  r=-0.00119670
    lag=  34139  r=-0.00119630
    Max |r| = 0.00138490; 2-sigma noise floor = 0.00063246
- Interpretation: All autocorrelations are within noise floor — no detectable linear structure.
- Next Step: If any lag is significant, investigate block patterns around that lag; otherwise proceed to block frequency analysis
