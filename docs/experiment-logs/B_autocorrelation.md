# Experiment Log

- Date: 2026-03-27
- Title: Autocorrelation Scan of Rule 30 Center Column (lags 1..100,000)
- Goal: Detect any linear dependence between the center column and lagged copies of itself
- Setup: 10M-bit center column mapped to +/-1; FFT-based autocorrelation; NumPy CPU
- Method: Compute normalised autocorrelation r(lag) = E[x_t * x_(t+lag)] / Var(x) via FFT for lags 1..100,000; rank by |r|
- Result: Top-20 lags by |autocorrelation|:
    lag=    284  r=+0.81973400
    lag=     14  r=+0.81970220
    lag=   1485  r=+0.81965030
    lag=    794  r=+0.81964920
    lag=    204  r=+0.81964900
    lag=    209  r=+0.81964470
    lag=    867  r=+0.81964450
    lag=    415  r=+0.81963550
    lag=    176  r=+0.81962680
    lag=    446  r=+0.81962500
    lag=    533  r=+0.81960270
    lag=    329  r=+0.81959950
    lag=    892  r=+0.81959480
    lag=    766  r=+0.81958720
    lag=    409  r=+0.81958130
    lag=    375  r=+0.81957430
    lag=    348  r=+0.81957360
    lag=    231  r=+0.81957230
    lag=     77  r=+0.81956930
    lag=   1250  r=+0.81955480
    Max |r| = 0.81973400; 2-sigma noise floor = 0.00063246
- Interpretation: Some autocorrelations EXCEED noise floor — potential linear structure detected!
- Next Step: If any lag is significant, investigate block patterns around that lag; otherwise proceed to block frequency analysis
