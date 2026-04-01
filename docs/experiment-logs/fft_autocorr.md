# Experiment Log — Full-Range FFT Autocorrelation

- Date: 2026-04-01
- Title: FFT Autocorrelation, all lags 1–23,000,000
- Goal: Definitively resolve the period-42795 candidate (windowed search z=4.66, match 52.33%)
- Data: center_col_46M.bin (46,000,000 bits)
- Method: Convert bits -> +-1; zero-pad before FFT; use irfft to recover exact linear autocovariance; divide lag k by (N-k) and lag-0 variance.
  Under H0: lag k has sigma~1/sqrt(N-k); lag-1 sigma=0.00014744.
- Result:
  - Period candidate lag 42795: r=-0.00010621, z=0.72sigma
  - Max |r| overall: 0.00105464 at lag 22,347,439 (z=5.13)
  - Lags >3sigma: 62,184 observed vs ~62,100 expected
  - Lags >5sigma: 8 observed vs ~13 expected
- Interpretation:
  Lag 42795 is NOT significant (z=0.72sigma). The windowed period search result does not survive exact linear autocorrelation.
- Elapsed: 12s
