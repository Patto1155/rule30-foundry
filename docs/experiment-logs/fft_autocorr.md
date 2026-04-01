# Experiment Log — Full-Range FFT Autocorrelation

- Date: 2026-04-01
- Title: FFT Autocorrelation, all lags 1–23,000,000
- Goal: Definitively resolve the period-42795 candidate (windowed search z=4.66, match 52.33%)
- Data: center_col_46M.bin (46,000,000 bits)
- Method: Convert bits -> +-1; rfft; |F|²; irfft; normalise by lag-0 variance.
  Under H₀: each lag r ~ N(0, 1/N), 1sigma=0.00014744, 5sigma=0.00073721.
- Result:
  - Period candidate lag 42795: r=-0.00010643, z=0.72sigma
  - Max |r| overall: 0.00083261 at lag 1,199,928
  - Lags >3sigma: 62,579 observed vs ~62,100 expected
  - Lags >5sigma: 16 observed vs ~13 expected
- Interpretation:
  Lag 42795 is NOT significant (z=0.72sigma). The windowed period search result was a statistical artifact of the limited window. No periodic structure found across all 23M lags.
- Elapsed: 4s
