# 2026-11-23 — the exact period search tested pure periodicity only

## Goal

Audit the Certificate "No period `p <= 5,000,000` exists in the first 10M
center-column bits" against what Prize 1 actually asks: *eventual*
periodicity, i.e. `S[i] == S[i+p]` for all `i >= k`, for some preperiod `k`.

## Defect

`experiments/period_search_exact.py::find_periods` keeps a candidate `p` only
if the 64-bit window at position `p` equals the window at position **0**, and
confirms it only if `S[i] == S[i+p]` for **every** `i` from 0. That decides
pure periodicity (`k = 0`). The 2026-08-30 log argued it also covers eventual
periodicity because "any `p` that is eventually a period would show as a long
— though not full-length — overlap, and the longest overlap found anywhere is
26 bits". That is false: `Z[p]` is the common prefix *from position 0*, so a
transient of length `k` caps it below `k`.

Reproduced, not argued: a 2M-bit stream that is exactly periodic with `p=1000`
after a 7-bit transient gives the old method **0** confirmed periods and a
longest self-overlap of **0** bits. Even a 1-bit transient defeats it. So the
certificate as worded excluded only `k = 0`, a sliver of the Prize 1 class, and
its stated defence of the wider reading would not have caught a real hit.

## Fix

`find_eventual_periods` anchors at the **end** of the prefix. With `T[p]` the
number of trailing positions satisfying `S[i] == S[i+p]`, the minimal
preperiod consistent with the prefix is `k(p) = n - p - T[p]`. Reversing the
stream makes `T` the forward overlap of the reversed stream, so the existing
window code applies unchanged. The gate now plants an eventual period
(`k <= 7`, `p = 1000`) and requires the tail test to recover it with its
preperiod, and records that the position-0 test misses it.
`tests/test_period_search_exact.py` checks the recovered preperiod against a
naive scan for `k = 0, 1, 5, 40, 300` and the tail maximum against brute force.

## Result (canonical 10M, restored and verified against the golden file)

`python experiments/period_search_exact.py --out data/prize/2026-11-23-period-search-eventual.json`

- Pure periods: unchanged, 0 survivors of a 64-bit window.
- Eventual periods, any preperiod: **0** survivors of a 64-bit trailing
  window over all `p <= n - 64`.
- Longest tail agreement `T_max = 23` bits (shift 3,751,918), exact for every
  `p <= n - 16` (16-bit narrow scan, 143 candidates, not truncated). The i.i.d.
  expectation is `~log2 n = 23.3`.

Hence: **the first 10^7 bits admit no eventual period `(k, p)` with
`k + p <= 9,999,976`**. This strictly contains the old pure-period statement
and covers `p` up to the whole prefix, not just `p <= n/2`.

## Interpretation

The old conclusion survives; the old certificate did not establish it. Under
Rule 1 this is an `exact-exclusion`, not evidence of complexity: the class of
eventually periodic words with `k + p <= n - 24` has `log2|M| ~ n - 24 + log2 n
< n`, and `T_max ~ log2 n` is exactly what a random word gives. It cannot
resolve Problem 1, which is asymptotic.

## Next step

Run the same script on the 46M stream once A3 is unblocked (no code change).
