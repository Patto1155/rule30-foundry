# Experiment Log

- Date: 2026-08-30
- Title: Exhaustive **Exact** Period Search in the 10M Center Column (Problem 1)
- Claim Level: **Certificate** — exhaustive and exact over the stated range,
  with no sampling and no statistical threshold.
- Run date: 2026-08-30 (152 s total, including 50 control streams)
- Artifact: `data/prize/2026-08-30-period-search-exact.json`
- Verifier: `python experiments/period_search_exact.py --gate-only`

## Goal

Item 13 of `docs/handover/2026-08-30-next-session-plan.md`. Prize Problem 1 asks
whether the center column is eventually periodic; a found period is a $30,000
counterexample. The plan calls this "the one place in this repo where 'run it
bigger' is genuinely the right move", and notes correctly that the counting
bound does not blunt it — this is a direct search for a specific structure, not
a negative over a model class, so the Admission Rule does not apply.

## Why the method changed, not just the size

The plan framed this as a VRAM problem: push `N` from 10^6 toward 10^7–10^8 and
"reuse the Bonferroni threshold discipline already in
`experiments/period_search.py` (best `z = 4.66 < 5.61`)".

On reading that script, the binding problem is not size. It is that **an exact
question was being answered statistically.** Experiment E estimates, for each
candidate period `p`, the match rate over 10,000 randomly sampled positions,
and compares a z-score against a Bonferroni threshold. But `p` is a period of
the observed prefix iff `S[i] == S[i+p]` for *every* valid `i`. **One mismatch
refutes it.** There is nothing to estimate, no null distribution, and no
multiple-testing correction to get right.

That last point matters practically. The plan warns that a larger search "needs
a correspondingly larger threshold, and forgetting that would manufacture a
false positive". An exact test removes that failure mode entirely rather than
managing it.

Sampling is also the *more* expensive choice. A random-looking stream disagrees
with its own shift within a handful of positions, so an exact test costs O(1)
per candidate in practice, against 10,000 sampled comparisons for an estimate
that is strictly weaker.

## Method

A period `p` survives only if the first `W` bits agree with the bits at offset
`p`. So pack the `W`-bit window at every position into a single integer and
take the positions whose window equals the window at 0. For `W = 64` on a
random-looking stream the expected number of survivors is `n / 2^64` — none —
so a single vectorised equality refutes every candidate at once. Any survivor
is then checked position-by-position to the end of its overlap.

This decides **every** `p <= n - W` exactly, rather than a chosen ceiling.

Headline claims are restricted to `p <= n/2`, where at least `n/2` positions
confirm the period; a "period" supported by a handful of positions is vacuous.

## Correctness gate (run before the search)

A search that cannot find a period that is really there proves nothing.

| Check | Result |
|---|---|
| planted period 3 is found, and reported as 3 | PASS |
| planted period 1000 is found, and reported as 1000 | PASS |
| planted period 65536 is found, and reported as 65536 | PASS |
| one flipped bit refutes an otherwise perfect period | PASS (0 confirmed) |
| bitstream decodes LSB-first to OEIS A051023 | PASS (`110111001100010`) |

The fourth check is the important one: it confirms the test is exact rather
than tolerant. A sampled test with 10,000 samples would have missed a single
flipped bit in 2,000,000 with probability ~99.5%.

## Result

```
bits searched                      10,000,000  (hash-verified, LSB-first)
periods decided exactly             9,999,936
candidates surviving a 64-bit window        0
confirmed periods                           0
confirmed periods p <= 5,000,000            0
window build                             2.09 s
candidate scan                           0.01 s
```

**No period. Exactly, exhaustively, over every `p` up to 9,999,936.**

Because *zero* candidates survived even the 64-bit window, every `p` is refuted
with at least `min(64, n-p)` positions of disagreement — the refutation is not
marginal anywhere in the range.

### Longest self-overlap

A yes/no period search discards how *close* the sequence comes to repeating.
`Z[p]` is the longest common prefix of the stream and the stream shifted by
`p`; `max_p Z[p]` is that quantity.

```
center column        26 bits, at shift 2,085,567
50 random controls   band [20, 29], median 22
expected maximum     ~log2(n) = 23.3
empirical p-value    0.098
union-bound p-value  0.138
verdict              consistent with chance
```

**A correction worth recording.** The first version of this script compared the
observed 26 against `2*log2(n) = 46.5` and against a 7-seed band of `[21, 23]`,
which made 26 look like a mild anomaly sitting above the controls. Both were
wrong:

- `2*log2(n)` is the longest common substring between two **independent**
  length-`n` sequences, where there are `n^2` position pairs. Comparing a
  stream against its own `n` shifts gives `~log2(n) = 23.3`. Using the doubled
  form inflates the null and would make a genuine anomaly look ordinary.
- Seven seeds is too small a sample for a maximum statistic. At 50 seeds the
  band is `[20, 29]` and 26 sits comfortably inside it.

The correct reading is that a maximum self-overlap of 26 bits is exactly what a
fair coin does at `n = 10^7`.

## Interpretation

**The Rule 30 center column has no period `p <= 5,000,000` in its first
10,000,000 bits.** This is a finite, checkable, exhaustive fact, established
without a threshold, a null distribution, or a multiple-testing correction.

Against the previous state of the ledger — "no period found up to 10^6",
sampled, with a best z of 4.66 against a Bonferroni threshold of 5.61 — this is
a 5x wider range, an exact verdict instead of a statistical one, and a claim
with no correction to forget.

Scope, plainly:

- **This cannot resolve Problem 1.** Eventual periodicity is an asymptotic
  property; no finite prefix can establish aperiodicity. What is excluded is a
  period that manifests within the first 10^7 bits. A period longer than the
  data, or a pre-period longer than the data, is untouched.
- The search assumes the period starts at position 0 in the sense that it tests
  `S[i] == S[i+p]` for all `i` from 0. That is the right test for *eventual*
  periodicity over the observed window as well: any `p` that is eventually a
  period would show as a long — though not full-length — overlap, and the
  longest overlap found anywhere is 26 bits.
- 10^7 is what this machine could reach. The 46M bitstream would extend the
  range to `p <= 2.3 x 10^7` with no change to the method and a few minutes of
  compute.

## Cost note, against the plan's expectation

The plan anticipated this being VRAM-bound and the best case for rented
hardware: "a 24 GB card quadruples the reachable horizon over the local 6 GB
GTX 1060." That is true for *generating* a longer bitstream, which is quadratic
in the horizon and genuinely expensive. It is not true of the search itself.

The scan over all 10^7 candidate periods took **2.1 seconds** on four CPU
cores, and 149 of the 152 seconds went to the 50 control streams. The binding
constraint on Problem 1 is the cost of simulating a longer center column, not
the cost of searching it. Rented GPU time should be aimed at the simulation.

## Next Step

1. **Re-run on the 46M bitstream.** No code change, a few minutes, and it
   extends the exact range to `p <= 2.3 x 10^7`.
2. **Then the constraint really is simulation.** Reaching `p <= 10^8` needs
   2 x 10^8 center bits, which is `(20)^2 = 400x` the 10M generation cost.
   That is the run to buy hardware for — and the search that follows it is
   seconds.
3. Retire `experiments/period_search.py`'s sampled scan, or keep it explicitly
   as the historical record. Two period searches with different methods and
   different answers in the same repo is how a ledger starts lying.
