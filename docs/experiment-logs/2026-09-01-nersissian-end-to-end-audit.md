# 2026-09-01 — Nersissian end-to-end shortcut audit

## Goal

Audit the complete cost of computing the Rule 30 center bit `c_n` from `n`, rather than reporting only the lookup cost after an n-dependent row/support object has already been constructed.

This is Prize **Problem 3** work: the computational-effort question.

## Hypotheses

**H0:** cold-start computation retains at least linear work because the support machinery required by the fast query must itself be constructed.

**H1:** the published method can be implemented end-to-end in `o(n)` work from `n` alone, without hiding `Omega(n)` n-dependent preprocessing or storage.

Memory variant: test separately whether a sublinear query hides linear n-dependent state.

## Source and interpretation

See `docs/references/nersissian-rule30.md`.

The source separates a fast query within a mathematically evaluated row from generation of the object being queried. Therefore this audit treats three regimes separately:

1. **cold isolated:** no n-specific cached state;
2. **warm:** required support already materialised;
3. **sequential:** state reused while advancing through neighboring indices.

## Implementation

`experiments/nersissian_audit.py` implements a transparent explicit-support baseline:

- published support recurrence;
- OR-convolution over `F_2`;
- Lucas-theorem binomial parity;
- exact operation counters for construction;
- explicit separation of construction and query accounting;
- independent validation against `prize_lab.center_bits_int`.

This implementation does **not** claim to reproduce the paper's compressed masked-dyadic representation. That distinction is intentional and enforced in output metadata.

## Small-case structural check

The reconstructed recurrence produces:

| m | S_m |
|---|---|
| 1 | `{0}` |
| 2 | `{1}` |
| 3 | `{1}` |
| 4 | `{2}` |
| 5 | `{2,3,4}` |
| 6 | `{3,5,7}` |
| 7 | `{3,5,8}` |
| 8 | `{4,6,8,9,12,14,16}` |

The resulting Lucas evaluation reproduces the known Rule 30 center prefix in the bounded regression tests.

## Deterministic construction accounting

For the explicit baseline, representative cold-start counts are:

| center index n | `|S_(n+1)|` | OR-pairs processed | peak support size | max support index |
|---:|---:|---:|---:|---:|
| 4 | 3 | 3 | 3 | 4 |
| 8 | 13 | 45 | 13 | 25 |
| 12 | 29 | 1,305 | 29 | 63 |
| 16 | 55 | 4,147 | 55 | 128 |
| 18 | 95 | 9,867 | 95 | 254 |

These are deterministic counts from the explicit recurrence, not timing-based asymptotic claims.

## Interpretation

The first concrete result is methodological: the advertised fast query and the full `n -> c_n` computation are different complexity questions.

The explicit recurrence already shows why this matters. A warm query can be cheap relative to reconstructing the n-dependent support object, while sequential reuse can make neighboring queries appear much cheaper than an isolated random-access query.

However, this baseline does **not** establish `Omega(n)` for Nersissian's compressed representation, and it certainly does not prove Prize Problem 3. The compressed dyadic representation could change both construction and storage scaling. That is now the critical object to reconstruct faithfully.

## Failure / correction during implementation

An initial unit-test horizon of `n=40` was too aggressive for the deliberately uncompressed recurrence and turned correctness validation into a scaling run. The test horizon was reduced to `n=18`. Larger horizons belong in the experiment runner, not CI.

## Next step

Reconstruct the source's masked/dyadic block representation exactly enough to implement the same interface:

- cold construction counts/time/space;
- warm single-cell query counts/time;
- sequential incremental cost;
- isolated random-access cost;
- representation size as a function of n.

Only after that reconstruction is independently validated should growth-class fitting be treated as informative.
