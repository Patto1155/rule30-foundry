# Handover — 2026-09-01 Nersissian end-to-end audit

## What changed

Created `research/nersissian-audit` from `claude/keen-sagan-21rzbr` at `6af8c9c8b5e9ab589b1999ce5c503c60a78f9dff`.

Added:

- `experiments/nersissian_audit.py`
- `tests/test_nersissian_audit.py`
- `docs/references/nersissian-rule30.md`
- `docs/experiment-logs/2026-09-01-nersissian-end-to-end-audit.md`
- `docs/handover/2026-09-01-high-value-directions.md`
- this handover

## Source audit

No Nersissian source material was present in the repository when searched by author name or `S_m` notation.

Public source material was located and recorded in `docs/references/nersissian-rule30.md`.

The key conceptual distinction is now explicit in code and docs:

- query complexity after a mathematical row/support representation exists;
- construction complexity of that n-dependent representation;
- total cold-start complexity from `n` to `c_n`.

Prize Problem 3 needs the third quantity.

## Implementation status

The current implementation is deliberately the explicit support-set recurrence, not a claimed reconstruction of the source's compressed masked/dyadic representation.

It implements:

`S_m = Inc((S_(m-1) * S_(m-2)) Δ S_(m-1) Δ S_(m-2))`

with OR-convolution multiplicity reduced modulo two, followed by Lucas-theorem parity evaluation.

Instrumentation records:

- layers built;
- OR-pair operations;
- support toggles;
- increment operations;
- peak and final support sizes;
- explicit warm-query Lucas tests;
- cold vs sequential reuse.

## Correctness checks

Small published supports through `S_8` are pinned in unit tests.

The support method is checked against the repository's independent integer center-column engine over a bounded prefix.

An initial `n=40` unit-test target was intentionally reduced to `n=18`; the explicit uncompressed recurrence is the object being measured and CI should not become a scaling experiment.

## Current conclusion

No prize claim is established.

The audit does establish the correct accounting framework and a reproducible baseline. It makes it impossible to cite a warm query cost as if it were an end-to-end random-access algorithm without also reporting how the required n-dependent support was obtained.

The explicit baseline shows material preprocessing cost, but that does not imply the compressed representation has the same scaling.

## Highest-value research roadmap

The ranked programme is now documented in:

`docs/handover/2026-09-01-high-value-directions.md`

Current effort allocation:

1. Nersissian compressed-representation audit — **30%**.
2. Problem 1 periodicity-to-contradiction search — **25%**.
3. Corrected circuit/index-function programme — **20%**.
4. Cross-rule controls — **15%**.
5. DFAO / grammar / GF(2) diagnostics — **7%**.
6. Larger simulations / discrepancy — **3%**.

The roadmap includes concrete deliverables, decision gates, stop conditions, and mandatory engineering gates before any paid compute.

## Highest-value next action

Faithfully reconstruct the masked/dyadic block representation from the primary source, behind the same cold/warm/sequential accounting interface, and validate every returned center bit against the existing independent Rule 30 engine.

Only then run logarithmically spaced scaling points and compare candidate growth classes. Do not infer an asymptotic lower bound from regression alone.

## Still outstanding from prior handover

- GPU simulator checkpoint/resume remains unimplemented.
- The existing `verify_all` SKIP still needs to be identified before any paid compute run.
- Do not rent compute for this audit.
