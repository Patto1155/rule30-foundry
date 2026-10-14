# Period-two worker report

**RULE 90 TEST: PASS in scope.** The audited zero-fiber proof uses the
monotone Rule 30 OR latch, which Rule 90 lacks. Checker verifies the single
seed under Rule 90 becomes constant zero, and the symmetric pair has zero
trace. No general left-permutive exclusion is claimed.

**ZERO TEST: PASS in scope.** Zero is retained as a constant-zero fiber.
Finite nonzero rows cannot vanish because both support endpoints expand.
Alternating centre hypothesis excludes the zero row directly.

## Returned files

- `docs/theory/period2_attempt.md`: analytical reconstruction of Condrey
  Theorems 2–3, Corollaries 4–5 and horizon Theorems 6–7; precise p=2 OR
  reset and first-one-statistic failure; fully quantified open bridge.
- `experiments/period2_identity_check.py`: independent elementary-rule
  truth-table checks and source-claim corroboration.
- This report. No commits or changes to state/log files.

## Acceptance

`python experiments/period2_identity_check.py`: PASS. Constant-fiber checks
cover 32 finite right words, both centre values, left depth 32. Exhaustive
nonzero initial rows radius 1..6 reproduce both horizon indices and counts.
32 right cones and 64 negative prefixes check two-step identities. Right
integers 1 and 7 first differ in the forced initial left completion at depth
9. Rule 90 and zero controls pass.

`python tools/lint_ledger.py`: PASS (45 ledger rows, 57 experiment logs).
`git diff --check`: PASS.

## Parent decisions supported, not assigned

The source's main period-one analytic proof can be reconstructed without
relying on computations or Lean. Parent should check the argument before
choosing an external-audited theorem grade. The §5 inference from a linear
lower bound to absence of a bounded law at fixed radius is unsupported and
not used in this reconstruction. Horizon constants use onset zero.

No new exclusion of period two was obtained. The reset identity and transfer
map formulation overlap existing recovered-branch work. The exact nine-cell
counterexample only locates failure of a first-right-one fiber classification.
A density lower bound depending on the entire fixed finite right seed is an
explicit stronger sufficient obligation; its weaker unbounded-weight version
is the original period-two question in another notation.

## Everything not run

Full verifier; SAT or DRAT; GPU; Lean or Mathlib; outside-provider/council
calls; large prefix or survivor searches; recovered branch invariant/core,
weighted transport, or fixed-weight reproducers; D1 script; and proof of
physical support growth. No independent referee or formal verification is
claimed. Existing research snapshot documents were read via git show only.
