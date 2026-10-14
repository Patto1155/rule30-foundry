# Nonlinear obligation review

Outputs:

- `docs/theory/nonlinear_obstruction.md`
- `experiments/nonlinear_obstruction_check.py`
- this report, `queue/results/period-ladder-nonlinear-review.md`

## Argument and proof filters

**Rule 90 filter: passes explicitly.** Rule 90 has propagator H=z⁻¹+z;
Rule 150 has L=H+1. For the switched update H a+μ a+λ Q(a), the entire
Duhamel remainder and the superposition defect carry λ. Both vanish when
λ=0; Rule 90 additionally has μ=0. No periodicity or complexity assertion is
made for Rule 90.

**Zero filter: passes explicitly.** The zero row stays zero and every forcing
term vanishes. The derivation makes no claim of aperiodicity or hardness for
zero. The script includes zero at all times.

**Result.** The claimed original obstruction was misworded: Rule 90 and Rule
30 differ by a linear centre term as well as the quadratic term. The centre
term preserves an additive binary-time formula (Rule 150). The nonlinear
term destroys superposition and adds the exactly derived forcing remainder
Σ L^(t−1−s)Q_s. The first centre correction on the prize seed occurs at t=2.
All-time formulas are proved by elementary polynomial algebra and induction;
finite checks exercise implementations of those formulas.

## Acceptance actually run

| Command | Result |
|---|---|
| `python experiments/nonlinear_obstruction_check.py` | PASS, 32 five-cell initial rows, 2,080 seed/time pairs at every t=0..64, rules 30/90/150, 40 local identity cases. About 0.13 s here. |
| `python tools/lint_ledger.py` | PASS, 45 existing ledger rows and 57 logs; all cited paths present. |
| `python -m py_compile experiments/nonlinear_obstruction_check.py` | PASS. |
| `git diff --check` | PASS. New files are untracked, so this command checks only tracked diffs; it is not a whitespace audit of the added files. |

Independent dynamics use Wolfram truth-table bits on an infinite integer
lattice. Closed forms use GF(2) polynomial convolution, actual binomial
coefficient parities, and binary products. No shared ANF recurrence drives
the reference simulation. Scope is exhaustive width-five seeds, not
exhaustive width-129 light cones. None of the tests infers an all-time theorem
from a finite prefix.

## Grade recommendation for the parent

The corrected Duhamel identity and superposition defect qualify as **Theorem
(elementary identity, no prize implication)** if the parent checks the short
algebraic proofs. Their novelty has not been established; do not advertise
them as a new mathematical discovery. The earlier blanket formulation
“precisely and only the quadratic term separates Rule 30 from Rule 90”
should be corrected, not promoted. No Prize 3 hardness grade increases, no
period rung is excluded, and no change-of-variable barrier is proved.

The existing nonlinear obligation should remain **open** with a narrowed
target. Suggested replacement fields, for the parent to apply:

```json
{
  "id": "nonlinear-term-obstruction",
  "prize": "3",
  "lemma": "After separating Rule 90's extra centre linear term (the additive Rule 150 propagator), determine whether the exact single-seed centre forcing sum sum_{s<t} [z^0] L^{t-1-s} Q_s admits sublinear random access, or establish a lower bound for a precisely named computation or representation model.",
  "implication": "The exact Rule 30 Duhamel identity locates the failure of homogeneous additive propagation, not a complexity barrier. Any Prize 3 implication requires a seed-specific compression theorem or a lower bound in a specified model.",
  "refutation": "For a proposed lower-bound mechanism, exhibit a compressed representation of the seed-specific forcing sum or a shortcut within its specified model. A claim that changes of variables cannot remove the forcing requires the transformation class to be defined first.",
  "status": "open",
  "progress": "Exact all-time Rule 150 Duhamel identity and quadratic superposition defect proved algebraically; independently checked on all 32 five-cell seeds at all t<=64. Rule 90 and zero filters pass. No computational lower bound or variable-change obstruction proved. Evidence: docs/theory/nonlinear_obstruction.md and experiments/nonlinear_obstruction_check.py."
}
```

This is a proposed state update, not an edit of an authoritative state file.

## Everything not run or not verified

- `python tools/verify_all.py`: deliberately not run concurrently; the parent
  owns the session-wide verifier and the before-commit gate.
- Counting-bound search: none run because this is symbolic derivation with
  exhaustive identity checks, not a searched-and-found-no-fit experiment.
- GPU, CuPy, SAT, DRAT, Lean, Mathlib, provider calls, and network requests:
  none used for this task.
- Literature novelty review: not run; the derivation does not claim novelty.
- All seeds wider than five cells, times greater than 64, or exhaustive
  space-time diagrams through t=64: not computed. The proved algebraic
  identities themselves hold for all finite rows and all times.
- No asymptotic runtime lower bound, impossibility of compressed forcing,
  exclusion of eventual period two, or non-removability under a variable
  transformation was verified or claimed.
- No commit or merge performed; no STATUS, CLAIM_LEDGER, obligations, or
  experiment-log file edited.
