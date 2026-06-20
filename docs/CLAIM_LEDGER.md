# Claim Ledger

Use this ledger to keep agents honest about what the repo actually knows. A
claim is not prize-facing just because it has a large run behind it.

## Levels

| Level | Meaning | Merge standard |
|---|---|---|
| Observation | One experiment or exploratory score. | Record setup, command, result, and next check. |
| Robust observation | Controls/nulls and at least one robustness check. | Include verification, baselines, and failure mode. |
| Certificate | Finite artifact another agent can verify. | Include verifier command and artifact path. |
| Proof candidate | Theorem-shaped argument with checkable lemmas. | Include assumptions, proof gaps, and reproduction path. |

## Current Claims

| Claim | Level | Evidence | What would promote it |
|---|---|---|---|
| Packed/GPU Rule 30 kernels are usable for research runs. | Robust observation | `docs/GPU_KERNELS.md`, reference checks in `experiments/rule30_open_utils.py`, fused-kernel branch validation. | A tiny standalone verifier that runs naive vs packed vs GPU across seeded edge cases in one command. |
| b=2 coarse-grain closure for Rule 30 is generic chaotic leakage, not Rule-30-specific reducibility. | Robust observation | `docs/experiment-logs/2026-06-13-coarse-grain-same-statistics-null.md`, `data/coarse_grain_rule_null.json`. | Independent rerun on multiple fields/radii, plus archived command output. |
| b=3 coarse-grain search does not find Rule-30-specific reducibility at r=1 and tested shears. | Robust observation | `docs/experiment-logs/2026-06-14-b3-coarse-grain-verdict.md`, `data/coarse_grain_b3_verdict.json`. | Multi-seed confidence band, larger `r`, alternate optimizers, and saved best projection artifacts. |
| The center column behaves statistically close to fair/random over tested prefixes. | Observation | README A-L summary and historical experiment logs. | Reframe as prize-specific discrepancy bounds or finite certificates, not more aggregate randomness tests. |
| No shortcut for the nth center bit is known in this repo. | Observation | Negative Markov/ML/coarse-grain experiments. | A `prize_lab.py` shortcut-search harness that emits candidate programs/transducers or checkable negative artifacts. |

## Promotion Rule

When adding a new result, state the next promotion step explicitly. If there is
no plausible promotion path, the experiment is probably not worth expanding.
