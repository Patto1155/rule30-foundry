# Period-2 horizon SAT instrument review (2026-10-14)

**Rule 90 test:** PASS by construction and checked control. The instrument uses Rule 30's complete truth table, including the OR latch, and has a separate Rule 90 truth table. A radius-zero single cell under Rule 90 yields `1000...` (64 bits), certified SAT and independently simulated; no exclusion of eventual periodicity is asserted.

**Zero test:** PASS by explicit nonzero constraint and checked control. Every search excludes the all-zero initial row. Removing that constraint admits the zero configuration with a 64-bit constant zero trace. No initial-condition-invariant theorem is asserted.

## Design and gates

This is finite-configuration lemma discovery, not a prize measurement. Initial rows are nonzero, supported in `[-R,R]`, with exactly zero exterior on the infinite line. Time starts at `T=0`. `L` means observed bits, so the alternating target is `0101...` of length `L`, not `2L`. The search grid is `L=8..64` for alternating, zero, and one, with radius cap 64.

The theory gate was accepted by the lead before dispatch. The counting gate ran `python experiments/counting_bound.py --pretty` before any query. New helper `finite_configuration_verdict` computes the relevant upper bound `log2(2^(2R+1)-1)` on trace-class size; `runs/period-horizon-2026-10-14/counting-gate.json` records the design arithmetic. Every query reruns that helper. Sparse negatives are **exact-exclusion**, never evidence of complexity. Counting does not guarantee a negative.

The observation light cone is encoded by full truth-table clauses. Initial support variables include cells outside the observation cone, so a remote nonzero cell is permitted without accidentally altering the trace. SAT witnesses are checked by a separate sparse, cell-by-cell infinite-line simulation. UNSAT queries retain both CNF and DRAT, checked using drat-trim. All 539 UNSAT queries retain their CNF, ASCII DRAT proof, and original checker log in one deterministic tar stream compressed with xz preset 3. Its 135,976,180 compressed bytes are split into 28 pieces of at most 5,000,000 bytes. Results include a SHA-256 digest for every chunk, the full compressed stream, and every raw CNF/proof/checker log. The verifier authenticates and concatenates the chunks, allows only the exact expected regular-file members, and extracts into a temporary directory. SAT-only CNFs are deterministically regenerable from the query metadata and checked against stored hashes.

Support classes are nested. A SAT witness at `R` remains supported at every larger radius; UNSAT at `R-1` excludes all smaller radii. An exact minimum therefore requires those adjacent certificates. UNKNOWN is retained as censored evidence and never counted as UNSAT.

## Controls

Before the grid, all 24 Condrey controls passed: for every `R=1..6`, SAT at zero-prefix length `2ceil(R/2)+1` and one-prefix length `2floor(R/2)+2`, and checked UNSAT at each length plus one. Independent exhaustive enumeration also reproduced those maxima and extremizer counts `2^R-1` (zero) and `2^R` (one). This is reproduction of Condrey preprint Theorems 6/7, not a novel result. Rule 90 and zero controls were checked separately. Four seeded random target controls use lengths 8, 16, 32, and 64.

## Bounded findings

All **171** grid points (`57 lengths × 3 targets`) have certified exact minimum radii; none is UNKNOWN or censored. The run used CaDiCaL 3.0.1, 10-second per-query limits, and a 900-second total search budget, and completed in 271.504 seconds. It made 1,249 distinct queries: 710 SAT and 539 UNSAT. Every UNSAT was checked during the run and is replayable from retained certificates. Tool binary hashes are recorded in `results.json`.

| Prefix length L | Alternating minimum R | Zero minimum R | One minimum R |
|---:|---:|---:|---:|
| 8 | 5 | 7 | 6 |
| 16 | 9 | 15 | 14 |
| 24 | 16 | 23 | 22 |
| 32 | 20 | 31 | 30 |
| 40 | 33 | 39 | 38 |
| 48 | 38 | 47 | 46 |
| 56 | 46 | 55 | 54 |
| 64 | 54 | 63 | 62 |

Across the whole grid, zero and one radii reproduce Condrey's laws: `R_zero(L)=2floor(L/2)-1` and `R_one(L)=2floor((L-1)/2)`. The new alternating curve has finite plateaus (for example, radius 20 for lengths 27–35) and reaches radius 54 at length 64. The four seeded random targets have minimum radii 3, 13, 27, and 57 at lengths 8, 16, 32, and 64. Four random words are a small instrument control, not a statistical comparison.

Suggested grade ceiling: **bounded computational finding** for the alternating curve; the constant curves are independent reproduction of a cited preprint. This grid supplies no unbounded-growth, fixed-radius infinite-horizon, or eventual-periodicity theorem.

## Acceptance commands

```bash
python experiments/period_horizon_sat.py --self-test
python experiments/period_horizon_sat.py --run --cadical third_party/cadical --drat-trim third_party/drat-trim --budget 900 --timeout 10
python experiments/period_horizon_sat.py --compact runs/period-horizon-2026-10-14/results.json
python experiments/period_horizon_sat.py --verify runs/period-horizon-2026-10-14/results.json --drat-trim third_party/drat-trim
python experiments/period_horizon_sat.py --tamper-test runs/period-horizon-2026-10-14/results.json --drat-trim third_party/drat-trim
```

Verification reconstructs every CNF, checks its hash, independently simulates every SAT row, checks every UNSAT DRAT, validates target/length/rule/support identity, and requires the exact full grid and all controls. Tamper checks reject altered target, length, minimum, query identity, initial witness, censored bound, and proof digest.

## Verification outcome

PASS self-test, including both truth tables, every initial row at radius 0–3 through ten steps, and independently enumerated Condrey maxima/counts at radius 1–6. PASS strengthened full verification against the final 28-chunk artifact: all 1,249 CNFs, 710 independent SAT witness simulations, 539 DRAT checks, full grid, controls, and minimum-radius evidence (136.123 seconds; `runs/period-horizon-2026-10-14/verification.txt`). PASS all seven tamper-rejection cases. PASS `python tools/lint_ledger.py`. The lead must rerun the acceptance commands and own the grade.

## Scope and not run

No eventual period-2 exclusion follows from this grid. The window is `T=0`; no other start times or the opposite alternating phase were scanned. No theorem bounding horizon for every fixed finite seed is proved. No large-radius asymptotics, GPU experiments, Lean verification, external-model review, or single-seed prize measurement were run by this worker. No full repository verifier was run concurrently; that remains the lead's gate before commit. No state files or experiment logs were edited and no commit was made.
