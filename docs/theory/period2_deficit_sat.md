# Physical finite-seed deficit SAT

`experiments/deficit_sat.py` asks the exact bounded question

> Does some integer `0 <= r < 2^K` have some `1 <= n <= N` with
> `n - 7 W_n >= C`, when its left completion forces centre `t mod 2`?

The free-column encoding supplies triangular inversion. An additional physical
right causal cone evolves the actual Rule 30 truth table, driven by that
centre, from the K seed bits and an identically zero exterior at time zero.
Only even column-1 cells are equated to the reduced free inputs. The odd
representatives in the reduced encoding are arbitrary and must not be equated
to physical odd cells. No later right boundary is independently selectable.
Bits initially beyond the observation cone are irrelevant, but allocated when
K exceeds N. There is no finite-left-support restriction beyond the completion
prefix: setting the exterior of that prefix to zero gives the required finite
forward trace through N.

For each prefix j, exact unary counters encode `C[j,k] iff W_j >= k`.
The update is `C[j,k] = C[j-1,k] OR (e_j AND C[j-1,k-1])`, with both
implications encoded. The final clause is the disjunction of
`NOT C[j, floor((j-C)/7)+1]` over every j from C through N. Therefore the
query covers every prefix, rather than testing only its endpoint.

Counting gate: an unrestricted physical seed class has exactly `2^K` members
and K independent initial bits. A fixed-seed control has one member and zero
free input bits (its storage width remains K). This is an exact-exclusion experiment; negative
answers would not constitute a complexity or randomness inference.

Small validation exhaustively fixes every 3-bit seed, each of rules 30/90,
each alternating/zero target, and compares three thresholds with independent
packed completion. It checks both admitted model completions and the exact
maximum over every prefix. The reference implementation independently replays
SAT witnesses, including full dictionary-cell forward simulation.

The positive control fixes `r=0x98f9d8a0` and recovers `N=190, W=26`, deficit
8. Zero Rule30 and finite zero-target Rule90 controls have growing deficit,
so this is neither a generic initial-condition argument nor a Rule90 argument.
Artifact verification regenerates each semantic CNF byte-for-byte from the saved
parameters, pins reference and encoder source digests, checks fixed-seed bounds
and canonical maximal-deficit witness metadata, and rejects certificates or
witnesses attached to UNKNOWN. Tamper tests alter K, horizon, saved W, source
digests, and status and require rejection.

The numerical runs are in `runs/p2-deficit-sat-2026-10-15/results.json`.
UNKNOWN is not an exclusion. A solver UNSAT result is reportable as UNSAT only
after external ASCII DRAT verification and hashing the exact CNF and proof.

Acceptance after merging A and B's dependencies:

```sh
python -m unittest tests.test_deficit_sat -v
python -m experiments.deficit_sat --cadical third_party/cadical --drat-trim third_party/drat-trim --timeout 15
```

Before merge, supply `DEFICIT_ENCODING` to the test and `--encoding` plus
`--reference` to the experiment. No global theorem, asymptotic density result,
or exclusion for all finite seed sizes follows from this instrument.
