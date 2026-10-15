# Bounded physical deficit queries

Purpose: exact exclusion only. No complexity or asymptotic inference.

| Query | K | maximum N | required deficit | Result |
|---|---:|---:|---:|---|
| q00 fixed positive control | 64 | 190 | 8 | SAT, r=0x98f9d8a0, W190=26 |
| q01 Rule30 zero target | 0 | 12 | 1 | SAT control |
| q02 Rule90 zero target, r=1 | 2 | 12 | 1 | SAT control |
| q03 physical search | 64 | 300 | 9 | UNKNOWN at 15 seconds |
| q04 physical search | 128 | 300 | 9 | UNKNOWN at 15 seconds |
| q05 physical search | 256 | 300 | 9 | UNKNOWN at 15 seconds |
| q06 fixed r=0 regression | 0 | 300 | 9 | UNSAT, external DRAT checked |

Each physical search includes **every** prefix `1 <= n <= maximum N` in one
CNF. Initial positive cells beyond K are zero. Later right cells are physical
CA evolution, not arbitrary boundary values. The left completion is free to
extend through N and forced by triangular inversion.

`results.json` preserves exact query dimensions, hashes, tools, timings, SAT
witnesses, and honest UNKNOWNs. Replay regenerates every exact semantic CNF,
checks source pins, and validates fixed-seed and canonical witness metadata. `certificates.tar.xz` contains all q*.cnf,
solver logs, and q06's checked ASCII DRAT plus checker output. The archive is
7.5 MiB. q06 excludes only the single right seed zero. It gives no exclusion
for K64/128/256. SAT witnesses were compared with Agent A's independent
completion and dictionary-cell full forward simulation.

After merging the reference and free-u encoder dependencies, replay without
rerunning the expensive searches:

```sh
python -m unittest tests.test_deficit_sat -v
python -m experiments.deficit_sat --verify runs/p2-deficit-sat-2026-10-15/results.json --cadical third_party/cadical --drat-trim third_party/drat-trim
```

The saved query ordering comes from the initial main run q00..q05, followed by
a separate q06 regression query. The current CLI places that regression before
the three search queries; CNFs are deterministic per query parameters.

Not run: full repository verifier/test suite, Lean, GPU, stronger requested-K
queries beyond 15s each, all finite K, or an independent model-lineage audit.
The lead owns repository-wide verifier and ledger updates.
