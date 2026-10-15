# Free right-boundary completion density

The asymptotic positive/zero question remains **undecided**. This run proves
20 finite minima, at N=4,8,...,80, and retains censored bounds at eight larger
sizes through N=500. These universal finite-boundary statements are potential
lemma discovery for all finite configurations; they are not a measurement of
the single prize seed or evidence about randomness/complexity.

## Exact scope and independent checks

The centre is fixed to c(t)=t mod 2. The free bits are physical **right**
column u(s)=x(1,2s), chronological, with ceil(N/2) independent variables.
Odd right-column bits can be zero representatives because the first OR latch
erases them. Column j is x(-j,t); its length is N-j+1. The SAT gates encode

`x(-j,t) = x(-(j-1),t+1) XOR (x(-(j-1),t) OR x(-(j-2),t))`.

The cardinality totalizer constrains the N initial left bits to weight <=k.
Each SAT word is replayed through A's independent packed implementation.
Each UNSAT claim has ASCII DRAT checked externally by drat-trim. The JSON
records each exact CNF and proof SHA256, tool hashes, bounds, elapsed time,
and every UNKNOWN query. The archive retains all solver/checker logs and
CNFs/proofs, including UNKNOWN CNFs. UNKNOWN proofs were deleted.

Counting preflight: the actual search class is 2^ceil(N/2) free words, and the
purpose is exact-exclusion. The finite-configuration radius count is inapplicable.
The counting-bound gate supplies no complexity or randomness interpretation.

The supplied N81 weight9 and no-11 N79 weight10 examples reproduce as
**witnesses**, not independently certified minima at those exact horizons.
Exhaustive tests cross-check all free inputs N<=12 and show each completion
bit unique; all minima N<=16 agree with exhaustive enumeration.

## Finite bounds

An exact row has a replayed feasible word and a checked UNSAT bound one below.
A bracket has a certified lower bound and replayed feasible upper bound;
UNKNOWN/time caps leave a hole. Certified bounds at smaller N also apply to
larger N by truncating any free word and its completion. The JSON records
which smaller horizon supplies each inherited lower bound. Upper witnesses
reuse all retained SAT words and the supplied N81 word, zero-padded or
truncated to the required length.

| N | min W bracket | min W/N bracket | status |
|---|---|---|---|
| 4 | 1–1 | 0.25000–0.25000 | exact |
| 8 | 2–2 | 0.25000–0.25000 | exact |
| 12 | 2–2 | 0.16667–0.16667 | exact |
| 16 | 2–2 | 0.12500–0.12500 | exact |
| 20 | 3–3 | 0.15000–0.15000 | exact |
| 24 | 4–4 | 0.16667–0.16667 | exact |
| 28 | 4–4 | 0.14286–0.14286 | exact |
| 32 | 4–4 | 0.12500–0.12500 | exact |
| 36 | 5–5 | 0.13889–0.13889 | exact |
| 40 | 6–6 | 0.15000–0.15000 | exact |
| 44 | 6–6 | 0.13636–0.13636 | exact |
| 48 | 6–6 | 0.12500–0.12500 | exact |
| 52 | 6–6 | 0.11538–0.11538 | exact |
| 56 | 7–7 | 0.12500–0.12500 | exact |
| 60 | 8–8 | 0.13333–0.13333 | exact |
| 64 | 9–9 | 0.14062–0.14062 | exact |
| 68 | 9–9 | 0.13235–0.13235 | exact |
| 72 | 9–9 | 0.12500–0.12500 | exact |
| 76 | 9–9 | 0.11842–0.11842 | exact |
| 80 | 9–9 | 0.11250–0.11250 | exact |
| 84 | 9–11 | 0.10714–0.13095 | censored bracket |
| 100 | 9–15 | 0.09000–0.15000 | censored bracket |
| 117 | 9–17 | 0.07692–0.14530 | censored bracket |
| 126 | 9–18 | 0.07143–0.14286 | censored bracket |
| 190 | 9–28 | 0.04737–0.14737 | censored bracket |
| 300 | 9–43 | 0.03000–0.14333 | censored bracket |
| 400 | 9–58 | 0.02250–0.14500 | censored bracket |
| 500 | 9–72 | 0.01800–0.14400 | censored bracket |

The exact finite ratios fluctuate rather than decrease monotonically.
No asymptotic law is fitted. The large-horizon intervals are too wide to
choose positive density versus decay. Twenty finite exact points cannot
resolve either infinite alternative. A deficit beyond eight rejects that
specific uniform credit; it does not refute seed-dependent positive slope,
or a weaker universal free-system slope.

## Controls and proof obligations

The full zero target retains a zero-boundary, zero-completion witness for both
Rule30 and Rule90. Rule90 uses N free full temporal right-column bits because
the OR erasure does not apply. Tests replay its gates and show why that
control cannot silently reuse ceil(N/2) inputs. The main encoding uses OR
explicitly. Those controls do not themselves establish an asymptotic result.

A small exact identity is `e1=1-u0, e2=u0`, so every free completion has one
one in its first two positions. This gives a bounded weight credit only;
restarting that pair at later spatial cuts is not justified. No valid
potential with bounded terminal credit was obtained. A positive-density
proof would still need that all-horizon argument, and a zero-density answer
would need an infinite construction or another proof.

## Reproduce

Use the repository's Python environment with python-sat installed. After A's
reference module is integrated:

```bash
python -m unittest tests.test_free_u_density -v
python experiments/free_u_density.py --verify runs/p2-free-density-2026-10-15/results.json --drat-trim third_party/drat-trim
python experiments/free_u_density.py --cadical third_party/cadical --drat-trim third_party/drat-trim --timeout 8 --budget 240 --output /tmp/free-u-reproduction
```

The original run used `--reference` pointing to A's concurrent worktree;
the JSON pins its content hash. The final script reuses earlier SAT words
during the sweep; scheduling and solver timeouts can alter later brackets,
while the saved exact claims are deterministically reverified. Verification
extracts the archive to a temporary directory, checks safe member names,
regenerates every CNF, checks hashes, replays all witnesses, and externally
checks all UNSAT proofs. SHA256SUMS pins the results and compressed archive.

Not run: full repository suite/verifier (parent owns it), GPU, Lean, infinite
density/exclusion proof, new network installations, or a fitted growth law.
