# Physical even-column factors and finite language relaxations

Here `u_s=x[1,2s]`, with prescribed centre `c_t=t mod 2`, and initial
right cell 1 in integer bit zero. This is a forced-centre necessary
condition, not a result about the single-seed centre column.

## Exact local factor certificates

`experiments/u_language.py` enumerates all `r<2^20`, observing the first
32 `u` bits of each row, and 20,000 seeded random 256-bit rows, observing
256 `u` bits per row. The vectorized exhaustive computation truncates to
64 spatial bits; every observation has time at most 62, so a missing
cell beyond position 64 cannot affect it. Independent naive truth-table
simulation and Agent A's reference check the orientation and outputs.

The observed factor counts at lengths 1 through 12 are

```
2, 3, 5, 8, 12, 17, 25, 36, 50, 68, 91, 119.
```

The following are the minimal absent factors relative to that sample.
Each is also excluded by its **exact finite dependency cone**, with a
retained CNF and ASCII DRAT proof checked by `drat-trim`:

| Length | Minimal forbidden factors |
| --- | --- |
| 2 | `11` |
| 5 | `00000` |
| 6 | `101001` |
| 7 | `0100101` |
| 9 | `010010001` |
| 10 | `0101000101`, `0101010000` |
| 11 | `01010001001`, `10010001001` |
| 12 | `010010000101`, `010100010001`, `100100010000` |

A length-L observation depends on exactly `2L-1` initial right cells.
The encoder leaves every one of these initial bits independent, fixes
only the centre, and applies the elementary Rule-30 truth table on the
shrinking triangle. Its output constraints demand the candidate word.
UNSAT therefore excludes the word for **every** physical right snapshot,
including infinitely supported ones. Starting at any even time gives the
same cone and centre phase, so these bans hold for every factor of `u`.
The two original candidates are genuine local bans, not sampling artifacts.

These certificates and the observed factors establish the exact factor
sets **through length 12**. To see completeness, take any absent word of
length at most 12: if it is not minimally absent, it has a proper absent
prefix or suffix. Descend until reaching one of the listed minimal words,
all of which are certified impossible. Every observed word is already
realized by a sampled finite row. The JSON retains the finite sampling
parameters and certificate hashes.

There is no all-length characterization here. Minimal absences continue
at length 12. Neither a finite experimental cutoff nor a gap at length 8
would prove regularity or the shift-of-finite-type property.

## The proposed L1/L2/L3 ladder is insufficient

Let L1 ban `11`, L2 additionally ban `00000`, and L3 additionally ban
`101001` and `0100101`. The explicit relaxed input

```
u = (01)^45 000101
N = 191, W_N = 26, N - 7 W_N = 9
```

belongs to all three finite languages. Agent A's completion recurrence
independently reproduces its weight. Thus **none of L1, L2, L3 implies
`7 W_N >= N-8` even through N=191**. This is a counterexample to a language
relaxation, not a physical counterexample: it contains the separately
certified forbidden word `0101000101` starting at zero-based offset 86.
It cannot be an actual physical `u` prefix.

The bounded SAT scan asks exactly `W_N <= floor((N-9)/7)` for each
N=1..300 and each language, with 50 ms solver interrupts and a global
300-second exploration budget. Its finite independent input class has
`ceil(N/2)` bits before restrictions; the JSON also counts each restricted
language exactly by suffix dynamic programming. The purpose is
**exact-exclusion**, with no complexity or randomness inference. UNSAT
answers have retained independently checked proofs; SAT witnesses are
replayed through Agent A's reference. UNKNOWN and NOT_RUN_BUDGET establish
nothing. This scan does not compute exact maxima or prove any all-N bound.
Its final coverage and status counts are recorded in `ladder/summary.json`.
The scan completed all 900 queries; the verified contiguous prefixes are
N<=51 for L1, N<=60 for L2, and N<=54 for L3. Higher checked isolated
instances do not close the intervening UNKNOWN gaps.

L4 adds all twelve certified minimal forbidden factors through length 12.
Targeted 10-second CaDiCaL queries at N=191 and N=300 both returned UNKNOWN.
They retain their compressed CNFs and hashes, but establish no L4 bound.
The proposed uniform L4 inequality remains a conjecture.

A separate L1 checkpoint at N=79 excludes W<=9 with checked DRAT.
Together with the witness `u=(01)^17 0^6` of weight10, this certifies the
exact minimum W79=10 in L1. It does not establish a larger-range bound.

## Controls, replay, and limits

With the OR latch replaced by Rule 90, `11` is realizable (right integer 5).
With the centre constantly zero, Rule 30 admits `00000` (right integer 0).
Thus the exclusions use both the Rule-30 nonlinearity and the alternating
centre hypothesis; they do not apply indiscriminately to zero initial data.

Reproduce factors and local certificates with repository-installed tools:

```bash
python experiments/u_language.py --out runs/p2-u-language-replay \
  --solver third_party/cadical --checker third_party/drat-trim
python experiments/u_language.py --verify --out runs/p2-u-language-2026-10-15 \
  --checker third_party/drat-trim
python -m unittest tests.test_u_language -v
```

Reproduce the threshold scan with `python experiments/u_language.py --ladder
--out runs/p2-u-ladder-replay --checker third_party/drat-trim`. The default
encoder and reference paths are the sibling experiment modules; `--encoder`
and `--reference` allow explicit selections. `--verify` regenerates saved
CNFs, checks hashes and every proof, and replays all realizing witnesses;
it does not repeat the million-row scan. Proof archives, when present, extract into their
containing directory; hashes refer to the original uncompressed files.

Not run by this worker: full verifier or full unittest suite, Lean, GPU,
all-length regularity/SFT decision, exact language minima through 300,
longer-than-12 exact factor classification, or a uniform density proof.
