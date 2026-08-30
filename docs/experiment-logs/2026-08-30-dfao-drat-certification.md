# Experiment Log

- Date: 2026-08-30
- Title: Machine-Checkable DRAT Certification of the `s*(n)` Curve
- Claim Level: **Certificate** — every bound is backed by an artifact a third
  party can check without trusting this repo, its solver, or its encoder.
- Run date: 2026-08-30
- Artifact: `data/prize/2026-08-30-dfao-drat-proofs.json`
- Verifier: `python experiments/dfao_drat_proofs.py --curve data/prize/2026-08-15-dfao-min-state-curve.json --out /tmp/recheck.json`

## Goal

Promote the `s*(n)` minimal-DFAO-size curve
(`docs/experiment-logs/2026-08-15-dfao-min-state-curve.md`) from Robust
observation to Certificate. That log states its own ceiling plainly:

> Not a Certificate: the SAT side ships a re-verified witness, but the UNSAT
> lower bounds rest on trusting the solver.

Two things had to be removed from the trusted base: **CaDiCaL's UNSAT verdicts**
(the lower bounds) and, on inspection, **`prize_lab`'s own evaluator** (the
upper bounds — see "Scope creep, and why" below).

## The planned recipe does not work

`docs/handover/2026-08-30-next-session-plan.md` item 7 specifies:

> The installed `pysat` exposes it — `Cadical153(with_proof=True)` (confirmed
> present in the `__init__` signature). Emit the DRAT log for each UNSAT
> instance, check it with `drat-trim`.

The flag is present. What it produces does not verify.

```
PHP(7,6), 133 clauses, genuinely UNSAT:
  pysat Cadical153(with_proof=True).get_proof()  -> 1815 lines, last line 'd 2 28 1 0'
  drat-trim php.cnf php_pysat.drat               -> s NOT VERIFIED   (rc=1)

  cadical --no-binary php.cnf php_cadical.drat   -> 1813 lines, last line '0'
  drat-trim php.cnf php_cadical.drat             -> s VERIFIED       (rc=0)
```

The pysat proof has **no terminating empty clause**; the standalone one does.
Appending `0` by hand does not rescue it, and reading the buffer after the
solver flushes and closes (via a `dup`ed fd) returns the same 13548 bytes, so
it is not a truncated write. pysat's own source, at the line that produces
this, reads:

```python
# stripping may cause issues here!
return Solver._proof_bin2text(bytearray(self.prfile.read()).strip())
```

`.strip()` on a *binary* DRAT buffer accounts for one lost trailing byte here,
not for the missing empty clause, so the root cause is upstream of that — but
the comment is a fair warning.

**This is the load-bearing finding.** Following the recipe as written, seeing
`with_proof=True` accepted and a 1815-line proof come back, and recording
"DRAT proofs emitted" would have produced a Certificate resting on refutations
that fail their own checker. Nothing in the pipeline would have complained: the
proofs exist, they are large, and they are wrong.

Proofs therefore come from a **standalone solver binary** (`cadical` 3.0.1,
built by `tools/build_sat_toolchain.sh`). That is strictly better for the
purpose anyway: it removes pysat from the trusted base entirely, which was the
objective, and it leaves a reader needing only two standard tools rather than a
pinned pysat.

## Scope creep, and why

The plan scopes item 7 to the UNSAT lower bounds. Two extensions were necessary
for the artifact to certify what a Certificate row would claim:

**1. Every implied verdict is re-proved.** The original run discharged most
instances as `UNSAT_IMPLIED` via monotonicity of `s*` in `n` — if no `s`-state
DFAO fits an `n`-bit prefix, none fits a longer one, since the constraint set
for the shorter prefix is a subset. The argument is sound. It is also a hand
argument sitting in the trusted base, and it was doing more work than it looks:

```
center/msd n=48:  direct UNSAT 0,  UNSAT_IMPLIED 11
```

At `n=48` MSD the entire lower bound `s*(48) >= 12` rested on monotonicity with
**no direct UNSAT verdict at all**. All 207 instances are now solved directly on
their own full prefix; 167 of them (87 for the center column) had previously
been implied. The monotonicity lemma is no longer load-bearing.

**2. The upper bounds are certified too.** `s*(n) <= k` rests on a witness DFAO
reproducing the bits. The original run re-verified each witness — with
`prize_lab.run_dfao`, the same module whose `dfao_sat_cnf` produced the witness.
A mis-encoding shared by both would validate itself. `experiments/dfao_drat_proofs.eval_dfao`
re-implements DFAO evaluation from the definition and shares no code with the
encoder, the same independence principle as `tools/gen_golden_reference.py`
versus `gpu/`. It agrees with `run_dfao` on all 54 recorded witnesses, and all
54 reproduce their prefix.

Certifying only the lower half would have left the row half-promoted.

## Method

For every row of the curve with an exact `s*(n)`, and every `s` in `1..s*-1`:

1. Regenerate the bit prefix and **assert its sha256 matches the recorded one**
   (66/66 do), so the instance provably concerns the same sequence.
2. Regenerate the DIMACS with `prize_lab.dfao_sat_cnf` and hash it.
3. `cadical --no-binary instance.cnf instance.drat`; require exit 20 (UNSAT).
4. `drat-trim instance.cnf instance.drat`; require `s VERIFIED`.
5. Separately, evaluate the row's witness DFAO independently over `0..n-1`.

### Detection power of the checker

A checker that prints VERIFIED unconditionally is worse than none, so
`--self-test` asserts both directions. The **first version of the negative
control was vacuous** and is worth recording: it truncated the proof of
`thue-morse n=64 s=1`, which is UNSAT by unit propagation alone. drat-trim
derives the empty clause from the CNF itself there, so it accepts *any* proof,
including an empty one — the control passed while testing nothing. The controls
now are:

| Check | Result |
|---|---|
| `thue-morse n=64 s=2 msd` is SAT | PASS |
| `center n=16 s=4 msd` is UNSAT (needs real search, 155 proof lines) | PASS |
| its DRAT proof verifies | PASS |
| drat-trim **rejects** that proof truncated by half | PASS |
| drat-trim **rejects** refuting a satisfiable formula (`{x1},{x2}` ⊢ ⊥) | PASS |

## Result

```
207 / 207 instances UNSAT and DRAT-verified          0 failures
 54 /  54 lower bounds certified
 54 /  54 upper bounds certified (independent evaluator)
 54 /  54 values of s*(n) fully certified
167 / 207 had previously been UNSAT_IMPLIED, now proved directly
```

Center column, per prefix:

```
 dir     n   s*   instances  re-proved  solve_s  check_s  proof_MB
 msd     8    3       2          0        0.01     0.18       0.0
 msd    12    4       3          2        0.01     0.28       0.0
 msd    16    5       4          3        0.01     0.37       0.0
 msd    20    6       5          4        0.02     0.48       0.0
 msd    24    8       7          5        0.08     0.71       0.3
 msd    28    9       8          7        0.26     0.94       1.2
 msd    32   10       9          8        1.25     1.90       6.0
 msd    40   12      11          9       18.74    19.37      74.4
 msd    48   12      11         11       23.15    24.76      86.0
 lsd     8    4       3          0        0.02     0.27       0.0
 lsd    12    5       4          3        0.01     0.37       0.0
 lsd    16    6       5          4        0.02     0.47       0.0
 lsd    20    6       5          5        0.02     0.48       0.0
 lsd    24    7       6          5        0.04     0.64       0.1
 lsd    28    8       7          6        0.26     0.84       1.1
 lsd    32    9       8          7        1.06     1.52       4.5
 lsd    40   10       9          8        5.18     4.50      20.0
```

Total across all three sequences: 58.3 s solving, 73.3 s checking, **133 s
wall clock**, 232 MB of proofs generated, largest single proof 62.6 MB.

**Independent cross-check of the encoder.** Every directly-solved instance that
the August run also solved directly reproduces its exact variable and clause
counts (`encoding_matches_original_run: true`). The CNF generator on this branch
and the one that ran on 2026-08-29 agree bit-for-bit on instance size.

## What is archived, and what is not

Proofs total 232 MB and are **not** stored. The artifact records each
instance's `proof_sha256`, and says in the artifact itself that this is
provenance for one run and **not** a reproducibility anchor — a different
solver emits a different, equally valid refutation.

The reproducible object is the **CNF**: `cnf_sha256` per instance. An
independent checker regenerates the instance from `(sequence, n, direction,
base, states)`, confirms the hash, and runs *its own* solver and *its own* DRAT
checker. That is a check requiring trust in neither this repo nor its
toolchain, which is what separates a Certificate from a well-run experiment.

## Interpretation

The scientific content of `s*(n)` is unchanged — this run re-derives exactly the
same curve. What changed is its epistemic status. Before: "CaDiCaL says no
`s`-state DFAO fits, and we re-ran the witnesses with the same library that
found them." After: "here are 207 refutations an independent checker accepts,
and 54 witnesses an independent evaluator confirms."

The negative therefore now stands as a Certificate: **on prefixes to n=48, in
base 2, both digit directions, the Rule 30 center column admits no DFAO smaller
than the measured `s*(n)`, and that fact is machine-checkable.**

Scope is unchanged and still narrow: n ≤ 48, base 2 only, DFAO class only. The
Admission Rule is satisfied as before — measured minima (10 at n=32) sit above
the counting-null band (6–7), so these are real minima and not an artefact of a
class too small to fit anything.

## An unexpected practical result

The standalone binary did **more** work than the August run — 207 full solves
versus 105 solver calls, most of the extra ones being the large-`n` instances
monotonicity had skipped — in **133 s against 3747 s**, a ~28x speedup, with
proof logging switched on. The 120 s per-instance timeout that stopped the
original curve at n=48 was not measuring the difficulty of the instances; it
was substantially measuring pysat's in-process overhead and the child-process
harness built around its non-interruptible solve.

This materially changes the cost estimate for **item 10** (extend `s*(n)` past
n=48). The plan treats n=56/64 as blocked on solver time. On this evidence the
next few prefix lengths are likely minutes, not hours — though instance
difficulty does grow steeply in `s` (n=48 s=11 alone is 23 s and 86 MB), so
this is an invitation to measure, not a promise.

## Next Step

1. **Item 10, re-costed.** Extend the curve past n=48 using the standalone
   solver. Keep the state budget growing with `n` — raising `n` at fixed
   `--max-states` makes the negative *more* vacuous, which is how the original
   DFAO claim got retracted.
2. **Item 9, bases 3/4/5.** Automaticity is base-dependent (Cobham), so this
   Certificate is narrower than it reads. The counting null shifts with `b`
   (`|M(s,b)| <= s^(s*b) * 2^s`), so re-derive the band per base.
3. Report the pysat proof defect upstream. `Cadical153(with_proof=True)`
   silently produces proofs no DRAT checker accepts, and any project using it
   to certify UNSAT is certifying nothing.
