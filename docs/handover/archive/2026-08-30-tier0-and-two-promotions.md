# Handover — Tier 0 Landed, `s*(n)` Certified, `g(n)` Measured

**Date:** 2026-08-30
**Branch:** `claude/keen-sagan-21rzbr`, stacked on `fix/data-hygiene` (PR #18), PR #19
**Read first:** [`2026-08-30-next-session-plan.md`](2026-08-30-next-session-plan.md)
(the plan this session executed), then `AGENTS.md` and `docs/CLAIM_LEDGER.md`.

**Start here:** `python tools/verify_all.py` — one command, ~2 s.
**Planning a paid run?** [`docs/COMPUTE_PLAN.md`](../COMPUTE_PLAN.md) first — the standing plan's VRAM premise is wrong, and most of the best work is free.

---

## TL;DR

Plan items **1, 2–6, 7, 8, 11 and 13** are done, from a GPU-less Linux
container. Item **14** was taken **in parallel from the owner's laptop**
(session `01F8jNAdw6Lcn6dqQ5y2k5Cq`, commits `3aedad9`, `75c75da`, `a1a1f0e` on
this branch) and is **partly** done — the tool exists and its gates pass, but
the walk has not reached the prediction window and deliberately carries no
ledger row yet. Items **9**, **10**, **12** and **15** are untouched.

Read the "Work from the laptop session" section below before planning: it fixed
a defect in this session's own code, added two verification stages, and left
item 14 at a well-defined halfway point.

Five findings that change what the next session should do:

1. **The plan's item-7 recipe is broken.** `Cadical153(with_proof=True)` emits
   proofs that `drat-trim` rejects. Following it as written would have shipped a
   Certificate backed by refutations that fail their own checker.
2. **`tools/make_manifest.py` was silently deleting the bitstream hash anchors**
   on any machine without the gitignored `.bin` files — i.e. every fresh clone.
3. **The SAT work is ~28x cheaper than assumed**, which re-costs item 10 from
   "blocked on solver time" to "probably minutes".
4. **The 10M bitstream can be regenerated without a GPU.** The independent CPU
   reference reproduces it byte-for-byte (145 min), which is how items 8 and 13
   got unblocked on a GPU-less machine. Do not assume "needs the GPU box"
   without checking whether the CPU reference suffices.
5. **The I–L neural suite is blind to long-lag XOR structure** — the structure
   class Rule 30 most plausibly has. The neural nulls are much narrower than
   the README used to imply.

---

## Environment — read before planning compute

This session ran in an **ephemeral Linux container: 4 CPUs, 15 GB, no GPU, no
cupy**, cloned fresh from GitHub. Two consequences worth knowing in advance:

- `gpu/rule30_sim.py` cannot run at all. Anything needing the canonical
  bitstreams is impossible here. `data/center_col_10M.bin` and
  `center_col_46M.bin` are gitignored and are not in a fresh clone. (An earlier
  draft of this file said they "live on the Windows box". That phrasing was an
  inference, and `a1a1f0e` corrected a ledger row that had leaned on it to
  assert the original run's host OS. Nothing in this repo records the host OS
  of the GTX 1060 run. Do not re-introduce the claim.)
- `verify_data.py` reports **165 verified, 2 absent** rather than 167. That is
  the correct fresh-clone state, not a failure.

`pip install` works (numpy, python-sat installed on demand), and outbound git
clone works, so building a SAT toolchain from source is fine here.

**Correction to an earlier draft of this file:** it said items 8, 13 and 14
belong on the GPU box. That was wrong. `tools/gen_golden_reference.py` is a
CPU-only implementation that regenerates the 10M center column in ~145 minutes
and, as it turned out, reproduces the canonical artifact **byte-for-byte** — so
`data/center_col_10M.bin` can be recreated anywhere, and items 8 and 13 were
both completed here. `torch` and `sklearn` install fine from PyPI (though
`download.pytorch.org` is blocked by the egress proxy — use plain PyPI).

What genuinely still needs better hardware is the **46M** stream: extending the
independent reference to it costs ~21x the 10M run, about **50 hours** of CPU.

---

## What landed

### Items 2–6 — Tier 0 tooling

| Tool | What it stops |
|---|---|
| `tools/lint_bitorder.py` | a sixth bare `np.unpackbits` |
| `tests/test_bitorder.py` | the LSB/MSB convention drifting silently |
| `tools/make_manifest.py` (rewritten) | anchors being dropped; CRLF manifests |
| `tests/test_manifest_determinism.py` | `csv.writer` number eleven |
| `tools/lint_ledger.py` | the ledger citing evidence that is not there |
| `tools/verify_all.py` | trust costing four commands and tribal knowledge |
| `tools/build_sat_toolchain.sh` | the DRAT toolchain being un-rebuildable |

Two **live defects** were found by writing these, not hypothesised:

- **`make_manifest.py` deleted anchors.** It wrote `# UNANCHORED` for any
  canonical bitstream absent from local disk. Since those are gitignored,
  regenerating the manifest on a fresh clone erased the recorded hashes
  `6f8670b4…` and `f281f11b…` — the only anchors tying the A–H results to
  specific bytes. Anchors are now carried forward, and the emitted line is
  identical whether or not the file is present, so the manifest is
  machine-independent.
- **`data/MANIFEST.sha256` was CRLF.** `Path.write_text` translates `\n` to
  `os.linesep`, so the manifest came out CRLF on Windows and LF elsewhere:
  byte-different files with identical content. Exactly the `csv.writer` defect
  from August, sitting in the integrity file itself. Now written with
  `newline=""`.

Design note for anyone extending the lints: the bit-order lint is **stricter
than the plan specified**. The plan keyed it off whether a file mentions
`center_col`; a new reader reaching the bitstream through a helper would slip
past that. It now flags every bare call under `experiments/`, `gpu/`, `tools/`,
with a mandatory-reason `# bitorder-exempt: <reason>` escape. Five sites are
annotated — two in `gen_golden_reference.py` (deliberate MSB-first), three that
re-pack an already-decoded array for zlib.

Both lints' tests **reproduce the historical failures**. A lint that passes on
today's tree but would not have caught the incident it was written for is worth
nothing.

### Item 7 — `s*(n)` is now a Certificate

`docs/experiment-logs/2026-08-30-dfao-drat-certification.md`,
`data/prize/2026-08-30-dfao-drat-proofs.json`.

```
207 / 207 lower-bound instances UNSAT and DRAT-verified   0 failures
 54 /  54 upper bounds certified by an independent evaluator
167 / 207 had been UNSAT_IMPLIED, now proved directly
133 s wall clock
```

**The plan's recipe does not work.** `Cadical153(with_proof=True)` returns a
proof with no terminating empty clause; `drat-trim` says `s NOT VERIFIED` on
instances where the same solver as a standalone binary produces a proof that
verifies. pysat's source carries `# stripping may cause issues here!` at the
point where it `.strip()`s a *binary* DRAT buffer as text. The flag is present
and returns a large plausible proof, so nothing complains — which is precisely
why this needed checking rather than wiring up.

Proofs now come from a standalone `cadical` binary. That removes pysat from the
trusted base, which was the objective all along.

Scope was widened twice, deliberately:

- **Every implied verdict re-proved.** At `n=48 msd` the entire lower bound
  rested on the monotonicity lemma with *no direct UNSAT at all*. The lemma is
  sound, but it was doing more work than it looked. It is no longer load-bearing.
- **Upper bounds certified too.** They rested on witnesses re-verified with
  `prize_lab.run_dfao` — the same module whose `dfao_sat_cnf` produced them, so
  a shared mis-encoding would validate itself.

### Item 1 — golden reference extended to all 10M bits

`docs/experiment-logs/2026-08-30-golden-reference-10M.md`. The CPU reference,
sharing no code, no bit-order convention and no tape geometry with `gpu/`,
regenerated all 10,000,000 bits and its LSB-first repacking is **byte-identical**
to the canonical artifact (`6f8670b4…` both sides). The independent check is now
a hash comparison over the whole stream instead of a diff over the first 10%.

Those same bytes now have **three** independent reproductions, and they cover
different failure modes, which is why the combination is worth more than any
one of them:

| reproduction | excludes |
|---|---|
| GTX 1060 (Pascal), original anchor | — the baseline |
| RTX 2050 (Ampere, CUDA 12.9), laptop session `3aedad9` | hardware / driver / toolchain-dependent bugs |
| CPU reference, no shared code or conventions | shared-source bugs, which reproduce identically on any hardware |

The second axis alone cannot catch a bug in shared source; the third alone
cannot catch a toolchain bug. Together they close both. The ledger row states
this scoping explicitly and makes no claim about the original run's host OS,
which is not recorded anywhere in the repo.
`verify_all.py` went from 7 passed / 2 skipped to **9 passed / 1 skipped** (the count also grew by the laptop session's clone-integrity stage).

### Item 13 — exact exhaustive period search

`docs/experiment-logs/2026-08-30-period-search-exact.md`. All **9,999,936**
candidate periods decided **exactly**; zero survived even a 64-bit window. The
plan framed this as VRAM-bound with a Bonferroni threshold to manage, but a
period is refuted by one mismatch — so the sampled test was answering an exact
question statistically. The scan took **2.1 s** on four CPU cores. Problem 1 is
bound by the cost of *simulating* a longer column, not searching it.

### Item 8 — I–L re-run, and a blind spot

`docs/experiment-logs/2026-08-30-rerun-il-bitorder.md`. The bit-order bug
changed nothing (largest paired difference 0.0024 across 24 configurations), so
I–L are un-retracted. But the positive controls the originals never had turned
up something worse than the bug: on `s[i] = s[i-13] ⊕ s[i-27]` — fully
determined by 27 bits inside a 64-bit context — every model fails, across six
budgets spanning 5x the parameters, more data, more epochs and short contexts.
The same suite learns period-31 and `s[i-3] ⊕ s[i-5]` instantly.

Rule 30 is left-permutive and Experiment S measured `L(n) = n/2`, so this blind
spot covers the most plausible structure class. I, K and L are reinstated but
**scoped down** in the README. Experiment J is exempt and stronger: its probe
detects a 0.30→0.70 bias ramp at 57.7% against a 10% floor.

### Item 11 — `g(n)`, the smallest-grammar curve

`docs/experiment-logs/2026-08-30-grammar-min-size-curve.md`,
`data/prize/2026-08-30-grammar-min-size-curve.json`.

```
      n  center  rand min  rand max   TM  g_null  c*log2(n)/n  r*log2(n)/n
     64      25        22        29   15      12        2.344        2.531
   4096     746       725       740   34     300        2.186        2.139
  65536    8175      8155      8228   47    3210        1.996        1.997
```

Inside the 7-seed band at 8 of 11 lengths, above at 3, **never below**. Both
curves sit at the `2n/log2(n)` rate of an incompressible string and *converge*
(ratio 0.9996 at n=65536). Thue-Morse stays flat at 47 rules — a 174x
separation — so detection power does not decay with scale.

Graded **Robust observation, not Certificate**: exact smallest grammar is
NP-hard, so this is what Re-Pair found. A shortcut hiding inside the heuristic's
worst-case logarithmic gap is not excluded.

---

## Work from the laptop session (`3aedad9`, `75c75da`, `a1a1f0e`)

Merged into this branch. Three things, and the first is a correction to work
done in this session.

### It found a live defect in this session's tests

`experiments/dfao_min_states.py` raised `SystemExit` at **module scope** when
pysat was absent. `SystemExit` inherits from `BaseException`, not `Exception`,
so the `except Exception -> unittest.SkipTest` guards written here in
`tests/test_dfao_drat_proofs.py` and `tests/test_grammar_min_size.py` could not
catch it. Measured with pysat blocked: **65 tests ran instead of 88**, and the
run printed `FAILED (errors=2)` — which reads as two broken tests, not
twenty-three absent ones.

The 23 that vanished are the ones checking *which* instances the `s*(n)`
certificate must prove: the coverage logic that stops a certificate reporting
"207/207 verified" over a silently smaller set. None of them need a solver.
That is the worst possible 23 to lose silently, and the guards written here
were the reason they were lost quietly rather than loudly.

Fixed at source (a soft `PYSAT_AVAILABLE` flag, `SystemExit` moved into
`main()`) and pinned by `tests/test_import_safety.py`, an AST scan forbidding
`raise SystemExit` reachable at import time across `experiments/`, `tools/`
and `gpu/`.

**The generalisable lesson:** an `except Exception` guard around an optional
import is not a guard. Catch `(Exception, SystemExit)` explicitly, or better,
do not let an import decide to exit a process.

### `tools/check_clone_integrity.py` — a Windows trap worth knowing

`core.autocrlf=true` beats `.gitattributes` (`data/** -text`) during git
clone's **initial checkout only**. So 162 tracked `data/` files land CRLF and
fail their manifest anchors, while `git status` reports a clean tree because
they round-trip. It presents as a wall of hash mismatches indistinguishable
from corrupted data — the exact failure `.gitattributes` was added to prevent,
arriving through a door it does not cover.

Now `verify_all.py`'s first stage. It detects the `i/lf w/crlf` signature and
prints the repair, and deliberately does not flag `i/crlf` files, which are
CRLF in the index and therefore byte-identical everywhere.

### Item 14 — the period-32 pattern-map walk, halfway

`experiments/pattern_map_walk.py`. The ledger's stated promotion path for
`period(d) ~ 2*log2(d)` is "confirm the 32->64 event predicted near
`d ~ 8.6e9`", and `period16_walk.py` had rejected the map-walk route because
the map is partial at a zero word.

The commit dissolves that objection rather than working around it: in the
one-period composite `x[t+1] = u[t] XOR (v[t] OR x[t])`, wherever `v[t] = 1`
the update is constant in `x[t]`, so `v != 0` makes `w_d` unique — no branch.
Only `v == 0` branches, and that is a collision, a ~`2^-32` per-diagonal event.
Branch points and doubling events are therefore the *same* collisions separated
by parity, so a walk to `d ~ 1e10` expects only a couple. Branches are
**followed** rather than resolved: if every surviving branch doubles in the
same window, the transient that cannot be computed did not matter.

It also records a parity conflation caught mid-implementation, which is easy to
repeat: **minimal-period** parity answers "does the period double?" (Lemma B),
**full-p** parity answers "does a period-p solution still exist?" (the walk).
At `d=87866` these disagree — odd on the minimal period, even over all 32 bits —
so halting on the wrong one stops the walk at the known 16->32 event and
reports it as terminal.

Gates pass at `--diagonals 100000`: period histogram matches, first period-32
diagonal is 87867 (agreeing with the direct-simulation refutation), Lemma A
clean over 8,000 tests, map vs simulation clean over 8,000 consecutive
diagonals. **Walk validated to `d = 5e7` at 7.7 M steps/s, 0 branch points, no
doubling yet** — as expected for a `2^-32` event.

**Not run to the prediction window, and deliberately no ledger row until it
is.** That is the right call and should not be "tidied up" by writing a row.

---

## Traps — new ones, in addition to the 2026-08-29 list

- **`Cadical153(with_proof=True)` is not usable for certification.** Its proofs
  do not check. If you need DRAT, use `tools/build_sat_toolchain.sh`.
- **A negative control on a propagation-trivial instance tests nothing.** The
  first version of the DRAT self-test truncated the proof of
  `thue-morse n=64 s=1`, which is UNSAT by unit propagation, so drat-trim
  derives the empty clause from the CNF and accepts *any* proof — including an
  empty one. The control passed while testing nothing. Truncation controls need
  an instance that requires real search.
- **`SKIP` is not `PASS`.** `verify_all.py` separates them deliberately. On a
  machine without the bitstreams, two stages check nothing at all.
- **Six markdown files were cp1252**, not UTF-8 (`0x97` em dash). Converted, and
  a test now pins it. If you author docs on Windows, watch the encoding.
- **`tools/gen_golden_reference.py` does not checkpoint.** It writes only after
  the full loop — confirmed the hard way by a 145-minute run. A 46M run would be
  ~50 hours and is not realistically attemptable until this is fixed.
- **`periodic` is not a positive control for a non-stationarity probe.** A
  periodic stream is stationary, so chance accuracy is the *correct* answer and
  the control passes while testing nothing. Same shape of error as the DRAT
  truncation control below: a control on an input where the trivial answer is
  also the right answer measures nothing.
- **`except Exception` does not catch `SystemExit`.** An optional-dependency
  guard written that way silently deleted 23 tests here. See the laptop-session
  section above. Catch `(Exception, SystemExit)`, and do not `raise SystemExit`
  at import scope — `tests/test_import_safety.py` now forbids it.
- **`core.autocrlf=true` beats `.gitattributes` on a fresh clone's first
  checkout.** 162 tracked `data/` files land CRLF, fail their anchors, and
  `git status` still reports clean. `tools/check_clone_integrity.py` is
  `verify_all.py`'s first stage precisely because this looks like corruption.
- **Minimal-period parity and full-period parity answer different questions.**
  Lemma B uses the minimal period; the period-32 walk needs parity over all 32
  bits. They disagree at `d=87866`, and using the wrong one halts the walk at
  the already-known 16->32 event.
- **`pkill -f <pattern>` matches your own shell.** It killed the running shell
  twice in this session because the command line contained the pattern. Use
  `ps -eo pid,args | grep "[p]attern"` or kill by PID.

---

## Now open, ranked

1. **Finish the item-14 walk — it is ~26 minutes of CPU from a ledger row.**
   `experiments/pattern_map_walk.py` is validated to `d = 5e7` at 7.7 M steps/s
   with 0 branch points; the prediction window is `d ~ 8.6e9` and the commit
   measures reaching `d = 1.2e10` at about **26 min of CPU per branch**. This
   is the cheapest open item in the repo by a wide margin, and it closes the
   ledger's own stated promotion path for `period(d) ~ 2*log2(d)`.

   Two cautions carried from that commit. Branches are *followed*, not
   resolved, so the argument only holds if every surviving branch doubles in
   the same window — check that, do not assume it. And halting must use
   **full-32-bit** parity, not minimal-period parity, or the walk stops at the
   already-known 16->32 event and reports it as the answer.

   Keep in view what the ledger already says: left-edge structure is **disjoint
   from the prize object** (`settle(T) ~ 1.34*T > T`), so this is beautiful
   mathematics that cannot by itself yield a center-column shortcut. Cheap and
   conclusive, but not prize-facing.

2. **Item 10 — extend `s*(n)` past n=48, and it is much cheaper than the plan
   assumed.** The standalone solver did 207 solves in **133 s** where the pysat
   harness took **3747 s** for 105, with proof logging on. The 120 s
   per-instance timeout that stopped the curve at n=48 was substantially
   measuring harness overhead, not instance difficulty. Re-measure before
   budgeting. **Keep the state budget growing with `n`** — raising `n` at fixed
   `--max-states` makes the negative more vacuous, which is how the original
   DFAO claim got retracted.
3. **Item 9 — `s*(n)` in bases 3, 4, 5.** The new Certificate covers base 2
   only, and automaticity is base-dependent (Cobham), so the claim is narrower
   than it reads. `experiments/dfao_min_states.py` already takes `--base`; the
   counting null shifts with `b`, so re-derive the band per base.
4. **Exact `g*(n)` at small n.** The counting null and Re-Pair bracket the true
   curve about 2.5x apart. Solving the exact smallest grammar via SAT/ILP at
   n <= 64 — the same design as `s*(n)`, and the same route to a Certificate —
   would close it. Given how cheap the DFAO instances proved to be, this looks
   tractable.
5. **A parity-capable estimator on the center column.** This is the direct
   consequence of item 8's blind spot. Berlekamp–Massey (already in Experiment
   S) and GF(2) rank methods find exactly what the neural suite misses, and
   they should be pointed at the column with the same control discipline. The
   answer to a model-class blind spot is a different method, not a bigger
   network.
6. **Re-run the detection-power ladder at GPU scale** (5–7M bits, d_model 256,
   context 1024). If a large model learns `lfsr_13_27`, the I/K/L caveat
   weakens; if it still fails, the caveat hardens into a statement about what
   this class of experiment can establish at all.
7. **Extend the independent reference to 46M** (~50 h CPU), or state the 10M
   horizon explicitly in the 46M ledger row rather than letting it inherit
   confidence it has not earned.
8. **Report the pysat proof defect upstream.** Any project using
   `with_proof=True` to certify UNSAT is certifying nothing.

## Do not

- Push `n` higher on `g(n)`. The curves have already converged to three decimal
  places; another octave adds reach, not information. The binding limitation is
  the upper-bound gap.
- Trust a `verify_all.py` run with `SKIP` lines as a verified repo.
- "Fix" `tools/gen_golden_reference.py` to match the kernel's bit order.
