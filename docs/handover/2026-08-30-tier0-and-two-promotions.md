# Handover — Tier 0 Landed, `s*(n)` Certified, `g(n)` Measured

**Date:** 2026-08-30
**Branch:** `claude/keen-sagan-21rzbr`, stacked on `fix/data-hygiene` (PR #18), PR #19
**Read first:** [`2026-08-30-next-session-plan.md`](2026-08-30-next-session-plan.md)
(the plan this session executed), then `AGENTS.md` and `docs/CLAIM_LEDGER.md`.

**Start here:** `python tools/verify_all.py` — one command, ~2 s.

---

## TL;DR

Plan items **1, 2–6, 7, 8, 11 and 13** are done. Item **14** was landed in
parallel by another session on this same branch. Only item **9**, item **10**,
item **12** and item **15** remain untouched.

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
  `center_col_46M.bin` are gitignored and live only on the Windows box.
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
`verify_all.py` went from 7 passed / 2 skipped to 9 passed / 1 skipped.

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
- **`pkill -f <pattern>` matches your own shell.** It killed the running shell
  twice in this session because the command line contained the pattern. Use
  `ps -eo pid,args | grep "[p]attern"` or kill by PID.

---

## Now open, ranked

1. **Item 10 — extend `s*(n)` past n=48, and it is much cheaper than the plan
   assumed.** The standalone solver did 207 solves in **133 s** where the pysat
   harness took **3747 s** for 105, with proof logging on. The 120 s
   per-instance timeout that stopped the curve at n=48 was substantially
   measuring harness overhead, not instance difficulty. Re-measure before
   budgeting. **Keep the state budget growing with `n`** — raising `n` at fixed
   `--max-states` makes the negative more vacuous, which is how the original
   DFAO claim got retracted.
2. **Item 9 — `s*(n)` in bases 3, 4, 5.** The new Certificate covers base 2
   only, and automaticity is base-dependent (Cobham), so the claim is narrower
   than it reads. `experiments/dfao_min_states.py` already takes `--base`; the
   counting null shifts with `b`, so re-derive the band per base.
3. **Exact `g*(n)` at small n.** The counting null and Re-Pair bracket the true
   curve about 2.5x apart. Solving the exact smallest grammar via SAT/ILP at
   n <= 64 — the same design as `s*(n)`, and the same route to a Certificate —
   would close it. Given how cheap the DFAO instances proved to be, this looks
   tractable.
4. **A parity-capable estimator on the center column.** This is the direct
   consequence of item 8's blind spot. Berlekamp–Massey (already in Experiment
   S) and GF(2) rank methods find exactly what the neural suite misses, and
   they should be pointed at the column with the same control discipline. The
   answer to a model-class blind spot is a different method, not a bigger
   network.
5. **Re-run the detection-power ladder at GPU scale** (5–7M bits, d_model 256,
   context 1024). If a large model learns `lfsr_13_27`, the I/K/L caveat
   weakens; if it still fails, the caveat hardens into a statement about what
   this class of experiment can establish at all.
6. **Extend the independent reference to 46M** (~50 h CPU), or state the 10M
   horizon explicitly in the 46M ledger row rather than letting it inherit
   confidence it has not earned.
7. **Report the pysat proof defect upstream.** Any project using
   `with_proof=True` to certify UNSAT is certifying nothing.

## Do not

- Push `n` higher on `g(n)`. The curves have already converged to three decimal
  places; another octave adds reach, not information. The binding limitation is
  the upper-bound gap.
- Trust a `verify_all.py` run with `SKIP` lines as a verified repo.
- "Fix" `tools/gen_golden_reference.py` to match the kernel's bit order.
