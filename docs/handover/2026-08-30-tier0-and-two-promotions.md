# Handover — Tier 0 Landed, `s*(n)` Certified, `g(n)` Measured

**Date:** 2026-08-30
**Branch:** `claude/keen-sagan-21rzbr`, stacked on `fix/data-hygiene` (PR #18), PR #19
**Read first:** [`2026-08-30-next-session-plan.md`](2026-08-30-next-session-plan.md)
(the plan this session executed), then `AGENTS.md` and `docs/CLAIM_LEDGER.md`.

**Start here:** `python tools/verify_all.py` — one command, ~2 s.

---

## TL;DR

Plan items **2–6** (Tier 0 tooling), **7** (`s*(n)` → Certificate) and **11**
(`g(n)` grammar curve) are done. Item **1** is in flight. Items 8, 13, 14 were
not attempted and could not be: see "Environment" below.

Three findings that change what the next session should do:

1. **The plan's item-7 recipe is broken.** `Cadical153(with_proof=True)` emits
   proofs that `drat-trim` rejects. Following it as written would have shipped a
   Certificate backed by refutations that fail their own checker.
2. **`tools/make_manifest.py` was silently deleting the bitstream hash anchors**
   on any machine without the gitignored `.bin` files — i.e. every fresh clone.
3. **The SAT work is ~28x cheaper than assumed**, which re-costs item 10 from
   "blocked on solver time" to "probably minutes".

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

**Items 8, 13 and 14 belong on the machine with a GPU and the bitstreams.**
Item 8's blocker is not only that `torch`/`sklearn` are missing — it is that the
10M bitstream the I–L experiments were computed on is not something a fresh
clone has.

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
  the full loop, so a 2-hour run loses everything if interrupted. Worth fixing
  before the next long generation.

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
4. **Item 8 (re-run I–L)** and **items 13/14** — all need the GPU box.
5. **Report the pysat proof defect upstream.** Any project using
   `with_proof=True` to certify UNSAT is certifying nothing.

## Do not

- Push `n` higher on `g(n)`. The curves have already converged to three decimal
  places; another octave adds reach, not information. The binding limitation is
  the upper-bound gap.
- Trust a `verify_all.py` run with `SKIP` lines as a verified repo.
- "Fix" `tools/gen_golden_reference.py` to match the kernel's bit order.
