# Next-Session Plan — Maximising Insightful Experiments

**Date:** 2026-08-30
**Read first:** [`2026-08-29-data-integrity-and-dfao-curve.md`](2026-08-29-data-integrity-and-dfao-curve.md)
(what happened last session), then `AGENTS.md` and `docs/CLAIM_LEDGER.md`.

This file is an ordered work plan, not a menu. The ordering matters: Tier 0 makes
every later result trustworthy and cheap to verify, Tier 1 converts work already
done into citable claims, and Tier 2 is new science. Doing Tier 2 first is how
this repo accumulated five months of results computed on a byte-reversed
sequence.

---

## How to work here

- **Verification-first.** Every run gets an artifact under `data/`, a hash in
  `data/MANIFEST.sha256`, and a row or an update in `docs/CLAIM_LEDGER.md`.
- **Grade honestly.** Observation / Robust observation / Certificate / Theorem.
  A Certificate needs an artifact *another agent can verify without trusting
  you or your solver*. Last session downgraded its own headline result from the
  Certificate its log claimed to Robust observation, because the UNSAT bounds
  rest on trusting CaDiCaL. Do that kind of thing.
- **Apply the Admission Rule before proposing any search.** A negative over
  model class `M` on `n` bits is vacuous unless `log2|M| >= n`. See
  `docs/theory/finite-prefix-counting-bound.md`. Scaling `n` at a fixed model
  budget makes a negative *more* vacuous, not less.
- **Tests are stdlib `unittest`**, run with `python -m unittest discover -s tests -v`.
  There is no pytest and no CI. Match `tests/test_ca_lab_cli.py` for style.
- **Always `bitorder='little'`** when unpacking `data/center_col_*.bin`.

---

# Tier 0 — Make the data trustworthy and keep it that way

Nothing below this line is citable until these land. Items 2–5 are the
"enabling" work: they are what let later sessions move fast without re-auditing
everything by hand, which is most of what last session actually spent its time
on.

## 1. Extend the golden reference to the full 10M bits

**Why.** This is the single largest hole in the repo's evidence. The 10M and 46M
bitstreams are graded **Certificate**, but the *independent* reference
(`tools/gen_golden_reference.py`, which deliberately shares no code with `gpu/`)
covers only the **first 1,000,000 bits**. Above that, the evidence is two runs
of the same kernel family agreeing — which by construction cannot catch a bug
they share. The June 2026 fixes were **boundary and padding** bugs, which
surface at large `t`. So the unverified 90% is exactly where such a bug would
live.

**How.**
```bash
python tools/gen_golden_reference.py --steps 10000000 --out data/golden/center_col_golden_10M.bin
python tools/verify_data.py --bitstream data/center_col_10M.bin
```
The generator is naive NumPy, so budget generously and run it in the background.
If it is unusably slow, do not silently switch to the GPU kernel — that destroys
the independence that makes it worth anything. Instead widen it in stages
(2M, 5M) and record how far the independent check reaches.

**Done when.** `verify_data.py --bitstream` compares against >1M bits, the
manifest anchors the new golden file, and `GOLDEN_BITS` in `tools/verify_data.py`
reflects the new horizon.

**If it fails** — that is the most valuable outcome available in this repo right
now, and it invalidates far more than experiments I–L did. Report it loudly and
do not retry until it passes.

## 2. `tests/test_bitorder.py` — pin the convention that cost five months

**Why.** A bare `np.unpackbits(data)` on an LSB-first file returns the true
stream with every 8-bit block reversed. 49.95% of positions differ while the bit
mean is *identical*, so every aggregate check the repo had was blind to it. This
must never be catchable only by a human noticing.

**What to write.**
- Round-trip: `packbits(bits, bitorder='little')` then
  `unpackbits(..., bitorder='little')` recovers `bits` exactly.
- Cross-convention: assert that decoding `data/center_col_10M.bin` MSB-first is
  *exactly* the per-byte reversal of the LSB-first decode, and that the two have
  equal mean — encoding the trap as an executable fact.
- Anchor: first 15 bits of the LSB-first decode equal OEIS A051023
  `1,1,0,1,1,1,0,0,1,1,0,0,0,1,0`.
- Skip cleanly (`unittest.SkipTest`) when the bitstream is absent, so a fresh
  clone still passes.

## 3. `tools/lint_bitorder.py` — make the bug unwritable

**Why.** Fixing four call sites does not stop a fifth. This is a static check,
so it costs nothing per run and catches the defect at authoring time.

**What to write.** Walk the AST of `experiments/`, `gpu/`, `tools/`. Flag any
`np.packbits` / `np.unpackbits` call that lacks a `bitorder` keyword **and**
occurs in a file that mentions `center_col`. Allowlist
`tools/gen_golden_reference.py` with an inline comment explaining that its
MSB-first convention is deliberate and load-bearing. Exit non-zero on a hit.
Wrap it in `tests/test_lint_bitorder.py` so `unittest discover` runs it.

## 4. `tests/test_manifest_determinism.py` — pin byte-reproducibility

**Why.** `.gitattributes` (`data/** -text`) removed git's EOL normalisation,
which had been silently hiding that `csv.writer` defaults to CRLF. Re-running an
experiment then produced byte-different CSVs and broke manifest hashes while the
*content* was identical. All 10 writers now force `lineterminator='\n'`, but
nothing stops writer number 11.

**What to write.** Write a small CSV through the repo's own helper path, assert
the bytes contain no `\r`. Assert every tracked `data/*.csv` contains no `\r`.
Assert `tools/make_manifest.py` is idempotent — running it twice leaves
`MANIFEST.sha256` byte-identical.

## 5. `tools/verify_all.py` — one command for trust

**Why.** Trust is currently four commands and some tribal knowledge. If checking
the repo is cheap, agents will do it before every experiment; if it is
expensive, they will skip it and you get another five-month bug.

**What to write.** Run, in order, and exit non-zero on any failure: the golden
self-test, `verify_data.py` (manifest + golden), `--bitstream` on each present
canonical bitstream, `lint_bitorder`, and `unittest discover -s tests`. Print a
one-line PASS/FAIL summary per stage. Put it at the top of `AGENTS.md`.

## 6. `tools/lint_ledger.py` — stop the ledger from lying

**Why.** This is not hypothetical. Last session found that the retracted DFAO
row pointed at a replacement "`s*(n)` curve row below" **that did not exist**,
and the experiment log for it was a template full of `RESULT_SECTION`
placeholders under a "Claim Level: Certificate" header. The ledger is the
honesty mechanism; it had been quietly wrong for two weeks.

**What to write.** Parse `docs/CLAIM_LEDGER.md`. For every row, assert that
each backticked path in the Evidence column exists, that no row contains an
unfilled `*_SECTION` placeholder, and that every claim graded Certificate names
a verifier command whose script exists. Also scan `docs/experiment-logs/*.md`
for placeholder tokens. Exit non-zero with the offending row.

---

# Tier 1 — Convert existing work into citable claims

Cheap, bounded, and each one raises a ledger grade. Do these before starting new
science.

## 7. DRAT proofs → promote `s*(n)` to Certificate

**Why.** The `s*(n)` curve is the strongest prize-facing result in the repo and
is currently **Robust observation** for exactly one reason: the SAT side ships a
re-verified witness DFAO (so the *upper* bounds are already certificate-grade),
but the *lower* bounds are CaDiCaL's UNSAT verdicts taken on trust.

**How.** The installed `pysat` exposes it — `Cadical153(with_proof=True)`
(confirmed present in the `__init__` signature). Emit the DRAT log for each
UNSAT instance, check it with `drat-trim`, and store the checker's verdict plus
the proof hash in the artifact. Then minimality is verifiable without trusting
the solver.

**Done when.** Every UNSAT verdict backing an exact `s*(n)` value has a
`drat-trim`-verified proof recorded, and the ledger row moves to Certificate.

## 8. Re-run experiments I–L with the corrected bit order

**Why.** The honesty debt. The README currently carries ⚠ markers on four of its
twelve rows. **Blocked:** `torch` and `sklearn` are not installed.

**The non-obvious value.** This is also a self-test of the ML pipeline. The
reversed stream is a *deterministic, bijective, position-local* recoding, so a
model with ≥8 bits of context could in principle see through it. If BPT was
≈1.000 on the reversed stream and is materially different on the true stream,
that says something is wrong with the methodology rather than with Rule 30.
Both outcomes are informative — report whichever you get, plainly, and do not
retry until it agrees with the old numbers.

**Done when.** All four re-run, the README ⚠ markers removed or the conclusions
corrected, and the ledger row updated from Retracted.

---

# Tier 2 — New experiments, ordered by insight per hour

## 9. `s*(n)` in bases 3, 4, 5

**Why.** The measured result says the center column is not **2**-automatic on
n ≤ 48. Automaticity is base-dependent — that is the content of Cobham's theorem
— so the negative is narrower than it reads. A sequence can be `k`-automatic for
one `k` and not another.

**How.** `experiments/dfao_min_states.py` already takes `--base` (confirmed).
The gate and the null machinery are unchanged; run the matched random control
per base, because the counting null shifts with `b`
(`|M(s,b)| <= s^(s*b) * 2^s`).

**Done when.** A `s*(n)` curve per base with its own null band, and a ledger row
scoped to the bases actually tested.

## 10. Extend `s*(n)` past n=48 — with the state budget growing

**Why and the trap.** n=56 and n=64 exceeded the 120 s per-instance timeout.
**Do not** simply raise `n` at a fixed `--max-states`; that makes the negative
more vacuous, which is precisely how the original DFAO claim got retracted.

**How.** Raise `--timeout-s`, and lean on monotonicity — the last run collapsed
8 of 10 state levels to `UNSAT_IMPLIED` at n=32 for free via "no <k-state DFAO
fits the shorter prefix". Keep the counting-null columns in the output so
admissibility stays visible per row.

## 11. Smallest-grammar curve `g(n)` — a genuinely new model class

**Why.** This is the highest-value *new* experiment available. The repo has now
measured minimal-description curves for two classes: LFSRs (Experiment S,
`L(n) = n/2`) and DFAOs (`s*(n)`). Both used the same sound design. Grammar
compression is the natural third and is much closer to the actual prize
question — a small straight-line grammar *is* a small program that emits the
prefix, which is what "faster algorithm" means in Problem 2.

**How.** Compute the smallest-grammar size (Re-Pair or Sequitur as an
upper bound; exact minimal grammar is NP-hard, so state clearly that you are
measuring an upper-bound curve) for center-column prefixes at increasing `n`,
against the same matched random control and the same counting null. The
counting bound applies directly: grammars of size `g` number at most
`2^(O(g log g))`, so state the admissible region before running.

**Done when.** `g(n)` for center vs a multi-seed random band, with the
upper-bound caveat stated in the ledger row, not buried.

## 12. Prize Problem 3 as an explicit discrepancy bound

**Why.** The ledger says this in as many words: the "behaves statistically close
to fair/random" row should be *"reframed as prize-specific discrepancy bounds or
finite certificates, not more aggregate randomness tests"*. It is still an
Observation, and it is the claim the README's headline rests on.

**How.** Stop reporting "bias < 0.05%, consistent with fair". Compute the actual
discrepancy `D_N = max_k |S_k - k/2|` over the verified 46M bits, report its
growth against `sqrt(N log log N)` (law of the iterated logarithm), and state a
finite, checkable bound of the form "over the first N bits the imbalance never
exceeds X". That is a certificate; "looks fair" is not.

**Done when.** A finite discrepancy certificate with a verifier command,
replacing an aggregate observation.

## 13. Prize Problem 1 — extend the period search

**Why.** The one place in this repo where "run it bigger" is genuinely the right
move. A found period is a $30,000 counterexample, and unlike the model-search
experiments the counting bound does not blunt it — this is not a negative over a
model class, it is a direct search for a specific structure.

**How.** The ledger records "no period found up to 10^6". The binding constraint
is VRAM: the light cone spreads at speed 1, so `N` valid center bits needs
~`2N` cells. Push toward 10^7–10^8. Reuse the Bonferroni threshold discipline
already in `experiments/period_search.py` (best `z = 4.66 < 5.61`) — a larger
search needs a correspondingly larger threshold, and forgetting that would
manufacture a false positive.

**Note.** This is the strongest case for rented hardware: a 24 GB card
quadruples the reachable horizon over the local 6 GB GTX 1060.

## 14. Generalised 32-bit pattern map → test the `period(d)` growth law

**Why.** The sharpest falsifiable prediction the repo owns: `period(d) ~
2*log2(d)` predicts the 32→64 doubling near `d ~ 8.6e9`. Lemmas A, B and C are
already Theorems, and the same machinery refuted the period-16 conjecture, so
the method has a track record of producing real answers.

**How.** The current pattern map is on 16-bit words and is valid only for
`d < 87866`. Generalise it to 32-bit words. The ledger is explicit that direct
simulation is the wrong tool (`T ~ 1.2e10`) — this needs the map, plus
resolution of each branch point at a zero word, where the map is only partial.

**Caveat to keep in view.** The ledger grades left-edge structure as **disjoint
from the prize object** (`settle(T) ≈ 1.34·T > T`), so this is beautiful
mathematics that cannot by itself yield a center-column shortcut. Budget it
accordingly.

## 15. Turn the left-edge ceiling into a certificate

**Why.** The ledger's own stated promotion path for the "disjoint from the prize
object" row: *"Extend the incompressibility argument from these three estimators
to a named model class with a counting bound, turning the ceiling into a
certificate."*

**How.** The unsettled core is currently called incompressible on the strength
of three estimators (periodicity, block entropy, zlib vs a matched control).
Name a model class, state `log2|M|` against the region size, and check
admissibility *before* running. This is the same design as items 9–11 applied to
a 2D region rather than a 1D prefix.

---

## Anti-goals — do not spend a session on these

- **More aggregate randomness tests on the center column.** `docs/idea-bank/theoretical-reframe-2026-03-28.md`
  argues A–L hit a ceiling; the ledger agrees. Another "looks random" result
  changes nothing.
- **Raising `n` at a fixed model budget** to make a negative look stronger. It
  makes it weaker. See the Admission Rule.
- **Renting GPUs before a run is queued and measured.** Nothing last session
  found was compute-bound; the integrity question was settled by a `find` and a
  bit-order check. Item 13 is the one clear exception.
- **"Fixing" `tools/gen_golden_reference.py` to match the kernel's bit order.**
  Its independence is the entire point.

## Suggested session shape

Tier 0 items 1–6 are perhaps half a session and make everything after them
cheap. Item 1 runs in the background while 2–6 are written. Then take item 7 (a
promotion) and one Tier 2 experiment — 9 or 11 are the best value — rather than
starting three and finishing none.
