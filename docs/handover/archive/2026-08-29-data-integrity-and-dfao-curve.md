# Handover — Data Integrity Closed, Experiments I–L Retracted, `s*(n)` Measured

**Date:** 2026-08-29
**Repo:** `D:/APATPROJECTS/rule30-foundry`
**Branch:** `fix/data-hygiene` (pushed, PR #18 open against `main`)
**Entry points:** `docs/DATA_INTEGRITY.md`, `docs/CLAIM_LEDGER.md`,
`tools/verify_data.py`, `experiments/dfao_min_states.py`,
`docs/experiment-logs/2026-08-15-dfao-min-state-curve.md`

---

## TL;DR — three results, one good, one bad, one new

1. **The March-vs-June kernel gap is CLOSED, and it passes.** The 10M center
   column regenerated on the current kernel is **byte-identical** to the March
   2026 artifact. The packed-kernel fixes never changed center-column output.

2. **Experiments I, J, K, L are RETRACTED.** Not for the documented reason. Four
   scripts read the LSB-first bitstream with numpy's MSB-first default, so the
   LSTM, CNN, Transformer and ML-scaling runs were trained on the center column
   **with every consecutive 8-bit block reversed**. A–H are unaffected.

3. **The `s*(n)` minimal-DFAO-size curve is measured.** The ledger's retracted
   DFAO row had promised a replacement row that did not exist. It exists now:
   the center column's curve is **indistinguishable from a random control**,
   and unlike the claim it replaces, this negative is not vacuous.

Nothing here needed rented hardware. The integrity question was settled by a
`find` and a bit-order check.

---

## Result 1 — the integrity gap, closed in the passing direction

`docs/DATA_INTEGRITY.md` asked whether the fixed kernel still reproduces the
March bitstream the A–L results were computed on. Nobody had checked.

The March artifacts were **not lost**. They were in
`D:/APATPROJECTS/rule30-research/data/`, which `DATA_INTEGRITY.md` and
`AGENTS.md` both described as "a repo that no longer exists under that name".
It exists. Verified:

```
foundry  origin -> github.com/Patto1155/rule30-foundry.git
research origin -> github.com/Patto1155/rule30-foundry.git      # same remote
research HEAD 06ab32d is a STRICT ANCESTOR of foundry HEAD
foundry ahead: 49 commits      research-only commits: 0
```

It is a **stale checkout of this same repository**, so the 16 scripts that
hardcoded that path were pointing at an old copy of themselves. Both docs are
corrected.

Regenerating and comparing:

```
March sha256: 6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e
rerun sha256: 6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e
0 of 10,000,000 bits differ
```

Both canonical bitstreams now live in `data/` (gitignored) with hashes anchored
in `data/MANIFEST.sha256`. The `UNANCHORED` rows are gone.

**Independent confirmation:** `experiments/bit_distribution.py` re-run today on
the verified bitstream reproduces March exactly — bias `0.00044400` at
N=10,000,000, ratio `1.4041`, matching the archived CSV byte-for-byte and the
README's "+0.044% bias, 1.4x noise floor".

---

## Result 2 — the bit-order bug (READ THIS BEFORE TOUCHING ANY BITSTREAM)

This is the most important thing in this handover.

`gpu/rule30_sim.py:226` writes center-column dumps with
`packbits(..., bitorder='little')` — **LSB-first**. Numpy's default is
MSB-first. Four scripts omitted the argument:

| script | line | README experiment |
|---|---|---|
| `experiments/lstm_prediction.py` | 36 | I — LSTM |
| `experiments/cnn_nonstationarity.py` | 31 | J — CNN |
| `experiments/transformer_prediction.py` | 40 | K — Transformer |
| `experiments/scaling_laws_ml.py` | 51 | L — ML scaling laws |

Every other reader passes `bitorder='little'` correctly. Measured effect:

```
TRUE center col (LSB) : 110111001100010110010011
what I,J,K,L saw (MSB): 001110111010001111001001
499,516 of 1,000,000 bits differ (49.95%)
MSB-decode == per-byte reversal of the true stream: True
bit mean: 0.500768 either way        <-- identical, which is why nothing caught it
```

**Why it hid for five months:** the reversal is a permutation, so every
aggregate statistic the repo was watching — bias, mean, monobit — is invariant
under it. Only a positional comparison exposes it.

**What follows, stated carefully.** The reversal is deterministic, bijective and
position-local. It permutes structure within bytes rather than destroying it,
and a model with >= 8 bits of context could in principle see through it. The I–L
*conclusions* may well survive a re-run. What does not survive is the claim that
they tested the center column. They tested a variant of it.

The source is fixed in all four scripts, plus a copy-pasteable snippet in
`docs/center-col-46M-benchmark.md` that taught the same bug. **The recorded
results stay retracted until the experiments are actually re-run.**

`tools/verify_data.py` had the same defect and so **could not validate the
output of this repo's own simulator** — it reported `499,516 of 1,000,000 bits
differ, first divergence at bit 0`, which reads as a destroyed kernel. It now
tries both orders, names the one that matched, and flags a ~50% mismatch as an
uncorrelated-streams signature rather than corruption.

`tools/gen_golden_reference.py` is MSB-first **by deliberate, documented
exception** — it is a standalone reference that must not share conventions with
the kernel it checks. Do not "fix" it.

---

## Result 3 — the `s*(n)` curve (the ledger's missing row)

The retracted DFAO row said it was "replaced by the `s*(n)` curve row below".
There was no such row. The harness existed, the 7-seed random null had been run,
and commit `46aa72d` said so outright: *"Rule 30 curve not yet run"*. The log
was still a template with `RESULT_SECTION` placeholders under a Certificate
header.

Run 2026-08-29, 62 min, CaDiCaL 1.5.3 via pysat. Exact minima to n=48:

```
   n  center   null band over 7 random seeds   verdict
   8       3   [2, 3, 3, 4, 4, 4, 4]           inside
  12       4   [4, 4, 4, 5, 5, 5, 5]           inside
  16       5   [4, 5, 5, 5, 6, 6, 6]           inside
  20       6   [5, 5, 6, 6, 6, 8, 8]           inside
  24       8   [6, 6, 7, 8, 8, 8, 9]           inside
  28       9   [7, 8, 8, 9, 9, 9, 11]          inside
  32      10   [7, 9, 9, 9, 10, 10, 11]        inside
  40      12   [9, 10, 10, 10, 10, 11, 13]     inside
  48      12   [11, 12, 12, 12, 13, 13, 13]    inside
```

**9 of 9 inside the band, 0 below.** Where it deviates it deviates *upward*.

Three conditions hold together, which is what separates this from the vacuous
claim it replaces:

1. **Detection power.** Thue-Morse returns `s* = 2` at every n in both digit
   directions. A genuine automatic sequence is found immediately.
2. **Informative regime.** Measured minima (10 at n=32) sit *above* the
   counting-null threshold (6–7), so these are real minima, not an artefact of
   a class too small to fit anything.
3. **It measures a curve, not a yes/no.** The Admission Rule's warning about a
   matching control negative concerns both sides returning "nothing fits". Here
   both return **finite exact minima**; the agreement of two measured curves is
   the content.

Graded **Robust observation, not Certificate**, against the log's own
aspirational header: the SAT side ships a re-verified witness DFAO so the upper
bounds are certificate-grade, but the UNSAT **lower** bounds rest on trusting
CaDiCaL.

---

## Now open, ranked

1. **Re-run I–L with the corrected bit order.** This is the last thing between
   the README and an honest results table. **Blocked:** `torch` and `sklearn`
   are not installed on this machine. The models are small (d_model 32–256,
   context 64–1024) so a GTX 1060 is adequate; this is the one task here that
   is genuinely compute-shaped. Compare against the recorded results and say
   plainly whether the conclusions survive.

2. **Promote `s*(n)` to Certificate via DRAT proofs** for the UNSAT verdicts.
   Then `s*(n)` becomes independently checkable without trusting the solver.

3. **Extend `s*(n)` past n=48.** n=56/64 exceed the 120 s per-instance timeout.
   Raise it, or lean harder on monotonicity — the run already collapsed 8 of 10
   state levels to `UNSAT_IMPLIED` at n=32 for free.
   **Do not** simply push n higher at a fixed state budget; that makes the
   negative *more* vacuous, which is exactly what the counting bound exists to
   prevent.

4. **Extend the golden reference past 1M bits.** It currently covers only the
   first 10% of the 10M stream, so the independent check is partial.

5. **Absorb or delete `D:/APATPROJECTS/rule30-research`.** It is a stale clone
   of this repo. Nothing depends on it now that the bitstreams are in `data/`.

---

## Traps — things that already bit someone

- **Bit order.** See Result 2. Always `bitorder='little'` for `center_col_*.bin`.
- **CSV line endings.** `csv.writer` defaults to CRLF. Git used to hide this by
  normalising on commit; `.gitattributes` (`data/** -text`) removed that
  normalisation, so a re-run produced byte-different CSVs and broke manifest
  hashes while the content was identical. All 10 writers across 9 files now
  force LF. **If you add a CSV writer, pass `lineterminator` explicitly.**
- **`.gitattributes` is load-bearing.** Without `data/** -text`, 162 of 164
  manifest entries fail on a Windows clone even though the stored content is
  bit-identical. Do not "fix" a manifest failure by normalising inside
  `verify_data.py` — that stops the tool verifying the bytes actually on disk.
- **Paths.** All 46 hardcoded literals are now `REPO_ROOT`-relative
  (`Path(__file__).resolve().parent.parent`). Do not add a new absolute path.
- **A ~50% bit difference is never a kernel bug.** It means the streams are
  uncorrelated — a packing or seed mismatch. A *kernel* bug shows up as a late
  first divergence.
- **Trust, but verify, subagent reports.** The bit-order finding came from a
  subagent audit; every claim in it was re-derived by hand before being acted
  on. One count in it was off by one (16 files, not 17).

---

## Reproduction

```bash
python tools/verify_data.py                       # 167 verified, 0 absent, OK
python tools/gen_golden_reference.py --self-test  # naive == packed, OEIS prefix matches
python tools/verify_data.py --bitstream data/center_col_10M.bin   # LSB-first, agrees with golden

# regenerate the 10M column (~6 min on a GTX 1060, 27k steps/s)
python gpu/rule30_sim.py --cells 21000000 --steps 10000000 \
  --center --center-out data/center_col_10M.bin

# the s*(n) curve: gate first, then the run (~62 min)
python experiments/dfao_min_states.py --gate-only --timeout-s 120
python experiments/dfao_min_states.py --timeout-s 120 --max-states 20 \
  --out data/prize/2026-08-15-dfao-min-state-curve.json
```

---

## Environment (2026-08-29)

- GTX 1060 6GB, cupy 14.1.1. The `CUDA path could not be detected` warning is
  cosmetic — the kernels run.
- `tqdm` had to be installed; it is in the README requirements but was absent.
- `pysat` 1.9.dev14 with CaDiCaL 1.5.3, in-process. **`torch` and `sklearn` are
  NOT installed** — this blocks the I–L re-run.
- Throughput scales with **cells/sec** (~567 Gcells/s), not steps/sec. A 46M-step
  run at 96M cells is ~2.2 h, not ~28 min. Budget from the cell count.

## Commits on this branch

```
9e03575  Measure the s*(n) DFAO curve: no automatic shortcut, and not vacuous
de3bc43  Make experiments portable and byte-reproducible; fix the I-L bit order
8110954  Close the March/June integrity gap; retract experiments I-L
a0e61f4  Anchor 10M and 46M bitstream hashes
950c033  Make the bitstream verifier bit-order aware
78c4d64  Mark data tree non-text so the hash manifest is portable
af92cff  Untrack block_freq_k15..k20.csv (87 MB, regenerable)
a48bdbc  Fix data hygiene: golden reference, hash manifest, honest .gitignore
```
