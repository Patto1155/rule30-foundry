# Experiment Log

- Date: 2026-08-30
- Title: Experiments I–L Re-run on the Correct Bit Order, with Detection-Power Controls
- Claim Level: **Robust observation** — paired design, four controls, reduced
  scale stated explicitly. Not a Certificate: no ML null can be one.
- Run date: 2026-08-30 (527 s re-run + 733 s probe, CPU only)
- Artifacts: `data/prize/2026-08-30-rerun-il-bitorder.json`,
  `data/prize/2026-08-30-detection-power-probe.json`
- Verifier: `python experiments/rerun_il_bitorder.py --smoke`

## Goal

Item 8, the repo's outstanding honesty debt. Experiments I (LSTM), J (CNN),
K (Transformer) and L (ML scaling) were trained on the center column with every
consecutive 8-bit block reversed, because four scripts decoded an LSB-first
file with NumPy's MSB-first default. All four are Retracted and the README
carries four ⚠ markers.

## Why this is a paired comparison, not a reproduction

The originals ran on a GPU with 5–7M training bits, hidden sizes to 256 and
contexts to 1024. This session had no GPU. Comparing small CPU numbers against
large GPU numbers would confound **bit order** with **scale** — the one thing
the re-run exists to separate.

So every configuration is trained on **both decodes of the same bytes**, at
identical budget, seed and architecture:

- `center_true` — LSB-first, the actual center column
- `center_reversed` — MSB-first, exactly what I–L saw

The question "did the bug change the conclusion?" is then answered internally.

## Controls the originals never had

I–L are all "the model learned nothing" results. That says something about
Rule 30 only if the pipeline can learn *something*; otherwise BPT ≈ 1.000 is
equally consistent with a broken training loop, and nothing in the original
experiments distinguishes those two. Four extra streams at the same budget:

| stream | what it tests |
|---|---|
| `periodic31` | period 31, inside the 64-bit context — trivial |
| `lfsr_3_5` | `s[i] = s[i-3] ⊕ s[i-5]` — short-lag XOR |
| `lfsr_13_27` | `s[i] = s[i-13] ⊕ s[i-27]` — long-lag XOR |
| `random` | i.i.d. fair coin — the floor |

## Result 1 — the bit order did not change the conclusions

```
stream             lstm h32   lstm h64   tfmr c64   cnn acc
center_true        1.000251   1.000088   1.003814    0.1040
center_reversed    1.000080   1.000008   1.006221    0.1040
periodic31         0.238523   0.066865   0.000130    0.1003
lfsr_3_5           0.742797   0.554068   0.000009    0.1040
lfsr_13_27         1.000165   1.000164   1.001907    0.1003
random             1.000060   0.999992   1.004530    0.1003
```

Largest `center_true` vs `center_reversed` difference across all 24
configurations: **0.0024** (transformer BPT). The CNN accuracies are identical
to four decimals. `center_true` also reproduces the recorded BPT ≈ 1.000001.

The reversal is a deterministic, bijective, position-local recoding, and the
2026-08-29 handover predicted the conclusions "may well survive". They do.

## Result 2 — the suite is blind to XOR structure at lag 27

The controls did not all pass, and this is the substantive finding.

`lfsr_13_27` is **fully determined** by 27 preceding bits, well inside a 64-bit
context window. Every model in the suite scores exactly as it does on a fair
coin. `experiments/detection_power_probe.py` escalated capacity, data, epochs
and context to find out whether that was a budget artefact:

```
lstm as re-run       (h=64,  c=64, 200k, 2ep)  BPT 1.000165   50,497 params  not learned
lstm wider + data    (h=128, c=64, 300k, 3ep)  BPT 1.000085  199,297 params  not learned
lstm short ctx       (h=128, c=32, 300k, 3ep)  BPT 0.999971  199,297 params  not learned
tfmr as re-run       (d=64,  c=64, 200k, 2ep)  BPT 1.003608  100,161 params  not learned
tfmr wider + data    (d=128, c=64, 300k, 4ep)  BPT 1.003449  265,345 params  not learned
tfmr short ctx       (d=128, c=32, 300k, 6ep)  BPT 1.004136  265,345 params  not learned
```

**Six of six fail.** Four times the parameters, 1.5x the data, three times the
epochs, and a short context where both taps sit trivially inside the window —
all at BPT ≈ 1.000. The short-context rungs matter most: they rule out
"the model could not see the relevant bits". This is **architecture- or
optimisation-limited, not budget-limited**, across everything tested here.

The same suite finds `periodic31` (transformer BPT 0.000130, LSTM 100.00%
accuracy) and `lfsr_3_5` (transformer BPT 0.000009, 100.00% accuracy)
immediately. So the failure is specific to long-lag parity, not general.

**Why this bears directly on Rule 30.** Rule 30 is **left-permutive** — its ANF
is `a(t,i-1) ⊕ a(t,i) ⊕ a(t,i+1) ⊕ a(t,i)·a(t,i+1)`, so its update is XOR-like
(a Theorem in this ledger). Experiment S measured linear complexity
`L(n) = n/2`, i.e. any LFSR description would need ~n/2 taps. The structure
class most plausibly present in the center column is **exactly** the class
these models demonstrably cannot detect when it *is* present.

That does not make I, K and L wrong. It makes them much narrower than the
README's "No LSTM shortcut ✓" implied.

## Result 3 — Experiment J's probe does have power, and J is strengthened

A control gap in the re-run's own design, found while reading the results:
**`periodic31` is not a positive control for the CNN decile probe.** That probe
asks which tenth of the stream a window came from, and a periodic stream is
genuinely *stationary* — 10% is the correct answer there, not a failure. So
nothing in the re-run established any power for Experiment J.

The probe adds a stream whose bias ramps 0.30 → 0.70 across its length:

```
drift_0.30_to_0.70   accuracy 0.5767   (chance 0.1000)
random_stationary    accuracy 0.1003   (chance 0.1000)
```

The probe detects blatant non-stationarity at **5.8x chance**, and correctly
returns chance on a stationary stream. Its 10.40% on the center column is
therefore a **powered** null, not an uninformative one.

So J comes out of this *stronger* than it went in: it is the one of the four
whose verdict is now backed by a demonstrated ability to detect the thing it
claims is absent.

## Interpretation

The honesty debt is settled, and the answer is not uniform:

- **The bit-order bug changed nothing.** Every I–L conclusion is reproduced on
  the correct stream at matched budget.
- **J is vindicated and strengthened.** Powered null, ⚠ removed, verdict stands.
- **I, K and L are reinstated but scoped down.** They report a real absence of
  learnable structure for these architectures, but they cannot exclude
  XOR-type structure at moderate lags, because they fail to find it when it is
  planted. Their ✓ marks are weakened accordingly in the README.

Scope, plainly:

- Reduced scale. 200–300k training bits and models to 265k parameters, against
  the originals' 5–7M bits and contexts to 1024. The paired design makes the
  bit-order conclusion robust to this; the detection-power finding is stated
  only for the budgets tested, though six rungs spanning 5x the parameters all
  failed identically, which is not the signature of a budget wall.
- No ML null can be a Certificate. There is no artifact a third party can check
  that establishes "no network of any size finds structure".

## Next Step

1. **Re-run the ladder on the GPU box** at the originals' scale. If a large
   model does learn `lfsr_13_27`, the I/K/L caveat weakens and should be
   revised; if it still fails at 5–7M bits and d_model 256, the caveat hardens
   into a real statement about what this class of experiment can establish.
2. **A parity-capable probe.** The natural follow-up to a blind spot is an
   estimator that is not blind: Berlekamp–Massey (already in Experiment S) and
   GF(2) rank methods find exactly the structure these models miss, and they
   should be pointed at the center column with the same discipline.
3. Do not add more neural experiments on the center column. The 2026-03-28
   idea-bank note and the ledger both say A–L hit a ceiling; this run shows
   the ceiling is partly the *models'*, and the answer to that is a different
   method, not a larger network.
