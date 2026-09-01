# Experiment Log

- Date: 2026-08-30
- Title: Independent Golden Reference Extended to the Full 10M Bits
- Claim Level: **Certificate** — a hash another agent can check in seconds.
- Run date: 2026-08-30 (8693 s / 145 min, CPU only)
- Artifact: `data/golden/center_col_golden_10M.bin` (1,250,000 bytes, tracked)
- Verifier: `python tools/verify_data.py --bitstream data/center_col_10M.bin`

## Goal

Item 1 of `docs/handover/2026-08-30-next-session-plan.md`, described there as
"the single largest hole in the repo's evidence":

> The 10M and 46M bitstreams are graded **Certificate**, but the *independent*
> reference covers only the **first 1,000,000 bits**. Above that, the evidence
> is two runs of the same kernel family agreeing — which by construction cannot
> catch a bug they share. The June 2026 fixes were **boundary and padding**
> bugs, which surface at large `t`. So the unverified 90% is exactly where such
> a bug would live.

## Setup

`tools/gen_golden_reference.py` — standalone NumPy, sharing no code with
`gpu/`, on an open zero-padded tape of `2*steps + 130 = 20,000,130` cells.
No GPU was available for this run and none was needed.

```
python tools/gen_golden_reference.py --steps 10000000 \
    --out data/golden/center_col_golden_10M.bin
```

Cost is quadratic in the horizon (`~T^2/128` word operations), so the 10x
extension cost ~100x the 1M run: **8693 s** against roughly 90 s. Measured
scaling on the way up was `t(T) ~ 8.9e-6*T + 3.5e-11*T^2` seconds, which
predicted 145 min and was accurate to within a few percent.

## Result — three checks, all passing

### 1. It reproduces the canonical GPU artifact byte-for-byte

The decisive one, and stronger than the plan asked for. Repacking the golden
bit sequence LSB-first — the convention `gpu/rule30_sim.py` writes — gives:

```
independent CPU reference, repacked LSB-first:
  6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e
anchored canonical hash in data/MANIFEST.sha256:
  6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e
MATCH
```

**All 10,000,000 bits, not a sample.** The plan's "done when" was that
`verify_data.py --bitstream` compares against more than 1M bits. What actually
happened is that an implementation sharing no code with `gpu/` produced a
*bit-identical file*, so the check is a single hash comparison rather than a
bitwise diff.

This closes the March/June integrity question in the passing direction over the
whole stream. The June boundary and padding fixes did not perturb the center
column at any `t` up to 10^7.

**It is one of three axes, and the combination is the point.** In parallel, the
laptop session (`3aedad9`) regenerated the same bitstream on an **RTX 2050**
(Ampere, compute capability 8.6, CUDA runtime 12.9) against the anchor recorded
on a **GTX 1060** (Pascal, 6.1) — same sha256, 0 of 10,000,000 bits differ.

| reproduction | what it can catch | what it cannot |
|---|---|---|
| second GPU, different architecture and CUDA major version | hardware, driver and toolchain-dependent bugs | a bug in shared source — it reproduces identically everywhere |
| this CPU reference, no shared code, convention or geometry | shared-source bugs | nothing about GPU-specific execution, since it never runs any |

Neither axis is sufficient alone and each covers the other's blind spot. No
claim is made about the host OS of the original GTX 1060 run: it is not recorded
in `data/center_col_10M_results.json`, `DATA_INTEGRITY.md` or `GPU_KERNELS.md`,
and an earlier ledger row that asserted it was corrected in `a1a1f0e`.

### 2. Geometry cross-check

The 1M reference was generated on a 2,000,130-cell tape, the 10M one on
20,000,130 — a 10x different geometry. Their first 1,000,000 bits agree
exactly. A padding or edge bug in the *reference itself* that depended on tape
width would have shown up here, and does not.

### 3. It reproduces Experiment A's recorded statistic

```
ones 5,002,220 of 10,000,000; excess over N/2 = 2,220
(ones - N/2)/(N/2) = 0.00044400
```

The README and `data/bit_distribution.csv` record bias `0.00044400`. Exact
agreement, from a completely independent implementation.

## Consequences

`tools/verify_data.py` selects the widest golden reference on disk rather than
a hardcoded 1M, so the horizon moved by itself:

```
golden: data/golden/center_col_golden_1M.bin OK (1,000,000 bits)
golden: data/golden/center_col_golden_10M.bin OK (10,000,000 bits)
golden: independent check reaches 10,000,000 bits
bitstream: center_col_10M.bin agrees with center_col_golden_10M.bin over 10,000,000 bits  OK
```

`python tools/verify_all.py` now reports **8 passed, 1 skipped, 0 failed**. The
remaining skip is the 46M bitstream, which is not present on this machine.

## Interpretation

The plan said of a failure here that it "is the most valuable outcome available
in this repo right now, and it invalidates far more than experiments I–L did".
It passed instead — and passed in the strongest available form, byte-identity
rather than agreement.

What this does and does not settle:

- **Settled:** the 10M center column is correct at every position, against an
  implementation with no shared code, no shared bit-order convention, and no
  shared tape geometry. Experiments A–H rest on bytes that are now independently
  reproduced end to end.
- **Not settled:** the 46M bitstream. Its independent check still reaches only
  the first 10M bits — better than the 1M it had, but the top 78% of that stream
  is still two runs of the same kernel family agreeing. Extending the reference
  to 46M costs `(46/10)^2 ~ 21x` this run, about **50 hours** of CPU. That is a
  job for the GPU box or for a rewritten generator, not for a 4-core container.

## Next Step

1. **Extend to 46M** on hardware where it is affordable, or accept the 10M
   horizon explicitly in the ledger row for the 46M artifact. It currently
   inherits confidence from the 10M result that it has not earned.
2. **Make the generator checkpoint.** It writes only after the full loop, so a
   145-minute run loses everything if interrupted, and a 50-hour run is not
   realistically attemptable without resumption.
3. The generator is single-threaded NumPy at roughly 1.2e10 cell-updates/s.
   A blocked, multi-threaded C implementation would plausibly reach 4-5x that
   and bring 46M inside a day — but it must keep sharing no code with `gpu/`,
   or it stops being evidence.
