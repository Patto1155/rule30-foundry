# Pre-declared acceptance criteria (frozen before any result)

Written 2026-10-22, on branch `cline/9xa6tztz`, **before any JWST data was
acquired or fitted**. Rationale: the task forbids changing acceptance criteria
after seeing results, which is only enforceable if the criteria exist, are
dated, and are committed before the numbers do. Git history is the timestamp.

These criteria apply to Plan A's verdict. They do **not** license an
astrophysical claim.

## 1. What counts as "reproduced"

A measurement is **recovered** when our value agrees with the published one
within the combined uncertainty, *and* the reference case was run with the
frozen spec of `02-repro-spec-u13151.md` unchanged.

| Quantity | Frozen reference | Recovery criterion |
|---|---|---|
| z (NIRSpec/PRISM) | `9.7995 ± 0.0270` | `|z_ours − z_ref| ≤ 0.0270` (i.e. ≤ 1σ if our σ is comparable) |
| z (MIRI/LRS, Hβ+[O III]) | `9.8351 (+0.0413/−0.0436)` | ≤ 1σ combined |
| z (MIRI/LRS, Hα) | `9.7621 (+0.0318/−0.0314)` | ≤ 1σ combined |
| Hβ | `68.1 ± 10.5` | within 1σ combined; **note OPEN-1 first** |
| [O III]4959 | `78.5 ± 4.9` | within 1σ combined |
| [O III]5007 | `236.3 ± 14.7` | within 1σ combined |
| [O II]3727,3729 | `7.7 ± 3.7` | within 1σ combined |
| [Ne III]3869 | `24.1 ± 3.3` | within 1σ combined |
| Hβ+[O III] (LRS) | `342.6 ± 61.1` | within 1σ combined |
| Hα+[N II] (LRS) | `196.1 ± 33.3` | within 1σ combined |

Fluxes are compared **in both normalisations** (raw and ×1.706). A result that
matches only under the normalisation that happens to be right is not a
reproduction of the measurement; it is a reproduction of the calibration.

The [O III] 5007/4959 ratio must come out at 2.98 ± its own error in any fit
that claims to recover both lines. A ratio outside 2.98 ± 3σ means the two
lines are not being deblended as claimed, and the fit is rejected regardless of
its agreement with the total flux.

## 2. Verdict rules

| Verdict | Condition |
|---|---|
| **reproduced** | z recovered **and** every line in §1 recovered, under the frozen spec, with the sensitivity checks of §3 not shifting any quoted uncertainty by more than the tolerance in §4 |
| **partially reproduced** | z recovered; ≥ half the lines recovered; failures are confined to lines the available calibration cannot reach (e.g. [O III]5007 without the 5.5 µm extension) |
| **failed reproduction** | z not recovered, or a failure in a line the frozen spec says is reachable |
| **blocked** | data or software unobtainable; **this is the expected verdict if the GTO pipeline is unavailable** |

"Agreement with the published answer" alone is never sufficient — shared data
and shared methods reproduce shared biases. A **reproduced** verdict requires
the sensitivity and injection–recovery evidence of §3 and §4 to be present.

## 3. Mandatory sensitivity checks (each is a pass/fail gate, not decoration)

1. **Individual vs combined fits.** Fit each nod/exposure separately, then the
   combination. If the scatter between nods exceeds the formal uncertainty on
   the combination, the formal uncertainty is wrong and must be inflated to the
   observed scatter.
2. **Diagonal vs empirical covariance.** Compare a diagonal-noise fit with one
   using a covariance estimated from source-free residuals. Report the ratio of
   uncertainties. **Nod differences are not automatically independent** — they
   share sky background and detector state; correlation must be measured, not
   assumed.
3. **Background region variation** within a defensible range.
4. **Extraction aperture** variation around the documented 0.44 arcsec
   (MIRI/LRS) and the NIRSpec slitlet.
5. **Masking**: with and without the strongest neighbouring line blends.
6. **NIRCam normalisation as a nuisance parameter**, marginalised, not fixed at
   1.706. The result must be reported with and without it free.

## 4. Invalidation thresholds

Any **one** of the following demotes the verdict by one grade (reproduced →
partially) and must be reported in the headline, not a footnote:

- Background-region choice shifts a line flux by more than **0.2 × its
  published uncertainty**.
- Aperture choice shifts a flux by more than **0.3 × its published
  uncertainty**.
- Empirical covariance inflates an uncertainty by more than **1.5×** the
  diagonal value.
- Nod-to-nod scatter exceeds the formal error by more than **1.5×**.
- Marginalising the NIRCam normalisation shifts a recovered flux by more than
  **1σ**.

## 5. Injection–recovery (bias and coverage)

Run before quoting any uncertainty as meaningful:

- Inject synthetic lines of known flux at 0.2–3× the detected flux into
  **real** source-free residuals, at the real wavelength sampling.
- Require: median recovered/injected flux ratio within **±15%** across that
  range; **68% interval coverage between 0.55 and 0.81** (nominal 0.68);
  no coverage collapse at the faint end.
- ≥ 200 injections per flux bin. Report Monte Carlo uncertainty on every
  recovered fraction.

A pipeline that recovers fluxes with correct 1σ errors on injections but fails
the noise-floor test on the real data has a background problem, not a
calibration win.

## 6. What would falsify this whole exercise

If the frozen spec's numbers cannot be recovered **even when we re-fit the
published spectra with the published code and settings**, then either the
published numbers depend on unpublished processing (most likely the GTO
pipeline), or the spec is wrong. Both are reportable. Neither is evidence for
new astrophysics, and neither may be reported as a discrepancy in the source
paper without first excluding our own error — the O32 episode in `02` §7 is the
worked example of why.
