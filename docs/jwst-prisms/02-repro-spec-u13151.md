# Reproduction specification — U13151 (frozen)

Source: **arXiv:2610.26560v1**, Rinaldi et al., submitted 2026-10-19.
Every field below is transcribed from the `.tex`. Fields marked `OPEN-n` cannot
be resolved from the paper alone and must be closed **before** any fit is
believed. This file is the target; it is not a result.

Machine-checked for internal consistency by
`python tools/jwst_u13151_consistency.py` (10 checks, 12 unit tests).

## 1. Observations

| Element | Value | Provenance |
|---|---|---|
| Target | UNCOVER-ABELL2744-13151 (U13151) | §I |
| Lens | Abell 2744, `μ = 12.8 ± 0.7` | Furtak+23, Price+24 |
| NIRSpec | **PID 2561**, UNCOVER (PIs Labbé & Bezanson), **PRISM** | §I, Fig. 1 caption |
| MIRI spectroscopy | **PID 8051**, PRISMS (PIs Álvarez-Márquez & Colina), **LRS** | §I, §II.1.2 |
| MIRI imaging | **PID 5578**, MIRIC (PIs Iani & Rinaldi), F770W + F1000W | §II.1.1 |
| NIRCam | UNCOVER public DR: PIDs 2756, 2561, 2883, 3538, 4111, 3516; 20 bands | §II.2.1 |
| Exposure IDs | **not stated in the paper** | OPEN-2 |

## 2. Reduction

### NIRSpec/PRISM — the critical dependency
Reduced with "the most recent realization of the data reduction pipeline
provided by the **NIRSpec GTO team**" (Scholtz et al. 2025, JADES), which
"extends the extracted and calibrated spectral traces on the detector
significantly beyond the nominal red-end boundary (i.e. beyond 5.3 µm up to
5.5 µm)". Compared with the earlier Fujimoto+24 reduction, this is what brings
Hβ and [ O III ] 4959,5007 into the usable range at z ≈ 9.8.

This is the single most important sentence in the paper for Plan A. See
`03-feasibility-and-budget.md` §2.

### MIRI/LRS
Standard Stage 1–2 with cosmic-ray shower correction in jump detection; then
custom wavelength masking, master-background construction and subtraction,
removal of residual wavelength-dependent background along the slit, and sigma
clipping. Stage 3 extraction with a **0.44 arcsec aperture** and the standard
point-source aperture correction. Full description deferred to Álvarez-Márquez
et al. 2026.

| Parameter | Value |
|---|---|
| MIRI imaging pipeline | `jwst` v1.20.0, `pmap 1439` |
| MIRI/LRS | same pipeline family, `pmap 1464`; exact version **not stated** → OPEN-3 |
| Photometry apertures | circular, filter-dependent 0.15–0.25 arcsec |
| Photometric background | 3 × 3 arcsec local box; noise from non-adjacent pixels 5 px away |
| Aperture corrections | Robertson et al. 2026 |

### Absolute cross-instrument normalisation
NIRSpec is normalised to the NIRCam photometry through synthetic photometry of
the spectrum (`sedpy`):
`F_NIRSpec,corr = 1.706 × F_NIRSpec`, with **1.706 ± 0.032**.
Raw and normalised measurements must be reported separately (task requirement,
and OPEN-1).

## 3. Line fitting

| Element | Value |
|---|---|
| Code | **MSAEXP** (Brammer 2023; Heintz 2025) |
| Continuum | Gaussian spline, **`nsplines = 11`** |
| Redshift search | z = 9–10, minimum reduced χ² |
| MIRI/LRS | MSAEXP with the appropriate spectral-resolution curve |
| Uncertainties | posterior sampling, `emcee`, **10 000 MCMC steps** |
| Simultaneous fits | Hγ + [ O III ]4363; [ Ne III ]3869 + H8+He I 3889 |

## 4. Published measurements to reproduce

Fluxes in `1e-20 erg s⁻¹ cm⁻²`, **observed** (not lensing-corrected). Upper
limits are limits, never detections.

| Line | Flux | Error |
|---|---|---|
| N IV]1486 | < 67 | — |
| C IV 1549 | 60.9 | 20.2 |
| O III]1660,1666 + He II 1640 | 78.5 | 17.2 |
| N III]1750 | < 46 | — |
| C III]1907,1909 | 41.9 | 13.1 |
| [O II]3727,3729 | 7.7 | 3.7 |
| [Ne III]3869 | 24.1 | 3.3 |
| He I 3889 + H8 | 9.6 | 2.6 |
| [Ne III]3968 + Hε | 15.5 | 2.8 |
| Hδ | 11.4 | 5.5 |
| Hγ | 21.9 | 9.0 |
| [O III]4363 | 8.6 | 4.3 |
| Hβ | 68.1 | 10.5 |
| [O III]4959 | 78.5 | 4.9 |
| [O III]5007 | 236.3 | 14.7 |
| Hβ + [O III] (LRS, unresolved) | 342.6 | 61.1 |
| Hα + [N II] (LRS, unresolved) | 196.1 | 33.3 |

Redshifts:

| Measurement | z |
|---|---|
| NIRSpec/PRISM (**adopted**) | `9.7995 ± 0.0270` (incl. 750 km/s wl-cal) |
| MIRI Hβ+[O III] | `9.8351 (+0.0413 / -0.0436)` |
| MIRI Hα | `9.7621 (+0.0318 / -0.0314)` |
| Fujimoto+24 (prior) | `9.880 (+0.012 / -0.001)` |
| Lyα break (this work) | `9.814 (+0.047 / -0.044)` |

Derived properties (targets for the diagnostics, not for the line fits):
`12+log(O/H)_SL = 7.29 ± 0.05`; `_Te = 7.44 (+0.33/-0.17)`;
`log U = -1.90 ± 0.35`; `A_V(Hα/Hβ) = 0.1 ± 0.62` (adopted 0);
`SFR(Hα) = 0.45 ± 0.08`; `SFR(UV) = 0.29 ± 0.02`; `M*/SFR = 10.6 Myr`;
`sSFR = 94 Gyr⁻¹`; `log ξ_ion = 25.64 ± 0.12`.

Expected diagnostics computed from the table above (paper's Table 1
definitions): `R3 = +0.540`, `O32 = +1.612` (their convention; `+1.487` in the
common one), `R23 = +0.675`, `Ne3O2 = +0.496`, `Ne3O2Hd = +0.446`,
`O3Hg = -0.406`, `O33 = +1.439`. These are arithmetic restatements of the
published fluxes — useful as a transcription check, worthless as evidence.

## 5. Open questions — close these before fitting

| ID | Question | Why it blocks | How to close |
|---|---|---|---|
| OPEN-1 | Do Table 2 fluxes include the ×1.706 normalisation? | Changes every flux by 0.23 dex, and changes **every ratio by nothing** — so a wrong choice is invisible in the diagnostics and fatal in a comparison table | Read Álvarez-Márquez+26 / Scholtz+25; or compare Table 2 to the NIRCam synthetic photometry |
| OPEN-2 | Which exposures/visits contain U13151? | A provenance manifest cannot be written without it | MAST `astroquery` query, `03` §3 |
| OPEN-3 | Exact `jwst` pipeline version and pmap for MIRI/LRS | Stage-3 extraction is version-sensitive | PRISMS methods paper |
| OPEN-4 | Is the NIRSpec-GTO extended-trace pipeline public? | Decides partial vs full reproduction | `03` §2 |
| OPEN-5 | Are the LRS flux complexes lensing-corrected? | They sit in the same table as unlensed NIRSpec fluxes | Table notes; paper says fluxes are *not* lens-corrected |

## 6. Standing refusals (encoded in the checker)

1. **No Balmer decrement** from Hα+[N II] (MIRI/LRS) over Hβ (NIRSpec):
   different instruments, apertures, normalisations. The paper's own
   `A_V = 0.1 ± 0.62` mag is consistent with zero and is set to zero.
2. **Upper limits never enter a ratio.** `log10(N/O) < -0.76` requires an ICF;
   it is not arithmetic on this table.
3. **No line flux is lensing-corrected.** Only daggered quantities are.

## 7. One verified non-issue, kept as a trap

The O32 convention offset is a rounding, not a discrepancy: the paper's
log10(1.3) versus log10(1.332) = 0.125 dex from the 2.98 doublet. Re-deriving it
from **wavelengths** instead of **fluxes** yields 0.299 dex and a spurious
0.185 dex "error" — this was produced once while writing the checker and is now
pinned by a unit test. Ratios of lines are ratios of fluxes.
