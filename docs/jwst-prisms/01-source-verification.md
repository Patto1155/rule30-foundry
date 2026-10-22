# Source verification

Method: abstract pages fetched live, then both LaTeX source bundles downloaded
from arXiv (`https://arxiv.org/e-print/<id>`) and searched directly. Every
quoted number below comes from the `.tex`, not from a summary, because the point
of this stage is to avoid building an experiment on an unverified digest.

Verified 2026-10-22.

## A. arXiv:2610.26560 — the Plan A target

| Field | Value |
|---|---|
| Title | *PRISMS/U13151. A Glimpse into the Earliest Phases of Galaxy Formation through an Ultra-Faint System ≈ 500 Myr after the Big Bang* |
| Authors | Rinaldi, Álvarez-Márquez, Crespo Gómez, Colina, Carniani, … (38 total) |
| Submitted | 2026-10-19 15:19 UTC |
| Versions | **v1 only** — no revision history to reconcile |
| Status | Submitted to ApJ; 37 pages, 15 figures, 3 tables |
| Subjects | astro-ph.GA |
| Licence | CC Zero |

Exists, current, and is about U13151. **Verified.**

Core numbers, quoted from the source rather than paraphrased:

- `z_spec = 9.7995 ± 0.0270` (NIRSpec/PRISM, adopted throughout).
- `M_UV = -17.44` (lensing-corrected), `μ = 12.8 ± 0.7`.
- `log10(M*/Msun) ~ 6.68`, age `5.5 ± 2.8 Myr`, `β ~ -2.64`.
- `log10(N/O) < -0.76` — **a limit**, from non-detections of N IV]1486 and
  N III]1750. The abstract's "undetected nitrogen emission" is this limit.
- `log10(C/O) = -0.75 ± 0.19` (ICF-corrected), `log10(Ne/O) = -0.51 ± 0.07`.

## B. arXiv:2610.28717 — the Plan B source

| Field | Value |
|---|---|
| Title | *Chemical abundance patterns at z=1-7: Carbon, Nitrogen, Neon, Sulphur, and Argon abundances from JWST spectroscopy* |
| Authors | Cataldi, Belfiore, Curti, … (25 total) |
| Submitted | 2026-10-20 18:00 UTC |
| Versions | **v1 only** |
| Status | Submitted to A&A; 19 pages, 9 figures |
| Licence | CC BY 4.0 |

Exists, current, and is about exactly the claimed subject. **Verified.**

- 244 galaxies with [O III]4363 detections (direct Te), plus 514 via new
  strong-line calibrations; MARTA (PID 1879, PI Curti) + archival.
- `N/O` enhanced by `~0.18 dex` in median relative to local galaxies, stronger
  at low metallicity.
- Appendix D is a **forward model of observational selection**, and Section 6.3
  is devoted to selection effects. This is decisive for Plan B — see
  `05-plan-b-selection.md`.

## C. Four claims in the task prompt that failed verification

The prompt said to treat its project context as potentially stale. It was.
Recording these because each would have misdirected work.

1. **"Candidate programme IDs are NIRSpec 2561 and MIRI 8051."** Half right, and
   the wrong half is the dangerous one. Verified from the source:
   - NIRSpec/PRISM **PID 2561** — correct. It is *UNCOVER* (PIs Labbé &
     Bezanson), not a PRISMS programme; PRISMS supplied the MIRI spectroscopy.
   - MIRI/LRS **PID 8051** — correct. This is *PRISMS* (PIs Álvarez-Márquez &
     Colina).
   - **MIRI imaging is PID 5578** (*MIRIC*, PIs Iani & Rinaldi) — a *third*
     programme. The prompt's two-PID framing silently merges the F770W/F1000W
     photometry into 8051, which would put the wrong exposures in a provenance
     manifest.
2. **"an earlier approval allowed a 6 GiB cap for a specific SMACS calibration
   blocker."** The target here is **Abell 2744**, not SMACS 0723. A budget
   approved for one calibration blocker does not transfer to a different
   cluster, instrument set and programme. Treated as **no approved budget**;
   see `03-feasibility-and-budget.md` for how to request a sized one.
3. **Plan A and Plan B do not share a dataset.** Plan A's object sits at
   z = 9.7995; Plan B's sample is z = 1–7. U13151 cannot be a member of the
   Plan B parent population, and it is also a *nitrogen-poor* object
   (`log N/O < -0.76`), so it cannot serve as Plan B's positive control without
   switching to the optical diagnostic.
4. **"the paper's reported ... fluxes" — there is an ambiguity the prompt does
   not anticipate.** The NIRSpec spectrum is multiplied by `1.706 ± 0.032` to
   match NIRCam photometry, but Table 2 does not state whether its fluxes are
   pre- or post-factor. Both are recorded in the spec (`OPEN-1`); it is
   arithmetic, not physics, and must be settled by reading
   Álvarez-Márquez et al. 2026 / Scholtz et al. 2025 before any fit.

## D. What could not be verified from here

- **Which exposures** contain U13151. The paper states the programmes and shows
  the spectra, but exposure-level association requires a MAST query from a
  machine with `astroquery` — not available here. Exact command in
  `03-feasibility-and-budget.md` §3.
- **Whether PID 8051 data are out of exclusive access.** PRISMS is a GTO
  programme; the 12-month proprietary window must be checked at MAST, not
  assumed. Unverified, and it gates Plan A's MIRI leg.
- **The data-availability statement.** Neither paper has a dedicated
  "Data availability" section. U13151's acknowledgements state the data were
  obtained from MAST and that programmes 5578 and 8051 were supported; the
  Cataldi sample is assembled from archival programmes tabulated in its
  Appendix A. No catalogue or code release is promised in either paper. Treat
  "public data, no published value-added catalogue" as the working assumption
  and confirm before promising a like-for-like reproduction.

## E. Claim grading

| Claim | Grade |
|---|---|
| Both papers exist, are current v1, and cover the stated subjects | **Verified** (primary source) |
| U13151 z = 9.7995 ± 0.0270 from NIRSpec/PRISM | **Quoted** — reproduced in the spec, not independently measured |
| Table 2 fluxes and their uncertainties | **Quoted** — internally consistent (`tools/jwst_u13151_consistency.py`) |
| The published reduction is reproducible from public products | **Unverified, and doubted** — see `03` |
| Selection can explain part of the low-Z N/O enhancement | **Already published by the source paper** (Appendix D), not a finding of this work |
