# Plan B — selection effects on nitrogen enhancement

**Question as posed:** could our detection and extraction procedure produce
apparent nitrogen enhancement from a population without intrinsic enhancement?

## 1. The finding that changes the plan

The source paper **already answers this**, in Appendix D, with a forward model,
and reports it in the abstract and conclusions. Verified quotes from
arXiv:2610.28717v1 (`model.tex`):

> "At low metallicity, the detectable simulated galaxies preferentially have
> enhanced N/O, producing a distribution consistent with the trend observed in
> the direct-method sample. This suggests that selection effects could account
> for part of the N/O enhancement at low O/H, even if the underlying population
> follows the local abundance–metallicity relation." (§6.3)

> "…but preferentially selects galaxies with enhanced N/O at low metallicity,
> with a weaker effect for S/O." (abstract)

So Plan B as written is **not a novel hypothesis test**. It is a replication of
an existing forward-model result, using a different injection engine. That is
worth doing — independent replication of a selection claim is exactly the kind
of evidence that is usually missing — but it must be labelled a replication, and
its deliverable is "consistent / inconsistent with Cataldi et al. Appendix D",
not "selection can/cannot explain N-enhancement".

Two further constraints from the paper that the prompt anticipated only
partially:

**The diagnostic split is real and the paper makes it explicitly.** §6.1:
JWST has found "galaxies with strongly enhanced nitrogen abundances, often
selected through N IV λ1486 and N III λ1750 emission", and "these extreme UV
N-emitters must be distinguished from the moderate N/O enhancement found through
rest-optical diagnostics". The paper's own Te sample and its 0.18 dex N/O result
are **rest-optical**; the extreme emitters are **UV-selected**. Conclusions
cannot be transferred between them without modelling the ionisation physics, as
the prompt warns — and the paper says the same thing.

**The paper's forward model is already specific.** Reproducing its assumptions
rather than choosing our own is the only way to make the comparison meaningful:

| Element | Frozen value from Appendix D |
|---|---|
| Redshift | fixed, single |
| Stellar mass grid | 10^7–10^11 M☉, 0.1 dex bins, uniform in log |
| Realisations | 300 per mass bin |
| Star-forming sequence | DiCesare+26, σ_MS = 0.37 dex |
| Mass–metallicity relation | Curti+24 (z = 3–6 branch), σ = 0.07 dex; cross-check Sanders+21 |
| Hβ from SFR | Kennicutt+98, Chabrier IMF |
| [O II], [O III] | Cataldi+25 strong-line calibrations (R2, R3); cross-check Sanders+26 |
| [O III]4363 | this work's empirical calibration vs O/H (2nd-order polynomial) |
| Detection | S/N > 3 in the diagnostic line **and** in [O III]4363 |
| Exposure | 5 h NIRSpec, "median exposure time of the DJA programmes" |

## 2. Why a naive Plan B would produce a misleading answer

Three failure modes, each of which the prompt's design would hit:

1. **Deriving the null population from the selected sample.** If the null is
   built from the detected galaxies, it already carries the selection and the
   test is circular — it will always find "no bias". The null must come from the
   MZR and the star-forming sequence, never from the observed sample.
2. **One representative exposure time.** The prompt is right that depths must be
   sampled. The source uses a single 5 h figure as representative; using the
   actual per-programme depth distribution is a genuine improvement and is the
   one place where this work can add value over Appendix D.
3. **Ignoring nondetections.** The whole effect lives in the nondetected tail.
   A completeness-corrected comparison that conditions on detection reproduces
   the bias it is meant to measure.

**Data blocker, independent of the environment:** injecting into "representative
real residuals, ideally before extraction" requires real 2D NIRSpec residuals
from the DJA/MARTA programmes. That needs the same missing environment as
Plan A. Injection into *simulated* noise is possible but forfeits the main
advantage over the published forward model, which is already simulation-based.

## 3. What a legitimate, non-duplicative Plan B looks like

Kept deliberately narrow, and gated on Plan A producing usable noise:

- **Scope to one diagnostic**, rest-optical N2O2 (N/O via [N II]/[O II]) **or**
  the Te-based N/O — not both, and not the UV N IV] lines.
- **Differentiate from Appendix D by mechanism, not by repetition**: inject into
  real 2D residuals and run the *unchanged* extraction+fit pipeline, so the
  selection includes extraction losses, deblending failures and covariance —
  none of which a flux-level forward model contains. That is the defensible
  increment.
- **Predeclare** the null population, relation, scatter, thresholds and
  simulation count before running (`04` gives the pattern).
- **Report** completeness, false-positive rate, flux bias, interval coverage and
  the apparent N/O shift induced by selection, with Monte Carlo uncertainty.
- **Bounded conclusion**: whether selection can explain the effect *under the
  tested assumptions* — never "selection explains N enhancement".

## 4. Verdict

**Deferred — not blocked on science, blocked on the same environment, and
reduced in novelty by a published result.**

The honest sequencing is: Plan A first (it produces the noise model and the
validated extraction), then decide whether a residual-selection replication adds
enough to justify the compute. On current evidence it adds one real thing —
extraction-level selection with real covariances — and duplicates the rest.
