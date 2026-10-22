# JWST/PRISMS — U13151 reproduction (Plan A) and N/O selection test (Plan B)

**Scope warning, read first.** This directory does not belong to rule30-foundry.
It holds work prepared for a JWST spectroscopy environment
(`JWST_scnience_env_2`) that **is not present on this machine**. There is no
`jwst`, `astropy`, `astroquery` or `numpy` installed, no James Webb data, and no
reduction pipeline. Everything here was produced without touching JWST data.

The material is parked in its own directory so it can be lifted out whole:

```bash
git archive cline/9xa6tztz docs/jwst-prisms tools/jwst_u13151_consistency.py \
    tests/test_jwst_u13151_consistency.py | tar -x -C /path/to/JWST_env_2
```

Nothing outside these paths depends on it, and it registers no stage in
`tools/verify_all.py`. `python tools/lint_ledger.py` and
`tools/lint_bitorder.py` stay green with it in place (checked).

## What was actually achieved, and what was not

| Item | Status |
|---|---|
| Repository inventory of the intended environment | **Blocked** — the environment does not exist here |
| Both arXiv sources verified against primary text (not abstracts) | **Done** |
| Published measurements extracted with provenance | **Done** — see `02-repro-spec-u13151.md` |
| Frozen spec checked for internal consistency, executable here | **Done** — `tools/jwst_u13151_consistency.py`, 12 tests |
| Plan A data inventory and download estimate | **Blocked** — needs MAST access from the real environment; procedure specified |
| Plan A reproduction (redshift, fluxes) | **Blocked** — see `03-feasibility-and-budget.md` |
| Plan B selection test | **Deferred, with a finding**: the source paper already contains this forward model — `05-plan-b-selection.md` |

There is no astrophysical result in this directory. There is a verified target,
a specification, a predeclared acceptance criterion, and two blockers named
precisely enough to attack in one afternoon on the right machine.

## Index

| File | Contents |
|---|---|
| `01-source-verification.md` | Both papers verified: title, ID, date, version, method, quoted numbers, data availability. Includes the four claims in the task prompt that **failed** verification. |
| `02-repro-spec-u13151.md` | The frozen reproduction specification: PIDs, reduction software, apertures, masks, normalisation, line model, every published flux, plus five open questions that must be closed before fitting. |
| `03-feasibility-and-budget.md` | What is missing, how big it is, how to measure that without guessing, and the two software blockers including the ~5.5 µm calibration question. |
| `04-predeclared-criteria.md` | Acceptance criteria frozen **before** any result exists: recovery tolerances, injection–recovery coverage, and the nod/extraction variation that would invalidate a quoted uncertainty. |
| `05-plan-b-selection.md` | Why Plan B as specified duplicates the source paper's Appendix D, and the bounded redesign that does not. |
| `CHECKPOINT.md` | Branch/SHA, commands that ran, outputs, failures, single next action. |

## The one-line summary

Plan A is **blocked on environment, not on science**: the target is fully
specified and internally consistent, the data are public, but the published
NIRSpec reduction depends on a NIRSpec-GTO pipeline realisation that extends the
traces to 5.5 µm — without it the [O III] 4959,5007 detection that anchors
z = 9.7995 is out of reach, so a stock-pipeline attempt is a **partial
reproduction** by construction and must be labelled as one.
